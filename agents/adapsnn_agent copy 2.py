from collections import defaultdict, deque
import json
import random
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron


class GridCellSystem:
    """
    Simulates the medial entorhinal cortex to provide a geometric sense of space.
    """
    def __init__(self, height, width, n_grids=12, device='cpu'):
        self.height = height
        self.width = width
        self.n_grids = n_grids
        self.device = device

        print("Pre-computing Grid Cell activation maps...")
        self.activation_maps = self._precompute_maps()
        print("Grid Cell maps ready.")

    def _precompute_maps(self):
        """
        Generates the parameters and activation maps for multiple grids
        with different scales and orientations.
        """
        rng = np.random.default_rng(seed=42)
        scales = rng.uniform(3, 10, self.n_grids)
        orientations = rng.uniform(0, np.pi / 3, self.n_grids)
        phases = rng.uniform(0, 2 * np.pi, (self.n_grids, 2))

        maps = torch.zeros(self.height, self.width, self.n_grids)

        y, x = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        pos_grid = np.stack([y.ravel(), x.ravel()], axis=1)

        for k in range(self.n_grids):
            rot_matrix = np.array([
                [np.cos(orientations[k]), -np.sin(orientations[k])],
                [np.sin(orientations[k]), np.cos(orientations[k])]
            ])
            rotated_grid = (rot_matrix @ pos_grid.T).T

            pi2 = 2 * np.pi
            term1 = np.cos(pi2 * rotated_grid[:, 0] / scales[k] + phases[k, 0])
            term2 = np.cos(pi2 * (-0.5 * rotated_grid[:, 0] + np.sqrt(3)/2 * rotated_grid[:, 1]) / scales[k] + phases[k, 1])
            term3 = np.cos(pi2 * (-0.5 * rotated_grid[:, 0] - np.sqrt(3)/2 * rotated_grid[:, 1]) / scales[k] + phases[k, 0])

            activation = (term1 + term2 + term3).reshape(self.height, self.width)
            maps[:, :, k] = torch.tensor(activation, dtype=torch.float32)

        return maps.to(self.device)

    def get_grid_cell_input(self, pos):
        """
        Calculates the grid cell input current for a given position.
        Returns a tensor of activations for all neurons.
        """
        y, x = int(pos[0]), int(pos[1])

        # Get the scalar activation value for the current position
        total_grid_activation_value = torch.sum(self.activation_maps[y, x, :])
        normalized_value = max(0, total_grid_activation_value.item()) / self.n_grids

        # Return a constant vector modulated by the activation value,
        # acting as a global background signal.
        return torch.full((self.height * self.width,), normalized_value, device=self.device)


class Hippocampus:
    """
    Represents the fast memory system (Hippocampus and associated cortices).

    Manages the spiking neural network (SNN) for spatial mapping, including
    Place Cells, Grid Cells, and synaptic learning via a manual STDP rule.
    """

    def __init__(self, height, width, feature_vector_size, device, T=100,
                 input_current=1.0, reward_bonus=0.05):
        """Initializes the SNN brain components."""
        self.device = device
        self.maze_height = height
        self.maze_width = width
        self.T = T
        self.input_current = input_current
        self.reward_bonus = reward_bonus

        num_place_cells = self.maze_height * self.maze_width
        self.num_neurons = num_place_cells

        # Neuronal Components
        self.sensory_projection = nn.Linear(feature_vector_size, num_place_cells)
        self.sensory_projection.weight.requires_grad = False
        
        self.place_cells = neuron.LIFNode(tau=2.0, v_threshold=1.0)
        
        self.synaptic_layer = nn.Linear(num_place_cells, num_place_cells, bias=False)
        torch.nn.init.constant_(self.synaptic_layer.weight, 0.0)

        # Manual STDP Components
        self.tau_pre = 2.0
        self.tau_post = 2.0
        self.lr_stdp = 1e-4  # STDP learning rate
        self.trace_pre = torch.zeros(1, num_place_cells, device=self.device)
        self.trace_post = torch.zeros(1, num_place_cells, device=self.device)

        # Geometric Perception System
        self.grid_cell_system = GridCellSystem(
            self.maze_height, self.maze_width, device=self.device
        )
        self.to(self.device)

    def to(self, device):
        """Moves all PyTorch components to the specified device."""
        self.sensory_projection.to(device)
        self.place_cells.to(device)
        self.synaptic_layer.to(device)
        # Note: STDPLearner doesn't have a .to() method, it operates on layers
        # that have already been moved.
        self.grid_cell_system.device = device
        self.grid_cell_system.activation_maps = \
            self.grid_cell_system.activation_maps.to(device)
        return self

    def process_sensory_input(self, sensory_input):
        """
        Procesa la entrada sensorial para que siempre tenga forma (1, num_neurons).

        sensory_input: tensor o array de cualquier forma
        """
        # Convertir a tensor si no lo es
        if not torch.is_tensor(sensory_input):
            sensory_input = torch.tensor(sensory_input, dtype=torch.float32)

        # Aplanar todo excepto batch
        sensory_input = sensory_input.flatten()

        # Ajustar tamaño según num_neurons
        if sensory_input.numel() > self.num_neurons:
            sensory_input = sensory_input[:self.num_neurons]  # recorta si hay más
        elif sensory_input.numel() < self.num_neurons:
            pad_size = self.num_neurons - sensory_input.numel()
            sensory_input = torch.cat([sensory_input, torch.zeros(pad_size)])  # rellena con ceros

        # Asegurar forma (1, num_neurons)
        sensory_input = sensory_input.view(1, self.num_neurons)

        return sensory_input

    @torch.no_grad()
    def get_local_prediction(self, current_pos, valid_actions, prev_pos):
        """Calculates the best local action based on synaptic strength."""
        num_neurons = self.maze_height * self.maze_width
        current_idx = self._pos_to_idx(current_pos)

        thought_spike = torch.zeros(1, num_neurons, device=self.device)
        thought_spike[0, current_idx] = 1.0
        downstream_activity = self.synaptic_layer(thought_spike)

        best_action, max_activation = -1, -float('inf')
        action_deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        for action in valid_actions:
            dy, dx = action_deltas[action]
            neighbor_pos = (current_pos[0] + dy, current_pos[1] + dx)
            
            activation_score = -float('inf') if neighbor_pos == prev_pos else \
                               downstream_activity[0, self._pos_to_idx(neighbor_pos)].item()

            if activation_score > max_activation:
                max_activation = activation_score
                best_action = action

        return best_action if best_action != -1 else random.choice(valid_actions)

    def plan_path(self, start_pos, goal_pos, weight_threshold=0.1):
        """Finds the shortest path using BFS on the learned synaptic map."""
        if start_pos is None or goal_pos is None:
            return None

        start_idx = self._pos_to_idx(start_pos)
        goal_idx = self._pos_to_idx(goal_pos)

        with torch.no_grad():
            adj_matrix = (self.synaptic_layer.weight.data > weight_threshold)

        queue = deque([(start_idx, [start_idx])])
        visited = {start_idx}

        while queue:
            current_idx, path_indices = queue.popleft()
            if current_idx == goal_idx:
                return [self._idx_to_pos(i) for i in path_indices]

            neighbors = torch.where(adj_matrix[:, current_idx])[0]
            for neighbor_idx in neighbors:
                if neighbor_idx.item() not in visited:
                    visited.add(neighbor_idx.item())
                    new_path = path_indices + [neighbor_idx.item()]
                    queue.append((neighbor_idx.item(), new_path))
        return None

    def reinforce_path(self, path):
        """Strengthens the synapses along a successful path."""
        with torch.no_grad():
            for i in range(len(path) - 1):
                idx_a = self._pos_to_idx(path[i])
                idx_b = self._pos_to_idx(path[i+1])
                current_weight = self.synaptic_layer.weight.data[idx_b, idx_a]
                self.synaptic_layer.weight.data[idx_b, idx_a] = torch.clamp(
                    current_weight + self.reward_bonus, max=1.0
                )

    def get_state(self):
        """Returns the state of all PyTorch modules."""
        return {
            'place_cells_state_dict': self.place_cells.state_dict(),
            'synaptic_layer_state_dict': self.synaptic_layer.state_dict(),
        }

    def load_state(self, state):
        """Loads the state into the PyTorch modules."""
        self.place_cells.load_state_dict(state['place_cells_state_dict'])
        self.synaptic_layer.load_state_dict(state['synaptic_layer_state_dict'])

    def _pos_to_idx(self, pos):
        """Converts (row, col) coordinates to a flat neuron index."""
        return int(pos[0]) * self.maze_width + int(pos[1])

    def _idx_to_pos(self, idx):
        """Converts a neuron index to (row, col) coordinates."""
        return (idx // self.maze_width, idx % self.maze_width)


class SensoryEncoder:
    """ analoguous to visual neocortex
    """
    def encode_to_spikes(self, features, t, dt=0.01, base_freq=5.0):
        """
        Encodes features into spike-like signals.

        features: array from process_sensors
        t: current simulation time (seconds)
        dt: timestep size
        base_freq: base oscillation frequency in Hz
        """
        dist_up, dist_down, dist_left, dist_right, \
            obs_up, obs_down, obs_left, obs_right, dist_goal = features

        # Normalize distances (avoid too large phases)
        max_dist = max(dist_up, dist_down, dist_left, dist_right, 1.0)
        norm_distances = np.array([dist_up, dist_down, dist_left, dist_right]) / max_dist

        # Encode distances into sinusoidal phases (grid-cell like)
        distance_phases = np.sin(2 * np.pi * base_freq * t + norm_distances * np.pi)

        # Encode obstacles as spikes (1 at event time, 0 otherwise)
        obstacles = np.array([obs_up, obs_down, obs_left, obs_right])
        spike_train = (np.random.rand(4) < (obstacles * 0.8)).astype(float)

        # Encode goal distance as frequency modulation
        goal_freq = base_freq + (1.0 / (dist_goal + 0.1)) * 10.0  # closer → faster
        goal_phase = np.sin(2 * np.pi * goal_freq * t)

        # Concatenate all encodings
        encoded = np.concatenate([distance_phases, spike_train, [goal_phase]])
        return encoded

    def process_sensors(self, obs, env):
        """
        obs: posición actual [x, y]
        env: referencia al entorno (VisualMazeEnv)
        Devuelve un vector de características sensoriales simuladas.
        """
        x, y = int(obs[0]), int(obs[1])
        size = env.size

        # Distances to edges (like proximity sensors)
        dist_up = x
        dist_down = size - 1 - x
        dist_left = y
        dist_right = size - 1 - y

        # Immediate obstacle detection (0/1)
        obs_up = 1 if (x-1 >= 0 and env.maze[x-1, y] == 1) else 0
        obs_down = 1 if (x+1 < size and env.maze[x+1, y] == 1) else 0
        obs_left = 1 if (y-1 >= 0 and env.maze[x, y-1] == 1) else 0
        obs_right = 1 if (y+1 < size and env.maze[x, y+1] == 1) else 0

        # Euclidean distance to the goal
        goal_dx = env.goal_pos[0] - x
        goal_dy = env.goal_pos[1] - y
        dist_goal = np.sqrt(goal_dx**2 + goal_dy**2)

        # Vector final
        features = np.array([
            dist_up, dist_down, dist_left, dist_right,
            obs_up, obs_down, obs_left, obs_right,
            dist_goal
        ], dtype=np.float32)

        return features


class Neocortex:
    def __init__(self, map_size=(50, 50)):
        """Initializes cortical memory as a 2D wave map."""
        self.map_size = map_size
        self.cortical_wave = np.zeros(map_size, dtype=float)  # wave-based memory
        self.familiarity = defaultdict(int)  # familiarity count

    def _add_spike(self, pos, amplitude=1.0, sigma=1.5):
        """Adds a Gaussian spike (wave) to the cortical memory."""
        x = np.arange(0, self.map_size[0])
        y = np.arange(0, self.map_size[1])
        X, Y = np.meshgrid(x, y)
        gx, gy = pos
        gauss = amplitude * np.exp(-((X - gx) ** 2 + (Y - gy) ** 2) / (2 * sigma ** 2))
        self.cortical_wave += gauss

    def consolidate_memory(self, path, total_reward, difficulty):
        """Consolidates a path as waves from spikes."""
        if not path or len(path) <= 1:
            return
        if total_reward < 0 and difficulty > 5:
            return

        efficiency = total_reward / len(path)
        amplitude = max(0.1, efficiency)  # avoid zero amplitude

        for step in path:
            self._add_spike(step, amplitude=amplitude)

    def decay_memory(self, decay_rate=0.99):
        """Applies decay to simulate forgetting."""
        self.cortical_wave *= decay_rate

    def get_exploration_action(self, position, valid_actions, prev_pos):
        """Chooses exploration action using low-wave-intensity areas."""
        scored_moves = []
        action_deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        for action in valid_actions:
            dy, dx = action_deltas[action]
            neighbor = (int(position[0] + dy), int(position[1] + dx))
            if 0 <= neighbor[0] < self.map_size[0] and 0 <= neighbor[1] < self.map_size[1]:
                wave_value = self.cortical_wave[neighbor]  # memory strength
                score = -1.0 if neighbor == prev_pos else (1 / (1 + wave_value))
                scored_moves.append((score, action))

        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves[0][1]

    def export_memory(self):
        """Exports cortical wave as JSON-serializable."""
        familiarity_str_keys = {str(k): v for k, v in self.familiarity.items()}

        return json.dumps({
            "cortical_wave": self.cortical_wave.tolist(),
            "familiarity": familiarity_str_keys
        })

    def import_memory(self, json_data):
        """Loads cortical wave and familiarity map from JSON."""
        data = json.loads(json_data)
        self.cortical_wave = np.array(data["cortical_wave"])
        
        familiarity_from_json = data["familiarity"]

        # <<<< CAMBIO: Un parser más robusto para las claves >>>>
        familiarity_tuple_keys = {}
        for k, v in familiarity_from_json.items():
            try:
                # Limpiar el string de paréntesis y espacios, luego dividir
                clean_parts = k.strip("() ").split(',')
                # Convertir cada parte a float y luego a int
                key_tuple = tuple(map(int, map(float, clean_parts)))
                familiarity_tuple_keys[key_tuple] = v
            except (ValueError, TypeError):
                # Ignorar claves mal formadas si las hubiera
                print(f"Skipping malformed key: {k}")
                continue
        
        self.familiarity = defaultdict(int, familiarity_tuple_keys)


class PrefrontalCortex:
    """
    Neuromorphic agent that combines a fast memory system (Hippocampus SNN)
    and an executive function system (Neocortex).
    """

    def __init__(self, name, difficulty, env_info, save_path="./models"):
        self.name = name
        self.difficulty = difficulty
        self.env = env_info.get('env')
        self.goal = self.env.goal_pos
        self.save_path = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)

        feature_vector_size = 9
        
        # Initialize the two brain systems
        self.hippocampus = Hippocampus(
            height=self.env.size,
            width=self.env.size,
            feature_vector_size=feature_vector_size,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.neocortex = Neocortex()
        self.sensory_encoder = SensoryEncoder()

        # State variables for agent control
        self.last_position = None
        self.previous_position = None
        self.current_plan = None

    def reset(self):
        """Resets the agent's state for a new episode."""
        self.current_plan = None
        self.previous_position = None

    def train(self, epsilon=0.1):
        """Runs a full episode of training and learning."""
        with torch.no_grad():
            obs, _ = self.env.reset()
            self.last_position = tuple(map(int, obs))
            self.previous_position = None
            total_reward, path = 0, [self.last_position]
            terminated, truncated = False, False

            while not (terminated or truncated):
                # 1. The Hippocampus processes the input and learns with STDP
                static_features = self.sensory_encoder.process_sensors(
                    self.last_position, self.env
                )
                time_series_input = torch.zeros(
                    self.hippocampus.T,
                    static_features.shape[0],
                    device=self.hippocampus.device
                )

                dt = 0.01  # Timestep de la simulación SNN
                for t_snn in range(self.hippocampus.T):
                    encoded_slice = self.sensory_encoder.encode_to_spikes(
                        static_features,
                        t=t_snn * dt
                    )
                    time_series_input[t_snn] = torch.from_numpy(
                        encoded_slice
                    ).to(self.hippocampus.device)

                self.hippocampus.process_sensory_input(
                    time_series_input
                )

                # 2. The Neocortex decides the exploration action
                valid_actions = self.env.get_possible_actions(
                    self.last_position
                )
                if not valid_actions:
                    valid_actions = list(range(4))

                if random.random() < epsilon:
                    action = self.neocortex.get_exploration_action(
                        self.last_position,
                        valid_actions,
                        self.previous_position
                    )
                else:
                    action = self.hippocampus.get_local_prediction(
                        self.last_position,
                        valid_actions,
                        self.previous_position
                    )

                # 3. Interaction with the environment
                next_obs, reward, terminated, truncated, _ = self.env.step(
                    action
                )
                next_pos = tuple(map(int, next_obs))

                # 4. Update state and familiarity memory
                self.neocortex.familiarity[self.last_position] += 1
                total_reward += reward
                path.append(next_pos)
                self.previous_position = self.last_position
                self.last_position = next_pos

            # 5. Post-Episode Consolidation and Reinforcement
            if terminated:
                print("FINISHED - goal reached")
                self.neocortex.consolidate_memory(
                    path,
                    total_reward,
                    self.difficulty
                )
                self.hippocampus.reinforce_path(path)

            return total_reward

    def _get_cortical_guidance(self, current_pos, valid_actions):
        """
        Searches the neocortex's wave memory to find the most promising
        next step based on past successful paths.
        """
        best_action = None
        max_wave_intensity = -1 # Start with a value lower than any possible wave intensity

        action_deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        for action in valid_actions:
            dy, dx = action_deltas[action]
            neighbor = (int(current_pos[0] + dy), int(current_pos[1] + dx))
            
            # Check if the neighbor is within the map bounds
            if (0 <= neighbor[0] < self.neocortex.map_size[0] and
                0 <= neighbor[1] < self.neocortex.map_size[1]):
                
                # Get the intensity of the memory wave at the neighbor's location
                wave_intensity = self.neocortex.cortical_wave[neighbor]
                
                if wave_intensity > max_wave_intensity:
                    max_wave_intensity = wave_intensity
                    best_action = action
        
        # Only return an action if we found a "memorized" spot with some intensity
        if max_wave_intensity > 0:
            return best_action
        
        # If all neighbors have zero memory, we have no guidance to offer
        return None

    def predict(self, obs):
        """Makes the best possible decision in exploitation mode."""
        current_pos = tuple(obs)
        valid_actions = self.env.get_possible_actions(current_pos)

        # Level 1: Check for a memorized action.
        action = self._get_cortical_guidance(current_pos, valid_actions)
        if action is not None:
            self.previous_position = current_pos
            return action

        # Level 2: Neocortex asks the Hippocampus for a long-term plan.
        if self.current_plan is None or current_pos not in self.current_plan:
            self.current_plan = self.hippocampus.plan_path(
                current_pos, self.goal
            )

        if self.current_plan and len(self.current_plan) > 1:
            next_pos = self.current_plan[1]
            plan_action = self._direction_from_to(current_pos, next_pos)
            if plan_action in valid_actions:
                self.previous_position = current_pos
                return plan_action

        # Level 3: Neocortex asks the Hippocampus for a local decision.
        local_action = self.hippocampus.get_local_prediction(
            current_pos, valid_actions, self.previous_position
        )
        self.previous_position = current_pos
        return local_action

    def save(self, save_dir=None):
        """Saves the state of both brain systems."""
        if save_dir is None:
            save_dir = self.save_path
        file_path = Path(save_dir) / f"{self.name}.pth"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'hippocampus_state': self.hippocampus.get_state(),
            'neocortex_state': self.neocortex.export_memory(), 
            'agent_name': self.name
        }
        torch.save(state, file_path)
        print(f"Agent saved to {file_path}")

    def load(self, load_dir=None):
        """Loads the state of both brain systems."""
        if load_dir is None:
            load_dir = self.save_path
        file_path = Path(load_dir) / f"{self.name}.pth"

        if not file_path.exists():
            print(f"No save file found at {file_path}.")
            return

        state = torch.load(file_path, map_location=self.hippocampus.device,
                           weights_only=False)
        self.hippocampus.load_state(state['hippocampus_state'])
        self.neocortex.import_memory(state['neocortex_state'])
        print(f"Agent loaded from {file_path}")

    def calculate_fingerprint(self, weight_threshold=0.1):
        """
        Analyzes the synaptic matrix to generate a structural "fingerprint"
        of the maze.
        """
        with torch.no_grad():
            weights = self.hippocampus.synaptic_layer.weight.data

            # Metric 1: Number of strong connections (measures map density)
            strong_connections = weights > weight_threshold
            num_strong_connections = strong_connections.sum().item()

            # Metric 2: Average strength of
            # those connections (measures map maturity)
            weights_cal = weights[strong_connections].mean().item()
            avg_strength = weights_cal if num_strong_connections > 0 else 0.0

            # Metric 3: Number of neural "hubs"
            # (measures intersection complexity)
            outgoing_strength = weights.sum(dim=0)
            hubs_threshold = outgoing_strength.mean() + outgoing_strength.std()
            num_hubs = (outgoing_strength > hubs_threshold).sum().item()

            # The fingerprint is a tuple
            # of these metrics, rounded for comparison.
            fingerprint = (
                num_strong_connections,
                round(avg_strength, 2),
                num_hubs
            )
            return fingerprint

    def _direction_from_to(self, current, next_pos):
        """Helper function to calculate the action from movement."""
        dy = next_pos[0] - current[0]
        dx = next_pos[1] - current[1]
        action_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
        return action_map.get((dy, dx), random.randint(0, 3))
