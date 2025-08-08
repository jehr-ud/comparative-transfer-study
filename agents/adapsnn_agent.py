from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.learning import STDPLearner
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import pickle
import random
import os
from pathlib import Path
from collections import deque, defaultdict


class GridCellSystem:
    def __init__(self, height, width, n_grids=12, device='cpu'):
        self.height = height
        self.width = width
        self.n_grids = n_grids
        self.device = device

        # We pre-compute the activation maps to speed up the process
        print("Pre-computing Grid Cell activation maps...")
        self.activation_maps = self._precompute_maps()
        print("Grid Cell maps ready.")

    def _precompute_maps(self):
        # We generate the parameters for the grids
        scales = np.random.uniform(3, 10, self.n_grids)
        orientations = np.random.uniform(0, np.pi / 3, self.n_grids)
        phases = np.random.uniform(0, 2 * np.pi, (self.n_grids, 2))

        # We create a tensor to store all activation maps
        maps = torch.zeros(self.height, self.width, self.n_grids)

        # We create a coordinate grid for the entire maze
        y, x = np.meshgrid(
            np.arange(self.height),
            np.arange(self.width),
            indexing='ij'
        )
        pos_grid = np.stack([y.ravel(), x.ravel()], axis=1)

        for k in range(self.n_grids):
            # We rotate the coordinates according to the grid's orientation
            rot_matrix = np.array(
                [
                    [np.cos(orientations[k]), -np.sin(orientations[k])],
                    [np.sin(orientations[k]), np.cos(orientations[k])]
                ]
            )
            rotated_grid = (rot_matrix @ pos_grid.T).T

            # We use a cosine wave function to simulate the hexagonal pattern
            # (This is a standard mathematical approximation)
            term1 = np.cos(
                2 * np.pi * rotated_grid[:, 0] / scales[k] + phases[k, 0]
            )
            term2 = np.cos(2 * np.pi * (-0.5 * rotated_grid[:, 0] + np.sqrt(3)/2 * rotated_grid[:, 1]) / scales[k] + phases[k, 1])
            term3 = np.cos(2 * np.pi * (-0.5 * rotated_grid[:, 0] - np.sqrt(3)/2 * rotated_grid[:, 1]) / scales[k] + phases[k, 0])

            # Activation is high if the position is on a grid "hotspot"
            activation = (term1 + term2 + term3).reshape(
                self.height,
                self.width
            )
            maps[:, :, k] = torch.tensor(activation)

        return maps.to(self.device)

    def get_grid_cell_input(self, pos):
        # Now it's a super-fast lookup in the pre-computed map
        y, x = int(pos[0]), int(pos[1])
        # We sum the activations of all grids for the position (y, x)
        total_grid_activation = torch.sum(self.activation_maps[y, x, :])

        # We only consider positive activations and normalize them
        return max(0, total_grid_activation.item()) / self.n_grids


class Hippocampus:
    """
    Represents the fast memory system (Hippocampus and associated cortices).

    Manages the spiking neural network (SNN) for spatial mapping, including
    Place Cells, Grid Cells, and synaptic learning via STDP.
    """

    def __init__(self, height, width, device, T=100, input_current=1.0,
                 reward_bonus=0.05):
        """Initializes the SNN brain components."""
        self.device = device
        self.maze_height = height
        self.maze_width = width
        self.T = T
        self.input_current = input_current
        self.reward_bonus = reward_bonus

        # Neuronal Components
        num_place_cells = self.maze_height * self.maze_width
        self.place_cells = neuron.LIFNode(tau=2.0, v_threshold=1.0)
        self.synaptic_layer = nn.Linear(
            num_place_cells, num_place_cells, bias=False
        )
        torch.nn.init.constant_(self.synaptic_layer.weight, 0.0)

        # Learning System
        self.stdp_learner = STDPLearner(
            step_mode='s', synapse=self.synaptic_layer, sn=self.place_cells,
            tau_pre=2.0, tau_post=2.0
        )

        # Geometric Perception System
        self.grid_cell_system = GridCellSystem(
            self.maze_height, self.maze_width, device=self.device
        )

        # Move all PyTorch modules to the correct device
        self.to(self.device)

    def to(self, device):
        """Moves all PyTorch components to the specified device."""
        self.place_cells.to(device)
        self.synaptic_layer.to(device)
        # Note: STDPLearner doesn't have a .to() method, it operates on layers
        # that have already been moved.
        self.grid_cell_system.device = device
        self.grid_cell_system.activation_maps = \
            self.grid_cell_system.activation_maps.to(device)
        return self

    def process_sensory_input(self, position):
        """
        Processes the current location, runs the SNN simulation, and the
        STDP learning.
        """
        num_neurons = self.maze_height * self.maze_width
        current_idx = self._pos_to_idx(position)

        place_cell_current = torch.zeros(num_neurons, device=self.device)
        place_cell_current[current_idx] = self.input_current

        grid_cell_current = self.grid_cell_system.get_grid_cell_input(position)

        total_external_input = place_cell_current + grid_cell_current

        self.place_cells.reset()
        self.stdp_learner.reset()
        previous_spikes = torch.zeros(1, num_neurons, device=self.device)

        for _ in range(self.T):
            recurrent_input = self.synaptic_layer(previous_spikes)
            total_input = total_external_input.unsqueeze(0) + recurrent_input
            current_spikes = self.place_cells(total_input)
            self.stdp_learner.step()
            previous_spikes = current_spikes

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

            activation_score = -float('inf') if neighbor_pos == prev_pos \
                else downstream_activity[0, self._pos_to_idx(neighbor_pos)].item()

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
            'stdp_learner_state_dict': self.stdp_learner.state_dict(),
        }

    def load_state(self, state):
        """Loads the state into the PyTorch modules."""
        self.place_cells.load_state_dict(state['place_cells_state_dict'])
        self.synaptic_layer.load_state_dict(state['synaptic_layer_state_dict'])
        self.stdp_learner.load_state_dict(state['stdp_learner_state_dict'])

    def _pos_to_idx(self, pos):
        """Converts (row, col) coordinates to a flat neuron index."""
        return int(pos[0]) * self.maze_width + int(pos[1])

    def _idx_to_pos(self, idx):
        """Converts a neuron index to (row, col) coordinates."""
        return (idx // self.maze_width, idx % self.maze_width)


class Neocortex:
    """
    Represents the slow memory system and executive functions (Neocortex).

    Manages episodic memory, memory consolidation, and high-level
    exploration and decision strategies.
    """

    def __init__(self):
        """Initializes memories and strategies."""
        self.cortical_memory = []  # Episodic memory of successful routes
        self.familiarity = defaultdict(int)  # Familiarity memory

    def consolidate_memory(self, path, total_reward, difficulty):
        """Saves a route to episodic memory if it is efficient."""
        if not path or len(path) <= 1:
            return
        if total_reward < 0 and difficulty > 5:
            return

        efficiency = total_reward / len(path)
        self.cortical_memory.append((path, efficiency))
        self.cortical_memory.sort(key=lambda x: -x[1], reverse=True)
        if len(self.cortical_memory) > 100:
            self.cortical_memory = self.cortical_memory[:100]

    def get_exploration_action(self, position, valid_actions, prev_pos):
        """Chooses an action for exploration, prioritizing novelty."""
        scored_moves = []
        action_deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        for action in valid_actions:
            dy, dx = action_deltas[action]
            neighbor = (position[0] + dy, position[1] + dx)
            score = -1.0 if neighbor == prev_pos \
                else 1 / (1 + self.familiarity.get(neighbor, 0) ** 2)
            scored_moves.append((score, action))

        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves[0][1]

    def get_state(self):
        """Returns the state of the cortical memories."""
        return {
            'cortical_memory': self.cortical_memory,
            'familiarity': dict(self.familiarity)
        }

    def load_state(self, state):
        """Loads the state into the cortical memories."""
        self.cortical_memory = state.get('cortical_memory', [])
        self.familiarity = defaultdict(int, state.get('familiarity', {}))


class AdapSNNAgent:
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

        # Initialize the two brain systems
        self.hippocampus = Hippocampus(
            height=self.env.size,
            width=self.env.size,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.neocortex = Neocortex()

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
            self.last_position = tuple(obs)
            self.previous_position = None
            total_reward, path = 0, [self.last_position]
            terminated, truncated = False, False

            while not (terminated or truncated):
                # 1. The Hippocampus processes the input and learns with STDP
                self.hippocampus.process_sensory_input(self.last_position)

                # 2. The Neocortex decides the exploration action
                valid_actions = self.env.get_possible_actions(self.last_position)
                if not valid_actions:
                    valid_actions = list(range(4))

                if random.random() < epsilon:
                    action = self.neocortex.get_exploration_action(
                        self.last_position, valid_actions, self.previous_position
                    )
                else:
                    action = self.hippocampus.get_local_prediction(
                        self.last_position, valid_actions, self.previous_position
                    )

                # 3. Interaction with the environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_pos = tuple(next_obs)

                # 4. Update state and familiarity memory
                self.neocortex.familiarity[self.last_position] += 1
                total_reward += reward
                path.append(next_pos)
                self.previous_position = self.last_position
                self.last_position = next_pos

            # 5. Post-Episode Consolidation and Reinforcement
            if terminated:
                self.neocortex.consolidate_memory(path, total_reward, self.difficulty)
                self.hippocampus.reinforce_path(path)

            return total_reward

    def predict(self, obs):
        """Makes the best possible decision in exploitation mode."""
        current_pos = tuple(obs)
        valid_actions = self.env.get_possible_actions(current_pos)

        # Level 1: Neocortex - Is there a memorized route?
        for path, _ in self.neocortex.cortical_memory:
            if current_pos in path:
                idx = path.index(current_pos)
                if idx + 1 < len(path):
                    next_pos = path[idx + 1]
                    action = self._direction_from_to(current_pos, next_pos)
                    if action in valid_actions:
                        self.previous_position = current_pos
                        return action

        # Level 2: Neocortex asks the Hippocampus for a long-term plan
        if self.current_plan is None or current_pos not in self.current_plan:
            self.current_plan = self.hippocampus.plan_path(current_pos, self.goal)

        if self.current_plan and len(self.current_plan) > 1:
            next_pos = self.current_plan[1]
            action = self._direction_from_to(current_pos, next_pos)
            if action in valid_actions:
                self.previous_position = current_pos
                return action

        # Level 3: Neocortex asks the Hippocampus for a local decision
        action = self.hippocampus.get_local_prediction(
            current_pos, valid_actions, self.previous_position
        )
        self.previous_position = current_pos
        return action

    def save(self, save_dir=None):
        """Saves the state of both brain systems."""
        if save_dir is None:
            save_dir = self.save_path
        file_path = Path(save_dir) / f"{self.name}.pth"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'hippocampus_state': self.hippocampus.get_state(),
            'neocortex_state': self.neocortex.get_state(),
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
        self.neocortex.load_state(state['neocortex_state'])
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

            # Metric 2: Average strength of those connections (measures map maturity)
            avg_strength = weights[strong_connections].mean().item() if num_strong_connections > 0 else 0.0

            # Metric 3: Number of neural "hubs" (measures intersection complexity)
            outgoing_strength = weights.sum(dim=0)
            hubs_threshold = outgoing_strength.mean() + outgoing_strength.std()
            num_hubs = (outgoing_strength > hubs_threshold).sum().item()

            # The fingerprint is a tuple of these metrics, rounded for comparison.
            fingerprint = (num_strong_connections, round(avg_strength, 2), num_hubs)
            return fingerprint

    def _direction_from_to(self, current, next_pos):
        """Helper function to calculate the action from movement."""
        dy = next_pos[0] - current[0]
        dx = next_pos[1] - current[1]
        action_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
        return action_map.get((dy, dx), random.randint(0, 3))


def calculate_fingerprint_distance(fp1, fp2):
    """Calcula la distancia euclidiana entre dos huellas digitales."""
    return np.linalg.norm(np.array(fp1) - np.array(fp2))


class PrefrontalCortex:
    """
    Orquestador de alto nivel que gestiona el aprendizaje y la adaptación
    del agente a través de múltiples tareas.
    """

    def __init__(self, atlas_path="snn_atlas.pkl"):
        self.atlas_path = Path(atlas_path)
        self.cerebral_atlas = self._load_atlas()
        self.active_agent = None  # El agente listo para la tarea actual
        print(
            f"Cerebral Atlas initialized with {len(self.cerebral_atlas)} "
            "known mazes."
        )

    def _load_atlas(self):
        if self.atlas_path.exists():
            with open(self.atlas_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_atlas(self):
        with open(self.atlas_path, 'wb') as f:
            pickle.dump(self.cerebral_atlas, f)
        print(f"Cerebral Atlas updated and saved to {self.atlas_path}")

    def handle_task(self, maze_name, env_info, identification_episodes=30):
        """
        Prepara a un agente para una nueva tarea usando la jerarquía completa:
        Reconocimiento, Adaptación (Fine-Tuning), o Aprendizaje desde Cero.
        """
        print(f"\n--- HANDLING NEW TASK: Maze '{maze_name}' ---")

        # 1. Fase de Identificación con un agente de diagnóstico
        diag_agent = AdapSNNAgent(
            name=f"diag_{maze_name}",
            difficulty=env_info['env'].size,
            env_info=env_info
        )
        print(f"Starting identification phase ({identification_episodes} episodes)...")
        for _ in range(identification_episodes):
            diag_agent.train(epsilon=0.95)

        fingerprint = diag_agent.calculate_fingerprint()
        print(f"Fingerprint for '{maze_name}' calculated: {fingerprint}")

        # 2. Jerarquía de Decisión Estratégica
        # Nivel 1: Reconocimiento Exacto
        if fingerprint in self.cerebral_atlas:
            model_path = self.cerebral_atlas[fingerprint]
            print(f"CONTEXT RECOGNIZED! Loading expert brain from: {model_path}")
            # Re-instanciamos el agente con su nombre correcto para cargar
            self.active_agent = AdapSNNAgent(
                name=Path(model_path).stem,
                difficulty=env_info['env'].size,
                env_info=env_info
            )
            self.active_agent.load(load_dir=Path(model_path).parent)
            print("Expert agent is now active.")
            return

        # Nivel 2: Transferencia por Similitud (Fine-Tuning)
        if self.cerebral_atlas:
            best_match_fp, min_distance = None, float('inf')
            for known_fp, path_str in self.cerebral_atlas.items():
                distance = calculate_fingerprint_distance(fingerprint, known_fp)
                if distance < min_distance:
                    min_distance = distance
                    best_match_fp = known_fp

            model_path = self.cerebral_atlas[best_match_fp]
            print(f"NEW CONTEXT. Most similar brain found (Dist: {min_distance:.2f}).")
            print(f"Performing 'brain transplant' from: {model_path}")

            # Re-instanciamos el agente con el nombre del *nuevo* laberinto
            # pero cargamos el cerebro del *donante*.
            self.active_agent = AdapSNNAgent(
                name=f"agent_{maze_name}",
                difficulty=env_info['env'].size,
                env_info=env_info
            )
            self.active_agent.load(load_dir=Path(model_path).parent)
            print("Brain transplant complete. Agent is ready for fine-tuning.")
            return

        # Nivel 3: Aprendizaje desde Cero
        print("Atlas is empty or no similar brain found. Preparing for learning from scratch.")
        self.active_agent = AdapSNNAgent(
            name=f"agent_{maze_name}",
            difficulty=env_info['env'].size,
            env_info=env_info
        )
        print("New agent is ready for training.")

    def train_active_agent(self, maze_name, training_episodes=200):
        """
        Ejecuta el bucle de entrenamiento completo para el agente que ya fue preparado.
        """
        if not self.active_agent:
            raise RuntimeError("No active agent. Call handle_task first.")

        print(f"\n--- Starting Training Session for '{maze_name}' ---")
        episode_rewards = []
        initial_epsilon = 1.0
        min_epsilon = 0.01
        epsilon_decay_rate = 0.995
        current_epsilon = initial_epsilon

        for i in range(training_episodes):
            current_epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay_rate ** i))
            reward = self.active_agent.train(current_epsilon)
            episode_rewards.append(reward)

        print(f"Training complete for '{maze_name}'.")

        # Guardar el cerebro recién entrenado/adaptado y actualizar el Atlas
        fingerprint = self.active_agent.calculate_fingerprint()
        expert_dir = self.active_agent.save_path / f"expert_{maze_name}"
        self.active_agent.save(save_dir=expert_dir)
        self.cerebral_atlas[fingerprint] = str(expert_dir)
        self._save_atlas()

        return self, episode_rewards

    def predict(self, obs, **kwargs):
        """Delega la llamada de predicción al agente activo."""
        if self.active_agent is None:
            raise RuntimeError("No active agent. Call handle_task to prepare an agent.")
        return self.active_agent.predict(obs)

    def reset(self):
        """Resetea el estado del agente activo."""
        if self.active_agent:
            self.active_agent.reset()