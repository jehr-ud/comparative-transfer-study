import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.learning import STDPLearner

import random
import os
from pathlib import Path
from collections import deque


class GridCellSystem:
    def __init__(self, height, width, n_grids=12, device='cpu'):
        self.height = height
        self.width = width
        self.n_grids = n_grids
        self.device = device
        self.num_neurons = height * width

        self.scales = np.random.uniform(3, 10, n_grids)  # Diferentes escalas/espaciados
        self.orientations = np.random.uniform(0, np.pi / 3, n_grids) # Diferentes orientaciones
        self.phases = np.random.uniform(0, 2 * np.pi, (n_grids, 2)) # Desfases aleatorios

    def get_grid_cell_input(self, pos):
        """
        Calcula la corriente de entrada de las grid cells para una posición dada.
        """
        pos = np.array(pos)
        total_grid_activation = torch.zeros(self.num_neurons, device=self.device)

        # Se itera sobre todas las neuronas/posiciones del mapa
        for i in range(self.height):
            for j in range(self.width):
                neuron_idx = i * self.width + j
                neuron_pos = np.array([i, j])
                activation = 0

                # Sumamos la activación de cada una de las rejillas
                for k in range(self.n_grids):
                    # Rotamos las coordenadas según la orientación de la rejilla
                    rot_matrix = np.array([[np.cos(self.orientations[k]), -np.sin(self.orientations[k])],
                                           [np.sin(self.orientations[k]), np.cos(self.orientations[k])]])
                    rotated_pos = rot_matrix @ neuron_pos

                    # Usamos una función de onda cosenoidal para simular el patrón hexagonal
                    # (Esta es una aproximación matemática estándar)
                    term1 = np.cos(2 * np.pi * rotated_pos[0] / self.scales[k] + self.phases[k, 0])
                    term2 = np.cos(2 * np.pi * (-0.5 * rotated_pos[0] + np.sqrt(3)/2 * rotated_pos[1]) / self.scales[k] + self.phases[k, 1])
                    term3 = np.cos(2 * np.pi * (-0.5 * rotated_pos[0] - np.sqrt(3)/2 * rotated_pos[1]) / self.scales[k] + self.phases[k, 0])

                    # La activación es alta si la posición está en un "hotspot" de la rejilla
                    activation += (term1 + term2 + term3)

                # Solo consideramos activaciones positivas y las normalizamos
                total_grid_activation[neuron_idx] = max(0, activation)

        # Si una neurona corresponde a la posición actual del agente, su activación se suprime
        # para dar prioridad a la 'place cell'.
        current_idx = int(pos[0]) * self.width + int(pos[1])
        total_grid_activation[current_idx] = 0

        return total_grid_activation / self.n_grids


class SNNCITgent:
    def __init__(self, name, difficulty, env_info, save_path="./snn_models"):
        self.name = name
        self.env = env_info.get('env')
        self.goal = self.env.goal_pos
        self.maze_height = self.env.size
        self.maze_width = self.env.size

        self.save_path = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)

        # --- Parámetros de la Simulación SNN ---
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.T = 100
        self.input_current = 1.0

        # 1. PLACE CELLS
        num_place_cells = self.maze_height * self.maze_width
        self.place_cells = neuron.LIFNode(tau=2.0, v_threshold=1.0).to(self.device)

        # 2. SINAPSIS
        self.synaptic_layer = nn.Linear(num_place_cells, num_place_cells, bias=False).to(self.device)
        torch.nn.init.constant_(self.synaptic_layer.weight, 0.0)

        # 3. APRENDIZAJE (STDP)
        self.stdp_learner = STDPLearner(
            step_mode='s',
            synapse=self.synaptic_layer,  # Le pasamos la capa sináptica
            sn=self.place_cells,         # Le pasamos la capa de neuronas post-sinápticas
            tau_pre=2.0,
            tau_post=2.0
        )

        # --- Variables de estado ---
        self.last_position = None
        self.previous_position = None
        self.current_plan = None  # Para guardar el plan a largo plazo

        self.cortical_memory = []  # Lista de rutas consolidadas

        self.grid_cell_system = GridCellSystem(
            self.maze_height,
            self.maze_width,
            device=self.device
        )

    def _pos_to_idx(self, pos):
        return int(pos[0]) * self.maze_width + int(pos[1])

    def consolidate(self, path, total_reward):
        """Consolida una ruta aprendida si fue eficiente."""
        if not path or len(path) <= 1:
            return

        if total_reward < 0:
            return

        efficiency = total_reward / len(path)
        self.cortical_memory.append((path, efficiency))
        self.cortical_memory.sort(key=lambda x: -x[1])

        # Limitamos la cantidad de recuerdos consolidados
        if len(self.cortical_memory) > 100:
            self.cortical_memory = self.cortical_memory[:100]

    @torch.no_grad()
    def predict_snn_action(self, current_pos, valid_actions, prev_pos=None):
        """
        Usa el mapa sináptico aprendido para decidir la mejor acción a tomar,
        evitando devolverse a la posición anterior.
        """
        num_neurons = self.maze_height * self.maze_width
        current_idx = self._pos_to_idx(current_pos)

        thought_spike = torch.zeros(1, num_neurons, device=self.device)
        thought_spike[0, current_idx] = 1.0

        downstream_activity = self.synaptic_layer(thought_spike)

        best_action = -1
        max_activation = -float('inf')

        if not valid_actions:
            return random.randint(0, 3)

        for action in valid_actions:
            action_deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # Arriba, Derecha, Abajo, Izquierda
            # Corrección del mapeo de acciones
            dy, dx = action_deltas[action]
            neighbor_pos = (current_pos[0] + dy, current_pos[1] + dx)

            # <<<< LÓGICA ANTI-PING-PONG INTEGRADA >>>>
            if neighbor_pos == prev_pos:
                activation_score = -float('inf')  # Se devuelve
            else:
                neighbor_idx = self._pos_to_idx(neighbor_pos)
                activation_score = downstream_activity[0, neighbor_idx].item()

            if activation_score > max_activation:
                max_activation = activation_score
                best_action = action

        if best_action == -1:
            return random.choice(valid_actions)

        return best_action

    def plan_path_snn(self, start_pos, goal_pos, weight_threshold=0.1):
        """
        Usa Búsqueda en Anchura (BFS) sobre el mapa sináptico aprendido para
        encontrar un camino desde el inicio hasta el objetivo.
        """
        print(f"Planificando ruta desde {start_pos} hasta {goal_pos}...")

        start_idx = self._pos_to_idx(start_pos)
        goal_idx = self._pos_to_idx(goal_pos)

        # Obtenemos una versión 'booleana' del mapa sináptico: solo las conexiones fuertes
        with torch.no_grad():
            adj_matrix = (self.synaptic_layer.weight.data > weight_threshold)

        # Cola para BFS: (índice_actual, [camino_de_índices])
        queue = deque([(start_idx, [start_idx])])
        visited = {start_idx}

        while queue:
            current_idx, path_indices = queue.popleft()

            if current_idx == goal_idx:
                # Convertimos el camino de índices a un camino de posiciones
                path_pos = [self._idx_to_pos(i) for i in path_indices]
                print(f"¡Ruta encontrada! Longitud: {len(path_pos)}")
                return path_pos

            # Encontrar vecinos: son las neuronas 'j' a las que la neurona 'current_idx'
            # tiene una conexión fuerte.
            # En nuestra matriz de peso W[i, j], 'j' es pre-sináptica (desde) e 'i' es post-sináptica (hacia).
            # La conexión desde 'current_idx' hacia 'neighbor_idx' está en W[neighbor_idx, current_idx].
            neighbors = torch.where(adj_matrix[:, current_idx])[0]

            for neighbor_idx in neighbors:
                neighbor_idx = neighbor_idx.item()
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    new_path = path_indices + [neighbor_idx]
                    queue.append((neighbor_idx, new_path))

        print("No se pudo encontrar una ruta al objetivo.")
        return None  # no se encuentra camino

    def train(self, epsilon=0.1):
        with torch.no_grad():
            obs, _ = self.env.reset()
            self.last_position = tuple(obs)
            self.previous_position = None

            total_reward = 0
            path = [self.last_position]

            terminated = False
            truncated = False

            while not (terminated or truncated):
                # --- La simulación SNN no cambia ---
                current_idx = self._pos_to_idx(self.last_position)
                num_neurons = self.maze_height * self.maze_width
                input_current_per_step = torch.zeros(
                    num_neurons,
                    device=self.device
                )
                input_current_per_step[current_idx] = self.input_current
                self.place_cells.reset()
                self.stdp_learner.reset()
                previous_step_spikes = torch.zeros(
                    1,
                    num_neurons,
                    device=self.device
                )
                for _ in range(self.T):
                    recurrent_input = self.synaptic_layer(previous_step_spikes)
                    total_input = input_current_per_step.unsqueeze(0) + recurrent_input
                    current_step_spikes = self.place_cells(total_input)
                    self.stdp_learner.step()
                    previous_step_spikes = current_step_spikes

                # --- La selección de acción inteligente (Epsilon-Greedy) no cambia ---
                valid_actions = self.env.get_possible_actions(self.last_position)
                if not valid_actions:
                    valid_actions = list(range(4))

                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    action = self.predict_snn_action(self.last_position, valid_actions)

                # --- Interacción con el entorno ---
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_pos = tuple(next_obs)

                total_reward += reward
                path.append(next_pos)

                self.previous_position = self.last_position
                self.last_position = next_pos

            if terminated:
                print(f"[OK] Meta alcanzada. Reward: {total_reward}. Pasos: {len(path) - 1}")
                self.consolidate(path, total_reward)
            else:
                print(f"[FAIL] No alcanzó la meta. Reward: {total_reward}")

            print(f"Pesos sinápticos máx: {self.synaptic_layer.weight.max().item():.4f}")
            return total_reward

    def predict(self, obs):
        """
        Toma una acción usando una jerarquía:
        1. Sigue un plan a largo plazo si existe y es válido.
        2. Si no, crea un nuevo plan hacia el objetivo.
        3. Si no puede planificar, recurre a la decisión local paso a paso.
        """
        current_pos = tuple(obs)
        valid_actions = self.env.get_possible_actions(current_pos)

        # --- ESTRATEGIA 1: Seguir el plan existente ---
        # Si ya tenemos un plan y todavía estamos en él, lo seguimos.
        if self.current_plan and current_pos in self.current_plan:
            idx = self.current_plan.index(current_pos)
            if idx + 1 < len(self.current_plan):
                next_pos = self.current_plan[idx + 1]
                action = self.direction_from_to(current_pos, next_pos)
                if action in valid_actions:
                    print(f"[Predict] Siguiendo plan: {current_pos} -> {next_pos}")
                    self.previous_position = current_pos
                    return action

        # --- ESTRATEGIA 2: Crear un nuevo plan ---
        # Si no hay plan, o nos hemos desviado, creamos uno nuevo.
        self.current_plan = self.plan_path_snn(current_pos, self.goal)

        # Si se encontró un plan, intentamos seguirlo inmediatamente
        if self.current_plan and len(self.current_plan) > 1:
            next_pos = self.current_plan[1] # El siguiente paso después del actual
            action = self.direction_from_to(current_pos, next_pos)
            if action in valid_actions:
                print(f"[Predict] Nuevo plan creado. Moviendo: {current_pos} -> {next_pos}")
                self.previous_position = current_pos
                return action

        print("[Predict] No se pudo planificar. Usando decisión local...")
        action = self.predict_snn_action(current_pos, valid_actions, prev_pos=self.previous_position)

        self.previous_position = current_pos
        return action

    def direction_from_to(self, current, next_pos):
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]

        if dx == 0 and dy == -1: return 0  # arriba
        if dx == 1 and dy == 0: return 1   # derecha
        if dx == 0 and dy == 1: return 2   # abajo
        if dx == -1 and dy == 0: return 3  # izquierda
        return random.randint(0, 3)

    def save(self, save_dir=None):
        """
        Guarda el estado completo del agente SNN en un directorio específico.
        """
        if save_dir is None:
            save_dir = self.save_path

        file_path = Path(save_dir) / f"{self.name}.pth"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Guardando estado del agente en {file_path}...")

        state = {
            'place_cells_state_dict': self.place_cells.state_dict(),
            'synaptic_layer_state_dict': self.synaptic_layer.state_dict(),
            'stdp_learner_state_dict': self.stdp_learner.state_dict(),
            'cortical_memory': self.cortical_memory,
            'grid_cell_system': self.grid_cell_system,
            'name': self.name,
            'maze_dims': (self.maze_height, self.maze_width)
        }

        try:
            torch.save(state, file_path)
            print("Agente SNN guardado exitosamente.")
        except Exception as e:
            print(f"Error al guardar el agente: {e}")

    def load(self, file_path=None):
        """
        Carga el estado del agente SNN desde un archivo.
        """
        if file_path is None:
            file_path = self.save_path

        file_path = file_path / f"{self.name}.pth"

        if not Path(file_path).exists():
            print(f"No se encontró un archivo de guardado en {file_path}. Iniciando agente desde cero.")
            return

        print(f"Cargando estado del agente desde {file_path}...")

        try:
            state = torch.load(file_path, map_location=self.device, weights_only=False)

            self.place_cells.load_state_dict(state['place_cells_state_dict'])
            self.synaptic_layer.load_state_dict(state['synaptic_layer_state_dict'])
            self.stdp_learner.load_state_dict(state['stdp_learner_state_dict'])
            self.cortical_memory = state.get('cortical_memory', [])
            if 'grid_cell_system' in state:
                self.grid_cell_system = state['grid_cell_system']
            if 'maze_dims' in state:
                loaded_dims = state['maze_dims']
                current_dims = (self.maze_height, self.maze_width)
                if loaded_dims != current_dims:
                    print(f"¡ADVERTENCIA! Dimensiones de laberinto no coinciden.")
            print("Agente SNN cargado exitosamente.")

        except Exception as e:
            print(f"Error al cargar el agente: {e}. Iniciando desde cero.")
