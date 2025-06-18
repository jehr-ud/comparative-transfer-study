import pickle
import random
import os
from pathlib import Path
from collections import deque, defaultdict


class CITgent:
    def __init__(self, name, difficulty, env_info, save_path="./models"):
        self.difficulty = difficulty
        self.model_name = name
        self.save_path = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)

        self.env = env_info.get('env')
        self.env_info = env_info
        self.hippocampus = {}  # Grafo del hipocampo
        self.goal = env_info.get("goal")
        self.last_position = None
        self.explored = set()
        self.memory_replay = []  # Neocorteza: rutas con recompensa
        self.familiarity = defaultdict(int)
        self.cortical_memory = [] # Lista de rutas consolidadas

    def consolidate(self, path, total_reward):
        """Consolida una ruta aprendida si fue eficiente."""
        if not path or len(path) <= 1:
            return
        
        if total_reward < -100:
            return

        efficiency = total_reward / len(path)
        self.cortical_memory.append((path, efficiency))
        self.cortical_memory = sorted(self.cortical_memory, key=lambda x: -x[1])  # Mejor eficiencia primero

        if len(self.cortical_memory) > 100:
            self.cortical_memory = self.cortical_memory[:100]

    def train(self, epsilon=0.1):
        obs, _ = self.env.reset()
        self.last_position = tuple(obs)
        total_reward = 0
        path = [self.last_position]
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = self.explore(self.last_position, epsilon)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_pos = tuple(next_obs)

            self.update_hippocampus(self.last_position, next_pos)
            self.last_position = next_pos
            path.append(next_pos)

            total_reward += reward

        if terminated:
            print(f"[OK] Meta alcanzada. Reward: {total_reward}")
            self.consolidate(path, total_reward)
        else:
            print(f"[FAIL] No alcanzó la meta. Reward: {total_reward}")

        return total_reward

    def predict(self, obs):
        current = tuple(obs)

        # Rutas que incluyen la posición actual
        candidate_paths = [(path, eff) for path, eff in self.cortical_memory if current in path]

        if candidate_paths:
            # Ordenar por (1) posición de `current` en el path (para rutas que lo usan antes)
            # y (2) eficiencia (opcional)
            candidate_paths.sort(key=lambda x: (x[0].index(current), -x[1]))

            best_path, _ = candidate_paths[0]
            idx = best_path.index(current)
            if idx + 1 < len(best_path):
                next_pos = best_path[idx + 1]
                print(f"[DEBUG] Usando ruta consolidada desde {current} a {next_pos}")
                return self.direction_from_to(current, next_pos)

        # Si no hay ruta conocida, usar planificación con el grafo (hipocampo)
        if current not in self.hippocampus:
            return self.explore(current)

        path = self.plan_path(current, self.goal)
        if len(path) < 2:
            return self.explore(current)

        next_pos = path[1]
        return self.direction_from_to(current, next_pos)

    def plan_path(self, start, goal):
        """
        Usa búsqueda en anchura (BFS) sobre el grafo del hipocampo
        para encontrar el camino más corto desde `start` hasta `goal`.
        """
        if start not in self.hippocampus or goal not in self.hippocampus:
            return []

        visited = set()
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path

            for neighbor in self.hippocampus.get(current, {}).get("neighbors", []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # No path found

    def update_hippocampus(self, current, next_pos):
        self.explored.add(current)
        self.familiarity[next_pos] += 1  # +1 cada vez que se visita
        self.familiarity[current] += 1

        if current not in self.hippocampus:
            self.hippocampus[current] = {"neighbors": set()}
        if next_pos not in self.hippocampus:
            self.hippocampus[next_pos] = {"neighbors": set()}

        self.hippocampus[current]["neighbors"].add(next_pos)
        self.hippocampus[next_pos]["neighbors"].add(current)

    def explore(self, position, epsilon=0.1):
        """Explora el entorno considerando solo movimientos válidos y con motivación dopaminérgica."""
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # arriba, derecha, abajo, izquierda

        # 1. Obtener acciones válidas
        valid_actions = self.env.get_possible_actions(position)

        scored_moves = []

        for action in valid_actions:
            dx, dy = directions[action]
            neighbor = (position[0] + dx, position[1] + dy)
            novelty_score = 1 / (1 + self.familiarity.get(neighbor, 0) ** 2)
            scored_moves.append((novelty_score, action))

        # 3. ε-greedy: exploración aleatoria
        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        # 2. Si no hay movimientos válidos, quedarse quieto o devolver random
        if not scored_moves:
            return random.choice([0, 1, 2, 3])


        # 4. Escoge acción con mayor novedad
        scored_moves.sort(reverse=True)  # mayor novedad primero
        return scored_moves[0][1]

    def neocortex_replay(self, start, goal):
        if start == goal or goal is None:
            return [start]

        visited = set()
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.hippocampus.get(current, {}).get("neighbors", []):
                if neighbor == goal:
                    return path + [neighbor]
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return [start]

    def direction_from_to(self, current, next_pos):
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]

        if dx == 0 and dy == -1: return 0  # arriba
        if dx == 1 and dy == 0: return 1   # derecha
        if dx == 0 and dy == 1: return 2   # abajo
        if dx == -1 and dy == 0: return 3  # izquierda
        return random.randint(0, 3)

    def save(self, path=None):
        if not path:
            path = self.save_path
        
        save_path = Path(path) / f"{self.model_name}.pkl"
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump({
                "hippocampus": self.hippocampus,
                "familiarity": dict(self.familiarity),
                "cortical_memory": self.cortical_memory
            }, f)

    def load(self, path=None):
        if not path:
            path = self.save_path

        save_path = Path(path) / f"{self.model_name}.pkl"

        if not save_path.exists():
            print(f"No se encontró archivo en {save_path}, comenzando desde cero.")
            return

        with open(save_path, "rb") as f:
            data = pickle.load(f)
            self.hippocampus = data.get("hippocampus", {})
            self.familiarity = defaultdict(int, data.get("familiarity", {}))
            self.cortical_memory = data.get("cortical_memory", [])
