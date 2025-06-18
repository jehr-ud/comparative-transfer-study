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
        
        self.previous_position = None # Memoria del paso inmediatamente anterior para evitar el ping-pong

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
        self.previous_position = None
        self.last_position = tuple(obs)
        total_reward = 0
        path = [self.last_position]
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = self.explore(self.last_position, epsilon, prev_pos=self.previous_position)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_pos = tuple(next_obs)

            self.update_hippocampus(self.last_position, next_pos)
            self.previous_position = self.last_position
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
        valid_actions = self.env.get_possible_actions(current)

        next_previous_position = current

        candidate_paths = [(path, eff) for path, eff in self.cortical_memory if current in path]
        if candidate_paths:
            candidate_paths.sort(key=lambda x: (x[0].index(current), -x[1]))
            best_path, _ = candidate_paths[0]
            idx = best_path.index(current)

            if idx + 1 < len(best_path):
                next_pos = best_path[idx + 1]
    
                # <<<<<<< LÓGICA ANTI-PING-PONG >>>>>>>>
                # ¡No te devuelvas! Si el siguiente paso es volver a donde estaba, ignora este plan.
                if next_pos == self.previous_position:
                    print(f"[DEBUG - ANTI PING-PONG] La memoria sugiere volver a {self.previous_position}. Ignorando.")
                    # Como este plan es malo, pasamos a la siguiente estrategia (planificación)
                else:
                    desired_action = self.direction_from_to(current, next_pos)
                    if desired_action in valid_actions:
                        self.previous_position = next_previous_position # Actualizar memoria antes de salir
                        return desired_action
                    else:
                        print(f"[DEBUG - ERROR DE MEMORIA] La ruta sugiere un movimiento inválido a {next_pos}. Forzando exploración.")
                        # Si hay un error, reseteamos la memoria de "previous" para no bloquear la exploración.
                        self.previous_position = None 
                        return self.explore(current)

        # --- ESTRATEGIA 2: PLANIFICACIÓN DELIBERADA ---
        path_to_goal = self.plan_path(current, self.goal)
        if path_to_goal and len(path_to_goal) > 1:
            next_pos = path_to_goal[1]

            # <<<<<<< LÓGICA ANTI-PING-PONG >>>>>>>>
            if next_pos == self.previous_position:
                print(f"[DEBUG - ANTI PING-PONG] El plan sugiere volver a {self.previous_position}. Ignorando.")
            else:
                desired_action = self.direction_from_to(current, next_pos)
                if desired_action in valid_actions:
                    self.previous_position = next_previous_position # Actualizar memoria antes de salir
                    return desired_action
                else:
                    self.previous_position = None
                    return self.explore(current)

        # --- ESTRATEGIA 3: ÚLTIMO RECURSO ---
        print(f"[DEBUG - Predict] Modo Fallback: Explorando.")
        # Cuando exploramos, también es buena idea evitar devolverse.
        # Puedes modificar `explore` para que también use `self.previous_position`.
        self.previous_position = None # Reseteamos para no limitar la exploración
        return self.explore(current)

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

    def explore(self, position, epsilon=0.1, prev_pos=None):
        """Explora el entorno, evitando devolverse inmediatamente."""
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # arriba, derecha, abajo, izquierda
        valid_actions = self.env.get_possible_actions(position)
        
        if not valid_actions:  # Seguridad por si el agente se queda atrapado sin salidas
            return random.randint(0, 3)

        scored_moves = []
        for action in valid_actions:
            dx, dy = directions[action]
            neighbor = (position[0] + dx, position[1] + dy)
            
            # <<<<<<< LÓGICA ANTI-PING-PONG EN EXPLORACIÓN >>>>>>>>
            if neighbor == prev_pos:
                novelty_score = -1.0
            else:
                novelty_score = 1 / (1 + self.familiarity.get(neighbor, 0) ** 2)
                
            scored_moves.append((novelty_score, action))

        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        if not scored_moves:
            return random.choice(valid_actions)

        scored_moves.sort(key=lambda x: x[0], reverse=True) # Ordenar por puntuación (novedad)
        return scored_moves[0][1]

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
