import gymnasium as gym
import numpy as np
import pygame


class NormalizeObs(gym.ObservationWrapper):
    """
    Escala cada eje x,y a [0,1].
    """
    def __init__(self, env):
        super().__init__(env)
        self.low = self.env.observation_space.low
        self.high = self.env.observation_space.high
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, obs):
        return (obs - self.low) / (self.high - self.low + 1e-8)

    def __getattr__(self, name):
        return getattr(self.env, name)


class VisualMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, config):
        super().__init__()
        self.name = config.get("name", "default_maze")
        self.size = config.get("size", 5)
        self.render_mode = config.get("render_mode", None)
        self.obstacles = config.get("obstacles", None)
        self.shared_window = config.get("shared_window", None)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.size - 1,
            shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)

        self.steps_taken = 0
        num_obstacles = len(self.obstacles) \
            if isinstance(self.obstacles, list) else 1

        steps_constant = 3
        self.max_steps = steps_constant * (self.size ** 2 + 2 * num_obstacles)

        self.num_obstacles = num_obstacles
        self.agent_pos = [self.size - 1, self.size - 1]
        self.goal_pos = [0, 0]

        self.maze = np.zeros((self.size, self.size), dtype=int)

        if self.obstacles:
            for (i, j) in self.obstacles:
                if [i, j] != self.agent_pos and [i, j] != self.goal_pos:
                    self.maze[i, j] = 1

        self.cell_size = 50
        self.window_size = self.size * self.cell_size
        self.window = self.shared_window
        self.clock = None

        self.last_pos = None
        self.last_action = None
        self.stuck_counter = 0
        self.visited = np.zeros_like(self.maze, dtype=np.int32)

        if self.render_mode == "human":
            print("[DEBUG] Initializing Pygame window")
            self.clock = pygame.time.Clock()

    def get_obstacle_count(self):
        return self.num_obstacles

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [self.size - 1, self.size - 1]
        self.steps_taken = 0
        return np.array(self.agent_pos, dtype=np.float32), {}

    def get_possible_actions(self, current_pos=None):
        """
        Calcula y retorna una lista de acciones válidas 
        desde la posición actual del agente.
        Las acciones se definen como:
        0: Arriba, 1: Derecha, 2: Abajo, 3: Izquierda.
        """
        if current_pos is None:
            current_pos = self.agent_pos

        valid_actions = []
        row, col = current_pos[0], current_pos[1]

        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        for action, (dr, dc) in enumerate(moves):
            new_row = int(row + dr)
            new_col = int(col + dc)
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                if self.maze[new_row, new_col] != 1:
                    valid_actions.append(action)

        return valid_actions

    def valid_position(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def step(self, action):
        old_pos = tuple(self.agent_pos)
        new_pos = list(old_pos)

        if action == 0:       # Arriba
            new_pos[0] -= 1
        elif action == 1:     # Derecha
            new_pos[1] += 1
        elif action == 2:     # Abajo
            new_pos[0] += 1
        elif action == 3:     # Izquierda
            new_pos[1] -= 1

        print(
            f"[DEBUG] Agent at {old_pos}, action {action}, new_pos {new_pos}"
        )

        self.steps_taken += 1
        timeout = False

        # 1. Recompensa base por existir (costo de vida)
        reward = -0.01

        # 2. Penalización por chocar o salir del mapa
        if not self.valid_position(new_pos):
            reward = -1.0  # Penalización fuerte por salir
        elif self.maze[tuple(new_pos)] == 1:
            reward = -0.75  # Penalización fuerte por chocar
        else:
            # 3. Movimiento válido
            self.agent_pos = new_pos
            self.visited[tuple(self.agent_pos)] += 1

            # 4. Recompensa por exploración (la más importante)
            if self.visited[tuple(self.agent_pos)] == 1:
                reward += 0.5  # Recompensa alta por descubrir una celda nueva
            else:
                # Penalización suave por visitar celdas ya conocidas
                reward -= 0.05
        # 5. Recompensa final por alcanzar la meta
        done = self.agent_pos == self.goal_pos
        if done:
            reward = 10.0  # Recompensa muy alta por ganar
        elif self.steps_taken >= self.max_steps:
            timeout = True

        info = {
            "steps_taken": self.steps_taken,
            "timeout": timeout
        }

        obs = np.array(self.agent_pos, dtype=np.float32)
        print(f"[DEBUG] obs {obs}")
        print(f"[DEBUG] steps_taken {self.steps_taken} max {self.max_steps}")

        return obs, reward, done, timeout, info

    def render(self):
        if self.render_mode == "human":
            self._draw()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return pygame.surfarray.array3d(self.window)

    def _draw(self, surface=None):
        if surface is None:
            surface = self.window

        surface.fill((255, 255, 255))

        for row in range(self.size):
            for col in range(self.size):
                x = col * self.cell_size
                y = row * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                color = (50, 50, 50) \
                    if self.maze[row, col] == 1 else (240, 240, 240)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (0, 0, 0), rect, 2)

        agent_center = (
            self.agent_pos[1] * self.cell_size + self.cell_size // 2,
            self.agent_pos[0] * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(
            surface,
            (0, 102, 204),
            agent_center, self.cell_size // 3
        )

        goal_rect = pygame.Rect(
            self.goal_pos[1] * self.cell_size + self.cell_size // 4,
            self.goal_pos[0] * self.cell_size + self.cell_size // 4,
            self.cell_size // 2,
            self.cell_size // 2
        )
        pygame.draw.rect(surface, (0, 204, 102), goal_rect)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
