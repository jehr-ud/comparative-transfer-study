import time

import gymnasium as gym
import numpy as np
import pygame


class VisualMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, size=5, render_mode=None, obstacles=None):
        super().__init__()
        self.size = size
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4)

        self.steps_taken = 0
        self.max_steps = 1000

        self.agent_pos = [size - 1, size - 1]
        self.goal_pos = [0, 0]

        self.maze = np.zeros((size, size), dtype=int)

        if obstacles:
            for (i, j) in obstacles:
                if [i, j] != self.agent_pos and [i, j] != self.goal_pos:
                    self.maze[i, j] = 1

        self.cell_size = 60
        self.window_size = self.size * self.cell_size
        self.window = None
        self.clock = None

        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Maze Env")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [self.size - 1, self.size - 1]
        self.steps_taken = 0
        return np.array(self.agent_pos, dtype=np.int32), {}

    def step(self, action):
        old_pos = tuple(self.agent_pos)
        new_pos = list(old_pos)

        if action == 0: new_pos[0] -= 1
        elif action == 1: new_pos[0] += 1
        elif action == 2: new_pos[1] -= 1
        elif action == 3: new_pos[1] += 1

        print(f"[DEBUG] Agent at {old_pos}, action {action}, new_pos {new_pos}")

        self.steps_taken += 1

        reward = -0.01
        moved = False

        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            if self.maze[tuple(new_pos)] == 0:
                self.agent_pos = new_pos
                moved = True
                print("[DEBUG] Moved to", self.agent_pos)
            else:
                print("[DEBUG] Hit obstacle at", new_pos)
                reward = -0.2
        else:
            print("[DEBUG] Invalid move out of bounds to", new_pos)
            reward = -0.5

        if not moved and reward == -0.01:
            reward -= 0.03  # Penalización por no hacer progreso

        done = self.agent_pos == self.goal_pos
        timeout = False

        if done:
            reward = 1.0
            print("[DEBUG] Goal reached!")
        elif self.steps_taken >= self.max_steps:
            done = True
            timeout = True

        info = {"timeout": timeout}
        return np.array(self.agent_pos, dtype=np.int32), reward, done, False, info

    def render(self):
        if self.render_mode != "human":
            return

        self.window.fill((255, 255, 255))

        for row in range(self.size):
            for col in range(self.size):
                x = col * self.cell_size
                y = row * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                if self.maze[row, col] == 1:
                    color = (50, 50, 50)
                else:
                    color = (240, 240, 240)

                pygame.draw.rect(self.window, color, rect) # Draw the background
                pygame.draw.rect(self.window, (0, 0, 0), rect, 2) # Draw the border

        agent_center = (
            self.agent_pos[1] * self.cell_size + self.cell_size // 2,
            self.agent_pos[0] * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(self.window, (0, 102, 204), agent_center, self.cell_size // 3)

        goal_rect = pygame.Rect(
            self.goal_pos[1] * self.cell_size + self.cell_size // 4,
            self.goal_pos[0] * self.cell_size + self.cell_size // 4,
            self.cell_size // 2,
            self.cell_size // 2
        )
        pygame.draw.rect(self.window, (0, 204, 102), goal_rect)

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        time.sleep(0.1)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
