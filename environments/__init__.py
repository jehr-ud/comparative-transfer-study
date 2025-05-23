from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environments.drivers.visual_maze_env import VisualMazeEnv


def create_env(name, render_mode=None, name_file=None, shared_window=None):
    if name == 'simple':
        env = VisualMazeEnv(
            size=6,
            obstacles=[(1, 1), (2, 3), (3, 1)],
            render_mode=render_mode,
            shared_window=shared_window
        )
    elif name == 'medium':
        env = VisualMazeEnv(
            size=8,
            obstacles=[(1, 2), (2, 4), (5, 5), (6, 3)],
            render_mode=render_mode,
            shared_window=shared_window
        )
    elif name == 'complex':
        env = VisualMazeEnv(
            size=12,
            obstacles=[(2, 2), (3, 7), (6, 6), (8, 9), (10, 4)],
            render_mode=render_mode,
            shared_window=shared_window
        )
    else:
        raise ValueError("Unknown environment name")

    print("Env was created......")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = name_file if name_file else f"results/monitor/visualmaze_{name}_{timestamp}"

    env = Monitor(env, filename=filename)
    return DummyVecEnv([lambda: env])
