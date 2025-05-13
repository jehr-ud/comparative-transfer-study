from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environments.drivers.visual_maze_env import VisualMazeEnv


def create_env(name, render=False, name_file=None, vectorize=True):
    render_mode = "human" if render else None

    if name == 'simple':
        env = VisualMazeEnv(
            size=6,
            obstacles=[(1, 1), (2, 3), (3, 1)],
            render_mode=render_mode
        )
    elif name == 'medium':
        env = VisualMazeEnv(
            size=8,
            obstacles=[(1, 2), (2, 4), (5, 5), (6, 3)],
            render_mode=render_mode
        )
    elif name == 'complex':
        env = VisualMazeEnv(
            size=12,
            obstacles=[(2, 2), (3, 7), (6, 6), (8, 9), (10, 4)],
            render_mode=render_mode
        )
    else:
        raise ValueError("Unknown environment name")

    print("Env was created......")

    filename = f"visualmaze_{name}"
    if name_file:
        filename = name_file

    env = Monitor(env, filename=filename)

    if vectorize:
        env = DummyVecEnv([lambda: env])  # for SB3 agents

    return env
