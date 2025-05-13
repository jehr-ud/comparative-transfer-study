from pathlib import Path

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.q_learning import QLearning
from environments import create_env, VisualMazeEnv
from stages.train.agents.base_sb_agent import train_sb_agent
from stages.train.agents.base_q_learning_agent import train_q_learning_agent


envs = {
    "simple": {
        "train": create_env("simple", vectorize=False),
        "validation": create_env("simple", render=True, vectorize=False)
    },
    "medium": {
        "train": create_env("medium", vectorize=False),
        "validation": create_env("medium", render=True, vectorize=False)
    },
    "complex": {
        "train": create_env("complex", vectorize=False),
        "validation": create_env("complex", render=True, vectorize=False)
    },
}

algorithms2 = [
    {
        "name": "DQN",
        "class": DQN,
        "train_function": train_sb_agent,
        "params_predict": {
            "deterministic": True
        },
        "type": "classical"
    },
    {
        "name": "PPO",
        "class": PPO,
        "train_function": train_sb_agent,
        "params_predict": {
            "deterministic": True
        },
        "type": "base"
    },
    {
        "name": "A2C",
        "class": A2C,
        "train_function": train_sb_agent,
        "params_predict": {
            "deterministic": True
        },
        "type": "classical"
    },
]

algorithms = [
    {
        "name": "Q-Learning",
        "class": QLearning,
        "train_function": train_q_learning_agent,
        "params_predict": {"deterministic": True},
        "type": "classical"
    }
]

transfer_envs = [
    DummyVecEnv([lambda: VisualMazeEnv(size=6, obstacles=[(1, 1), (2, 3), (3, 1)])]),
    DummyVecEnv([lambda: VisualMazeEnv(size=8, obstacles=[(1, 2), (2, 4), (5, 5), (6, 3)])]),
    DummyVecEnv([lambda: VisualMazeEnv(size=12, obstacles=[(2, 2), (3, 7), (6, 6), (8, 9), (10, 4)])]),
]

model_paths = [
    str(Path("models") / "{model}_simple"),
    str(Path("models") / "{model}_medium"),
    str(Path("models") / "{model}_complex"),
]
