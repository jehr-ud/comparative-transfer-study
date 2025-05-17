from pathlib import Path

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.q_learning import QLearning
from environments import create_env, VisualMazeEnv
from stages.train.agents.base_sb_agent import train_sb_agent
from stages.train.agents.base_q_learning_agent import train_q_learning_agent


envs = {
    "simple": {
        "train": create_env("simple"),
        "validation": create_env("simple", render=True)
    },
    "medium": {
        "train": create_env("medium"),
        "validation": create_env("medium", render=True)
    },
    "complex": {
        "train": create_env("complex"),
        "validation": create_env("complex", render=True)
    },
}

algorithms2 = [
    {
        "name": "Q-Learning",
        "class": QLearning,
        "train_function": train_q_learning_agent,
        "params_predict": {"deterministic": True},
        "type": "classical"
    },
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
        "name": "DQN",
        "class": DQN,
        "train_function": train_sb_agent,
        "params_predict": {
            "deterministic": True
        },
        "type": "classical"
    }
]

transfer_envs = [
    {
        "source": {
            "env": create_env("medium"),
            "name": "Medium"
        },
        "target_model_paths": [
            {"name": "simple", "path": str(Path("models") / "{model}_simple")},
        ]
    },
    {
        "source": {
            "env": create_env("complex"),
            "name": "Complex"
        },
        "target_model_paths": [
            {"name": "simple", "path": str(Path("models") / "{model}_simple")},
            {"name": "Medium", "path": str(Path("models") / "{model}_medium")},
        ]
    }
]
