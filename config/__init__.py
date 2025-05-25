from pathlib import Path


from agents.rl_classical import (
    RLLibAgent as PPO,
    RLLibAgent as SAC,
    RLLibAgent as DQN
)
from agents.imitation_agent import ImitationMarWilTransfer

from stages.train.agents.base_classical_agent import train_rllib_agent
from stages.train.agents.imitation_transfer import (
    train_imitation_transfer_agent
)
from environments.visual_maze_env import VisualMazeEnv
from ray.tune.registry import register_env


def env_creator(cfg):
    return VisualMazeEnv(cfg)


register_env("simple", env_creator)
register_env("medium", env_creator)
register_env("complex", env_creator)


simple_env_conf = {
    "name": "simple",
    "size": 6,
    "obstacles": [(1, 1), (2, 3), (3, 1)],
    "render_mode": None,
    "shared_window": None
}

medium_env_conf = {
    "name": "medium",
    "size": 8,
    "obstacles": [(1, 2), (2, 4), (5, 5), (6, 3)],
    "render_mode": None,
    "shared_window": None
}

complex_env_conf = {
    "name": "complex",
    "size": 12,
    "obstacles": [(2, 2), (3, 7), (6, 6), (8, 9), (10, 4)],
    "render_mode": None,
    "shared_window": None
}

simple_env = env_creator(simple_env_conf)
medium_env = env_creator(medium_env_conf)
complex_env = env_creator(complex_env_conf)

envs = [
    {
        "name": "simple",
        "config": simple_env_conf,
        "env": simple_env
    },
    {
        "name": "medium",
        "config": medium_env_conf,
        "env": medium_env
    },
    {
        "name": "complex",
        "config": complex_env_conf,
        "env": complex_env
    }
]

ppo = {
    "name": "PPO",
    "class": PPO,
    "train_function": train_rllib_agent,
    "params_predict": {},
    "type": "classical",
}

classical_algorithms = [
    {
        "name": "DQN",
        "class": DQN,
        "train_function": train_rllib_agent,
        "params_predict": {},
        "type": "classical",
    },
    ppo,
    {
        "name": "SAC",
        "class": SAC,
        "train_function": train_rllib_agent,
        "params_predict": {},
        "type": "classical",
    },
]

transfer_algorithms = [
    {
        "name": "Imitation-MarWil",
        "class": ImitationMarWilTransfer,
        "train_function": train_imitation_transfer_agent,
        "params_predict": {},
        "params_train": {
            "expert": {
                "model": ppo,
                "path": str(Path("models") / "{model}_{env}")
            },
        },
        "type": "transfer",
    },
    {
        "name": "Imitation-MarWil",
        "class": ImitationMarWilTransfer,
        "train_function": train_imitation_transfer_agent,
        "params_predict": {},
        "params_train": {
            "expert": {
                "model": ppo,
                "path": str(Path("models") / "{model}_{env}")
            },
        },
        "type": "transfer",
    }
]

classical_transfer_envs = [
    {
        "target": {
            "env": medium_env,
            "name": "Medium",
        },
        "source_model_paths": [
            {
                "name": "Simple",
                "path": str(Path("models") / "{model}_simple"),
            },
        ],
    },
    {
        "target": {
            "env": complex_env,
            "name": "Complex",
        },
        "source_model_paths": [
            {
                "name": "Simple",
                "path": str(Path("models") / "{model}_simple"),
            },
            {
                "name": "Medium",
                "path": str(Path("models") / "{model}_medium"),
            },
        ],
    },
]

transfer_envs = [
    {
        "target": {
            "env": simple_env,
            "name": "Simple",
        },
        "source_model_paths": [
            {
                "name": "simple",
                "path": str(Path("models") / "{model}_simple"),
            },
        ],
    },
    {
        "target": {
            "env": medium_env,
            "name": "Medium",
        },
        "source_model_paths": [
            {
                "name": "medium",
                "path": str(Path("models") / "{model}_medium"),
            },
        ],
    },
    {
        "target": {
            "env": complex_env,
            "name": "Complex",
        },
        "source_model_paths": [
            {
                "name": "Complex",
                "path": str(Path("models") / "{model}_complex"),
            },
        ],
    },
]
