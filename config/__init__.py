from pathlib import Path


from agents.ray_agent import (
    RLLibAgent as PPO,
    RLLibAgent as IMPALA,
    RLLibAgent as DQN
)
from agents.imitation_agent import ImitationMarWilTransfer

from stages.train.agents.base_classical_agent import train_rllib_agent
from stages.train.agents.imitation_transfer import (
    train_imitation_transfer_agent
)
from environments.visual_maze_env import VisualMazeEnv


def env_creator(cfg):
    return VisualMazeEnv(cfg)


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

simple_env_info = {
    "name": "simple",
    "config": simple_env_conf,
    "env": simple_env
}

medium_env_info = {
    "name": "medium",
    "config": medium_env_conf,
    "env": medium_env
}

complex_env_info = {
    "name": "complex",
    "config": complex_env_conf,
    "env": complex_env
}

envs = [
    simple_env_info,
    medium_env_info,
    complex_env_info
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
        "name": "IMPALA",
        "class": IMPALA,
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

classical_transfer_experiments = [
    {
        "target_env": medium_env_info,
        "source_model_paths": [
            {
                "difficulty": "simple",
                "path": str(Path("models") / "{model}_simple"),
            }
        ],
    },
    {
        "target_env": complex_env_info,
        "source_model_paths": [
            {
                "difficulty": "simple",
                "path": str(Path("models") / "{model}_simple"),
            },
            {
                "difficulty": "medium",
                "path": str(Path("models") / "{model}_medium"),
            },
        ],
    },
]

transfer_experiments = [
    {
        "target_env": simple_env_info,
        "source_model_paths": [
            {
                "difficulty": "simple",
                "path": str(Path("models") / "{model}_simple"),
            },
        ],
    },
    {
        "target_env": medium_env_info,
        "source_model_paths": [
            {
                "difficulty": "medium",
                "path": str(Path("models") / "{model}_medium"),
            },
        ],
    },
    {
        "target_env": complex_env_info,
        "source_model_paths": [
            {
                "difficulty": "complex",
                "path": str(Path("models") / "{model}_complex"),
            },
        ],
    },
]
