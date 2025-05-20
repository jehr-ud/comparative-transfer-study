from pathlib import Path

from stable_baselines3 import DQN, PPO, A2C

from agents.q_learning import QLearning
from agents.imitation_agent import ImitationTransfer
from agents.cross_domain import CrossDomainTransfer
from environments import create_env
from stages.train.agents.base_sb_agent import train_sb_agent
from stages.train.agents.base_q_learning_agent import train_q_learning_agent
from stages.train.agents.imitation_transfer import (
    train_imitation_transfer_agent
)
from stages.train.agents.cross_domain_transfer import (
    train_cross_domain_transfer_agent
)

envs1 = {
    "simple": {
        "train": create_env("simple"),
        "validation": create_env("simple", render=True)
    },
    "medium": {
        "train": create_env("medium"),
        "validation": create_env("medium", render=True)
    },
}

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

classical_algorithms = [
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
        "type": "classical"
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

transfer_algorithms = [
    {
        "name": "Imitation",
        "class": ImitationTransfer,
        "train_function": train_imitation_transfer_agent,
        "params_predict": {},
        "type": "transfer"
    },
    {
        "name": "CrossDomain",
        "class": CrossDomainTransfer,
        "train_function": train_cross_domain_transfer_agent,
        "params_predict": {},
        "type": "transfer"
    },
]

classical_transfer_envs = [
    {
        "target": {
            "env": create_env("medium"),
            "name": "Medium"
        },
        "source_model_paths": [
            {"name": "Simple", "path": str(Path("models") / "{model}_simple")},
        ]
    },
    {
        "target": {
            "env": create_env("complex"),
            "name": "Complex"
        },
        "source_model_paths": [
            {"name": "Simple", "path": str(Path("models") / "{model}_simple")},
            {"name": "Medium", "path": str(Path("models") / "{model}_medium")},
        ]
    }
]


transfer_envs = [
    {
        "target": {
            "env": create_env("simple"),
            "name": "Simple"
        },
        "source_model_paths": [
            {"name": "simple", "path": str(Path("models") / "{model}_simple")},
        ]
    },
    {
        "target": {
            "env": create_env("medium"),
            "name": "Medium"
        },
        "source_model_paths": [
            {"name": "medium", "path": str(Path("models") / "{model}_medium")},
        ]
    },
    {
        "target": {
            "env": create_env("complex"),
            "name": "Complex"
        },
        "source_model_paths": [
            {
                "name": "Complex",
                "path": str(Path("models") / "{model}_complex")
            }
        ]
    }
]
