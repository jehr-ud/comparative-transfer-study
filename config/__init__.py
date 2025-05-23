from pathlib import Path

from stable_baselines3 import DQN, PPO, A2C

from agents.q_learning import QLearning
#from agents.imitation_agent import ImitationTransfer
#from agents.cross_domain import CrossDomainTransfer
from environments import create_env

from stages.train.agents.base_sb_agent import train_sb_agent
from stages.train.agents.base_q_learning_agent import train_q_learning_agent
#from stages.train.agents.imitation_transfer import (
#    train_imitation_transfer_agent
#)
#from stages.train.agents.cross_domain_transfer import (
#    train_cross_domain_transfer_agent
#)

simple_env = create_env("simple")
medium_env = create_env("medium")
complex_env = create_env("complex")


envs = [
    {
        "name": "simple",
        "env": simple_env
    },
    {
        "name": "medium",
        "env": medium_env
    },
    {
        "name": "complex",
        "env": complex_env
    }
]

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
        "params_predict": {"deterministic": True},
        "type": "classical",
    },
    {
        "name": "PPO",
        "class": PPO,
        "train_function": train_sb_agent,
        "params_predict": {"deterministic": True},
        "type": "classical",
    },
    {
        "name": "A2C",
        "class": A2C,
        "train_function": train_sb_agent,
        "params_predict": {"deterministic": True},
        "type": "classical",
    },
]

transfer_algorithms = [
    {
        "name": "Imitation",
        "class": ImitationTransfer,
        "train_function": train_imitation_transfer_agent,
        "params_predict": {},
        "params_train": {
            "expert": {
                "model": "PPO",
                "path": str(Path("models") / "{model}_{env}")
            },
        },
        "type": "transfer",
    },
    {
        "name": "CrossDomain",
        "class": CrossDomainTransfer,
        "train_function": train_cross_domain_transfer_agent,
        "params_predict": {},
        "params_train": {
            "expert": {
                "model": "PPO",
                "path": str(Path("models") / "{model}_{env}")
            },
        },
        "type": "transfer",
    },
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
