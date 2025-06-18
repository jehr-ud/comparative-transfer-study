from pathlib import Path


from agents.base_agent import (
    ClassicalAgent as PPO,
    ClassicalAgent as A2C,
    ClassicalAgent as DQN
)
from agents.q_learning import QLearning
from agents.imitation_agent import ImitationMarWilTransfer
from agents.cit_agent import CITgent

from stages.train.agents.base_classical_agent import train_basical_agent
from stages.train.agents.cit_agent import train_agent
from stages.train.agents.imitation_transfer import (
    train_imitation_transfer_agent
)
from stages.train.agents.base_q_learning_agent import (
    train_q_learning_agent
)
from environments.visual_maze_env import VisualMazeEnv


def env_creator(cfg):
    return VisualMazeEnv(cfg)


simple_env_conf = {
    "name": "simple",
    "size": 6,
    "obstacles": [(1, 1), (2, 3), (3, 1)],
    "render_mode": 'None',
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
    # medium_env_info,
    # complex_env_info
]

ppo = {
    "name": "PPO",
    "class": PPO,
    "train_function": train_basical_agent,
    "params_predict": {},
    "type": "classical",
}

qlearning = {
    "name": "QLearning",
    "class": QLearning,
    "train_function": train_q_learning_agent,
    "params_predict": {"deterministic": True},
    "type": "classical",
}

dqn = {
    "name": "DQN",
    "class": DQN,
    "train_function": train_basical_agent,
    "params_predict": {},
    "type": "classical",
}

a2c = {
    "name": "A2C",
    "class": A2C,
    "train_function": train_basical_agent,
    "params_predict": {},
    "type": "classical",
}

classical_algorithms = [
    ppo,
    dqn,
    qlearning,
    a2c
]


marwil = {
    "name": "Imitation-MarWil",
    "class": ImitationMarWilTransfer,
    "train_function": train_imitation_transfer_agent,
    "params_predict": {},
    "params_train": {
        "expert": {
            "expert_info": ppo,
            "path": str(Path("models") / "{model}_{env}")
        },
    },
    "type": "transfer",
}

# Perception-Action Agent
cpa = {
    "name": "CIT",
    "class": CITgent,
    "train_function": train_agent,
    "params_predict": {},
    "params_train": {
    },
    "type": "transfer",
}

transfer_algorithms = [
    # marwil,
    cpa
]

classical_transfer_experiments = [
    {
        "target_env": medium_env_info,
        "source_model_paths": [
            {
                "difficulty": "simple",
                "path": str(Path("models") / "{experiment}_{model}_simple" / "{experiment}_{model}_simple"),
            }
        ],
    },
    {
        "target_env": complex_env_info,
        "source_model_paths": [
            {
                "difficulty": "simple",
                "path": str(Path("models") / "{experiment}_{model}_simple" / "{experiment}_{model}_simple"),
            },
            {
                "difficulty": "medium",
                "path": str(Path("models") / "{experiment}_{model}_medium" / "{experiment}_{model}_medium"),
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
                "path": str(Path("models") / "{experiment}_{model}_simple"),
            },
        ],
    },
    # {
        # "target_env": medium_env_info,
        # "source_model_paths": [
            # {
                # "difficulty": "complex",
                # "path": str(Path("models") / "{experiment}_{model}_complex"),
            # },
        # ],
    # },
    # {
        # "target_env": complex_env_info,
        # "source_model_paths": [
            # {
                # "difficulty": "complex",
                # "path": str(Path("models") / "{experiment}_{model}_complex"),
            # },
        # ],
    # },
]
