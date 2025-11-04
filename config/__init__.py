from pathlib import Path


from agents.base_agent import (
    ClassicalAgent as PPO,
    ClassicalAgent as A2C,
    ClassicalAgent as DQN
)
from agents.snncit_agent import SNNCITgent
from agents.adapsnn_agent import SNNAgent as MazeSolver

from stages.train.agents.base_classical_agent import train_basical_agent
from stages.train.agents.snncit_agent import train_agent as citsnn_train_agent
from stages.train.agents.adapsnn_agent import (
    train_agent as adapsnn_train_agent
)
from environments.visual_maze_env import (
    VisualMazeEnv
)


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
    medium_env_info,
    complex_env_info
]

ppo = {
    "name": "PPO",
    "class": PPO,
    "train_function": train_basical_agent,
    "params_predict": {},
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

adap = {
    "name": "ADAP-SSN",
    "class": MazeSolver,
    "train_function": adapsnn_train_agent,
    "params_predict": {},
    "params_train": {
    },
    "type": "transfer",
}

classical_algorithms = [
    adap,
    ppo,
    dqn,
    a2c,
]

# Perception-Action Agent
cit = {
    "name": "CIT-SSN",
    "class": SNNCITgent,
    "train_function": citsnn_train_agent,
    "params_predict": {},
    "params_train": {
    },
    "type": "transfer",
}


transfer_algorithms = [
    # cit,
    # ppo,
    # dqn,
    # a2c,
    adap
]

simple = "{experiment}_{model}_simple"
complex = "{experiment}_{model}_medium"
simple_model = str(Path("models") / simple / simple)
complex_model = str(Path("models") / complex / complex)

transfer_experiments = [
    {
        "target_env": medium_env_info,
        "source_model_paths": [
            {
                "difficulty": "simple",
                "path": simple_model,
            }
        ]
    },
    {
        "target_env": complex_env_info,
        "source_model_paths": [
            {
                "difficulty": "simple",
                "path": simple_model,
            },
            {
                "difficulty": "medium",
                "path": complex_model,
            },
        ]
    }
]
