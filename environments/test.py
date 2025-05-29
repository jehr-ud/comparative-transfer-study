import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from environments.visual_maze_env import VisualMazeEnv
import torch.distributions as D


def env_creator(env_config):
    return VisualMazeEnv(env_config)


register_env("env1", env_creator)

# Configuración y entrenamiento
config = PPOConfig()
config.environment(
    env="env1",
    env_config={
        "name": "my-maze",
        "size": 12,
        "obstacles": [(2, 2), (3, 7), (6, 6), (8, 9), (10, 4)],
        "render_mode": None,
        "shared_window": None
    }
)

algo = config.build()
algo.train()

# Crear entorno para inferencia
env = VisualMazeEnv({
    "name": "my-maze",
    "size": 12,
    "obstacles": [(2, 2), (3, 7), (6, 6), (8, 9), (10, 4)],
    "render_mode": None,
    "shared_window": None
})

# Obtener el módulo del agente
module = algo.get_module("default_policy")

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    obs_array = torch.tensor([obs], dtype=torch.float32)  # Shape: [1, obs_dim]
    output = module.forward_inference({"obs": obs_array})

    logits = output["action_dist_inputs"]
    dist = D.Categorical(logits=logits)
    action = dist.sample().item()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Total reward: {total_reward}")
