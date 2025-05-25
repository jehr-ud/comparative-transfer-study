from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
import torch
import torch.distributions as D


class RLLibAgent:
    def __init__(self, name, difficulty, env_info, save_path="./models"):
        self.name = name
        self.difficulty = difficulty
        self.env_info = env_info
        self.save_path = Path(save_path)

        self.config = self._get_config()
        self.agent = self.config.build()

    def train(self):
        self.agent.train()

    def _get_config(self):
        config = None

        if self.name == "PPO":
            config = PPOConfig()
        elif self.name == "DQN":
            config = DQNConfig()
        elif self.name == "SAC":
            config = SACConfig()
        else:
            raise ValueError("Agentt not configured.")

        config.environment(
            env=self.difficulty,
            env_config=self.env_info.get('config')
        )
        config.env_runners(num_env_runners=1)
        config.training(
            gamma=0.9,
            lr=3e-4,
            train_batch_size=256,
            train_batch_size_per_learner=256
        )
        return config

    def predict(self, obs):
        module = self.agent.get_module("default_policy")

        obs_array = torch.tensor([obs], dtype=torch.float32)  # Shape: [1, obs_dim]
        output = module.forward_inference({"obs": obs_array})

        if 'actions' in output:
            return output.get('actions') 
        
        logits = output["action_dist_inputs"]
        dist = D.Categorical(logits=logits)
        return dist.sample().item()

    def load(self, path):
        self.agent.restore(str(path))

    def save(self, path=None):
        if self.agent is None:
            raise ValueError("Agenet not initialized. Train or load a model first.")

        self.save_path.mkdir(parents=True, exist_ok=True)
        save_to = Path(path) if path else self.save_path / f"{self.name}_{self.difficulty}"
        self.agent.save(str(save_to))
        print(f"Model saved to {save_to}")
