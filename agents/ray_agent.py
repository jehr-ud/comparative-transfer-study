from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.impala import IMPALAConfig
import torch
import torch.distributions as D


params = {
    "simple": {
        "PPO": {
            "gamma": 0.95,
            "lr": 1e-4,
            "train_batch_size": 256,
            "num_sgd_iter": 5,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "clip_param": 0.2,
            "train_batch_size_per_learner": 256,
        },
        "DQN": {
            "gamma": 0.95,
            "lr": 1e-4,
            "target_network_update_freq": 250,
            "train_batch_size_per_learner": 256
        },
        "IMPALA": {
            "lr": 1e-4,
            "gamma": 0.95,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "train_batch_size_per_learner": 256,
        }
    },
    "medium": {
        "PPO": {
            "gamma": 0.97,
            "lr": 2e-4,
            "train_batch_size": 512,
            "num_sgd_iter": 8,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.02,
            "clip_param": 0.2,
            "train_batch_size_per_learner": 512,
        },
        "DQN": {
            "gamma": 0.97,
            "lr": 2e-4,
            "target_network_update_freq": 400,
            "train_batch_size_per_learner": 512
        },
        "IMPALA": {
            "lr": 2e-4,
            "gamma": 0.97,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "train_batch_size_per_learner": 512,
        }
    },
    "complex": {
        "PPO": {
            "gamma": 0.99,
            "lr": 3e-4,
            "train_batch_size": 1024,
            "num_sgd_iter": 10,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.05,
            "clip_param": 0.2,
            "train_batch_size_per_learner": 1024,
        },
        "DQN": {
            "gamma": 0.99,
            "lr": 3e-4,
            "target_network_update_freq": 500,
            "train_batch_size_per_learner": 1024
        },
        "IMPALA": {
            "lr": 3e-4,
            "gamma": 0.99,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "train_batch_size_per_learner": 1024,
        }
    }
}

ENV_RUNNER_CONFIG = {
    "simple":  {"num_env_runners": 1, "num_envs_per_env_runner": 2},  # 2 envs
    "medium":  {"num_env_runners": 2, "num_envs_per_env_runner": 2},  # 4 envs
    "complex": {"num_env_runners": 2, "num_envs_per_env_runner": 3}   # 5 envs
}


class RLLibAgent:
    def __init__(
        self,
        name,
        difficulty,
        env_info,
        save_path="./models",
        explore=False
    ):
        self.name = name
        self.difficulty = difficulty
        self.env_info = env_info
        self.save_path = Path(save_path)

        self.explore = explore

        self.config = self._get_config()
        self.agent = self.config.build()

    def train(self):
        return self.agent.train()

    def _get_config(self):
        config = None
        algo_params = params[self.env_info["name"]][self.name]

        if self.name == "PPO":
            config = PPOConfig()
        elif self.name == "DQN":
            config = DQNConfig()
        elif self.name == "IMPALA":
            config = IMPALAConfig()
        else:
            raise ValueError("Agentt not configured.")

        config.environment(
            env="visual_env",
            env_config=self.env_info.get('config')
        )

        config.learners(num_learners=2)

        info_runners = ENV_RUNNER_CONFIG.get(self.difficulty, {})
        info_runners["explore"] = self.explore

        config.env_runners(**info_runners)

        config = config.training(**algo_params)

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

    def stop(self):
        self.agent.stop()
