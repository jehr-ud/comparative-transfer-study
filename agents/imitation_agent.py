from pathlib import Path

import pandas as pd
import ray
import torch
import torch.distributions as D
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.tune.registry import register_env
from environments.visual_maze_env import VisualMazeEnv


CONFIG_BY_DIFFICULTY = {
            "simple": {
                "beta": 0.3,
                "lr": 1e-4,
                "gamma": 0.95,
                "train_batch_size": 256
            },
            "medium": {
                "beta": 0.7,
                "lr": 5e-5,
                "gamma": 0.98,
                "train_batch_size": 512
            },
            "complex": {
                "beta": 1.0,
                "lr": 1e-5,
                "gamma": 0.99,
                "train_batch_size": 1024
            }
}

ENV_RUNNER_CONFIG = {
    "simple":  {"num_env_runners": 1, "num_envs_per_env_runner": 2},  # 2 envs
    "medium":  {"num_env_runners": 2, "num_envs_per_env_runner": 2},  # 4 envs
    "complex": {"num_env_runners": 3, "num_envs_per_env_runner": 2}   # 6 envs
}


def env_creator(env_config):
    return VisualMazeEnv(env_config)


class ImitationMarWilTransfer:
    def __init__(
        self,
        name,
        difficulty,
        env_info,
        save_path="./models",
        explore=False
    ):
        self.env_info = env_info
        self.name = name
        self.difficulty = difficulty

        self.explore = explore

        self.policy = None
        self.save_path = save_path

        register_env("visual_env", env_creator)

        self.config = self._get_config()
        self.agent = self.config.build()

    def save(self, path=None):
        if self.agent is None:
            raise ValueError(
                "Agent not initialized. Train or load a model first."
            )

        self.save_path.mkdir(parents=True, exist_ok=True)
        save_to = Path(path) if path \
            else self.save_path / f"{self.name}_{self.difficulty}"
        self.agent.save(str(save_to))
        print(f"Model saved to {save_to}")

    def _get_config(self):
        if self.difficulty not in CONFIG_BY_DIFFICULTY:
            raise ValueError(f"Unknown difficulty: {self.difficulty}")

        cfg = CONFIG_BY_DIFFICULTY[self.difficulty]

        config = MARWILConfig()

        config.environment(
            env="visual_env",
            env_config=self.env_info.get('config')
        )

        config.learners(num_learners=2)

        info_runners = ENV_RUNNER_CONFIG.get(self.difficulty, {})
        info_runners["explore"] = self.explore

        config.env_runners(**info_runners)

        config = config.training(
            beta=cfg["beta"],
            lr=cfg["lr"],
            gamma=cfg["gamma"],
            train_batch_size_per_learner=cfg["train_batch_size"],
        )
        return config

    def load_trajectories(self, trajectories_path):
        print(f"[DEBUG] env to load {self.env_info.get('name')}")

        df = pd.read_parquet(trajectories_path)
        ds = ray.data.from_pandas(df)

        self.config.offline_data(
            input_=ds,
            dataset_num_iters_per_learner=1,
        )
        self.agent = self.config.build()

    def load(self, path):
        self.agent.restore(str(path))

    def predict(self, obs):
        module = self.agent.get_module("default_policy")

        obs_array = torch.tensor(
            [obs],
            dtype=torch.float32
        )  # Shape: [1, obs_dim]
        output = module.forward_inference({"obs": obs_array})

        if 'actions' in output:
            return output.get('actions')

        logits = output["action_dist_inputs"]
        dist = D.Categorical(logits=logits)
        return dist.sample().item()

    def learn(self):
        return self.agent.train()

    def stop(self):
        if self.agent:
            self.agent.stop()
