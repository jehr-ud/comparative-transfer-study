import os

import pandas as pd
import ray
import torch
import torch.distributions as D
from ray.rllib.algorithms.marwil import MARWILConfig


class ImitationMarWilTransfer:
    def __init__(self, name, difficulty, env_info, save_path="./models"):
        self.env_info = env_info
        self.name = name
        self.difficulty = difficulty
        self.imitation_model = None
        self.policy = None
        self.save_path = save_path

    def save(self, file_path):
        model_path = os.path.join(self.save_path, file_path)
        os.makedirs(model_path, exist_ok=True)
        self.imitation_model.save(model_path)

    def load(self, trajectories_path):
        print(f"[DEBUG] env to load {self.env_info.get('name')}")

        obs_space = self.env_info.get('env').observation_space
        act_space = self.env_info.get('env').action_space

        CONFIG_BY_DIFFICULTY = {
            "simple": {
                "beta": 0.3,
                "lr": 1e-4,
                "gamma": 0.95,
                "train_batch_size": 1000,
                "num_env_runners": 2,
                "num_envs_per_env_runner": 4,
            },
            "medium": {
                "beta": 0.7,
                "lr": 5e-5,
                "gamma": 0.98,
                "train_batch_size": 2000,
                "num_env_runners": 3,
                "num_envs_per_env_runner": 2,
            },
            "complex": {
                "beta": 1.0,
                "lr": 1e-5,
                "gamma": 0.99,
                "train_batch_size": 4000,
                "num_env_runners": 4,
                "num_envs_per_env_runner": 2,
            }
        }

        if self.difficulty not in CONFIG_BY_DIFFICULTY:
            raise ValueError(f"Unknown difficulty: {self.difficulty}")

        cfg = CONFIG_BY_DIFFICULTY[self.difficulty]

        config = (
            MARWILConfig()
            .environment(
                env="visual_env",
                env_config=self.env_info.get('config')
            )
            .framework("torch")
            .env_runners(
                num_env_runners=cfg["num_env_runners"],
                num_envs_per_env_runner=cfg["num_envs_per_env_runner"],
            )
            .training(
                beta=cfg["beta"],
                lr=cfg["lr"],
                gamma=cfg["gamma"],
                train_batch_size_per_learner=cfg["train_batch_size"],
            )
        )

        df = pd.read_parquet(trajectories_path)
        ds = ray.data.from_pandas(df)

        config.offline_data(
            input_=ds,
            dataset_num_iters_per_learner=1,
        )

        config.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )

        print(obs_space)
        print(act_space)

        config.observation_space = obs_space
        config.action_space = act_space

        self.imitation_model = config.build()

    def predict(self, obs):
        module = self.imitation_model.get_module("default_policy")

        obs_array = torch.tensor([obs], dtype=torch.float32)  # Shape: [1, obs_dim]
        output = module.forward_inference({"obs": obs_array})

        if 'actions' in output:
            return output.get('actions')

        logits = output["action_dist_inputs"]
        dist = D.Categorical(logits=logits)
        return dist.sample().item()

    def learn(self):
        return self.imitation_model.train()

    def stop(self):
        if self.imitation_model:
            self.imitation_model.stop()
