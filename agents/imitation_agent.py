import os
import numpy as np
import pickle

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.marwil import MARWILConfig
from environments.visual_maze_env import VisualMazeEnv


class ImitationMarWilTransfer:
    def __init__(self, env):
        self.env = env
        self.imitation_model = None
        self.policy = None

    def save(self, file_path):
        model_path = os.path.join("models", file_path)
        os.makedirs(model_path, exist_ok=True)
        self.imitation_model.save(model_path)

    def load(self, trajectories_path):
        def loader_fn():
            with open(trajectories_path, "rb") as f:
                trajs = pickle.load(f)

            obs = []
            actions = []
            rewards = []
            dones = []
            next_obs = []

            for traj in trajs:
                for i in range(len(traj["states"])):
                    obs.append(traj["states"][i])
                    actions.append(traj["actions"][i])
                    rewards.append(traj["rewards"][i])
                    dones.append(traj["dones"][i])
                    if i + 1 < len(traj["states"]):
                        next_obs.append(traj["states"][i + 1])
                    else:
                        next_obs.append(traj["states"][i])

            return SampleBatch({
                SampleBatch.OBS: obs,
                SampleBatch.ACTIONS: actions,
                SampleBatch.REWARDS: rewards,
                SampleBatch.DONES: dones,
                SampleBatch.NEXT_OBS: next_obs,
            })

        print(f"[DEBUG] env to load {self.env.envs[0].name}")

        dummy_env = VisualMazeEnv("simple", size=6, obstacles=[(1, 1), (2, 3), (3, 1)])
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space

        config = (
            MARWILConfig()
            .environment(self.env.envs[0].name, disable_env_checking=True)
            .offline_data(input_="dataset")
            .framework("torch")
            .rollouts(num_rollout_workers=0)
            .training(train_batch_size=200)
        )
        config.observation_space = obs_space
        config.action_space = act_space

        print("[DEBUG] INFO ENV")
        print(self.env.action_space)
        print(self.env.observation_space)

        config.model = {
            "fcnet_hiddens": [256, 256],  #  2 capas ocultas con 256 neuronas
            "fcnet_activation": "relu",
            "max_seq_len": 20,
            # "fcnet_input_shape": self.env.observation_space.shape,
        }

        config.input_config["loader_fn"] = loader_fn

        

        self.imitation_model = config.build()
        self.policy = self.imitation_model.get_policy()

    def predict(self, observation):
        action = self.policy.compute_single_action(observation)
        return action, None

    def learn(self):
        return self.imitation_model.train()
