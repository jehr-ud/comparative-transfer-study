from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class RewardCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages.
    """
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        for i, done in enumerate(self.locals['dones']):
            if done:
                # The 'infos' dictionary contains the episode reward and length
                episode_reward = self.locals['infos'][i]['episode']['r']
                self.rewards.append(episode_reward)
                if self.verbose > 0:
                    print(
                        f"Episode finished. Reward: {episode_reward:.2f}, Total episodes: {len(self.rewards)}"
                    )
        return True


class ClassicalAgent:
    def __init__(
        self,
        name,
        difficulty,
        env_info,
        save_path="./models",
        explore=False,
        params=None
    ):
        if not params:
            params = {}

        self.name = name
        self.difficulty = difficulty
        self.env_info = env_info
        self.save_path = Path(save_path)
        self.params = params
        self.explore = explore

        self.model = None
        self.algorithm_class = self._get_algorithm_class()

    def _get_algorithm_class(self):
        """Devuelve la clase del algoritmo (PPO, DQN, A2C) sin instanciarla."""
        if self.name == "PPO":
            return PPO
        elif self.name == "DQN":
            return DQN
        elif self.name == "A2C":
            return A2C
        else:
            raise ValueError("Agent not configured.")

    def setup_model(self):
        """Crea una nueva instancia del modelo. Llamar solo para un entrenamiento nuevo."""
        print(f"Setting up a new '{self.name}' model...")
        self.model = self.algorithm_class(
            "MlpPolicy",
            self.env_info.get('env'),
            verbose=1,
            **self.params
        )

    def train(self, total_timesteps):
        if self.model is None:
            raise ValueError("Model is not set up. Call setup_model() or load() first.")
        reward_callback = RewardCallback(verbose=1)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=reward_callback,
            reset_num_timesteps=False
        )
        rewards = reward_callback.rewards if reward_callback.rewards else []

        return rewards

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def load(self, path):
        """
        Carga un modelo desde el disco usando el método de clase correcto.
        """
        print(f"Loading model from {path}...")
        self.model = self.algorithm_class.load(
            path,
            env=self.env_info.get('env')
        )

    def save(self, path):
        if self.model is None:
            raise ValueError("Agent not initialized. Train or load a model first.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        print(f"Model saved to {save_path}")

    def stop(self):
        """Método para cerrar el entorno del agente."""
        if self.model and self.model.get_env():
            print("Closing agent's environment.")
            self.model.get_env().close()
