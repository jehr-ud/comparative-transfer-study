from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeInfoCallback(BaseCallback):
    """
    A custom callback to:
    1. Log the total reward for each episode.
    2. Stop training after N episodes.
    3. Optionally, print information about each episode to the console.

    :param n_episodes: The number of episodes to train on.
    :param log_to_console: If True, print the episode information.
    :param verbose: Verbosity level.
    """
    def __init__(
        self,
        n_episodes: int,
        log_to_console: bool = True,
        verbose: int = 0
    ):
        super(EpisodeInfoCallback, self).__init__(verbose)
        self.n_episodes = n_episodes
        self.log_to_console = log_to_console

        self.episodes_done = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """
        This method is called at every step of the environment.
        """
        # Iterate on the 'gifts' for handling vectorized environments
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.episodes_done += 1

                # Access episode information from the 'info' dictionary
                info = self.locals['infos'][i]

                if 'episode' in info:
                    reward = info['episode']['r']
                    length = info['episode']['l']

                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)

                    if self.log_to_console:
                        print(f"Episode {self.episodes_done} finished")
                        print(f"Reward: {reward:.2f}, Length: {length}")

        # Check if training should be stopped
        if self.episodes_done >= self.n_episodes:
            if self.verbose > 0:
                print(
                    f"Stopping training: {self.n_episodes} episodes reached."
                )
            return False  # Stop training

        return True  # Training continues


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
        """Returns the algorithm class (PPO, etc) without instantiating it."""
        if self.name == "PPO":
            return PPO
        elif self.name == "DQN":
            return DQN
        elif self.name == "A2C":
            return A2C
        else:
            raise ValueError("Agent not configured.")

    def setup_model(self):
        """Creates a new instance of the model. Call only for new training."""
        print(f"Setting up a new '{self.name}' model...")
        self.model = self.algorithm_class(
            "MlpPolicy",
            self.env_info.get('env'),
            verbose=0,
            **self.params
        )

    def train(self, target_episodes: int, max_timesteps: int):
        if self.model is None:
            raise ValueError(
                "Model is not set up. Call setup_model() or load() first."
            )

        # 1. The callback is created
        # with the correct number of target episodes.
        episode_callback = EpisodeInfoCallback(
            n_episodes=target_episodes,
            log_to_console=True,
            verbose=1
        )

        print(f"\nStarting training for {target_episodes} episodes")
        print(f"(limit of {max_timesteps} timesteps)...")

        self.model.learn(
            total_timesteps=max_timesteps,
            callback=episode_callback
        )

        rewards = episode_callback.episode_rewards \
            if episode_callback.episode_rewards else []
        return rewards

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def load(self, path):
        """
        Load a model from disk using the correct class method.
        """
        print(f"Loading model from {path}...")
        self.model = self.algorithm_class.load(
            path,
            env=self.env_info.get('env'),
            device='auto'
        )

    def save(self, path):
        if self.model is None:
            raise ValueError(
                "Agent not initialized. Train or load a model first."
            )

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        print(f"Model saved to {save_path}")

    def stop(self):
        """Method to close the agent environment."""
        if self.model and self.model.get_env():
            print("Closing agent's environment.")
            self.model.get_env().close()
