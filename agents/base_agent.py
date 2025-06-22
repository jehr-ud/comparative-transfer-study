from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class EpisodeInfoCallback(BaseCallback):
    """
    Una devolución de llamada personalizada para:
    1. Registrar la recompensa total de cada episodio.
    2. Detener el entrenamiento después de N episodios.
    3. Opcionalmente, imprimir información de cada episodio en la consola.

    :param n_episodes: El número de episodios en los que entrenar.
    :param log_to_console: Si es True, imprime la información del episodio.
    :param verbose: Nivel de verbosidad.
    """
    def __init__(self, n_episodes: int, log_to_console: bool = True, verbose: int = 0):
        super(EpisodeInfoCallback, self).__init__(verbose)
        self.n_episodes = n_episodes
        self.log_to_console = log_to_console
        
        # Contadores y almacenamiento
        self.episodes_done = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """
        Este método es llamado en cada paso del entorno.
        """
        # Itera sobre los 'dones' para manejar entornos vectorizados
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.episodes_done += 1
                
                # Accede a la información del episodio desde el diccionario 'info'
                info = self.locals['infos'][i]
                
                if 'episode' in info:
                    reward = info['episode']['r']
                    length = info['episode']['l']
                    
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)
                    
                    if self.log_to_console:
                        print(f"Episodio {self.episodes_done} terminado. Recompensa: {reward:.2f}, Longitud: {length}")

        # Comprueba si se debe detener el entrenamiento
        if self.episodes_done >= self.n_episodes:
            if self.verbose > 0:
                print(f"Deteniendo el entrenamiento: se alcanzaron los {self.n_episodes} episodios.")
            return False  # Detiene el entrenamiento

        return True # Continúa el entrenamiento


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
            verbose=0,
            **self.params
        )

    def train(self, target_episodes: int, max_timesteps: int):
        if self.model is None:
            raise ValueError("Model is not set up. Call setup_model() or load() first.")

        # 1. El callback se crea con el número correcto de episodios objetivo.
        episode_callback = EpisodeInfoCallback(n_episodes=target_episodes, log_to_console=True, verbose=1)

        print(f"\nIniciando entrenamiento para {target_episodes} episodios (límite de {max_timesteps} timesteps)...")
        self.model.learn(
            total_timesteps=max_timesteps,
            callback=episode_callback
        )

        rewards = episode_callback.episode_rewards if episode_callback.episode_rewards else []
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
