import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, max_episodes=500, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done and 'episode' in self.locals['infos'][i]:
                reward = self.locals['infos'][i]['episode']['r']
                self.episode_rewards.append(reward)
                if self.verbose:
                    print(f"Episodio {len(self.episode_rewards)} - reward: {reward:.2f}")
        # Detener entrenamiento si llegamos a 500 episodios
        return len(self.episode_rewards) < self.max_episodes

# Crear entorno
env = gym.make("CartPole-v1")

# Crear modelo
model = DQN("MlpPolicy", env, verbose=0)

# Instanciar callback con límite de 500 episodios
reward_callback = RewardCallback(max_episodes=500, verbose=1)

# Entrenar el modelo (timesteps muy altos, porque se detendrá por episodios)
model.learn(total_timesteps=1_000_000, callback=reward_callback)

# Resultado: lista de recompensas por episodio
rewards = reward_callback.episode_rewards
print(f"\nSe registraron {len(rewards)} episodios.")
print(f"Recompensa promedio: {sum(rewards)/len(rewards):.2f}")
