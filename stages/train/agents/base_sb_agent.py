import os
import numpy as np


def get_model_params(model_name, difficulty):
    configs = {
        "DQN": {
            "simple":  {"total_timesteps": 50000, "learning_rate": 1e-3, "buffer_size": 10000, "batch_size": 32},
            "medium":  {"total_timesteps": 1000000, "learning_rate": 5e-4, "buffer_size": 20000, "batch_size": 64},
            "complex": {"total_timesteps": 2500000, "learning_rate": 1e-4, "buffer_size": 50000, "batch_size": 128}
        },
        "PPO": {
            "simple":  {"total_timesteps": 50000, "learning_rate": 3e-4, "n_steps": 128, "batch_size": 32},
            "medium":  {"total_timesteps": 1000000, "learning_rate": 2.5e-4, "n_steps": 256, "batch_size": 64},
            "complex": {"total_timesteps": 2500000, "learning_rate": 2e-4, "n_steps": 512, "batch_size": 128}
        },
        "A2C": {
            "simple":  {"total_timesteps": 50000, "learning_rate": 7e-4, "n_steps": 5},
            "medium":  {"total_timesteps": 1000000, "learning_rate": 5e-4, "n_steps": 20},
            "complex": {"total_timesteps": 2500000, "learning_rate": 3e-4, "n_steps": 30}
        }
    }
    return configs[model_name][difficulty]


def train_sb_agent(model_class, model_name, env, difficulty, save_path="./models"):
    # configurations
    params = get_model_params(model_name, difficulty)
    total_timesteps = params.get('total_timesteps', 1000)
    del params['total_timesteps']

    # model init
    model = model_class("MlpPolicy", env, verbose=1, **params)

    # train
    model.learn(total_timesteps=total_timesteps)

    # pre-evaluation
    base_env = env.envs[0]
    episode_rewards = base_env.get_episode_rewards() if hasattr(base_env, "get_episode_rewards") else []
    mean_reward = np.mean(episode_rewards[-100:]) if episode_rewards else None
    print(f"[{model_name}-{difficulty}] Recompensa media (últimos 100 episodios): {mean_reward}")

    print(f"Episode rewards collected: {len(episode_rewards)}")

    # save model
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, f"{model_name}_{difficulty}.zip")
    model.save(model_file)
    print(f"Model save in: {model_file}")

    return model, episode_rewards
