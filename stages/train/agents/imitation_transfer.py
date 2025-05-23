import os
import tempfile

import numpy as np
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.offline import JsonWriter

from utils.load_model import load_model

TRAINING_CONFIG = {
    "simple": {
        "num_episodes": 1000,
    },
    "medium": {
        "num_episodes": 2000,
    },
    "complex": {
        "num_episodes": 3000
    }
}


def train_imitation_transfer_agent(
    model,
    model_name,
    env,
    difficulty,
    train_parameters={}
):
    print(f"Training MARWIL imitation model for environment: {difficulty}")

    # 1. Load expert
    expert_model_name = train_parameters.get("expert", {}).get("model", "PPO")
    expert_path_template = train_parameters.get("expert", {}).get("path", "")
    expert_path = expert_path_template.format(model=expert_model_name, env=difficulty)
    expert = load_model(expert_model_name, expert_path, env)
    print(f"Loaded expert from {expert_path}")

    # 2. Generate demonstrations and save
    tmp_dir = tempfile.mkdtemp()
    writer = JsonWriter(tmp_dir)

    print("Generating expert demonstrations...")

    config = TRAINING_CONFIG.get(difficulty, TRAINING_CONFIG["simple"])
    num_episodes = config["num_episodes"]

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = expert.predict(obs)
            next_obs, reward, done, info = env.step(action)
            writer.write({
                "obs": obs,
                "actions": action,
                "rewards": reward,
                "dones": done,
                "new_obs": next_obs,
            })
            obs = next_obs

    # 3. Configure MARWIL training
    config = (
        MARWILConfig()
        .environment(lambda config: env)
        .offline_data(input_="dataset")
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .training(train_batch_size=200)
    )
    config["input"] = tmp_dir

    # 4. Train with MARWIL
    algo = config.build()
    for _ in range(10):
        result = algo.train()
        print(f"Iteration reward: {result['episode_reward_mean']}")

    # 5. Save MARWIL model
    model_path = os.path.join("models", f"{model_name}_{difficulty}")
    os.makedirs(model_path, exist_ok=True)
    algo.save(model_path)

    # 6. Wrap the policy
    imitation_agent = model(policy=algo.get_policy(), path=model_path)

    # 7. Evaluate the imitation agent
    rewards = []
    for _ in range(5):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = imitation_agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    return imitation_agent, rewards
