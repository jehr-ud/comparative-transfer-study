import os

import pickle
import numpy as np

from stages.utils import load_model
from agents.imitation_agent import ImitationMarWilTransfer


TRAINING_EXPERT_CONFIG = {
    "simple": {"num_episodes": 1000},
    "medium": {"num_episodes": 2000},
    "complex": {"num_episodes": 3000}
}

TRAINING_LEARNER_CONFIG = {
    "simple": {"num_episodes": 1000},
    "medium": {"num_episodes": 2000},
    "complex": {"num_episodes": 3000}
}


def train_imitation_transfer_agent(
    model_class: ImitationMarWilTransfer,
    model_name,
    env,
    difficulty,
    params_train={},
    generate_demonstrations=True
):
    print(f"\n🔧 Training MARWIL imitation model for difficulty: {difficulty}")

    base_path = "models/demonstrations"
    os.makedirs(base_path, exist_ok=True)

    print(params_train)
    expert_info = params_train.get("expert", {}).get("model")

    expert_name = expert_info.get('name')
    expert_class = expert_info.get('class')

    expert_path_template = params_train.get("expert", {}).get("path", "")
    expert_path = expert_path_template.format(model=expert_name, env=difficulty)
    traj_file_path = f"{base_path}/{difficulty}_{expert_name}_trajectories.pkl"

    print("search in")
    print(expert_path)

    if generate_demonstrations:
        print("📦 Generating expert demonstrations...")
        expert = load_model(expert_path, expert_class, env)
        print(f"✅ Loaded expert from {expert_path}")

        trajectories = []
        num_episodes = TRAINING_EXPERT_CONFIG.get(difficulty, TRAINING_EXPERT_CONFIG["simple"])["num_episodes"]

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            traj = {"states": [], "actions": [], "rewards": [], "dones": []}

            while not done:
                obs = np.array(obs)
                action, _ = expert.predict(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                traj["states"].append(obs.copy())
                traj["actions"].append(action.copy())
                traj["rewards"].append(reward)
                traj["dones"].append(done)

                obs = next_obs

            for key in traj:
                traj[key] = np.array(traj[key])
            trajectories.append(traj)

        with open(traj_file_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(
            f"💾 Saved {len(trajectories)} expert trajectories to {traj_file_path}"
        )

    imitation_model = model_class(env)
    imitation_model = imitation_model.load(traj_file_path)

    print("🚀 Starting imitation training...")
    iterations = TRAINING_LEARNER_CONFIG[difficulty]

    rewards = []
    for i in range(iterations):
        result = imitation_model.learn()
        episode_reward_mean = result.get("module_episode_returns_mean", {}).get("default_policy", 0)
        print(f"[Iter {i}] Mean reward: {episode_reward_mean}")
        rewards.append(episode_reward_mean)

    imitation_model.save(f"{difficulty}_{model_name}")

    return imitation_model, rewards
