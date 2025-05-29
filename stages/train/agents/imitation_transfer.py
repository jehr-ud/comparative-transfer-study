import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from stages.utils import load_model
from agents.imitation_agent import ImitationMarWilTransfer


TRAINING_EXPERT_CONFIG = {
    "simple": {"num_iterations": 50},
    "medium": {"num_iterations": 100},
    "complex": {"num_iterations": 200}
}

TRAINING_LEARNER_CONFIG = {
    "simple": {"num_iterations": 100},
    "medium": {"num_iterations": 200},
    "complex": {"num_iterations": 400}
}


def train_imitation_transfer_agent(
    model_class: ImitationMarWilTransfer,
    model_name,
    env_info,
    difficulty,
    experiment_number,
    params_train={},
    generate_demonstrations=True,
    save_path="./models"
):
    print(f"\n🔧 Training MARWIL imitation model for difficulty: {difficulty}")

    base_path = "models/demonstrations"
    os.makedirs(base_path, exist_ok=True)

    print(params_train)
    expert = params_train.get("expert", {})
    expert_info = expert.get("expert_info")

    expert_name = expert_info.get('name')
    expert_class = expert_info.get('class')

    expert_path_template = params_train.get("expert", {}).get("path", "")
    expert_path = expert_path_template.format(
        model=expert_name,
        env=difficulty
    )
    path_traj = f"{experiment_number}_{difficulty}_{expert_name}"
    traj_file_path = f"{base_path}/{path_traj}_trajectories.parquet"

    print("search in")
    print(expert_path)

    if not generate_demonstrations:
        print("📦 Generating expert demonstrations...")
        expert = load_model(
            expert_name,
            expert_path,
            expert_class,
            difficulty,
            env_info
        )
        print(f"✅ Loaded expert from {expert_path}")

        trajectories = []
        num_iterations = TRAINING_EXPERT_CONFIG.get(
            difficulty, TRAINING_EXPERT_CONFIG["simple"]
        )["num_iterations"]

        env = env_info.get("env")

        for _ in range(num_iterations):
            obs, _ = env.reset()
            done = False
            traj = {"states": [], "actions": [], "rewards": [], "dones": []}

            while not done:
                obs = np.array(obs)
                action = expert.predict(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                print(info)
                done = terminated or truncated

                traj["states"].append(obs.copy())
                traj["actions"].append(action)
                traj["rewards"].append(reward)
                traj["dones"].append(done)

                obs = next_obs

            for key in traj:
                traj[key] = np.array(traj[key])
            trajectories.append(traj)

        df_rows = []
        for traj in trajectories:
            length = len(traj["states"])
            for i in range(length):
                row = {
                    "obs": traj["states"][i].tolist(),
                    "action": traj["actions"][i],
                    "reward": traj["rewards"][i],
                    "done": traj["dones"][i],
                    "next_obs": traj["states"][i + 1].tolist() if i + 1 < length else traj["states"][i].tolist(),
                }
                df_rows.append(row)

        df = pd.DataFrame(df_rows)
        df.to_parquet(traj_file_path, index=False, engine="pyarrow")
        print(f"💾 Saved expert trajectories to {traj_file_path}")

    print("🚀 Starting imitation training...")
    num_iterations = TRAINING_LEARNER_CONFIG.get(
        difficulty, TRAINING_LEARNER_CONFIG["simple"]
    )["num_iterations"]

    rewards = []
    i = 0

    while i < num_iterations:
        try:
            imitation_model = model_class(
                model_name,
                difficulty,
                env_info,
                save_path
            )
            imitation_model.load(traj_file_path)

            result = imitation_model.learn()
            reward_mean = result.get("module_episode_returns_mean", {}).get("default_policy", None)
            if not reward_mean:
                reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0)
            rewards.append(reward_mean)
            print(f"[Iter {i}] Mean reward: {reward_mean}")
            i += 1  # exitoso

            imitation_model.stop()
        except Exception as e:
            print(f"[⚠️ Iter {i}] Error during training: {e}")
            if imitation_model:
                imitation_model.stop()
            break

    imitation_model.save(f"{experiment_number}_{model_name}_{difficulty}")

    return imitation_model, rewards
