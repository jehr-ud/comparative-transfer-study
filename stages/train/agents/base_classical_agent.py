from pathlib import Path
import time
import os
import json
import shutil

from agents.ray_agent import RLLibAgent

TRAINING_CONFIG = {
    "simple": {"num_iterations": 100},
    "medium": {"num_iterations": 200},
    "complex": {"num_iterations": 400}
}


def get_progress_file(model_name, experiment_number, difficulty):
    return f"progress_train_{experiment_number}_{model_name}_{difficulty}.json"


def save_progress(iteration, model_name, difficulty, experiment_number):
    progress = {
        "iteration": iteration,
        "difficulty": difficulty,
        "experiment_number": experiment_number
    }
    with open(get_progress_file(model_name, experiment_number, difficulty), "w") as f:
        json.dump(progress, f, indent=2)


def load_progress(model_name, difficulty, experiment_number):
    file = get_progress_file(model_name, experiment_number, difficulty)
    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)
            return data.get("iteration", 0)
    return 0


def get_max_iterations(difficulty):
    return TRAINING_CONFIG[difficulty]["num_iterations"]


def train_rllib_agent(
    agent_class: RLLibAgent,
    model_name,
    difficulty,
    env_info,
    experiment_number,
    save_path="./models",
    params_train={}
):
    save_path = Path(save_path).resolve()
    final_model_path = save_path / f"{experiment_number}_{model_name}_{difficulty}"
    temp_model_path = save_path / "temporal" / f"{experiment_number}_{model_name}_{difficulty}"

    temp_model_path.mkdir(parents=True, exist_ok=True)
    final_model_path.mkdir(parents=True, exist_ok=True)

    start_iteration = load_progress(model_name, difficulty, experiment_number)
    max_iterations = get_max_iterations(difficulty)

    rewards = []
    i = start_iteration

    while i < max_iterations:
        agent = None
        try:
            agent = agent_class(
                model_name,
                difficulty,
                env_info,
                save_path,
                **params_train
            )

            if any(temp_model_path.iterdir()):
                print(f"Iteration {i}: Loading model from {temp_model_path}")
                agent.load(str(temp_model_path))
            else:
                print(f"Iteration {i}: No previous model found, starting from scratch...")

            print(f"Training agent (iteration {i})...")
            result = agent.train()

            reward_mean = result.get("module_episode_returns_mean", {}).get("default_policy")
            if reward_mean is None:
                reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0)
            rewards.append(reward_mean)

            print(f"Iteración {i}, Recompensa promedio: {rewards[-1]}")

            agent.save(str(temp_model_path))
            save_progress(i + 1, model_name, difficulty, experiment_number)

            i += 1

        except Exception as e:
            print(f"[ERROR] Iteration {i} failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

        finally:
            if agent:
                try:
                    agent.stop()
                except Exception as e:
                    print(f"[ERROR] Failed to stop agent cleanly: {e}")

    if agent:
        try:
            agent.save(str(final_model_path))
            shutil.rmtree(temp_model_path, ignore_errors=True)
            print(f"✅ Modelo final guardado en: {final_model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save final model: {e}")

    return agent, rewards
