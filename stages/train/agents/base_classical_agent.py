from pathlib import Path
import time

from agents.base_agent import ClassicalAgent
from stages.utils import (
    load_progress,
    get_max_iterations,
    save_progress
)


def get_sb_model_params(model_name, difficulty):
    configs = {
        "DQN": {
            "simple": {
                "total_timesteps": 63,
            },
            "medium": {
                "total_timesteps": 108,
            },
            "complex": {
                "total_timesteps": 	231,
            }
        },
        "PPO": {
            "simple": {
                "total_timesteps": 63,
            },
            "medium": {
                "total_timesteps": 108,
            },
            "complex": {
                "total_timesteps": 	231,
            }
        },
        "A2C": {
            "simple": {
                "total_timesteps": 63,
            },
            "medium": {
                "total_timesteps": 108,
            },
            "complex": {
                "total_timesteps": 	231,
            }
        }
    }

    return configs[model_name][difficulty]


def get_episode_targets(difficulty: str) -> int:
    """Returns the target number of episodes based on difficulty."""
    targets = {
        "simple": 500,
        "medium": 1000,
        "complex": 1500,
    }
    if difficulty not in targets:
        options = list(targets.keys())
        raise ValueError(
            f"Difficulty '{difficulty}' not found. Options: {options}"
        )
    return targets[difficulty]


def train_basical_agent(
    agent_class: ClassicalAgent,
    model_name,
    difficulty,
    env_info,
    experiment_number,
    save_path="./models",
    params_train={}
):
    params = get_sb_model_params(model_name, difficulty)
    total_timesteps_per_iteration = params.get('total_timesteps', 1000)
    del params['total_timesteps']

    save_path = Path(save_path).resolve()
    base_path = f"{experiment_number}_{model_name}_{difficulty}"
    final_model_path = save_path / f"{base_path}"
    temp_model_path = save_path / "temporal" / f"{base_path}"

    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_model_path.mkdir(parents=True, exist_ok=True)
    temp_model_file = temp_model_path / f"{base_path}.zip"

    start_iteration = load_progress(model_name, difficulty, experiment_number)
    max_iterations = get_max_iterations(difficulty)
    rewards = []

    print("Creating agent and environment...")
    try:
        agent: ClassicalAgent = agent_class(
            model_name, difficulty, env_info, save_path, **params_train
        )

        if start_iteration > 0 and temp_model_file.exists():
            agent.load(str(temp_model_file))
        else:
            agent.setup_model()

    except Exception as e:
        print(f"[FATAL ERROR] Could not create the agent: {e}")
        return None, []

    i = start_iteration
    try:
        print(f"--- Starting Iteration {i}/{max_iterations-1} ---")
        target_episodes = get_episode_targets(difficulty)

        rewards = agent.train(
            target_episodes=target_episodes,
            max_timesteps=total_timesteps_per_iteration
        )

        save_progress(i + 1, model_name, difficulty, experiment_number)

        i += 1

    except Exception as e:
        print(f"[ERROR] Iteration {i} failed: {e}. Retrying in 10 seconds...")
        time.sleep(10)

    try:
        print("Training complete. Saving final model...")
        agent.save(str(final_model_path / f"{base_path}.zip"))
        print(f"✅ Final model saved to: {final_model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save final model: {e}")

    return agent, rewards
