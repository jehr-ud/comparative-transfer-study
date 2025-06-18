from pathlib import Path
from agents.cit_agent import CITgent
from stages.utils import (
    load_progress,
    save_progress
)


config = {
    "simple": {
        "total_timesteps": 1000,
    },
    "medium": {
        "total_timesteps": 2_000,
    },
    "complex": {
        "total_timesteps": 3_000,
    }
}


def train_agent(
    model_class: CITgent,
    model_name,
    env_info,
    difficulty,
    experiment_number,
    params_train=None,
    save_path="./models"
):
    if params_train is None:
        params_train = {}

    agent = None

    rewards = []
    save_path = Path(save_path).resolve()
    path = f"{experiment_number}_{model_name}_{difficulty}"
    final_model_path = save_path / path
    temp_model_path = save_path / "temporal" / path

    temp_model_path.mkdir(parents=True, exist_ok=True)
    final_model_path.mkdir(parents=True, exist_ok=True)

    start_iteration = load_progress(model_name, difficulty, experiment_number)

    episode = start_iteration
    episodes = config.get(difficulty).get('total_timesteps')
    temporal = f"models/temporal/{experiment_number}"
    temp_model_file = Path(f"{temporal}_{model_name}")

    try:
        agent: CITgent = model_class(
            model_name,
            difficulty,
            env_info,
            save_path=save_path
        )
    except Exception as e:
        print(f"[FATAL ERROR] Could not create the agent: {e}")
        return None, []

    if start_iteration > 0 and temp_model_file.exists():
        print(f"[INFO] Loading model from: {temp_model_file}")
        agent.load(str(temp_model_file))

    for episode_num in range(start_iteration, episodes):
        reward = agent.train()
        rewards.append(reward)

        save_progress(
            episode + 1,
            model_name,
            difficulty,
            experiment_number
        )

        agent.save(
            str(temp_model_file)
        )

        print(
            f"Episode {episode_num + 1}. Reward: {reward}"
        )
        episode += 1

    if agent:
        try:
            agent.save(str(final_model_path))
        except Exception as save_e:
            print(f"[❌] Error saving final model: {save_e}")

    return agent, rewards
