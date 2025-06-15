from pathlib import Path
from agents.cit_agent import CITgent
from stages.utils import (
    load_progress,
    get_max_iterations,
    save_progress
)


config = {
    "simple": {
        "total_timesteps": 2_000,
    },
    "medium": {
        "total_timesteps": 4_000,
    },
    "complex": {
        "total_timesteps": 6_000,
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

    agent: CITgent = model_class(
        model_name,
        difficulty,
        env_info,
        save_path=save_path
    )

    rewards = []
    save_path = Path(save_path).resolve()
    path = f"{experiment_number}_{model_name}_{difficulty}"
    final_model_path = save_path / path
    temp_model_path = save_path / "temporal" / path

    temp_model_path.mkdir(parents=True, exist_ok=True)
    final_model_path.mkdir(parents=True, exist_ok=True)

    start_iteration = load_progress(model_name, difficulty, experiment_number)
    max_iterations = get_max_iterations(difficulty)

    episode = start_iteration

    while episode < max_iterations:
        episodes = config.get(difficulty).get('total_timesteps')
        for episode_num in range(episodes):
            reward = agent.train()
            rewards.append(reward)

            agent.save(str(temp_model_path))
            save_progress(
                episode + 1,
                model_name,
                difficulty,
                experiment_number
            )

            if (episode_num + 1) % 5 == 0:
                print(
                    f"Episode {episode_num + 1}. Reward: {reward:.2f}"
                )
                print("Guardando checkpoint del modelo...")
                temporal = f"models/temporal/{experiment_number}"
                agent.save(
                    f"{temporal}_checkpoint_{episode_num + 1}"
                )
                import time
                time.sleep(5)

            episode += 1

    if agent:
        try:
            agent.save(str(final_model_path))
        except Exception as save_e:
            print(f"[❌] Error saving final model: {save_e}")

    return agent, rewards
