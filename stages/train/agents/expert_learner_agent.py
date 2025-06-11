from pathlib import Path
from agents.cpa_agent import CPAgent
from stages.utils import (
    load_progress,
    get_max_iterations,
    save_progress
)


def train_agent(
    model_class: CPAgent,
    model_name,
    env_info,
    difficulty,
    experiment_number,
    params_train=None,
    save_path="./models"
):
    if params_train is None:
        params_train = {}

    agent: CPAgent = model_class(
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
        reward_mean = agent.train()
        rewards.append(reward_mean)

        agent.save(str(temp_model_path))
        save_progress(episode + 1, model_name, difficulty, experiment_number)
        episode += 1

    if agent:
        try:
            agent.save(str(final_model_path))
        except Exception as save_e:
            print(f"[❌] Error saving final model: {save_e}")

    return agent, rewards
