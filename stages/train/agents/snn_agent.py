from pathlib import Path
from agents.ssn_agent import SNNCITgent
from stages.utils import (
    load_progress,
    save_progress
)


config = {
    "simple": {
        "total_timesteps": 500,
    },
    "medium": {
        "total_timesteps": 1000,
    },
    "complex": {
        "total_timesteps": 1500,
    }
}


def train_agent(
    model_class: SNNCITgent,
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
    path_suffix = f"{experiment_number}_{model_name}_{difficulty}"
    final_model_dir = save_path / path_suffix
    temp_model_dir = save_path / "temporal" / path_suffix

    temp_model_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    start_iteration = load_progress(model_name, difficulty, experiment_number)

    episode = start_iteration
    episodes = config.get(difficulty).get('total_timesteps')
    temporal = f"models/temporal/{experiment_number}"
    temp_model_file = Path(f"{temporal}_{model_name}")

    try:
        agent: SNNCITgent = model_class(
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

    initial_epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay_rate = 0.995
    current_epsilon = initial_epsilon

    for episode_num in range(start_iteration, episodes):
        current_epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay_rate ** episode_num))

        reward = agent.train(epsilon=current_epsilon)
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
            f"Episode {episode_num + 1}. Epsilon: {current_epsilon:.4f}. Reward: {reward}"
        )
        episode += 1

    if agent:
        try:
            agent.save(save_dir=final_model_dir)
        except Exception as save_e:
            print(f"[❌] Error saving final model: {save_e}")

    return agent, rewards
