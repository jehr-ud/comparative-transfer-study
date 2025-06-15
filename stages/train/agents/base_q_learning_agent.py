from agents.q_learning import QLearning
from stages.utils import (
    load_progress,
    save_progress
)

config = {
    "simple": {
        "total_episodes": 6400,
    },
    "medium": {
        "total_episodes": 5400,
    },
    "complex": {
        "total_episodes": 3400
    }
}


def train_q_learning_agent(
    model: QLearning,
    model_name,
    difficulty,
    env_info,
    experiment_number,
    params_train={}
):
    env = env_info.get("env")
    model = model(env)
    episode_rewards = []

    print(f"Train: {model_name} in {difficulty}")

    start_episode = load_progress(model_name, difficulty, experiment_number)
    total_episodes = config.get(difficulty).get("total_episodes")

    for episode in range(start_episode, total_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = obs
            action_list, _ = model.predict(state, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action_list)
            next_state = obs
            done = terminated or truncated

            model.learn(state, action_list, reward, next_state, done)
            total_reward += reward

        model.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 500 == 0:
            print(
                f"Episode {episode + 1}: Total Reward: {total_reward}, "
                f"Epsilon: {model.epsilon:.4f}"
            )

            initial_obs, _ = env.reset()
            initial_state_for_debug = model.discretize(initial_obs[0])
            if initial_state_for_debug is not None:
                print(f"Q-values for initial state {initial_state_for_debug}: {model.q_table[initial_state_for_debug]}")
            else:
                print("Could not discretize initial state for debug.")

        if (episode + 1) % 1000 == 0 or (episode + 1) == total_episodes:
            save_progress(episode + 1, model_name, difficulty, experiment_number)

    model_path = f"models/{experiment_number}_{model_name}_{difficulty}.pkl"
    model.save(model_path)

    return model, episode_rewards
