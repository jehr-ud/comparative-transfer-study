from agents.q_learning import QLearning
from stages.utils import (
    load_progress,
    get_max_iterations,
    save_progress
)


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

    start_iteration = load_progress(model_name, difficulty, experiment_number)
    max_iterations = get_max_iterations(difficulty)

    episode = start_iteration

    while episode < max_iterations:
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action_list, _ = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, _ = env.step(action_list)
            done = terminated or truncated

            next_state = obs
            model.learn(obs, action_list, reward, next_state, done)
            total_reward += reward

        model.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 500 == 0:
            print(
                f"Episode {episode}: Total Reward: {total_reward}, "
                f"Epsilon: {model.epsilon:.4f}"
            )

            initial_obs, _ = env.reset()
            initial_state_for_debug = model.discretize(initial_obs[0])
            if initial_state_for_debug is not None:
                print(f"Q-values for initial state {initial_state_for_debug}: {model.q_table[initial_state_for_debug]}")
            else:
                print("Could not discretize initial state for debug.")

        save_progress(episode + 1, model_name, difficulty, experiment_number)
        episode += 1

    model_path = f"models/{experiment_number}_{model_name}_{difficulty}.pkl"
    model.save(model_path)

    return model, episode_rewards
