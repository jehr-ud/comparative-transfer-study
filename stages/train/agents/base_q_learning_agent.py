from agents.q_learning import QLearning


TRAINING_CONFIG = {
    "simple": {
        "num_episodes": 100_000,
    },
    "medium": {
        "num_episodes": 500_000,
    },
    "complex": {
        "num_episodes": 1_000_000,
    }
}


def train_q_learning_agent(model: QLearning, model_name, env, difficulty, params_train={}):
    model = model(env)
    episode_rewards = []

    print(f"Train: {model_name} in {difficulty}")

    config = TRAINING_CONFIG.get(difficulty, TRAINING_CONFIG["simple"])
    num_episodes = config["num_episodes"]

    for episode in range(num_episodes):
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
                f"Episode {episode + 1}: Total Reward: {total_reward}, "
                f"Epsilon: {model.epsilon:.4f}"
            )

            initial_obs, _ = env.reset()
            initial_state_for_debug = model.discretize(initial_obs[0])
            if initial_state_for_debug is not None:
                print(f"Q-values for initial state {initial_state_for_debug}: {model.q_table[initial_state_for_debug]}")
            else:
                print("Could not discretize initial state for debug.")

    model_path = f"models/{model_name}_{difficulty}.pkl"
    model.save(model_path)

    return model, episode_rewards
