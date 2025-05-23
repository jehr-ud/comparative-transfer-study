from agents.q_learning import QLearning


TRAINING_CONFIG = {
    "simple": {
        "num_episodes": 50000,
    },
    "medium": {
        "num_episodes": 1000000,
    },
    "complex": {
        "num_episodes": 2500000,
    }
}


def train_q_learning_agent(model: QLearning, model_name, env, difficulty):
    model = model(env)
    episode_rewards = []

    print(f"Train: {model_name} in {difficulty}")

    config = TRAINING_CONFIG.get(difficulty, TRAINING_CONFIG["simple"])
    num_episodes = config["num_episodes"]

    for episode in range(num_episodes):
        obs = env.reset()
        state = obs[0]  # Extrae la observación del primer (y único) entorno

        total_reward = 0
        done = False

        while not done:
            action_list, _ = model.predict(state, deterministic=False)

            obs, reward, done_array, info = env.step(action_list)
            print("information step")
            print(obs, reward, done_array, info)

            next_state = obs[0]
            reward = reward[0]
            done = done_array[0]

            model.learn(state, action_list, reward, next_state, done)

            state = next_state
            total_reward += reward

        model.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 500 == 0:
            print(
                f"Episode {episode + 1}: Total Reward: {total_reward}, "
                f"Epsilon: {model.epsilon:.4f}"
            )

            initial_obs = env.reset()
            initial_state_for_debug = model.discretize(initial_obs[0])
            if initial_state_for_debug is not None:
                print(f"Q-values for initial state {initial_state_for_debug}: {model.q_table[initial_state_for_debug]}")
            else:
                print("Could not discretize initial state for debug.")

    model_path = f"models/{model_name}_{difficulty}.pkl"
    model.save(model_path)

    return model, episode_rewards
