from agents.q_learning import QLearning


TRAINING_CONFIG = {
    "simple": {
        "num_episodes": 1000,
        "max_steps_per_episode": 100
    },
    "medium": {
        "num_episodes": 2000,
        "max_steps_per_episode": 200
    },
    "complex": {
        "num_episodes": 3000,
        "max_steps_per_episode": 300
    }
}


def train_q_learning_agent(
    model,
    model_name,
    env,
    difficulty
):
    model: QLearning = model(env)
    episode_rewards = []

    print(f"Train: {model_name} in {difficulty}")

    config = TRAINING_CONFIG.get(difficulty, TRAINING_CONFIG["simple"])
    num_episodes = config["num_episodes"]
    max_steps_per_episode = config["max_steps_per_episode"]

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        done = False

        for _ in range(max_steps_per_episode):
            action, _ = model.predict(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            print("Information of step", info)

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            model.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        model.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 500 == 0:
            print(
                f"Episode {episode + 1}: Total Reward: {total_reward}, "
                f"Epsilon: {model.epsilon:.4f}"
            )

    model_path = f"models/{model_name}_{difficulty}.pkl"
    model.save(model_path)

    return model, episode_rewards
