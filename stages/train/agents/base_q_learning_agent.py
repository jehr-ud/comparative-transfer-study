from agents.q_learning import QLearning


def train_q_learning_agent(model, model_name, env, difficulty, num_episodes=1000, max_steps_per_episode=200):
    model: QLearning = model(env)
    episode_rewards = []

    print(f"Train: {model_name} in {difficulty}")

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # En caso de que sea una tupla
            state = state[0]  # Usamos solo el primer valor del estado

        total_reward = 0
        done = False

        for _ in range(max_steps_per_episode):
            action, _ = model.predict(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            print("Information of step", info)

            if isinstance(next_state, tuple):  # En caso de que también next_state sea una tupla
                next_state = next_state[0]

            model.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        model.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {model.epsilon:.4f}")

    model_path = f"models/{model_name}_{difficulty}.pkl"
    model.save(model_path)

    return model, episode_rewards
