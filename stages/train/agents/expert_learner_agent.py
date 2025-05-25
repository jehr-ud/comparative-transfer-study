def train_agent(
    agent_class,
    model_name,
    difficulty,
    env,
    save_path="./models",
    params_train={}
):
    agent = agent_class(model_name, difficulty, env, save_path=save_path)

    initial_obs, _ = env.reset()
    detected_context = agent.learner.detect_context(initial_obs)
    agent.learner.current_context = detected_context
    agent.train(detected_context, num_iterations=params_train.get("num_iterations", 100))

    obs, _ = env.reset()
    done = False
    total_reward = 0

    episode_rewards = []

    while not done:
        action = agent.predict(obs)
        action_list
        print(info)
        total_reward += reward

    print(f"[{model_name}-{difficulty}] Evaluation reward: {total_reward}")
    episode_rewards.append(total_reward)

    return agent, episode_rewards
