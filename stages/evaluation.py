import csv
import matplotlib.pyplot as plt


def evaluate_agent(
    model,
    env,
    num_episodes=100,
    filename='results/evaluation_metrics.csv',
    params_predict={}
):
    success_count = 0
    total_rewards = []
    episode_data = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, **params_predict)
            print("States", _states)
            obs, reward, done, info = env.step(action)
            reward = float(reward)
            episode_reward += reward

        print("info =====")
        print(info)

        timeout = False
        if isinstance(info, list) and len(info) > 0:
            timeout = info[0].get("timeout", False)
        elif isinstance(info, dict):
            timeout = info.get("timeout", False)

        success = episode_reward > 0 and not timeout
        if success:
            success_count += 1
        total_rewards.append(episode_reward)

        episode_data.append({
            "Episode": episode + 1,
            "Reward": episode_reward,
            "Success": int(success),
            "Timeout": int(timeout)
        })

    success_rate = success_count / num_episodes
    average_reward = sum(total_rewards) / num_episodes

    # Save per-episode data
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Episode", "Reward", "Success", "Timeout"])
        writer.writeheader()
        for row in episode_data:
            writer.writerow(row)

    # Also save summary metrics
    summary_filename = filename.replace(".csv", "_summary.csv")
    with open(summary_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Success Rate', success_rate])
        writer.writerow(['Average Reward', average_reward])

    # Print summary
    print(f"✅ Success Rate: {success_rate:.2f}")
    print(f"📊 Average Reward: {average_reward:.2f}")

    # Plot bar chart
    metrics = ['Success Rate', 'Average Reward']
    values = [success_rate, average_reward]

    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=['green', 'blue'])
    plt.title('Agent Evaluation Metrics')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/evaluation_metrics_plot.png")

    # Plot reward per episode
    episode_numbers = [row["Episode"] for row in episode_data]
    episode_rewards = [row["Reward"] for row in episode_data]

    plt.figure(figsize=(10, 5))
    plt.plot(episode_numbers, episode_rewards, label='Reward', color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode (Evaluation)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/evaluation_reward_per_episode.png")

    return success_rate, average_reward


def load_model(model_path, model_class, env=None):
    if model_class.__name__ == "QLearning":
        assert env is not None, "QLearning needs an environment to load."
        model = model_class(env)
        model.load(model_path)
        return model
    else:
        return model_class.load(model_path)


def evaluate_transfer_learning(model_name, model_class, model_paths, environments, num_episodes=2, filename='results/evaluation_metrics.csv'):
    """
    Evaluate the trained agent in several different-sized environments and compare performance.
    """
    results = []

    for model_path in model_paths:
        model_path_result = model_path.format(model=model_name)

        for env in environments:
            model = load_model(model_path_result, model_class, env)
            success_rate, average_reward = evaluate_agent(
                model,
                env,
                num_episodes
            )

            results.append({
                'Model': model_path,
                'Env Size': env.envs[0].size,
                'Success Rate': success_rate,
                'Average Reward': average_reward
            })

    # save
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Env Size', 'Success Rate', 'Average Reward'])
        writer.writeheader()
        writer.writerows(results)

    # show results
    for r in results:
        print(f"📊 Model: {r['Model']}, Env Size: {r['Env Size']}, Success Rate: {r['Success Rate']:.2f}, Average Reward: {r['Average Reward']:.2f}")

    plot_comparison(results)


def plot_comparison(results):
    models = sorted(list(set(result['Model'] for result in results)))
    env_sizes = sorted(list(set(result['Env Size'] for result in results)))

    success_rates = {model: {size: 0 for size in env_sizes} for model in models}
    average_rewards = {model: {size: 0 for size in env_sizes} for model in models}

    for r in results:
        success_rates[r['Model']][r['Env Size']] = r['Success Rate']
        average_rewards[r['Model']][r['Env Size']] = r['Average Reward']

    # plot sucess rate
    plt.figure(figsize=(10, 6))
    for model in models:
        plt.plot(env_sizes, [success_rates[model][size] for size in env_sizes], marker='o', label=model)
    plt.xlabel('Environment Size')
    plt.ylabel('Success Rate')
    plt.title('Transfer Learning: Success Rate Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/success_rate_comparison.png')

    # plot average reward
    plt.figure(figsize=(10, 6))
    for model in models:
        plt.plot(env_sizes, [average_rewards[model][size] for size in env_sizes], marker='o', label=model)
    plt.xlabel('Environment Size')
    plt.ylabel('Average Reward')
    plt.title('Transfer Learning: Average Reward Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/average_reward_comparison.png')


def plot_learning_curves(curves_dict, output_file="results/learning_curves.png"):
    plt.figure(figsize=(12, 6))
    for label, rewards in curves_dict.items():
        plt.plot(rewards, label=label)
    plt.xlabel("Episodes")
    plt.ylabel("Reward per episode")
    plt.title("Learning curves by algorithm and environment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)


def save_learning_curves(curves_dict, filename="results/learning_curves.csv"):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm and Environment', 'Episode', 'Reward'])
        for label, rewards in curves_dict.items():
            for episode, reward in enumerate(rewards):
                writer.writerow([label, episode, reward])
