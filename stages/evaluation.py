import csv
import matplotlib.pyplot as plt
import numpy as np


def evaluate_agent(model, env, num_episodes=100, filename='results/evaluation_metrics.csv', params_predict={}):
    episode_data = []
    success_count = 0
    convergence_episode = None

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        timeout = False

        while not done:
            action, _states = model.predict(obs, **params_predict)
            obs, reward, done, info = env.step(action)
            reward = float(reward)
            episode_reward += reward
            step_count += 1

        # Check for timeout
        if isinstance(info, list) and len(info) > 0:
            timeout = info[0].get("timeout", False)
        elif isinstance(info, dict):
            timeout = info.get("timeout", False)

        success = episode_reward > 0 and not timeout
        if success:
            success_count += 1
            # First time reaching 100% success from this point?
            if convergence_episode is None and success_count == (episode + 1):
                convergence_episode = episode + 1

        episode_data.append({
            "Episode": episode + 1,
            "Reward": episode_reward,
            "Success": int(success),
            "Timeout": int(timeout),
            "Steps": step_count
        })

    # Metrics
    total_rewards = [d["Reward"] for d in episode_data]
    total_steps = [d["Steps"] for d in episode_data]
    success_rate = success_count / num_episodes
    average_reward = np.mean(total_rewards)
    average_steps = np.mean(total_steps)
    convergence_speed = convergence_episode if convergence_episode is not None else "Not reached"

    # Save episode-level data
    fieldnames = ["Episode", "Reward", "Success", "Timeout", "Steps"]
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_data)

    # Save summary
    summary_filename = filename.replace(".csv", "_summary.csv")
    with open(summary_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Success Rate', success_rate])
        writer.writerow(['Average Reward', average_reward])
        writer.writerow(['Average Steps per Episode', average_steps])
        writer.writerow(['Convergence Speed (Episode)', convergence_speed])

    # Print summary
    print(f"✅ Success Rate: {success_rate:.2f}")
    print(f"📊 Average Reward: {average_reward:.2f}")
    print(f"📈 Average Steps: {average_steps:.2f}")
    print(f"🚀 Convergence Speed: {convergence_speed}")

    # Plot bar chart of metrics
    plot_metrics({
        "Success Rate": success_rate,
        "Average Reward": average_reward,
        "Average Steps": average_steps / max(total_steps),  # Normalize for plotting
    }, "results/evaluation_metrics_plot.png")

    # Plot rewards over episodes
    plot_episode_rewards(episode_data, "results/evaluation_reward_per_episode.png")

    return {
        "success_rate": success_rate,
        "average_reward": average_reward,
        "average_steps": average_steps,
        "convergence_speed": convergence_speed
    }


def plot_metrics(metrics_dict, save_path):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['green', 'blue', 'purple'])
    plt.title('Agent Evaluation Metrics')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_episode_rewards(episode_data, save_path):
    episodes = [d["Episode"] for d in episode_data]
    rewards = [d["Reward"] for d in episode_data]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label='Reward', color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode (Evaluation)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_model(model_path, model_class, env=None):
    if model_class.__name__ == "QLearning":
        assert env is not None, "QLearning needs an environment to load."
        model = model_class(env)
        model.load(model_path)
        return model
    else:
        return model_class.load(model_path)


def evaluate_transfer_learning(
    model_name,
    model_class,
    environments,
    num_episodes=10
):
    """
    Evaluate a model across different environments and save performance metrics.
    """
    all_results = []

    for env_info in environments:
        source_env = env_info.get('source')
        for model_info in env_info.get('target_model_paths'):
            env_label = model_info.get('name')
            model_path_template = model_info.get('path')
            model_path = model_path_template.format(model=model_name)

            loaded_model = load_model(model_path, model_class, source_env.get('env'))
            metrics = evaluate_agent(
                loaded_model,
                source_env.get('env'),
                num_episodes,
                filename=f'results/{model_name}_{env_label}_metrics.csv'
            )

            result = {
                'Target Env': source_env.get('name'),
                'Model': model_name,
                'Env Name': env_label,
                'Env Size': source_env.get('env').envs[0].size
            }
            result.update(metrics)

            all_results.append(result)

    filename = f"results/{model_name}_evaluation_metrics.csv"

    save_transfer_results(all_results, filename)
    print_transfer_summary(all_results)
    plot_transfer_metrics(all_results)


def save_transfer_results(results, filename):
    """Save the evaluation results into a CSV file."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_transfer_summary(results):
    """Print the evaluation results in a readable format."""
    for r in results:
        print(f"📊 Model: {r['Model']}, Env: {r['Env Name']}, Size: {r['Env Size']}, "
              f"Success Rate: {r.get('Success Rate', 0):.2f}, "
              f"Avg Reward: {r.get('Average Reward', 0):.2f}")


def plot_transfer_metrics(results):
    """Plot comparison charts for each metric across models and environment sizes."""
    if not results:
        return

    models = sorted(set(r['Model'] for r in results))
    metric_names = [key for key in results[0].keys() if key not in {'Model', 'Env Name', 'Env Size'}]

    for metric in metric_names:
        plt.figure(figsize=(10, 6))
        for model in models:
            values = [r[metric] for r in results if r['Model'] == model]
            sizes = [r['Env Size'] for r in results if r['Model'] == model]
            plt.plot(sizes, values, marker='o', label=model)

        plt.xlabel('Environment Size')
        plt.ylabel(metric)
        plt.title(f'Transfer Learning: {metric} Comparison')
        plt.legend()
        plt.tight_layout()
        safe_metric = metric.replace(" ", "_").lower()
        plt.savefig(f'results/{safe_metric}_comparison.png')


def plot_learning_curves(env_name, curves_dict, output_file="results/learning_curves.png"):
    plt.figure(figsize=(12, 6))
    for label, rewards in curves_dict.items():
        plt.plot(rewards, label=label)
    plt.xlabel("Episodes")
    plt.ylabel("Reward per episode")
    plt.title(f"Learning curves by algorithm for envoiroment {env_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)


def save_learning_curves(curves_dict, filename="results/learning_curves.csv"):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Episode', 'Reward'])
        for label, rewards in curves_dict.items():
            for episode, reward in enumerate(rewards):
                writer.writerow([label, episode, reward])
