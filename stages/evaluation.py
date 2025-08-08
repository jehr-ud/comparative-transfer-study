import csv

import matplotlib.pyplot as plt
import numpy as np

from .utils import load_model


def evaluate_agent(
    model,
    model_name,
    env_info,
    filename,
    type_algorithms,
    experiment_number: int,
    num_episodes=100,
    params_predict=None
):
    if not params_predict:
        params_predict = {}

    episode_data = []
    success_count = 0
    convergence_episode = None

    env = env_info.get('env')

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        timeout = False

        if hasattr(model, 'reset') and callable(getattr(model, 'reset')):
            model.reset()

        print(f"evaluating {model}", model_name)
        if model_name == "ADAP-SSN":
            params_predict['env_info'] = env_info

        while not done:
            print("Calling predict with params:", params_predict)
            action = model.predict(obs, **params_predict)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
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
            "Experiment": experiment_number,
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
    convergence_speed = (
        convergence_episode if convergence_episode is not None else 0
    )

    # Save episode-level data
    fieldnames = [
        "Experiment",
        "Episode",
        "Reward",
        "Success",
        "Timeout",
        "Steps"
    ]
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
        writer.writerow(['Experiment', experiment_number])

    # Print summary
    print(f"✅ Success Rate: {success_rate:.2f}")
    print(f"📊 Average Reward: {average_reward:.2f}")
    print(f"📈 Average Steps: {average_steps:.2f}")
    print(f"🚀 Convergence Speed: {convergence_speed}")
    print(f"✅ Experiment: {experiment_number}")

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


def evaluate_transfer_learning(
    model_name,
    model_class,
    experiments,
    type_algorithms,
    experiment_number,
    num_episodes=100
):
    """
    Evaluate a model across different environments
    and save performance metrics.
    """
    all_results = []

    for experiment in experiments:
        target_env = experiment.get('target_env')
        source_models = experiment.get('source_model_paths')

        for source_model in source_models:
            difficulty = source_model.get('difficulty')
            model_path = source_model.get('path')

            print(
                f"[DEBUG] TL evaluation for {model_name} {difficulty}"
            )

            model_path = f"{model_path}"

            model_path = model_path.format(
                model=model_name,
                experiment=experiment_number
            )

            print(f"Loading {model_path}")
            print(target_env)
            loaded_model = load_model(
                model_name,
                model_path,
                model_class,
                difficulty,
                target_env
            )

            experiment = f"{experiment_number}_{model_name}"

            # for trasfer in base line
            if model_name in ["PPO", "DQN", "A2C"]:
                loaded_model.learn(total_timesteps=100000)
                loaded_model.save(f"models/{experiment}_expert_{difficulty}")

            name_file = f"{experiment}_{difficulty}_metrics.csv"
            file = f'results/{type_algorithms}/{name_file}'

            metrics = evaluate_agent(
                loaded_model,
                model_name,
                target_env,
                file,
                type_algorithms,
                num_episodes,
            )

            result = {
                'Target Env': target_env.get('name'),
                'Model': model_name,
                'Env Name': difficulty,
                'Env Size': target_env.get('config').get('size')
            }
            result.update(metrics)
            all_results.append(result)

            if hasattr(loaded_model, "env") and loaded_model.env is not None:
                try:
                    loaded_model.env.close()
                except Exception as e:
                    print(f"Could not close environment: {e}")

            if hasattr(loaded_model, "stop"):
                try:
                    loaded_model.stop()
                except Exception as e:
                    print(f"[ERROR] Could not stop model: {e}")

    path_name = "transfer_evaluation_metrics"
    filename = f"{experiment_number}_{difficulty}_{path_name}.csv"
    filename = f"results/{type_algorithms}/{filename}"

    save_transfer_results(all_results, filename)
    print_transfer_summary(all_results)
    plot_transfer_metrics(all_results, type_algorithms, experiment_number)


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
        print(f"📊 Model: {r['Model']}, Env: {r['Env Name']}")
        print("Size: {r['Env Size']} ")
        print(f"Success Rate: {r.get('Success Rate', 0):.2f}, "
              f"Avg Reward: {r.get('Average Reward', 0):.2f}")


def plot_transfer_metrics(results, type_algorithms, experiment_number):
    """ Plot comparison charts for each metric
        across models and environment sizes.
    """
    if not results:
        return

    models = sorted(set(r['Model'] for r in results))
    metric_names = [
        key for key in results[0].keys() if key not in
        {'Model', 'Env Name', 'Env Size'}
    ]

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
        file = "comparison.png"
        file_name = f"{experiment_number}_{safe_metric}_{file}"
        plt.savefig(
            f"results/{type_algorithms}/{file_name}"
        )


def plot_learning_curves(env_name, curves_dict, output_file):
    plt.figure(figsize=(12, 6))
    for label, rewards in curves_dict.items():
        plt.plot(rewards, label=label)
    plt.xlabel("Iterations")
    plt.ylabel("Reward per Iteration")
    plt.title(f"Learning curves by algorithm for envoiroment {env_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)


def save_learning_curves(curves_dict, filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Episode', 'Reward'])
        for label, rewards in curves_dict.items():
            for episode, reward in enumerate(rewards):
                writer.writerow([label, episode, reward])
