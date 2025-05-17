from stages.evaluation import (
    evaluate_agent,
    evaluate_transfer_learning,
    plot_learning_curves,
    save_learning_curves
)


def run_training_and_evaluation(
    envs,
    algorithms: list
):

    for env_name, env_pair in envs.items():
        print(f"\n📦 Evaluating environment: {env_name}")
        curves_dict = {}

        for algorithm in algorithms:
            algo_name = f"{algorithm.get('name')}"

            print(f"🚀 Training with {algo_name}")
            # necessary params: class, name of algorithm, env and difficulty
            function_train = algorithm.get('train_function')
            agent, rewards = function_train(
                algorithm.get('class'),
                model_name=algorithm.get('name'),
                env=env_pair['train'],
                difficulty=env_name
            )
            curves_dict[algo_name] = rewards
            print("✅ Training finished")

            print("📊 Evaluation starting..")
            evaluate_agent(
                agent,
                env_pair['validation'],
                filename=f"results/{algo_name}_{env_name}_metrics.csv",
                params_predict=algorithm.get('params_predict')
            )
            print("📊 Evaluation finished")

        plot_learning_curves(env_name, curves_dict, output_file=f"results/learning_curves_{env_name}.png")
        save_learning_curves(curves_dict, filename=f"results/learning_curves_{env_name}.csv")

    return curves_dict


def run_transfer_comparation(
    algorithms: list,
    transfer_envs=None
):
    # transfer evaluation
    if transfer_envs:
        print("\n🔄 Transfer learning evaluation")
        for algorithm in algorithms:
            evaluate_transfer_learning(
                algorithm.get('name'),
                algorithm.get('class'),
                transfer_envs,
            )
