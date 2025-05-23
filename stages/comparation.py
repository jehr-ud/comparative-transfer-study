from stages.evaluation import (
    evaluate_agent,
    evaluate_transfer_learning,
    plot_learning_curves,
    save_learning_curves
)


def run_training_and_evaluation(
    envs,
    algorithms: list,
    type_algorithms: str
):

    for env_info in envs:
        env = env_info.get('env')
        env_name = env_info.get('name')
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
                env=env,
                difficulty=env_name
            )
            curves_dict[algo_name] = rewards
            print("✅ Training finished")

            print("📊 Evaluation starting..")

            file_results = f"{algo_name}_{env_name}_metrics.csv"
            path = f"results/{type_algorithms}/{file_results}"
            evaluate_agent(
                agent,
                env,
                path,
                type_algorithms,
                params_predict=algorithm.get('params_predict')
            )
            print("📊 Evaluation finished")

        plot_learning_curves(
            env_name,
            curves_dict,
            f"results/{type_algorithms}/learning_curves_{env_name}.png"
        )
        save_learning_curves(
            curves_dict,
            f"results/{type_algorithms}/learning_curves_{env_name}.csv"
        )

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
                "transfer"
            )
