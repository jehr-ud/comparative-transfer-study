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
    curves_dict = {}

    for env_name, env_pair in envs.items():
        print(f"\n📦 Evaluating environment: {env_name}")
        for algorithm in algorithms:
            label = f"{algorithm.get('name')} - {env_name}"

            print(f"🚀 Training with {label}")
            # necessary params: class, name of algorithm, env and difficulty
            function_train = algorithm.get('train_function')
            agent, rewards = function_train(
                algorithm.get('class'),
                model_name=algorithm.get('name'),
                env=env_pair['train'],
                difficulty=env_name
            )
            curves_dict[label] = rewards
            print("✅ Training finished")

            print("📊 Evaluation starting..")
            evaluate_agent(
                agent,
                env_pair['validation'],
                filename=f"results/{label}_metrics.csv",
                params_predict=algorithm.get('params_predict')
            )
            print("📊 Evaluation finished")

    plot_learning_curves(curves_dict)
    save_learning_curves(curves_dict)

    return curves_dict


def run_transfer_comparation(
    algorithms: list,
    model_paths=None,
    transfer_envs=None
):
    # transfer evaluation
    if model_paths and transfer_envs:
        print("\n🔄 Transfer learning evaluation")
        for algorithm in algorithms:
            evaluate_transfer_learning(
                algorithm.get('name'),
                algorithm.get('class'),
                model_paths,
                transfer_envs,
            )
