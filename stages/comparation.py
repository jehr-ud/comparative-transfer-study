from environments.visual_maze_env import VisualMazeEnv

from stages.evaluation import (
    evaluate_agent,
    evaluate_transfer_learning,
    plot_learning_curves,
    save_learning_curves
)


def env_creator(cfg):
    return VisualMazeEnv(cfg)


def run_training_and_evaluation(
    envs,
    algorithms: list,
    type_algorithms: str,
    experiment_number: int
):
    for env_info in envs:
        env_name = env_info.get('name')

        print(f"\n📦 Evaluating environment: {env_name}")
        curves_dict = {}

        for algorithm in algorithms:

            algo_name = f"{algorithm.get('name')}"

            print(f"🚀 Training with {algo_name}")
            # necessary params: class, name of algorithm, env and difficulty
            function_train = algorithm.get('train_function')

            # call function
            agent, rewards = function_train(
                algorithm.get('class'),
                model_name=algorithm.get('name'),
                env_info=env_info,
                difficulty=env_name,
                params_train=algorithm.get('params_train', {}),
                experiment_number=experiment_number
            )
            curves_dict[algo_name] = rewards
            print("✅ Training finished")

            print("📊 Evaluation starting..")

            base_path = f"{experiment_number}_{algo_name}_{env_name}"
            file_results = f"{base_path}_metrics.csv"
            path = f"results/{type_algorithms}/{file_results}"

            if agent:
                print("📊 Evaluation start")
                evaluate_agent(
                    agent,
                    algorithm.get('name'),
                    env_info,
                    path,
                    type_algorithms,
                    experiment_number,
                    params_predict=algorithm.get('params_predict')
                )
                print("📊 Evaluation finished")

        path = f"results/{type_algorithms}/{experiment_number}"
        plot_learning_curves(
            env_name,
            curves_dict,
            f"{path}_learning_curves_{env_name}.png"
        )
        save_learning_curves(
            curves_dict,
            f"{path}_learning_curves_{env_name}.csv"
        )

    return curves_dict


def run_transfer_comparation(
    algorithms: list,
    experiments,
    experiment_number: int
):
    # transfer evaluation
    print("\n🔄 Transfer learning evaluation")

    for algorithm in algorithms:
        evaluate_transfer_learning(
            algorithm.get('name'),
            algorithm.get('class'),
            experiments,
            "transfer",
            experiment_number
        )
