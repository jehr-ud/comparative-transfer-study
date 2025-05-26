import gc
import atexit

from environments.visual_maze_env import VisualMazeEnv
import ray
from ray.tune.registry import register_env

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
    type_algorithms: str
):
    if not ray.is_initialized():
        ray.init()
        atexit.register(ray.shutdown)

    for env_info in envs:
        env_name = env_info.get('name')

        print(f"\n📦 Evaluating environment: {env_name}")
        curves_dict = {}

        for algorithm in algorithms:
            if not ray.is_initialized():
                ray.init()

            register_env("visual_env", env_creator)

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
                params_train=algorithm.get('params_train', {})
            )
            curves_dict[algo_name] = rewards
            print("✅ Training finished")

            print("📊 Evaluation starting..")

            file_results = f"{algo_name}_{env_name}_metrics.csv"
            path = f"results/{type_algorithms}/{file_results}"
            evaluate_agent(
                agent,
                algorithm.get('name'),
                env_info,
                path,
                type_algorithms,
                params_predict=algorithm.get('params_predict')
            )
            print("📊 Evaluation finished")

            if hasattr(agent, "env") and agent.env is not None:
                try:
                    agent.env.close()
                except Exception as e:
                    print(f"Could not close environment: {e}")

            if hasattr(agent, "stop"):
                try:
                    agent.stop()
                except Exception as e:
                    print(f"[ERROR] Could not stop model: {e}")

            ray.shutdown()

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
    experiments
):
    # transfer evaluation
    print("\n🔄 Transfer learning evaluation")
    if not ray.is_initialized():
        ray.init()
        atexit.register(ray.shutdown)

    register_env("visual_env", env_creator)

    for algorithm in algorithms:
        evaluate_transfer_learning(
            algorithm.get('name'),
            algorithm.get('class'),
            experiments,
            "transfer"
        )

    ray.shutdown()
    gc.collect()
