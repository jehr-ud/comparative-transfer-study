from pathlib import Path
import os
import json


def load_model(
    model_name,
    model_path,
    model_class,
    difficulty,
    env_info
):
    """
    Loads a trained model from disk.

    Args:
        model_path (str): Path to the saved model file.
        model_class (class): The class of the model to load.
        env (gym.Env, optional): The environment.

    Returns:
        model: An instance of the loaded model.
    """
    #base_path = f"{model_name}_{difficulty}"
    #load_to = Path(model_path) if model_path else model_path / base_path
    #load_to = load_to.resolve()

    print("[DEBUG] loading class:")
    print(model_class.__name__)

    model = model_class(model_name, difficulty, env_info)
    model.load(model_path)
    return model


def get_env_by_name(name, envs):
    return next((env for env in envs if env["name"] == name), None)


TRAINING_CONFIG = {
    "simple": {"num_iterations": 1},
    "medium": {"num_iterations": 1},
    "complex": {"num_iterations": 1}
}


def get_progress_file(model_name, experiment_number, difficulty):
    return f"results/advance/progress_train_{experiment_number}_{model_name}_{difficulty}.json"


def save_progress(iteration, model_name, difficulty, experiment_number):
    progress = {
        "iteration": iteration,
        "difficulty": difficulty,
        "experiment_number": experiment_number
    }
    with open(
        get_progress_file(model_name, experiment_number, difficulty),
        "w"
    ) as f:
        json.dump(progress, f, indent=2)


def load_progress(model_name, difficulty, experiment_number):
    file = get_progress_file(model_name, experiment_number, difficulty)
    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)
            return data.get("iteration", 0)
    return 0


def get_max_iterations(difficulty):
    return TRAINING_CONFIG[difficulty]["num_iterations"]
