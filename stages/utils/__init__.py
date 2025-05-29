from pathlib import Path


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
        model_class (class): The class of the model to load (e.g., PPO, QLearning).
        env (gym.Env, optional): The environment, required for models like QLearning.

    Returns:
        model: An instance of the loaded model.
    """
    # QLearning models require the environment to be passed during instantiation
    if model_class.__name__ == "QLearning":
        if env_info is None:
            raise ValueError("QLearning requires an environment to be loaded.")
        model = model_class(env_info)
        model.load(model_path)
        return model

    # Ray Model
    base_path = f"{model_name}_{difficulty}"
    load_to = Path(model_path) if model_path else model_path / base_path
    load_to = load_to.resolve()

    model = model_class(model_name, difficulty, env_info)
    model.load(str(load_to))
    return model


def get_env_by_name(name, envs):
    return next((env for env in envs if env["name"] == name), None)
