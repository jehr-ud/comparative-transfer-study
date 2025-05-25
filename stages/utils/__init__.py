def load_model(model_path, model_class, env=None):
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
        if env is None:
            raise ValueError("QLearning requires an environment to be loaded.")
        model = model_class(env)
        model.load(model_path)
        return model

    try:
        return model_class.load(model_path, env=env)
    except TypeError:
        # If the model class does not accept the 'env' argument during load
        return model_class.load(model_path)


def get_env_by_name(name, envs):
    return next((env for env in envs if env["name"] == name), None)
