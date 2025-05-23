def load_model(model_path, model_class, env=None):
    if model_class.__name__ == "QLearning":
        assert env is not None, "QLearning needs an environment to load."
        model = model_class(env)
        model.load(model_path)
        return model
    else:
        return model_class.load(model_path)
