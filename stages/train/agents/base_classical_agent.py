from pathlib import Path
from agents.ray_agent import RLLibAgent


TRAINING_CONFIG = {
    "simple": {
        "num_iterations": 100,
    },
    "medium": {
        "num_iterations": 1000,
    },
    "complex": {
        "num_iterations": 2500,
    }
}


def train_rllib_agent(
    agent_class: RLLibAgent,
    model_name,
    difficulty,
    env_info,
    save_path="./models",
    params_train={}
):
    agent: RLLibAgent = agent_class(
        model_name,
        difficulty,
        env_info,
        save_path
    )

    train_params = TRAINING_CONFIG[difficulty]

    rewards = []

    for i in range(train_params.get('num_iterations')):
        print(f"Iteration {i} for {model_name}")
        result = agent.train()
        rewards.append(result.get("module_episode_returns_mean", {}).get("default_policy", 0))
        print(f"Iteración {i}, Recompensa promedio: {rewards[-1]}")

    model_path = f"models/{model_name}_{difficulty}"
    save_to = Path(model_path) if model_path else model_path / f"{model_name}_{difficulty}"
    save_to = save_to.resolve()
    agent.save(str(save_to))

    return agent, rewards
