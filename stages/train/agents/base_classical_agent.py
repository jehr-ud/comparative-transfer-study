from pathlib import Path
from agents.rl_classical import RLLibAgent


def train_rllib_agent(
    agent_class: RLLibAgent,
    model_name,
    difficulty,
    env_info,
    save_path="./models",
    params_train={}
):
    agent: RLLibAgent = agent_class(model_name, difficulty, env_info, save_path)

    for i in range(1000):
        result = agent.train()
        print(f"Iter {i}: reward = {result['episode_reward_mean']:.2f}")

    agent.train()

    env = env_info.get('env')

    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(info)
        done = terminated or truncated
        total_reward += reward

    print(f"[{model_name}-{difficulty}] Evaluation reward: {total_reward}")

    model_path = f"models/{model_name}_{difficulty}"
    save_to = Path(model_path) if model_path else model_path / f"{model_name}_{difficulty}"
    save_to = save_to.resolve()
    agent.save(str(save_to))

    return agent, total_reward
