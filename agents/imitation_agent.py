from stable_baselines3 import PPO
from imitation.data import rollout
from imitation.algorithms import bc


def train_imitation(env):
    # El experto es un modelo previamente entrenado
    expert_model = PPO("MlpPolicy", env, verbose=1)
    expert_model.learn(total_timesteps=50000)

    # Guardamos el modelo experto para utilizarlo en el entrenamiento por imitación
    expert_model.save("expert_model")

    # Ahora, podemos obtener las demostraciones del experto para entrenamiento por imitación
    expert_rollouts = rollout.generate_rollouts(expert_model, env, n_episodes=10)

    # Usamos el Behavior Cloning (BC) para aprender de las demostraciones del experto
    bc_model = bc.BehaviorClone(expert_model.policy, expert_rollouts)
    bc_model.train(n_epochs=10)

    return bc_model


def train_transfer_imitation(env_source, env_target):
    """
    El agente aprende por imitación de un experto (BC) y luego ajusta su política usando refuerzo.
    """

    # Paso 1: Entrenamiento por imitación (Behavior Cloning)
    print("Training agent using imitation learning...")
    bc_model = train_imitation(env_source)

    # Paso 2: Transferencia de habilidades de imitación al entorno objetivo
    print("Transferring imitation policy to target environment...")
    # Inicializamos el modelo con la política aprendida por el BC (imitación)
    model = PPO("MlpPolicy", env_target, verbose=1)
    model.set_parameters(bc_model.policy.get_parameters())  # Transferimos los parámetros del BC

    # Paso 3: Ajuste fino usando refuerzo en el entorno objetivo
    print("Fine-tuning agent using reinforcement learning...")
    model.learn(total_timesteps=100000)  # Ajuste fino con refuerzo

    return model
