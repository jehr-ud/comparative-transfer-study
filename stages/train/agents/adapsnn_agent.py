from pathlib import Path
from agents.adapsnn_agent import PrefrontalCortex


def train_agent(
    model_class,
    model_name,
    env_info,
    difficulty,
    experiment_number,
    params_train=None,
    save_path="./models"
):
    if params_train is None:
        params_train = {}

    rewards = []
    save_path = Path(save_path).resolve()
    path_suffix = f"{experiment_number}_{model_name}_{difficulty}"
    final_model_dir = save_path / path_suffix
    temp_model_dir = save_path / "temporal" / path_suffix

    temp_model_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    pfc = PrefrontalCortex(atlas_path="atlas_cerebral_principal.pkl")

    training_params = {
        'training_episodes': 500,       # Episodios totales si aprende de cero
        'identification_episodes': 30   # Episodios para calcular la huella
    }

    expert_agent, rewards = pfc.execute_task(
        maze_name=f"{model_name}_{env_info.get('name')}",
        env_info=env_info,
        training_episodes=training_params['training_episodes'],
        identification_episodes=training_params['identification_episodes']
    )

    print("\n--- SESIÓN DE APRENDIZAJE A LO LARGO DE LA VIDA FINALIZADA ---")
    print(f"La Corteza Prefrontal ahora es experta en {len(pfc.cerebral_atlas)} tipos de laberintos.")
    print("Atlas Cerebral final:", pfc.cerebral_atlas)

    return expert_agent, rewards
