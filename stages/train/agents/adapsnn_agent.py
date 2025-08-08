from pathlib import Path
from agents.adapsnn_agent import PrefrontalCortex
from stages.utils import (
    save_progress
)


config = {
    "simple": {
        'training_episodes': 500,       # Episodios totales si aprende de cero
        'identification_episodes': 50   # Episodios para calcular la huella
    },
    "medium": {
        'training_episodes': 1000,       # Episodios totales si aprende de cero
        'identification_episodes': 100   # Episodios para calcular la huella
    },
    "complex": {
       'training_episodes': 1500,       # Episodios totales si aprende de cero
       'identification_episodes': 150   # Episodios para calcular la huella
    }
}


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

    pfc: PrefrontalCortex = model_class(
        atlas_path="models/atlas_cerebral_principal.pkl"
    )

    cong_diff = config.get(difficulty)
    identification_episodes = cong_diff.get('identification_episodes')
    training_episodes = cong_diff.get('training_episodes')

    pfc.handle_task(
        f"{model_name}_{env_info.get('name')}",
        env_info,
        identification_episodes=identification_episodes
    )

    expert_agent, rewards = pfc.train_active_agent(
        maze_name=f"{model_name}_{env_info.get('name')}",
        training_episodes=training_episodes
    )

    save_progress(
        1,
        model_name,
        difficulty,
        experiment_number
    )

    print("\n--- SESIÓN DE APRENDIZAJE A LO LARGO DE LA VIDA FINALIZADA ---")
    print(f"La Corteza Prefrontal ahora es experta en {len(pfc.cerebral_atlas)} tipos de laberintos.")
    print("Atlas Cerebral final:", pfc.cerebral_atlas)

    return expert_agent, rewards
