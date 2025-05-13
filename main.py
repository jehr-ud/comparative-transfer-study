from stages.comparation import (
    run_training_and_evaluation,
    run_transfer_comparation
)
from config import envs, algorithms, transfer_envs, model_paths


if __name__ == "__main__":
    run_training_and_evaluation(
        envs,
        algorithms
    )

    run_transfer_comparation(
        algorithms,
        model_paths,
        transfer_envs
    )
