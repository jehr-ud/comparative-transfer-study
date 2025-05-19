from stages.comparation import (
    run_training_and_evaluation,
    run_transfer_comparation
)
from config import (
    envs,
    classical_algorithms,
    transfer_envs,
    transfer_algorithms
)


def run_classical_methods():
    print("🎯 Running classical methods...")
    run_training_and_evaluation(envs, classical_algorithms)
    run_transfer_comparation(classical_algorithms, transfer_envs)


def run_transfer_methods():
    print("🎁 Running transfer methods...")
    run_training_and_evaluation(envs, transfer_algorithms)
    run_transfer_comparation(classical_algorithms, transfer_algorithms)


if __name__ == "__main__":
    choice = input(
        "What do you want to run? (classical / transfer / both): "
    ).strip().lower()

    if choice == "classical":
        run_classical_methods()
    elif choice == "transfer":
        run_transfer_methods()
    else:
        print("Invalid or empty option. Running both methods...")
        run_classical_methods()
        run_transfer_methods()
