from stages.comparation import (
    run_training_and_evaluation,
    run_transfer_comparation
)
from stages.usage import show_agent
from config import (
    envs,
    classical_algorithms,
    classical_transfer_envs,
    transfer_envs,
    transfer_algorithms
)


def run_classical_methods():
    print("🎯 Running classical methods...")
    run_training_and_evaluation(envs, classical_algorithms, "classical")
    run_transfer_comparation(classical_algorithms, classical_transfer_envs)


def run_transfer_methods():
    print("🎁 Running transfer methods...")
    run_training_and_evaluation(envs, transfer_algorithms, "transfer")
    run_transfer_comparation(classical_algorithms, transfer_envs)


if __name__ == "__main__":
    choice = input(
            "What do you want to run? (classical / transfer): "
        ).strip().lower()

    need_run_experiments = input(
        "do you want to train? (y/n): "
    ).strip().lower()

    if need_run_experiments == "y":
        if choice == "classical":
            run_classical_methods()
        elif choice == "transfer":
            run_transfer_methods()
    elif need_run_experiments == "n":
        difficulty = input(
            "What do you want to show? (simple / medium / complex): "
        ).strip().lower()

        if choice == "classical":
            show_agent(classical_algorithms, difficulty)
        elif choice == "transfer":
            show_agent(transfer_algorithms, difficulty)
    else:
        print("bye")
