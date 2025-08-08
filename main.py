import os
import shutil
import json

from stages.comparation import (
    run_training_and_evaluation
)
from stages.usage import show_agent
from config import (
    envs,
    classical_algorithms,
    transfer_algorithms,
)

REQUIRED_DIRS = [
    "results/classical",
    "results/transfer",
    "models",
    "models/temporal"
]

for dir_path in REQUIRED_DIRS:
    os.makedirs(dir_path, exist_ok=True)
    print(f"📂 Checked/created directory: {dir_path}")

RESULTS_DIRS = {
    "classical": "results/classical",
    "transfer": "results/transfer"
}

PROGRESS_FILE = "progress.json"
TOTAL_RUNS = 5


def clean_directories(type):
    for path in RESULTS_DIRS[type]:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"🧹 Cleaned directory: {path}")


def load_progress(choice):
    file_name = PROGRESS_FILE.replace(".json", f"_{choice}.json")
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return json.load(f)
    return {"total": 0}


def save_progress(choice, progress):
    file_name = PROGRESS_FILE.replace(".json", f"_{choice}.json")
    with open(file_name, "w") as f:
        json.dump(progress, f, indent=2)


def run_classical_methods(experiment_number):
    print("🎯 Running classical methods...")
    run_training_and_evaluation(
        envs,
        classical_algorithms,
        "classical",
        experiment_number
    )


def run_transfer_methods(experiment_number):
    print("🎁 Running transfer methods...")
    run_training_and_evaluation(
        envs,
        transfer_algorithms,
        "transfer",
        experiment_number
    )


if __name__ == "__main__":
    choice = input(
        "What do you want to run? (classical / transfer): "
    ).strip().lower()

    need_run_experiments = input(
        "Do you want to train? (y/n): "
    ).strip().lower()

    if need_run_experiments == "y":
        progress = load_progress(choice)
        while progress.get("total", 0) < TOTAL_RUNS:
            experiment_n = progress["total"] + 1

            print(
                f"🔁 Running {experiment_n} / {TOTAL_RUNS} for {choice}"
            )
            if choice == "classical":
                run_classical_methods(experiment_n)
            elif choice == "transfer":
                run_transfer_methods(experiment_n)
            progress["total"] += 1
            save_progress(choice, progress)
            print("✅ Done.")

        print(f"🏁 Finished {TOTAL_RUNS} runs for {choice}.")

    elif need_run_experiments == "n":
        difficulty = input(
            "What do you want to show? (simple / medium / complex): "
        ).strip().lower()

        if choice == "classical":
            show_agent(classical_algorithms, difficulty, envs)
        elif choice == "transfer":
            show_agent(transfer_algorithms, difficulty, envs)
    else:
        print("👋 Bye")
