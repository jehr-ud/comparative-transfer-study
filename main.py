from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import json

from stages.comparation import (
    run_training_and_evaluation,
    run_transfer_comparation
)
from stages.usage import show_agent
from config import (
    envs,
    classical_algorithms,
    classical_transfer_experiments,
    transfer_experiments,
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


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"classical": 0, "transfer": 0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def run_classical_methods(experiment_number):
    print("🎯 Running classical methods...")
    run_training_and_evaluation(
        envs,
        classical_algorithms,
        "classical",
        experiment_number
    )
    run_transfer_comparation(
        classical_algorithms,
        classical_transfer_experiments,
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
    run_transfer_comparation(
        transfer_algorithms,
        transfer_experiments,
        experiment_number
    )


if __name__ == "__main__":
    #choice = input(
    #    "What do you want to run? (classical / transfer): "
    #).strip().lower()
    
    choice = "transfer"

    need_run_experiments = "y"
    
    #input(
    #    "Do you want to train? (y/n): "
    #).strip().lower()

    if need_run_experiments == "y":
        progress = load_progress()
        while progress.get(choice, 0) < TOTAL_RUNS:
            experiment_n = progress[choice] + 1

            print(
                f"🔁 Running {experiment_n} / {TOTAL_RUNS} for {choice}"
            )
            if choice == "classical":
                run_classical_methods(experiment_n)
            elif choice == "transfer":
                run_transfer_methods(experiment_n)
            progress[choice] += 1
            save_progress(progress)
            print("✅ Done.")

        print(f"🏁 Finished {TOTAL_RUNS} runs for {choice}.")

    elif need_run_experiments == "n":
        difficulty = input(
            "What do you want to show? (simple / medium / complex): "
        ).strip().lower()

        if choice == "classical":
            show_agent(classical_algorithms, difficulty)
        elif choice == "transfer":
            show_agent(transfer_algorithms, difficulty)
    else:
        print("👋 Bye")
