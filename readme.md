# 🧠 Visual Maze Reinforcement Learning Project

This project implements training, evaluation, for RL algorithms (DQN and CIT-SNN) in a custom visual maze environment.


---

## 🛠️ Installation

require python 3.10

### 1. Clone the repository

- clone the repo
```bash
git clone url
cd visual-maze-rl
```

- envoiroment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

- intall dependencies

```bash
pip install -r requirements.txt
```

### 🚀 Running the Project

Modify this file:

```
config/__init__.py
```

```bash
python main.py
```

## How run the five experiments:

run main.py

For DQN, choose classical run type, and train 'y'

For CIT-SNN choose transfer run type, and train 'y'

After the train opcionally can put train in 'n' to show the maze.

## 📊 Output Artifacts

in folder results:

- learning_curves.csv & learning_curves.png – Training performance over episodes.

- evaluation_metrics.csv – Success rate and reward in unseen mazes.

In models/ 

– Trained models for each difficulty.

## 🤖 Algorithms Used

- DQN
- CIT-SNN


## 📚 Requirements

See requirements.txt. Key libraries include:

- stable-baselines3
- gym
- matplotlib
- numpy

### 📬 Contact
For questions, feel free to open an issue or email jehernandezr@udistrital.edu.co


## 📁 Project Structure
```
├── agents/
│ ├── base_agent.py # Training logic for this algorithm
│ ├── ssn_agent.py # it's the Custom Cognitive Architecture
│
├── environments/
│ └── init.py
│ └── visual_maze_env.py # Custom maze environment definition
│
├── stages/
│ ├── evaluation.py # Evaluation functions
│ ├── comparation.py # Generate the comparation 
│ │── train/ # functions to train according with the directory "agents"
│
├── results/
├── learning_curves.csv # CSV output with reward curves
├── learning_curves.png # Plot of learning progress
├── evaluation_metrics.csv # Evaluation results for transfer learning
│
├── models/ # Directory for saved models
│
├── main.py # Main training and evaluation runner
├── requirements.txt # Python dependencies
├── requirements-lock.txt # Python dependencies versions
├── README.md # You are here
```