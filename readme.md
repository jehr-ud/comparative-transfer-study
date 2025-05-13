# 🧠 Visual Maze Reinforcement Learning Project

This project implements training, evaluation, and transfer learning using popular RL algorithms (DQN, PPO, A2C) in a custom visual maze environment.


---

## 📁 Project Structure

├── agents/
│ ├── base_agent.py # Training and evaluation logic for Stable-Baselines3 agents
│
├── environments/
│ └── init.py
│ └── visual_maze.py # Custom maze environment definition
│ └── create_env.py # Factory function to create environments of different difficulty
│
├── stages/
│ ├── evaluation.py # Evaluation and transfer learning functions
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
├── README.md # You are here


## 🛠️ Installation

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

```bash
python main.py
```

## 📊 Output Artifacts

in folder results:

- learning_curves.csv & learning_curves.png – Training performance over episodes.

- evaluation_metrics.csv – Success rate and reward in unseen mazes.

In models/ 

– Trained .zip models for each difficulty.

## 🤖 Algorithms Used

- DQN
- PPO
- A2C


## 📚 Requirements

See requirements.txt. Key libraries include:

- stable-baselines3
- gym
- matplotlib
- numpy

### 📬 Contact
For questions, feel free to open an issue or email jehernandezr@udistrital.edu.co.