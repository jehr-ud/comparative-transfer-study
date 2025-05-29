import numpy as np
import os
import pickle
import gymnasium as gym

CONFIG_PER_ENV = {
    "env_size_6": {
        "bins": 6,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.05,
    },
    "env_size_8": {
        "bins": 8,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.05,
    },
    "env_size_12": {
        "bins": 12,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.05,
    }
}


class QLearning:
    def __init__(self, env):
        """
        Inicializa el agente Q-learning.
        :param env: El entorno en el que va a operar el agente.
        :param alpha: Tasa de aprendizaje.
        :param gamma: Factor de descuento.
        :param epsilon: Probabilidad de exploración (epsilon-greedy).
        :param epsilon_decay: Tasa de decaimiento de epsilon.
        :param epsilon_min: Valor mínimo de epsilon.
        :param bins: Número de bins para discretizar el espacio de observación.
        """
        self.env = env
        self.action_size = env.action_space.n
        env_size = self.env.size
        print(env_size)

        params = CONFIG_PER_ENV.get(f"env_size_{env_size}")

        self.alpha = params.get('alpha')
        self.gamma = params.get('gamma')
        self.epsilon = params.get('epsilon')
        self.epsilon_decay = params.get('epsilon_decay')
        self.epsilon_min = params.get('epsilon_min')
        self.bins = params.get('bins')

        # Discretización del espacio de observación
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.state_size = (env.size, env.size)
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.state_size = (self.bins,) * env.observation_space.shape[0]
            self.observation_space_low = env.observation_space.low
            self.observation_space_high = env.observation_space.high
        else:
            raise ValueError("Espacio de observación no soportado.")

        self.n_actions = env.action_space.n  # actions up, left, top, down
        self.q_table = np.zeros((self.bins, self.bins, self.n_actions))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def discretize(self, state):
        if isinstance(state, tuple) and len(state) > 0 and isinstance(state[0], (np.ndarray, list)):
            state = state[0]
        elif not isinstance(state, np.ndarray):
            state = np.array(state)

        if state.ndim > 1:
            state = state.flatten()

        state = np.clip(state, self.observation_space_low, self.observation_space_high)

        discrete_state_array = np.floor((state - self.observation_space_low) / (self.observation_space_high - self.observation_space_low) * self.bins).astype(int)
        discrete_state_array = np.clip(discrete_state_array, 0, self.bins - 1)
        return tuple(discrete_state_array)

    def learn(self, state, action, reward, next_state, done):
        state = self.discretize(state)
        next_state = self.discretize(next_state)

        if isinstance(action, (list, np.ndarray)):
            action = int(action[0])

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (self.gamma * self.q_table[next_state][best_next_action] if not done else 0)
        td_delta = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.alpha * td_delta

    def predict(self, state, deterministic=True):
        if isinstance(state, (list, np.ndarray)) and len(state) == 1:
            state = state[0]  # extract [[...]] → [...]

        discrete_state = self.discretize(state)

        print(f"[DEBUG] predict state raw: {state}")
        print(f"[DEBUG] discretized: {discrete_state}")

        if deterministic:
            action = int(np.argmax(self.q_table[discrete_state]))
        else:
            action = int(self.env.action_space.sample())

        info = {
            "state": discrete_state,
            "q_values": self.q_table[discrete_state]
        }

        return [action], info

    def save(self, filename):
        """
        Método para guardar el modelo (Q-table) en un archivo usando pickle.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Modelo guardado en: {filename}")

    def load(self, filename):
        """
        Método para cargar un modelo (Q-table) desde un archivo.
        """
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Modelo cargado desde: {filename}")
            self.epsilon = 0.0
        else:
            print(f"El archivo {filename} no existe.")

        print(f"Q-table cargada (primeras 5x5x4 entradas): \n{self.q_table[0:5, 0:5, :]}")
        print(f"Valores Q para estado inicial (0,0): {self.q_table[0, 0, :]}")
