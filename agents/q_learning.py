import numpy as np
import os
import pickle
import gymnasium as gym


class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, bins=10):
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
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.bins = bins  # Almacenar el número de bins

        # Discretización del espacio de observación
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.state_size = (env.observation_space.n,)
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.state_size = (bins,) * env.observation_space.shape[0]
            self.observation_space_low = env.observation_space.low
            self.observation_space_high = env.observation_space.high
        else:
            raise ValueError("Espacio de observación no soportado.")

        self.q_table = np.zeros(self.state_size + (self.action_size,))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def discretize(self, state):
        if isinstance(state, tuple):
            state = state[0]

        state = np.clip(state, self.observation_space_low, self.observation_space_high)

        discrete_state = np.floor((state - self.observation_space_low) / (self.observation_space_high - self.observation_space_low) * self.bins).astype(int)

        discrete_state = np.clip(discrete_state, 0, self.bins - 1)

        return tuple(discrete_state)

    def learn(self, state, action, reward, next_state, done):
        """
        Método para actualizar la Q-table usando la fórmula de Q-learning.
        """
        state = self.discretize(state)
        next_state = self.discretize(next_state)

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (self.gamma * self.q_table[next_state][best_next_action] if not done else 0)
        td_delta = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.alpha * td_delta

        if not done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def predict(self, state, deterministic=True):
        if isinstance(state, (list, np.ndarray)) and len(state) == 1:
            state = state[0]  # extract [[...]] → [...]

        discrete_state = self.discretize(state)

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
        else:
            print(f"El archivo {filename} no existe.")

    def update(self, state, action, reward, next_state, done):
        def scalarize(x):
            # Convierte arrays o listas de un solo valor a enteros
            if isinstance(x, (np.ndarray, list)) and len(x) == 1:
                return int(x[0])
            elif isinstance(x, np.ndarray):
                return int(x.item()) if x.size == 1 else int(x.flat[0])  # Primer valor
            return int(x)

        # Asegura que todos los valores estén en forma escalar (enteros)
        state = tuple(scalarize(s) for s in self.discretize(state))
        next_state = tuple(scalarize(s) for s in self.discretize(next_state))

        # Validar índices
        if any(s >= dim for s, dim in zip(state, self.q_table.shape[:-1])) or \
        any(s >= dim for s, dim in zip(next_state, self.q_table.shape[:-1])):
            print(f"Índices fuera de rango: {state}, {next_state}")
            return

        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max * (1 - int(done)) - old_value)
        self.q_table[state][action] = new_value


