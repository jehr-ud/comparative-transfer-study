import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import pickle
from d3rlpy.datasets import MDPDataset
from d3rlpy.models.encoders import Encoder
from d3rlpy.algos import DoubleDQNConfig
from d3rlpy.models.encoders import EncoderFactory
from spikingjelly.activation_based import neuron
from collections import defaultdict


# ------------------------- #
# --- MÓDULOS DEL MAPA COGNITIVO SNN --- #
# ------------------------- #

class GridCellModule(nn.Module):
    def __init__(self, map_size=(32, 32), device='cpu'):
        super().__init__()
        self.map_size = map_size
        self.device = device
        self.grid_neurons = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            detach_reset=True
        )
        self.membrane_potential = torch.zeros(1, *map_size, device=self.device)

    def reset(self):
        self.membrane_potential.zero_()
        center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
        self.membrane_potential[
            0, center_x-1:center_x+1, center_y-1:center_y+1
        ] = 1.5

    def update_state(self, velocity_vector):
        dx, dy = int(velocity_vector[0]), int(velocity_vector[1])
        self.membrane_potential = torch.roll(
            self.membrane_potential,
            shifts=(dx, dy),
            dims=(1, 2)
        )
        _ = self.grid_neurons(self.membrane_potential)
        self.membrane_potential = self.grid_neurons.v

    def get_current_spikes(self):
        return (
            self.membrane_potential > self.grid_neurons.v_threshold
        ).float()


class PlaceCellModule(nn.Module):
    def __init__(
        self,
        grid_map_size=(32, 32),
        num_place_cells=256,
        device='cpu'
    ):
        super().__init__()
        grid_flat_size = grid_map_size[0] * grid_map_size[1]
        self.fc = nn.Linear(grid_flat_size, num_place_cells).to(device)
        self.place_neurons = neuron.LIFNode(v_threshold=1.0, detach_reset=True)

    def forward(self, grid_spikes):
        x = grid_spikes.flatten(1).float()
        current_injection = self.fc(x)
        spikes = self.place_neurons(current_injection)
        return spikes


class SensoryNoveltyModule:
    def __init__(self, novelty_threshold=0.5):
        self.novelty_threshold = novelty_threshold
        self.place_memory = {}

    def get_sensory_novelty(self, place_cell_spikes, current_observation):
        if place_cell_spikes.sum() == 0:
            return 1.0

        place_cell_index = torch.argmax(place_cell_spikes).item()

        if place_cell_index not in self.place_memory:
            return 0.9

        remembered_obs = self.place_memory[place_cell_index]
        current_obs_np = current_observation.detach().cpu().numpy().flatten()
        remembered_obs_np = remembered_obs.detach().cpu().numpy().flatten()

        if current_obs_np.shape != remembered_obs_np.shape:
            return 1.0  # Error de forma, tratar como máxima novedad

        sensory_error = np.linalg.norm(current_obs_np - remembered_obs_np)
        return min(sensory_error, 1.0)  # Normalizar para que no sea mayor a 1

    def update_memory(self, place_cell_spikes, current_observation):
        if place_cell_spikes.sum() > 0:
            place_cell_index = torch.argmax(place_cell_spikes).item()
            if place_cell_index not in self.place_memory:
                self.place_memory[place_cell_index] = \
                    current_observation.clone()

# ------------------------- #
# --- MÓDULOS DEL AGENTE -- #
# ------------------------- #


class RecurrentEncoder(Encoder):
    def __init__(self, observation_shape: list[int], feature_size: int = 256):
        super().__init__()
        self.observation_shape = observation_shape
        self.feature_size = feature_size
        self.lstm = nn.LSTM(
            input_size=observation_shape[0],
            hidden_size=feature_size,
            batch_first=True
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        features, _ = self.lstm(x)
        return features[:, -1, :]

    def get_feature_size(self) -> int:
        return self.feature_size


class RecurrentEncoderFactory(EncoderFactory):
    TYPE = "recurrent"

    def create(self, observation_shape: list[int]) -> RecurrentEncoder:
        return RecurrentEncoder(
            observation_shape=observation_shape,
            feature_size=256
        )

    def get_type(self) -> str:
        return self.TYPE


class NeoCortex:
    def __init__(
        self,
        env_info,
        context_name,
        difficulty="simple",
        save_path="./models"
    ):
        self.context_name = context_name
        self.save_path = Path(save_path)
        self.env_info = env_info
        self.name = f"DQN-{difficulty}-{context_name}"
        self.difficulty = difficulty
        self.model = self._get_config()

    def _get_config(self):
        recurrent_encoder_factory = RecurrentEncoderFactory()

        dqn_config = DoubleDQNConfig(
            learning_rate=3e-4,
            gamma=0.995,
            encoder_factory=recurrent_encoder_factory,
            n_critics=1
        )
        dqn = dqn_config.create(device="cpu:0")

        env = self.env_info.get('env')
        dqn.build_with_env(env)

        return dqn

    def predict(self, raw_obs):
        return self.model.predict(np.expand_dims(raw_obs, axis=0))

    def load(self, path):
        try:
            self.model.load_model(str(path))
        except Exception as e:
            print("exception error", e)

    def save(self, path=None):
        if self.model is None:
            raise ValueError("Agent not initialized.")
        self.save_path.mkdir(parents=True, exist_ok=True)
        save_to = Path(path) if path \
            else self.save_path / f"{self.name}_{self.difficulty}.d3"
        self.model.save_model(str(save_to))


class Hippocampo:
    def __init__(
        self,
        env_info: dict,
        difficulty: str,
        snn_output_dim: int = 256
    ):
        self.env_info = env_info
        self.env = env_info.get('env')
        self.difficulty = difficulty
        self.snn_output_dim = snn_output_dim

        snn_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.grid_module = GridCellModule(map_size=(32, 32), device=snn_device)
        self.place_module = PlaceCellModule(
            num_place_cells=snn_output_dim,
            device=snn_device
        )
        self.novelty_module = SensoryNoveltyModule(novelty_threshold=0.3)

        self.action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.experts = {}
        self.current_context = "context_default"
        self.known_contexts_features = []
        self.context_threshold = 0.4
        self.experience_buffer = defaultdict(list)

    def _action_to_velocity(self, action):
        return self.action_map.get(action, (0, 0))

    def reset_episode(self):
        initial_observation, info = self.env.reset()
        self.grid_module.reset()
        return torch.tensor(initial_observation, dtype=torch.float32), info

    def get_expert(self, context_name: str):
        if context_name not in self.experts:
            self.experts[context_name] = NeoCortex(
                self.env_info,
                context_name,
                self.difficulty
            )
        return self.experts[context_name]

    def _detect_context(self, place_cell_spikes):
        if place_cell_spikes.sum() == 0:
            return "context_unknown"

        feature_vec = place_cell_spikes.detach().cpu().numpy().flatten()
        min_dist = float('inf')
        detected_context = None

        for name, known_feat in self.known_contexts_features:
            dist = np.linalg.norm(feature_vec - known_feat)
            if dist < min_dist:
                min_dist = dist
                detected_context = name

        if detected_context is None or min_dist > self.context_threshold:
            new_name = f"context_{len(self.known_contexts_features)}"
            self.known_contexts_features.append((new_name, feature_vec))
            return new_name
        return detected_context

    def train(self):
        obs, info = self.reset_episode()
        print(info)
        terminated, truncated = False, False
        trajectory = []

        total_episode_reward = 0

        context_for_episode = None

        while not (terminated or truncated):
            grid_spikes = self.grid_module.get_current_spikes()
            place_spikes = self.place_module(grid_spikes)
            novelty_score = self.novelty_module.get_sensory_novelty(
                place_spikes,
                obs
            )

            if novelty_score > self.novelty_module.novelty_threshold:
                action = self.env.action_space.sample()
                context_for_episode = self._detect_context(place_spikes)
            else:
                if context_for_episode is None:
                    context_for_episode = self._detect_context(place_spikes)
                expert = self.get_expert(self.current_context)
                action = expert.predict(obs.numpy())[0]

            next_obs, reward, terminated, truncated, info = \
                self.env.step(action)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)

            total_episode_reward += reward

            velocity_vector = self._action_to_velocity(action)
            self.grid_module.update_state(velocity_vector)

            next_grid_spikes = self.grid_module.get_current_spikes()
            next_place_spikes = self.place_module(next_grid_spikes)
            self.novelty_module.update_memory(next_place_spikes, next_obs)

            trajectory.append(
                (
                    obs.numpy(),
                    action,
                    reward,
                    next_obs.numpy(),
                    terminated,
                    truncated
                )
            )
            obs = next_obs

        if trajectory and context_for_episode is not None:
            self.experience_buffer[context_for_episode].extend(trajectory)
            self._update_expert_from_buffer(context_for_episode)

        print(trajectory)

        print("Training  episode finished.")
        return total_episode_reward

    def _update_expert_from_buffer(self, context_name):
        full_trajectory = self.experience_buffer[context_name]
        if not full_trajectory:
            return

        expert = self.get_expert(context_name)

        observations = np.array([t[0] for t in full_trajectory])
        actions = np.array([t[1] for t in full_trajectory])
        rewards = np.array([t[2] for t in full_trajectory])
        terminals = np.array([t[4] for t in full_trajectory])
        timeouts = np.array([t[5] for t in full_trajectory])

        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts
        )
        expert.model.fit(dataset, n_steps=2000)

    def predict(self, obs):
        grid_spikes = self.grid_module.get_current_spikes()
        place_spikes = self.place_module(grid_spikes)
        self.current_context = self._detect_context(place_spikes)
        expert = self.get_expert(self.current_context)
        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()
        return expert.predict(obs)[0]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.grid_module.state_dict(), path / "grid_module.pt")
        torch.save(self.place_module.state_dict(), path / "place_module.pt")

        for name, expert in self.experts.items():
            expert.save(path / f"expert_{name}.d3")

        with open(path / "known_contexts.pkl", "wb") as f:
            pickle.dump(self.known_contexts_features, f)
        with open(path / "place_memory.pkl", "wb") as f:
            pickle.dump(self.novelty_module.place_memory, f)

        with open(path / "experience_buffer.pkl", "wb") as f:
            pickle.dump(self.experience_buffer, f)

    def load(self, path):
        path = Path(path)
        if not path.exists():
            return

        if (path / "grid_module.pt").exists():
            self.grid_module.load_state_dict(
                torch.load(path / "grid_module.pt")
            )
        if (path / "place_module.pt").exists():
            self.place_module.load_state_dict(
                torch.load(path / "place_module.pt")
            )

        for expert_file in path.glob("expert_*.d3"):
            name = '_'.join(expert_file.stem.split('_')[1:])
            expert = self.get_expert(name)
            expert.load(expert_file)

        if (path / "known_contexts.pkl").exists():
            with open(path / "known_contexts.pkl", "rb") as f:
                self.known_contexts_features = pickle.load(f)
        if (path / "place_memory.pkl").exists():
            with open(path / "place_memory.pkl", "rb") as f:
                self.novelty_module.place_memory = pickle.load(f)
        if (path / "experience_buffer.pkl").exists():
            with open(path / "experience_buffer.pkl", "rb") as f:
                self.experience_buffer = pickle.load(f)


# Context inference trasfer (CIT)
class CITgent:
    def __init__(self, name, difficulty, env_info, save_path="./models"):
        self.difficulty = difficulty
        self.model_name = name
        self.save_path = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        self.learner = Hippocampo(env_info, difficulty)
        self.load()

    def train(self):
        return self.learner.train()

    def predict(self, obs):
        return self.learner.predict(obs)

    def save(self, path):
        self.learner.save(path)

    def load(self, path=None):
        load_location = path if path is not None else self.save_path
        self.learner.load(load_location)
