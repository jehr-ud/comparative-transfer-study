import os
import random

from pathlib import Path
import collections
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.sample_batch import SampleBatch
import gymnasium as gym

import numpy as np
import pickle
import pandas as pd

import torch.distributions as D
from spikingjelly.activation_based import neuron, functional, surrogate

torch, nn = try_import_torch()


ENV_RUNNER_CONFIG = {
    "simple":  {"num_env_runners": 1, "num_envs_per_env_runner": 2},
    "medium":  {"num_env_runners": 2, "num_envs_per_env_runner": 2},
    "complex": {"num_env_runners": 2, "num_envs_per_env_runner": 3}
}


class SNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.last_processed_features = None
        self.last_raw_observation_processed = None

        if len(input_shape) == 1:
            self.features = nn.Sequential(
                nn.Linear(input_shape[0], 128),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                nn.Linear(128, 128),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                nn.Linear(128, output_dim),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
            )
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

    def forward(self, x, time_steps=8):
        original_device = x.device if isinstance(x, torch.Tensor) else next(self.parameters()).device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(original_device)

        if x.dim() == len(self.input_shape):
            x = x.unsqueeze(0)

        self.last_raw_observation_processed = x.detach().cpu().numpy()

        # [T, B, *dims]
        x_time_series = x.unsqueeze(0).repeat(time_steps, 1, *([1] * (x.dim() -1 )))

        functional.reset_net(self)

        output_spikes_sum = torch.zeros(x.shape[0], self.output_dim).to(x.device)
        for t in range(time_steps):
            out_t = self.features(x_time_series[t])
            output_spikes_sum += out_t.float()

        result = output_spikes_sum / time_steps
        self.last_processed_features = result.detach().cpu().numpy()
        return result


class NeoCortex:
    def __init__(
        self,
        env_info,
        context_name,
        snn_hippocampus_processor,
        difficulty="simple",
        save_path="./models",
        explore=False
    ):
        self.context_name = context_name
        self.save_path = Path(save_path)
        self.env_info = env_info
        self.snn_hippocampus_processor = snn_hippocampus_processor
        self.name = f"PPO-{context_name}"
        self.explore = explore
        self.difficulty = difficulty
        self.config = self._get_config()
        self.model = self.config.build()

    def _get_config(self):
        config = PPOConfig()
        config.environment(
            env="visual_env",
            env_config=self.env_info.get('config')
        )

        return config

    def update_from_learner(self, weights):
        self.model.set_weights(weights)

    def predict(self, raw_obs):
        module = self.model.get_module("default_policy")

        obs_array = torch.tensor([raw_obs], dtype=torch.float32)
        output = module.forward_inference({"obs": obs_array})

        if 'actions' in output:
            return output.get('actions')

        logits = output["action_dist_inputs"]
        dist = D.Categorical(logits=logits)
        return dist.sample().item()

    def load(self, path):
        try:
            self.config = self._get_config()
            self.model = self.config.build()

            self.model.restore(str(path))
            print(f"Model weights for {self.name} loaded from {path}")
        except Exception as e:
            print(f"Failed to restore model weights for {self.name} from {path}: {e}. Model initialized with new weights.")

    def save(self, path=None):
        if self.model is None:
            raise ValueError(
                "Agent not initialized. Train or load a model first."
            )

        self.save_path.mkdir(parents=True, exist_ok=True)
        save_to = Path(path) if path else self.save_path / f"{self.name}_{self.difficulty}"
        self.model.save(str(save_to))
        print(f"Model saved to {save_to}")


class Hippocampo:
    def __init__(
        self,
        env_info: dict,
        difficulty,
        snn_output_dim: int = 128,
        context_threshold=1.0,
        fast_change_threshold=1.5
    ):
        self.env = env_info.get('env')
        self.env_info = env_info
        self.snn_output_dim = snn_output_dim

        snn_input_shape = self.env.observation_space.shape

        self.snn_feature_extractor = SNNFeatureExtractor(
            snn_input_shape,
            snn_output_dim
        )
        self.context_threshold = context_threshold
        self.fast_change_threshold = fast_change_threshold

        if torch.cuda.is_available():
            self.snn_feature_extractor.to(torch.device("cuda"))

        self.last_detection_info = {}
        self.last_prediction_info = {}
        self.experts = {}
        self.current_context = None
        self.expert_in_training = None
        self.known_contexts_features = []
        self.context_threshold = 0.4

        self.context_history = []
        self.total_context_steps = 0

        self.last_feature_for_detection = None
        
        self.difficulty = difficulty

        # Para memoria a corto plazo en detección de contexto
        self.feature_buffer_size = 3
        self.recent_features_buffer = collections.deque(
            maxlen=self.feature_buffer_size
        )

    def reset_episode_memory(self):
        self.recent_features_buffer.clear()
        print("Hippocampo's short-term feature buffer cleared for new episode.")

    def detect_context(self, obs):
        self.last_raw_observation_for_detection = obs

        device = next(self.snn_feature_extractor.parameters()).device
        observation_tensor = self._to_tensor(obs).to(device)
        feature_vector = self._extract_features(observation_tensor)

        self.recent_features_buffer.append(feature_vector)
        avg_feature = np.mean(self.recent_features_buffer, axis=0) if self.recent_features_buffer else feature_vector

        obstacle_count = self._get_obstacle_count()
        full_feature = np.concatenate([avg_feature, [obstacle_count]])

        # Cambio abrupto en el contexto
        if self.last_feature_for_detection is not None:
            delta = np.linalg.norm(full_feature - self.last_feature_for_detection)
            if delta > self.fast_change_threshold:
                print(f"[INFO] Cambio abrupto detectado: Δ={delta:.2f} (contexto actual: {self.current_context})")
        self.last_feature_for_detection = full_feature

        context_name, distance, distance_list = self._compare_with_known_contexts(full_feature)
        is_new = context_name is None or distance > self.context_threshold

        if is_new:
            context_name = self._register_new_context(full_feature)

        self._store_detection_info(feature_vector, avg_feature, full_feature, distance_list, context_name, is_new)
        self.current_context = context_name
        return context_name

    def _to_tensor(self, obs):
        if isinstance(obs, torch.Tensor):
            return obs.float()
        try:
            return torch.tensor(obs, dtype=torch.float32)
        except Exception as e:
            print(f"[ERROR] Fallo al convertir obs a tensor. Tipo: {type(obs)}, Contenido: {obs}")
            raise e

    def _extract_features(self, tensor_obs):
        spikes = self.snn_feature_extractor(tensor_obs, time_steps=8)
        return spikes.mean(dim=0).detach().cpu().numpy()

    def _get_obstacle_count(self):
        if hasattr(self.env, 'get_obstacle_count') and callable(getattr(self.env, 'get_obstacle_count')):
            return self.env.get_obstacle_count()
        return 0

    def _compare_with_known_contexts(self, feature):
        min_distance = float('inf')
        closest_context = None
        distances = []

        for name, known_feat in self.known_contexts_features:
            dist = np.linalg.norm(feature - known_feat)
            distances.append((name, dist))
            if dist < min_distance:
                min_distance = dist
                closest_context = name

        return closest_context, min_distance, distances

    def _register_new_context(self, feature):
        name = f"laberinto_C{len(self.known_contexts_features) + 1}"
        self.known_contexts_features.append((name, feature))
        return name

    def _store_detection_info(self, snn_features, avg_features, final_feature, distances, context_name, is_new):
        last_shape = getattr(self.snn_feature_extractor, 'last_raw_observation_processed', None)
        detection_info = {
            "raw_obs_shape_for_snn": last_shape.shape if last_shape is not None else "N/A",
            "snn_features_current_obs": snn_features,
            "avg_features_from_buffer": avg_features,
            "final_feature_for_detection": final_feature,
            "known_contexts_count": len(self.known_contexts_features),
            "distances_to_known_contexts": distances,
            "context_threshold": self.context_threshold,
            "detected_context_name": context_name,
            "is_new_context": is_new,
            "feature_buffer_len": len(self.recent_features_buffer),
            "step": self.total_context_steps
        }

        self.last_detection_info = detection_info
        self.context_history.append({
            "step": self.total_context_steps,
            "context": context_name,
            "features": final_feature.tolist()
        })
        self.total_context_steps += 1

    def save_context_history_csv(self, filename="context_history.csv"):
        df = pd.DataFrame(self.context_history)
        df.to_csv(filename, index=False)
        print(f"[INFO] Historial de contextos guardado en: {filename}")

    def save_context_history_pickle(self, filename="context_history.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.context_history, f)
        print(f"[INFO] Historial de contextos guardado en: {filename}")

    def get_expert(self, context_name: str, difficulty: str = "simple"):
        if context_name not in self.experts:
            self.experts[context_name] = NeoCortex(
                self.env_info,
                context_name,
                self.snn_feature_extractor,
                difficulty=difficulty
            )
        return self.experts[context_name]

    def train(
        self,
        max_steps=100
    ):
        env = self.env_info.get('env')
        state, _ = env.reset()
        paths = []
        success = False

        for _ in range(max_steps):
            # Exploración pura: acción aleatoria (mejorar con SNN)
            action = random.choice(env.get_valid_actions())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            paths.append({
                "obs": state,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated
            })

            state = obs

            if done:
                success = True
                break

        context_name = self.detect_context(state)

        if success:
            obs = [np.array(step["obs"]).flatten().tolist() for step in paths]
            actions = [step["action"] for step in paths]
            rewards = [step["reward"] for step in paths]
            terminateds = [step["terminated"] for step in paths]
            truncateds = [step["truncated"] for step in paths]

            next_obs = obs[1:] + [obs[-1]]

            batch_dict = {
                SampleBatch.OBS: obs,
                SampleBatch.ACTIONS: actions,
                SampleBatch.REWARDS: rewards,
                SampleBatch.TERMINATEDS: terminateds,
                SampleBatch.TRUNCATEDS: truncateds,
                SampleBatch.NEXT_OBS: next_obs,
                SampleBatch.EPS_ID: [0] * len(obs),
                SampleBatch.AGENT_INDEX: [0] * len(obs),
                SampleBatch.T: list(range(len(obs))),
            }

            df = pd.DataFrame(batch_dict)

            data_path = "mi_datos_offline.parquet"
            df.to_parquet(data_path)
            print(f"SampleBatch guardado en: {data_path}")

            return self._attempt_transfer(data_path, context_name)
        else:
            print("Exploración fallida, no se guarda experiencia.")

        return 0

    def _attempt_transfer(self, batch_file,  context_name: str):
        """Pasa experiencia a experto."""
        # expert: NeoCortex = self.get_expert(context_name, self.difficulty)
        # self.expert_in_training = expert

        config = (
            PPOConfig()
                .environment(
                    env="visual_env",
                    env_config=self.env_info.get('config'),
                    action_space=gym.spaces.Discrete(4),
                    observation_space=gym.spaces.Box(
                        low=0,
                        high=4 - 1,
                        shape=(2,), dtype=np.float32
                    )
                )
                .learners(num_learners=2)
                .env_runners(num_env_runners=2)
                .offline_data(
                    input_="file://" + batch_file
                )
                .training(
                    train_batch_size=256,
                    gamma=0.99
                )
                .build()
        )

        return config.train()

    def stop_experts(self):
        if self.expert_in_training and hasattr(self.expert_in_training.model, 'stop'):
            self.expert_in_training.model.stop()
        for expert_name in list(self.experts.keys()):
            expert = self.experts[expert_name]
            if hasattr(expert.model, 'stop'):
                expert.model.stop()

    def predict(self, raw_obs):
        if self.current_context is None:
            print("Advertencia: Contexto no detectado previamente. Intentando detectar ahora.")
            obs_for_detection = raw_obs
            if isinstance(raw_obs, np.ndarray) and raw_obs.ndim == len(self.snn_feature_extractor.input_shape):
                obs_for_detection = np.expand_dims(raw_obs, axis=0)
            elif isinstance(raw_obs, torch.Tensor) and raw_obs.ndim == len(self.snn_feature_extractor.input_shape):
                obs_for_detection = raw_obs.unsqueeze(0)

            self.detect_context(obs_for_detection)
            if self.current_context is None:
                raise ValueError("Contexto aún no detectado después del intento. Llama a detect_context explícitamente.")

        expert: NeoCortex = self.get_expert(self.current_context)
        action = expert.predict(raw_obs)

        self.last_prediction_info = {
            "raw_obs_shape_for_snn": self.snn_feature_extractor.last_raw_observation_processed.shape if self.snn_feature_extractor.last_raw_observation_processed is not None else "N/A",
            "snn_output_features_for_expert": self.snn_feature_extractor.last_processed_features if self.snn_feature_extractor.last_processed_features is not None else "N/A",
            "expert_used": self.current_context,
            "predicted_action": action
        }
        return action

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.known_contexts_features,
            path / "known_contexts_features.pt"
        )
        torch.save(
            self.snn_feature_extractor.state_dict(),
            path / "snn_feature_extractor.pt"
        )
        for context_name, expert_instance in self.experts.items():
            expert_instance.save(
                path
            )

    def load(self, path):
        path = Path(path)
        if not path.exists():
            return

        if (path / "known_contexts_features.pt").exists():
            self.known_contexts_features = torch.load(path / "known_contexts_features.pt")

        snn_weights_path = path / "snn_feature_extractor.pt"
        if snn_weights_path.exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.snn_feature_extractor.load_state_dict(torch.load(snn_weights_path, map_location=device))
            self.snn_feature_extractor.to(device)

        for context_dir_name, _ in self.known_contexts_features:
            expert_context_path = path / context_dir_name
            if expert_context_path.is_dir():
                difficulty_of_expert = "simple"
                checkpoint_name_pattern = f"PPO-{context_dir_name}_{difficulty_of_expert}"
                potential_checkpoint_path = expert_context_path / checkpoint_name_pattern

                if potential_checkpoint_path.exists():
                    expert_instance = NeoCortex(self.env_info, context_dir_name, self.snn_feature_extractor, difficulty=difficulty_of_expert)
                    expert_instance.load(path=potential_checkpoint_path)
                    self.experts[context_dir_name] = expert_instance
                else:
                    print(
                        f"Advertencia: Checkpoint para experto {context_dir_name} con dificultad {difficulty_of_expert} no encontrado en {potential_checkpoint_path}"
                    )


class CPAgent:
    def __init__(
        self,
        name,
        difficulty,
        env_info,
        save_path="./models",
        explore=False
    ):
        self.difficulty = difficulty
        self.model_name = name

        self.save_path = Path(save_path)
        os.makedirs(self.save_path, exist_ok=True)

        self.learner = Hippocampo(env_info, difficulty)
        self.learner.load(self.save_path)
        self.text_generator = None

        self.explore = explore

    def train(self):
        return self.learner.train()

    def predict(self, obs):
        return self.learner.predict(obs)

    def save(self, path):
        self.learner.save(path)

    def load(self):
        self.learner.load(self.save_path)

    def stop(self):
        self.learner.stop_experts()


class Explanation:
    def explain(self, obs_for_prediction=None):
        """
        Genera una explicación de cómo el agente determinó el contexto y
        predijo la acción para la última observación procesada.

        Si `obs_for_prediction` es None, usa la última observación procesada por el Hippocampo.
        Si se proporciona, el Hippocampo intentará procesarla y usar sus datos.
        """
        if not self.learner.last_detection_info:
            return "No hay información de detección de contexto disponible. Asegúrate de que el agente haya procesado al menos una observación con `detect_context`."
        if not self.learner.last_prediction_info:
            return "No hay información de predicción de acción disponible. Asegúrate de que el agente haya realizado al menos una predicción con `predict`."

        if obs_for_prediction is not None:
            _ = self.learner.detect_context(obs_for_prediction)
            _ = self.learner.predict(obs_for_prediction)

        context_data = self.learner.last_detection_info
        action_data = self.learner.last_prediction_info

        raw_obs_shape_display = context_data.get("raw_obs_shape", "N/A")
        snn_processed_features_display = context_data.get("processed_features_from_snn", np.array([]))
        current_feature_vector_avg_display = context_data.get("current_feature_avg", np.array([]))

        snn_input_to_expert_display = action_data.get("snn_input_to_expert", np.array([]))

        prompt = self._build_explanation_prompt(
            raw_obs_shape_display,
            snn_processed_features_display,
            current_feature_vector_avg_display,
            context_data.get("known_contexts_features", []),
            context_data.get("distances_to_known_contexts", []),
            context_data.get("context_threshold", "N/A"),
            context_data.get("detected_context_name", "N/A"),
            context_data.get("is_new_context", "N/A"),
            action_data.get("expert_used", "N/A"),
            snn_input_to_expert_display,
            action_data.get("predicted_action", "N/A")
        )

        if self.text_generator:
            explanation = self._call_generative_ai(prompt)
            return explanation
        else:
            return "El modelo de IA generativa no está cargado. No se puede generar una explicación."

    def _build_explanation_prompt(self,
                                  raw_obs_shape,
                                  snn_processed_features,
                                  current_feature_avg,
                                  known_contexts,
                                  distances_calculated,
                                  context_threshold,
                                  detected_context_name,
                                  is_new_context,
                                  expert_used,
                                  snn_input_to_expert,
                                  predicted_action):
        """
        Prompt estructurado para la IA generativa
        """
        prompt_parts = []
        prompt_parts.append("Eres un experto en inteligencia artificial y un narrador brillante, capaz de explicar sistemas complejos de forma clara y atractiva. Tu objetivo es explicar cómo un agente de IA con 'cerebro' dividido (Hipocampo y Neocorteza) percibió un entorno, lo clasificó y luego tomó una decisión.")
        prompt_parts.append("Usa analogías si te ayudan a aclarar, pero sé preciso con los datos proporcionados. Formatea tu respuesta con encabezados y puntos claros.")

        prompt_parts.append("\n\n## Proceso de Detección de Contexto (Hipocampo)")
        prompt_parts.append(f"Cuando el agente observó el laberinto (con una forma de datos de {raw_obs_shape}), su 'Percepción SNN' (Hipocampo) lo analizó. Esto es como si el Hipocampo formara una 'huella dactilar' única de este laberinto.")
        prompt_parts.append(f"Esta huella se representó como un vector numérico de 128 dimensiones. La versión promedio de esta huella para el laberinto fue: `{current_feature_avg[:5]}...` (mostrando los primeros 5 valores para referencia).")

        if known_contexts:
            prompt_parts.append("\nEl Hipocampo comparó esta huella con todas las 'memorias' de laberintos que ya conocía. Cada memoria también tiene su propia huella dactilar:")
            for name, dist in distances_calculated:
                prompt_parts.append(f"  - Memoria '{name}': Distancia de similitud = {dist:.2f}")
            prompt_parts.append(f"El umbral de decisión para considerar un laberinto como 'nuevo' o 'diferente' es `{context_threshold}`.")

        if is_new_context:
            prompt_parts.append(f"\nDado que la huella del laberinto actual no se parecía lo suficiente a ninguna memoria existente (la menor distancia fue superior al umbral o no había memorias previas), el Hipocampo identificó esto como un **NUEVO CONTEXTO**: `{detected_context_name}`.")
            prompt_parts.append("Para este nuevo contexto, se decidió que se necesitaría un nuevo 'NeoCortexo' especializado en la Neocorteza para aprender a resolver este tipo de laberinto.")
        else:
            prompt_parts.append(f"\nLa huella del laberinto actual era muy similar a la memoria de `{detected_context_name}` (la distancia fue `{min(d for _,d in distances_calculated if d is not None) if distances_calculated else 'N/A'}`). Por lo tanto, el Hipocampo reconoció este laberinto como un **CONTEXTO CONOCIDO**.")
            prompt_parts.append(f"Automáticamente, el Hipocampo activó al 'NeoCortexo' de la Neocorteza que ya está especializado en resolver laberintos de tipo `{detected_context_name}`.")

        prompt_parts.append("\n\n## Proceso de Toma de Acción (Neocorteza - NeoCortexo)")
        if expert_used == "N/A" or predicted_action == "N/A":
            prompt_parts.append("No se pudo obtener información completa sobre la predicción de la acción.")
        else:
            prompt_parts.append(f"Una vez que el Hipocampo determinó que el laberinto pertenecía al contexto `{expert_used}`, pasó la 'huella dactilar' procesada del laberinto (el vector `{snn_input_to_expert[:5]}...`) al 'NeoCortexo' correspondiente en la Neocorteza.")
            prompt_parts.append(f"Este 'NeoCortexo', que es un modelo PPO entrenado específicamente para el contexto `{expert_used}`, procesó esta huella. Basándose en todo el conocimiento que ha adquirido resolviendo laberintos de ese tipo, el NeoCortexo decidió la siguiente acción: **`{predicted_action}`**.")
            prompt_parts.append("Esta acción es el paso que el agente cree que es el más efectivo para avanzar en este laberinto específico.")

        prompt_parts.append("\n---")

        return "\n".join(prompt_parts)

    def _call_generative_ai(self, prompt):
        print("\n--- GENERANDO EXPLICACIÓN CON MODELO LLM ---")

        response = self.text_generator(
            prompt,
            max_new_tokens=600,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.text_generator.tokenizer.eos_token_id 
        )

        generated_text = response[0]['generated_text']

        explanation = generated_text.replace(prompt, "").strip() 
        if not explanation:
            explanation = generated_text.strip()

        print("----------------------------------------------------------------\n")
        return explanation