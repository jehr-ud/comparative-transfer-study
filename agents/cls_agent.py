import os
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate
import gymnasium as gym
import numpy as np
from transformers import pipeline


class Expert:
    def __init__(self, env, context_name, snn_hippocampus_processor, save_path="./models"):
        self.context_name = context_name
        self.save_path = Path(save_path)
        self.env = env
        self.snn_hippocampus_processor = snn_hippocampus_processor
        self.name = "PPO"
        self.model = self._build_model()

    def _build_model(self):
        snn_output_shape = (128,)

        self.config = (
            PPOConfig()
            .environment(env=self.env, observation_space=snn_output_shape)
            .framework("torch")
            .training(
                model={
                    "custom_model": "NeocortexPolicyNet",
                    "fcnet_hiddens": [256, 256],
                }
            )
            .rollouts(num_rollout_workers=0)
        )
        return self.config.build()

    def update_from_learner(self, learner_model):
        self.model.set_weights(learner_model.get_weights())

    def predict(self, raw_obs):
        processed_obs = self.snn_hippocampus_processor.process_observation(raw_obs)
        return self.model.compute_single_action(processed_obs)

    def load(self, path):
        config_cls = {
            "PPO": PPOConfig
        }[self.name]
        snn_output_shape = (128,)
        self.model = config_cls().environment(self.env, observation_space=snn_output_shape).framework("torch").build()
        self.model.restore(str(path))
        print(f"Model loaded from {path}")

    def save(self, path=None, difficulty="default"):
        if self.model is None:
            raise ValueError("Model not initialized. Train or load a model first.")

        self.save_path.mkdir(parents=True, exist_ok=True)
        save_to = Path(path) if path else self.save_path / f"{self.context_name}/{self.name}_{difficulty}"
        self.model.save(str(save_to))
        print(f"Model saved to {save_to}")


class SNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.last_processed_features = None
        self.last_raw_observation_processed = None

        if len(input_shape) == 3:
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                neuron.LIFNode(surrogate_function=surrogate.ATan()),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(self._get_flat_ft_size(input_shape), output_dim),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
            )
        elif len(input_shape) == 1:
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

    def _get_flat_ft_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.features[0](dummy_input)
            x = self.features[2](x)
            x = self.features[3](x)
            x = self.features[5](x)
            return x.numel()

    def forward(self, x, time_steps=8):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == len(self.input_shape):
            x = x.unsqueeze(0)

        self.last_raw_observation_processed = x.detach().cpu().numpy()

        if x.dim() == len(self.input_shape):
            x = x.unsqueeze(0)

        x_time_series = x.unsqueeze(0).repeat(time_steps, 1, *([1] * len(self.input_shape)))

        functional.reset_net(self)

        output_spikes_sum = torch.zeros(x.shape[0], self.output_dim).to(x.device)
        for t in range(time_steps):
            out_t = self.features(x_time_series[t])
            output_spikes_sum += out_t.float()

        result = output_spikes_sum / time_steps
        self.last_processed_features = result.detach().cpu().numpy()
        return result


class Learner:
    def __init__(self, env: gym.Env, snn_output_dim: int = 128):
        self.env = env
        self.snn_output_dim = snn_output_dim
        self.snn_feature_extractor = SNNFeatureExtractor(
            env.observation_space.shape,
            snn_output_dim
        )

        self.last_detection_info = {}
        self.last_prediction_info = {}
        self.experts = {}
        self.current_context = None
        self.expert_in_training = None

        self.known_contexts_features = []
        self.context_threshold = 0.8

    def detect_context(self, observation_batch):
        self.last_raw_observation_for_detection = observation_batch

        if not isinstance(observation_batch, torch.Tensor):
            observation_batch = torch.tensor(
                observation_batch,
                dtype=torch.float32
            )

        processed_features = self.snn_feature_extractor(observation_batch, time_steps=8)
        current_feature_avg = processed_features.mean(dim=0).detach().cpu().numpy()

        obstacle_count = self.env.get_obstacle_count()
        current_feature_avg = np.concatenate([current_feature_avg, [obstacle_count]])

        print("[DEBUG] features", current_feature_avg)

        min_distance = float('inf')
        detected_context_name = None
        distances_to_known_contexts = []

        for known_context_name, known_feature_vec in self.known_contexts_features:
            distance = np.linalg.norm(current_feature_avg - known_feature_vec)
            distances_to_known_contexts.append((known_context_name, distance))
            if distance < min_distance:
                min_distance = distance
                detected_context_name = known_context_name

        is_new_context = (detected_context_name is None or min_distance > self.context_threshold)

        if is_new_context:
            new_context_id = f"laberinto_C{len(self.known_contexts_features) + 1}"
            self.known_contexts_features.append((new_context_id, current_feature_avg))
            detected_context_name = new_context_id
            print(f"Detected NEW context: {new_context_id} (distance: {min_distance:.2f})")
        else:
            print(f"Detected KNOWN context: {detected_context_name} (distance: {min_distance:.2f})")

        self.last_detection_info = {
            "raw_obs_shape": self.snn_feature_extractor.last_raw_observation_processed.shape,
            "processed_features_from_snn": self.snn_feature_extractor.last_processed_features,
            "current_feature_avg": current_feature_avg,
            "known_contexts_features": self.known_contexts_features.copy(),
            "distances_to_known_contexts": distances_to_known_contexts,
            "context_threshold": self.context_threshold,
            "detected_context_name": detected_context_name,
            "is_new_context": is_new_context
        }

        self.current_context = detected_context_name
        return detected_context_name

    def get_expert(self, context_name: str):
        if context_name not in self.experts:
            print(f"Creating new Expert for context: {context_name}")
            self.experts[context_name] = Expert(self.env, context_name, self.snn_feature_extractor)
        return self.experts[context_name]

    def train(self, context_name: str, num_iterations: int = 100):
        expert: Expert = self.get_expert(context_name)
        print(f"Training Expert for context: {context_name} for {num_iterations} iterations...")
        self.expert_in_training = expert

        for i in range(num_iterations):
            result = expert.model.train()
            if i % 10 == 0:
                print(f"   Iteration {i}: episode_reward_mean={result['episode_reward_mean']:.2f}")

        print(f"Training for context {context_name} complete.")
        self.expert_in_training = None

    def stop(self):
        self.agent.stop()

    def predict(self, raw_obs):
        if self.current_context is None:
            raise ValueError("Context not detected. Call detect_context at episode start.")

        expert: Expert = self.get_expert(self.current_context)
        action = expert.predict(raw_obs)

        self.last_prediction_info = {
            "raw_obs_shape": self.snn_feature_extractor.last_raw_observation_processed.shape,
            "snn_input_to_expert": self.snn_feature_extractor.last_processed_features,
            "expert_used": self.current_context,
            "predicted_action": action
        }

        return action

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.known_contexts_features, path / "known_contexts_features.pt")

        for context_name, expert_instance in self.experts.items():
            expert_instance.save(path=path / f"expert_{context_name}")
        print(f"Learner state and experts saved to {path}")

    def load(self, path):
        path = Path(path)
        if not path.exists():
            print(f"Path {path} does not exist. Cannot load.")
            return

        if (path / "known_contexts_features.pt").exists():
            self.known_contexts_features = torch.load(path / "known_contexts_features.pt")
            print(f"Loaded {len(self.known_contexts_features)} known contexts.")

        for context_name, _ in self.known_contexts_features:
            expert_path = path / f"expert_{context_name}"
            if expert_path.exists():
                expert_instance = Expert(self.env, context_name, self.snn_feature_extractor)
                expert_instance.load(path=expert_path)
                self.experts[context_name] = expert_instance
                print(f"Loaded Expert for context: {context_name}")
            else:
                print(f"Warning: Expert for context {context_name} not found at {expert_path}")


class CLSAgent:
    def __init__(self, model_name, difficulty, env, save_path="./models"):
        self.env = env
        self.save_path = os.path.join(save_path, model_name, difficulty)
        os.makedirs(self.save_path, exist_ok=True)

        self.learner = Learner(env)
        self.learner.load(self.save_path)

        try:
            print("Cargando modelo generativo local (Hugging Face)... Esto puede tardar.")
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                torch_dtype=torch.float16,
                device=0 if torch.cuda.is_available() else -1
            )
            print("Modelo generativo cargado.")
        except Exception as e:
            print(f"Error al cargar el modelo generativo de Hugging Face: {e}")
            print("La funcionalidad de explicación generativa no estará disponible.")
            self.text_generator = None

    def train(self, detected_context, num_iterations=100):
        return self.learner.train(
            context_name=detected_context,
            num_iterations=num_iterations
        )

    def predict(self, obs):
        return self.learner.predict(obs)

    def save(self):
        self.learner.save(self.save_path)

    def load(self):
        self.learner.load(self.save_path)
        return self

    def explain(self, obs_for_prediction=None):
        """
        Genera una explicación de cómo el agente determinó el contexto y
        predijo la acción para la última observación procesada.

        Si `obs_for_prediction` es None, usa la última observación procesada por el Learner.
        Si se proporciona, el Learner intentará procesarla y usar sus datos.
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
            prompt_parts.append("Para este nuevo contexto, se decidió que se necesitaría un nuevo 'Experto' especializado en la Neocorteza para aprender a resolver este tipo de laberinto.")
        else:
            prompt_parts.append(f"\nLa huella del laberinto actual era muy similar a la memoria de `{detected_context_name}` (la distancia fue `{min(d for _,d in distances_calculated if d is not None) if distances_calculated else 'N/A'}`). Por lo tanto, el Hipocampo reconoció este laberinto como un **CONTEXTO CONOCIDO**.")
            prompt_parts.append(f"Automáticamente, el Hipocampo activó al 'Experto' de la Neocorteza que ya está especializado en resolver laberintos de tipo `{detected_context_name}`.")

        prompt_parts.append("\n\n## Proceso de Toma de Acción (Neocorteza - Experto)")
        if expert_used == "N/A" or predicted_action == "N/A":
             prompt_parts.append("No se pudo obtener información completa sobre la predicción de la acción.")
        else:
            prompt_parts.append(f"Una vez que el Hipocampo determinó que el laberinto pertenecía al contexto `{expert_used}`, pasó la 'huella dactilar' procesada del laberinto (el vector `{snn_input_to_expert[:5]}...`) al 'Experto' correspondiente en la Neocorteza.")
            prompt_parts.append(f"Este 'Experto', que es un modelo PPO entrenado específicamente para el contexto `{expert_used}`, procesó esta huella. Basándose en todo el conocimiento que ha adquirido resolviendo laberintos de ese tipo, el Experto decidió la siguiente acción: **`{predicted_action}`**.")
            prompt_parts.append("Esta acción es el paso que el agente cree que es el más efectivo para avanzar en este laberinto específico.")

        prompt_parts.append("\n---")

        return "\n".join(prompt_parts)

    def _call_generative_ai(self, prompt):
        print("\n--- GENERANDO EXPLICACIÓN CON MODELO LOCAL (Hugging Face) ---")

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