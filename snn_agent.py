import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

from spikingjelly.activation_based import neuron, functional, learning


# ============================
# 🔹 Encoder SNN (Neocortex)
# ============================
class SNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, tau=2.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.lif = neuron.LIFNode(tau=tau, detach_reset=True)

    def forward(self, x):
        x = self.fc(x)
        spikes = self.lif(x)
        return spikes


# ============================
# 🔹 Transfer Encoder para nueva tarea
# ============================
class TransferSNNEncoder(nn.Module):
    def __init__(self, input_dim, old_encoder=None, hidden_dim=32, tau=2.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.lif = neuron.LIFNode(tau=tau, detach_reset=True)

        if old_encoder is not None:
            # Copiar bias del encoder antiguo (transferencia parcial)
            self.fc.bias.data = old_encoder.fc.bias.data.clone()

    def forward(self, x):
        x = self.fc(x)
        spikes = self.lif(x)
        return spikes


# ============================
# 🔹 Decoder SNN (Hipocampo)
# ============================
class SNNDecoder(nn.Module):
    def __init__(self, hidden_dim, action_dim, tau=5.0):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.lif = neuron.LIFNode(tau=tau, detach_reset=True)

        # STDP para adaptación rápida
        self.stdp = learning.STDPLearner(
            synapse=self.fc,
            sn=self.lif,
            tau_pre=20.0,
            tau_post=20.0,
            f_pre=lambda x: 0.01 * x,
            f_post=lambda x: -0.01 * x,
            step_mode='s'
        )

    def forward(self, spikes):
        x = self.fc(spikes)
        out_spikes = self.lif(x)
        return out_spikes

    def stdp_step(self):
        # Llama a step en modo de actualización directa.
        # Esto usa los spikes del forward pass real y evita errores
        # de gradientes o de listas vacías.
        self.stdp.step(on_grad=False)


# ============================
# 🔹 Agente RL SNN
# ============================
class SNNAgent:
    def __init__(self, state_dim, action_dim, encoder=None, hidden_dim=32):
        self.encoder = encoder if encoder is not None else SNNEncoder(state_dim, hidden_dim)
        self.decoder = SNNDecoder(hidden_dim, action_dim)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.batch_size = 64
        self.action_dim = action_dim

    def act(self, state, eps=0.1):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        spikes = self.encoder(state)
        q_spikes = self.decoder(spikes)
        q_values = q_spikes.detach().cpu().numpy().flatten()
        if random.random() < eps:
            return random.randrange(self.action_dim)
        return int(np.argmax(q_values))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        for i in range(self.batch_size):
            reward = rewards[i].item()
            # Modulación STDP
            self.decoder.stdp.f_pre = lambda x, r=reward: 0.01 * max(0, r) * x
            self.decoder.stdp.f_post = lambda x, r=reward: -0.01 * max(0, r) * x

            # Forward completo para registrar spikes
            spikes = self.encoder(states[i].unsqueeze(0))
            _ = self.decoder(spikes)

            # STDP step
            self.decoder.stdp_step()


# ============================
# 🔹 Entrenamiento SNN
# ============================
def train_snn(env_name, episodes, encoder=None):
    env = gym.make(env_name)
    agent = SNNAgent(env.observation_space.shape[0], env.action_space.n, encoder)
    scores = []

    for ep in range(episodes):
        s = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        total_reward = 0
        for _ in range(500):
            a = agent.act(s)
            s2, r, done, _, _ = env.step(a)
            agent.remember(s, a, r, s2, done)
            agent.replay()
            s = s2
            total_reward += r
            if done:
                break
        scores.append(total_reward)
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes}, Score (avg last10): {np.mean(scores[-10:])}")

        # Reset neuronas SNN entre episodios
        functional.reset_net(agent.encoder)
        functional.reset_net(agent.decoder)

    env.close()
    return agent, scores


# ============================
# 🔹 Fase 1: CartPole
# ============================
print("🔹 Entrenando en CartPole con SNN...")
cartpole_agent, _ = train_snn("CartPole-v1", episodes=200)

# Guardamos encoder universal
universal_encoder = cartpole_agent.encoder

# ============================
# 🔹 Fase 2: Transferencia a MountainCar
# ============================
print("\n🔹 Transferencia a MountainCar con SNN...")

# Creamos Transfer Encoder adaptado a input_dim=2
mountaincar_encoder = TransferSNNEncoder(input_dim=2, old_encoder=universal_encoder, hidden_dim=32)

mountaincar_agent, _ = train_snn("MountainCar-v0", episodes=300, encoder=mountaincar_encoder)

print("✅ Entrenamiento completo con SNN y transferencia.")
