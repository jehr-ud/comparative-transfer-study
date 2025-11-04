import gym
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF

# =====================
# Corteza Entorrinal
# =====================
# Codifica estados del entorno a spikes (representación temporal)
def entorhinal_encoder(state, time_window=50):
    # codificación Poisson simple
    state = torch.tensor(state, dtype=torch.float)
    state = (state - state.min()) / (state.max() - state.min() + 1e-5)
    spike_train = torch.bernoulli(state.repeat(time_window, 1))
    return spike_train

# =====================
# Hipocampo (aprendizaje rápido, episódico)
# =====================
class Hippocampus(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        beta = 0.9
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

    def forward(self, x):
        mem1 = self.fc1(x)
        spk1, mem1 = self.lif1(mem1)
        mem2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(mem2)
        return spk2, mem2

# =====================
# Neocortex (aprendizaje lento, generalización)
# =====================
class Neocortex(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# =====================
# Ciclo de Aprendizaje
# =====================
env = gym.make("CartPole-v1")

hippocampus = Hippocampus(input_size=4, hidden_size=64, output_size=2)
neocortex = Neocortex(input_size=4, hidden_size=64, output_size=2)

optimizer_hip = optim.Adam(hippocampus.parameters(), lr=1e-3)
optimizer_neocortex = optim.Adam(neocortex.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

num_episodes = 50

for ep in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(200):
        # Corteza entorrinal → codificación a spikes
        spikes = entorhinal_encoder(state)

        # Hipocampo (decisión rápida)
        spk_out, mem_out = hippocampus(spikes.float())
        action = torch.argmax(mem_out).item()

        # Ejecución en el entorno
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Entrenamiento Hipocampo (rápido)
        target = torch.tensor([action])
        loss_hip = loss_fn(mem_out.unsqueeze(0), target)
        optimizer_hip.zero_grad()
        loss_hip.backward()
        optimizer_hip.step()

        # Transferencia: entrenamos el Neocortex con la experiencia
        pred = neocortex(torch.tensor(state, dtype=torch.float))
        target_nc = torch.tensor([action])
        loss_nc = loss_fn(pred.unsqueeze(0), target_nc)
        optimizer_neocortex.zero_grad()
        loss_nc.backward()
        optimizer_neocortex.step()

        state = next_state
        if done:
            break

    print(f"Episode {ep}, Total Reward: {total_reward}")

env.close()
