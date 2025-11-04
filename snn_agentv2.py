import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, functional, learning

# ====== Configuración ======
device = 'cpu'
num_neurons = 10
timesteps = 200

# ====== Neuronas de Memoria de Trabajo ======
# Usamos LIF con tau grande para retener información más tiempo
short_term_layer = neuron.LIFNode(
    tau=200.0,  # constante de tiempo grande → más persistencia
    v_threshold=1.0,
    v_reset=0.0,
    detach_reset=True
).to(device)

# ====== Sinapsis con Memoria a Largo Plazo (STDP) ======
stdp_synapse = nn.Linear(num_neurons, num_neurons, bias=False).to(device)
stdp_synapse.weight.data = torch.rand((num_neurons, num_neurons)) * 0.1

stdp = learning.STDPLearner(
    step_mode='s',
    synapse=stdp_synapse,
    sn=short_term_layer,
    tau_pre=20.0,
    tau_post=20.0,
    f_pre=lambda x: 0.01 * x,
    f_post=lambda x: -0.01 * x,
)

# ====== Estímulo ======
stimulus_time = 10
inputs = torch.zeros((timesteps, num_neurons))
inputs[stimulus_time] = torch.ones(num_neurons) * 1.5

# ====== Simulación ======
membrane_potentials = []
spike_record = []

for t in range(timesteps):
    x = inputs[t]
    spikes = short_term_layer(stdp.synapse(x))
    stdp(x, spikes)  # aprendizaje largo plazo
    membrane_potentials.append(short_term_layer.v.clone().cpu())
    spike_record.append(spikes.clone().cpu())

    # "Replay" para consolidar
    if t in [50, 100, 150]:
        replay_spikes = torch.ones(num_neurons) * 1.5
        short_term_layer(stdp.synapse(replay_spikes))
        stdp(replay_spikes, spikes)

functional.reset_net(short_term_layer)

# ====== Visualización ======
membrane_potentials = torch.stack(membrane_potentials).detach().numpy()
spike_record = torch.stack(spike_record).detach().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.title("Potenciales de Membrana (Memoria de Trabajo)")
plt.imshow(membrane_potentials.T, aspect='auto', cmap='viridis')
plt.colorbar(label="Voltaje")
plt.ylabel("Neurona")

plt.subplot(2, 1, 2)
plt.title("Spikes")
plt.imshow(spike_record.T, aspect='auto', cmap='binary')
plt.ylabel("Neurona")
plt.xlabel("Tiempo")

plt.tight_layout()
plt.show()

print("Pesos sinápticos (memoria a largo plazo):")
print(stdp_synapse.weight.data.detach().cpu())