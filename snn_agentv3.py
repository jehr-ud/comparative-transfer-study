import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, functional, learning
from environments.visual_maze_env import VisualMazeEnv


class GridCellSystem:
    """
    Grid Cell System for spatial encoding in a discrete 2D environment.

    This class generates activation maps for multiple "grid cells" that simulate
    the periodic and hexagonal firing patterns observed in the medial entorhinal cortex of mammals.

    Each grid cell is defined by random parameters (scale, orientation, and phase),
    and its activation for a position p = (x, y) is computed as the sum of three cosine functions
    oriented 60° apart, forming a hexagonal grid pattern.

    Mathematically, the activation of the i-th grid cell at position p is given by:

        g_i(p) = cos(2π / λ_i * (u_i · p) + φ_{i,0})
            + cos(2π / λ_i * (v_i · p) + φ_{i,1})
            + cos(2π / λ_i * (w_i · p) + φ_{i,0})

    Where:
        - λ_i is the spatial scale (wavelength) of grid cell i.
        - u_i, v_i, w_i are orientation vectors separated by 60° (defining the hexagonal lattice).
        - φ_{i,0}, φ_{i,1} are random phases to shift the grid.
        - The dot product (·) represents the projection of position vector p onto the orientation vectors.

    The class precomputes activation maps for each position in the maze to accelerate
    lookup during training or simulation.

    Main methods:
        - get_grid_cell_input(pos): returns a normalized vector of activations
        of all grid cells at the given position.

    Initialization parameters:
        - height: maze height (number of rows).
        - width: maze width (number of columns).
        - n_grids: number of grid cells to simulate.
        - device: PyTorch device to store maps (cpu or cuda).

    """
    def __init__(self, height, width, n_grids=12, device='cpu'):
        self.height = height
        self.width = width
        self.n_grids = n_grids
        self.device = device

        print("Pre-calculating Grid Cell Activation Maps...")
        self.activation_maps = self._precompute_maps()
        print("Grid Cell Maps Ready")

    def _precompute_maps(self):
        rng = np.random.default_rng(seed=42)
        scales = rng.uniform(3, 10, self.n_grids)
        orientations = rng.uniform(0, np.pi / 3, self.n_grids)
        phases = rng.uniform(0, 2 * np.pi, (self.n_grids, 2))

        maps = torch.zeros(self.height, self.width, self.n_grids)

        y, x = np.meshgrid(
            np.arange(self.height),
            np.arange(self.width),
            indexing='ij'
        )
        pos_grid = np.stack([y.ravel(), x.ravel()], axis=1)

        for k in range(self.n_grids):
            rot_matrix = np.array(
                [
                    [np.cos(orientations[k]), -np.sin(orientations[k])],
                    [np.sin(orientations[k]), np.cos(orientations[k])]
                ]
            )
            rotated_grid = (rot_matrix @ pos_grid.T).T

            pi2 = 2 * np.pi

            term1 = np.cos(
                pi2 * rotated_grid[:, 0] / scales[k]
                + phases[k, 0]
            )

            term2 = np.cos(
                pi2 * (
                    -0.5 * rotated_grid[:, 0]
                    + np.sqrt(3) / 2 * rotated_grid[:, 1]
                ) / scales[k]
                + phases[k, 1]
            )

            term3 = np.cos(
                pi2 * (
                    -0.5 * rotated_grid[:, 0]
                    - np.sqrt(3) / 2 * rotated_grid[:, 1]
                ) / scales[k]
                + phases[k, 0]
            )

            activation = (term1 + term2 + term3).reshape(
                self.height,
                self.width
            )
            maps[:, :, k] = torch.tensor(activation)

        return maps.to(self.device)

    def get_grid_cell_input(self, pos):
        y, x = int(pos[0]), int(pos[1])
        activation_vector = self.activation_maps[y, x, :]
        activation_vector = activation_vector / activation_vector.abs().max()
        return activation_vector.unsqueeze(0)  # batch dim


class PlaceCellLayer(nn.Module):
    """
    Place Cell Layer that receives input from Grid Cells and produces
    localized spatial activations ("place fields").

    The activation of the j-th place cell for position p is computed as:

        P_j(p) = σ( Σ_i w_ji * g_i(p) + b_j )

    where:
        - g_i(p) is the activation of the i-th grid cell at position p,
        - w_ji are the synaptic weights from grid cell i to place cell j,
        - b_j is the bias term,
        - σ is a non-linear activation function (e.g., LIF neuron output).

    The weights w_ji are trained using STDP to form place fields, i.e., selective
    firing when the agent is in a particular spatial location.

    This allows place cells to represent specific locations as a sparse code,
    derived from the periodic grid cell inputs.
    """
    def __init__(self, input_size, place_cell_count):
        super().__init__()
        self.fc = nn.Linear(input_size, place_cell_count, bias=True)
        self.lif = neuron.LIFNode(tau=2.0)
        self.stdp = learning.STDPLearner(
            step_mode='s',
            synapse=self.fc,
            sn=self.lif,
            tau_pre=20.0,
            tau_post=20.0,
            f_pre=lambda x: 0.01 * x,
            f_post=lambda x: -0.01 * x,
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # batch dim
        out = self.fc(x)
        out = self.lif(out)
        return out

    def stdp_step(self):
        self.stdp.step()


class MemorySNNModule(nn.Module):
    """
    Módulo SNN que combina:
    - short_term: LIF con tau grande (hippocampo-like / persistente durante episodio)
    - long_term: LIF (representa salida que STDP modifica)
    STDP modifica la sinapsis self.fc in-place.
    """
    def __init__(self, input_size, hidden_size, tau_short=50.0, tau_long=20.0):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size, bias=False)

        # Memoria "short" (a nivel de dinámica)
        self.short_term = neuron.LIFNode(tau=tau_short, detach_reset=True)

        # Memoria "long" (la salida que será evaluada / usada por siguientes módulos)
        self.long_term = neuron.LIFNode(tau=tau_long, detach_reset=True)

        # STDP actúa sobre self.fc y usa la neurona long_term como post-synaptic detector
        self.stdp = learning.STDPLearner(
            step_mode='s',
            synapse=self.fc,
            sn=self.long_term,
            tau_pre=20.0,
            tau_post=20.0,
            f_pre=lambda x: 0.01 * x,
            f_post=lambda x: -0.01 * x
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.fc(x)
        x = self.short_term(x)
        x = self.long_term(x)
        return x

    def stdp_step(self):
        self.stdp.step()


class BrainMazeSolver(nn.Module):
    """
    Arquitectura con separación hippocampo / neocortex y conservación
    de memoria larga solo si terminated == True.
    """
    def __init__(self, input_size, n_place_cells=32, device='cpu'):
        super().__init__()
        self.device = device

        # módulos
        self.visual = MemorySNNModule(input_size, 64)
        self.entorhinal = MemorySNNModule(64, 64)
        self.place_cells = PlaceCellLayer(64, n_place_cells)
        self.hippocampus = MemorySNNModule(n_place_cells, 32)  # volatile by design
        self.prefrontal = MemorySNNModule(32, 16)
        self.motor = MemorySNNModule(16, 4)

        # Lista de módulos corticales que consideramos para consolidación (neocortex)
        # Por defecto: no incluimos hippocampus (es volátil)
        self.cortical_modules = [
            ('visual', self.visual),
            ('entorhinal', self.entorhinal),
            ('place_cells.fc', self.place_cells.fc),  # place_cells fc es Linear
            ('prefrontal', self.prefrontal),
            ('motor', self.motor)
        ]

        # Diccionario para almacenar pesos consolidados (long-term memory)
        # será dict: module_key -> state_dict (tensors detached)
        self.long_term_memory = {}

    def forward(self, x):
        spikes_visual = self.visual(x)
        spikes_entorhinal = self.entorhinal(spikes_visual)
        spikes_place_cells = self.place_cells(spikes_entorhinal)
        spikes_hippocampus = self.hippocampus(spikes_place_cells)
        spikes_prefrontal = self.prefrontal(spikes_hippocampus)
        spikes_motor = self.motor(spikes_prefrontal)
        return spikes_motor

    def predict(self, input_spikes):
        """
        Realiza un forward y devuelve la acción más probable (explotación pura).
        No aplica exploración, ideal para evaluación.
        """
        with torch.no_grad():
            output_spikes = self(input_spikes)
            spikes_np = output_spikes.detach().cpu().numpy().flatten()
            action = int(spikes_np.argmax())
        return action
    
    # -------------------------
    # Funciones de consolidación
    # -------------------------
    def save_long_term(self):
        """
        Guarda (consolida) los pesos actuales de los módulos corticales
        en self.long_term_memory como copies detach-clone.
        """
        for key, mod in self.cortical_modules:
            if isinstance(mod, nn.Linear):
                # módulo es la capa fc del place_cells
                w = mod.weight.detach().clone().cpu()
                b = mod.bias.detach().clone().cpu() if mod.bias is not None else None
                self.long_term_memory[key] = {'weight': w, 'bias': b}
            else:
                # módulo es MemorySNNModule u otro nn.Module con .fc
                # intentamos guardar fc.weight y fc.bias si existen
                try:
                    sd = {}
                    fc = mod.fc
                    sd['weight'] = fc.weight.detach().clone().cpu()
                    if fc.bias is not None:
                        sd['bias'] = fc.bias.detach().clone().cpu()
                    else:
                        sd['bias'] = None
                    self.long_term_memory[key] = sd
                except Exception as e:
                    print(e)
                    # seguro no tiene fc; intentamos guardar state_dict
                    self.long_term_memory[key] = {
                        k: v.detach().clone().cpu() for k, v in mod.state_dict().items()
                    }

        print("✅ Long-term memory saved for modules:", list(self.long_term_memory.keys()))

    def restore_long_term(self):
        """
        Si existe memoria larga, carga esos pesos en los módulos (prepara episodio).
        """
        if not self.long_term_memory:
            return  # nada que restaurar

        for key, mod in self.cortical_modules:
            if key not in self.long_term_memory:
                continue
            self._restore_module_state(mod, self.long_term_memory[key])

    def _restore_module_state(self, module, state_dict):
        """Restaura los pesos de un módulo desde el diccionario guardado."""
        if isinstance(module, nn.Linear):
            self._copy_linear_params(module, state_dict)
        else:
            if not self._try_restore_fc(module, state_dict):
                self._restore_generic(module, state_dict)

    def _copy_linear_params(self, module, state_dict):
        """Copia pesos y bias a un módulo Linear."""
        module.weight.data.copy_(state_dict['weight'].to(module.weight.device))
        if module.bias is not None and state_dict.get('bias') is not None:
            module.bias.data.copy_(state_dict['bias'].to(module.bias.device))

    def _try_restore_fc(self, module, state_dict):
        """Intenta restaurar usando un atributo .fc, devuelve True si tuvo éxito."""
        if not hasattr(module, 'fc'):
            return False
        fc = module.fc
        fc.weight.data.copy_(state_dict['weight'].to(fc.weight.device))
        if fc.bias is not None and state_dict.get('bias') is not None:
            fc.bias.data.copy_(state_dict['bias'].to(fc.bias.device))
        return True

    def _restore_generic(self, module, state_dict):
        """Fallback para restaurar pesos directamente a los atributos coincidentes."""
        for name, tensor in state_dict.items():
            if hasattr(module, name):
                param = getattr(module, name)
                if isinstance(param, torch.Tensor):
                    param.data.copy_(tensor.to(param.device))

    def revert_to_long_term(self):
        """
        Reemplaza los pesos actuales por los guardados (deshace aprendizaje temporal).
        """
        self.restore_long_term()
        print("🔁 Reverted weights to last consolidated long-term memory.")

    # -------------------------
    # Utilidades
    # -------------------------
    def reset_short_term(self):
        """
        Resetea estados dinámicos de neuronas (memoria de corto plazo).
        """
        # Reset dinámicas de toda la red
        functional.reset_net(self)

    # -------------------------
    # Método de ayuda para debug/print
    # -------------------------
    def print_weights_summary(self):
        print("=== Weights summary (cortical modules) ===")
        for key, _ in self.cortical_modules:
            if key in self.long_term_memory:
                print(f"{key}: consolidated")
            else:
                print(f"{key}: (no consolidated)")


def train_agent(env, net, grid_cells, episodes=100, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.02):
    epsilon = epsilon_start  # valor inicial de exploración
    rewards = []
    
    for ep in range(episodes):
        obs, _ = env.reset()

        # Antes del episodio, restauramos memoria larga si existe
        net.restore_long_term()
        net.reset_short_term()

        total_reward = 0
        terminated, truncated = False, False
        step = 0
        episode_records = []  # registro opcional

        while not (terminated or truncated):
            # Codifica posición
            input_spikes = grid_cells.get_grid_cell_input(obs)
            output_spikes = net(input_spikes)

            spikes_np = output_spikes.detach().cpu().numpy().flatten()

            # ==============================
            #   Selección de acción ε-greedy
            # ==============================
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # aleatorio
            else:
                action = int(spikes_np.argmax())  # mejor acción

            # Ejecutar acción
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1

            # Guardar para replay
            episode_records.append((input_spikes.clone(), action, reward))

            # STDP en línea modulado por recompensa
            modulation = max(0, reward)
            modules_for_stdp = [
                net.visual, net.entorhinal, net.place_cells, net.hippocampus, net.prefrontal, net.motor
            ]

            for module in modules_for_stdp:
                if hasattr(module, 'stdp'):
                    module.stdp.f_pre = lambda x, m=modulation: 0.01 * m * x
                    module.stdp.f_post = lambda x, m=modulation: -0.01 * m * x
                    module.stdp.step()
                elif hasattr(module, 'stdp_step'):
                    module.stdp_step()

        rewards.append(total_reward)
        
        # Consolidación o revertir
        if terminated:
            print("FINISHED - goal reached")
            net.save_long_term()
            print(f"Episode {ep+1}: SUCCESS! Consolidated. Total Reward={total_reward:.2f} Steps={step}")
        else:
            net.revert_to_long_term()
            print(f"Episode {ep+1}: Failed. Reverted. Total Reward={total_reward:.2f} Steps={step}")

        net.reset_short_term()

        # Decaimiento de ε (no baja de epsilon_end)
        epsilon = max(epsilon_end, epsilon - epsilon_decay)
        print(f"Exploration rate (epsilon): {epsilon:.3f}")
        
        print("reward avg ", sum(rewards) / len(rewards))


if __name__ == "__main__":
    config = {
        "name": "simple",
        "size": 6,
        "obstacles": [(1, 1), (2, 3), (3, 1)],
        "render_mode": 'None',
        "shared_window": None
    }

    env = VisualMazeEnv(config)
    grid_cells = GridCellSystem(env.size, env.size, n_grids=12, device='cpu')

    input_size = 12  # n_grids
    brain = BrainMazeSolver(input_size)

    train_agent(env, brain, grid_cells, episodes=100)
