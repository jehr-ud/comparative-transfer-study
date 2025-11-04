# diayn_cls_toy.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

# -----------------------
# Hyperparámetros rápidos
# -----------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DIAYN_EPISODES = 120        # pretrain rápido (toy)
DIAYN_MAX_STEPS = 200
SKILLS = 4
MACRO_LEN = 6               # pasos por skill en transferencia
HIPPO_BUFFER_SIZE = 200     # episodic memory
CONSOLIDATION_EPOCHS = 80   # entrenamiento cortical (lento)
CORTEX_BATCH = 64

# -----------------------
# Reusable utils
# -----------------------
def one_hot(idx, dim):
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v

# -----------------------
# Redes: SkillPolicy & Discriminator
# -----------------------
class SkillPolicy(nn.Module):
    def __init__(self, state_dim, skill_dim, action_dim, fixed_input_dim=None):
        super().__init__()
        # si fixed_input_dim se pasa (p.ej. cartpole_state+skill_dim), usamos eso para compatibilidad
        if fixed_input_dim is not None:
            input_dim = fixed_input_dim
        else:
            input_dim = state_dim + skill_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state, skill):
        # state: [B, state_dim]; skill: [B, skill_dim]
        x = torch.cat([state, skill], dim=-1)
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, state_dim, skill_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, skill_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, s):
        return self.net(s)  # log-probs over skills

# -----------------------
# Hipocampo: episodic buffer
# -----------------------
Episode = namedtuple('Episode', ['skill_id', 'states', 'actions', 'rewards', 'total_reward'])

class EpisodicBuffer:
    def __init__(self, capacity=200):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, episode: Episode):
        self.buffer.append(episode)

    def sample_topk(self, k):
        # devuelve k episodios con mayor total_reward (si hay menos, devuelve todos)
        sorted_eps = sorted(self.buffer, key=lambda e: e.total_reward, reverse=True)
        return sorted_eps[:k]

    def __len__(self):
        return len(self.buffer)

# -----------------------
# Cortex: slow consolidator (aprende a elegir skill dado estado)
# Implementación simple: red que mapea state -> skill logits
# -----------------------
class Cortex(nn.Module):
    def __init__(self, state_dim, skill_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, skill_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, s):
        return self.net(s)

# -----------------------
# DIAYN pretraining (toy) en CartPole
# -----------------------
def diayn_pretrain_collect(env_name="CartPole-v1", skills=SKILLS, episodes=DIAYN_EPISODES, hippo_buffer=None):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    skill_dim = skills

    policy = SkillPolicy(state_dim, skill_dim, action_dim, fixed_input_dim=state_dim + skill_dim)  # fija input para compatibilidad
    disc = Discriminator(state_dim, skill_dim)

    opt_policy = optim.Adam(policy.parameters(), lr=1e-3)
    opt_disc = optim.Adam(disc.parameters(), lr=1e-3)

    for ep in range(episodes):
        obs = env.reset()
        # gym >=0.26 returns (obs, info)
        if isinstance(obs, tuple):
            state = obs[0]
        else:
            state = obs
        skill_id = np.random.randint(skill_dim)
        skill_vec = one_hot(skill_id, skill_dim)

        states, actions, rewards = [], [], []
        total_intrinsic = 0.0

        for step in range(DIAYN_MAX_STEPS):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            skill_t = torch.tensor(skill_vec, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                probs = policy(state_t, skill_t)
            action = torch.distributions.Categorical(probs).sample().item()

            next_obs = env.step(action)
            if len(next_obs) == 5:
                next_state, _, done, _, _ = next_obs
            else:
                next_state, _, done, _ = next_obs

            # train discriminator to predict skill from next_state
            opt_disc.zero_grad()
            logp = disc(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))  # log softmax
            loss_disc = nn.NLLLoss()(logp, torch.tensor([skill_id]))
            loss_disc.backward()
            opt_disc.step()

            # intrinsic reward = log q(z|s) - log p(z) -> p(z) uniform -> const ignored
            with torch.no_grad():
                intrinsic = logp[0, skill_id].item()
            total_intrinsic += intrinsic

            # update policy to maximize intrinsic (gradient ascent)
            opt_policy.zero_grad()
            # we use a simple surrogate: negative log-probability of skill as loss
            loss_policy = - intrinsic
            # wrap in tensor to create grad (toy)
            t = torch.tensor(loss_policy, requires_grad=True)
            t.backward()
            # apply gradients to policy parameters manually by stepping optimizer
            opt_policy.step()

            states.append(state.copy() if isinstance(state, np.ndarray) else np.array(state))
            actions.append(action)
            rewards.append(intrinsic)  # store intrinsic as reward for episode memory (toy)
            state = next_state
            if done:
                break

        ep_struct = Episode(skill_id=skill_id,
                            states=np.array(states, dtype=np.float32),
                            actions=np.array(actions, dtype=np.int32),
                            rewards=np.array(rewards, dtype=np.float32),
                            total_reward=float(sum(rewards)))
        if hippo_buffer is not None:
            hippo_buffer.add(ep_struct)

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"[DIAYN] ep {ep+1}/{episodes}  total_intrinsic={total_intrinsic:.3f}  hippo_size={len(hippo_buffer)}")

    env.close()
    return policy  # policy preentrenada (skill-conditioned)

# -----------------------
# Consolidación cortical (slow): imitar skills exitosos
# Seleccionamos episodios top-K del buffer y entrenamos cortex para predecir el skill usado
# -----------------------
def consolidate_cortex(cortex: Cortex, hippo_buffer: EpisodicBuffer, epochs=CONSOLIDATION_EPOCHS, batch=CORTEX_BATCH, top_k=40):
    opt = optim.Adam(cortex.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    # obtenemos los episodios top-K
    episodes = hippo_buffer.sample_topk(top_k)
    if not episodes:
        print("[Cortex] buffer vacío, nada que consolidar.")
        return

    # crear dataset: pares (state, skill_id) usando todos los estados de los episodios top
    X = []
    Y = []
    for ep in episodes:
        for s in ep.states:
            X.append(s)
            Y.append(ep.skill_id)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)

    # dataset simple en memoria
    n = len(Y)
    print(f"[Cortex] consolidando {n} ejemplos (top {len(episodes)} episodios)")

    for e in range(epochs):
        perm = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch):
            ids = perm[i:i+batch]
            if len(ids) == 0:
                continue
            s_batch = torch.tensor(X[ids], dtype=torch.float32)
            y_batch = torch.tensor(Y[ids], dtype=torch.long)
            probs = cortex(s_batch)
            loss = loss_fn(probs, y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (e+1) % 20 == 0 or e == 0:
            print(f"[Cortex] epoch {e+1}/{epochs}  loss={np.mean(losses):.4f}")

# -----------------------
# Transfer: usar cortex para elegir skill, ejecutar skill_policy como macroacción en MountainCar
# -----------------------
def transfer_with_cortex(policy: SkillPolicy, cortex: Cortex, env_name="MountainCar-v0",
                         episodes=20, macro_len=MACRO_LEN, skill_dim=SKILLS,
                         cartpole_state_dim=4):
    env = gym.make(env_name)
    mc_state_dim = env.observation_space.shape[0]

    for ep in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            state = obs[0]
        else:
            state = obs
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < DIAYN_MAX_STEPS:
            # cortex decide skill from (current) state (MountainCar state)
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # cortex was trained on cartpole-state-dim states; but we trained cortex with CartPole states
            # here we must either map MC state to same dim or train cortex on generic features.
            # Simpler: pad MountainCar state to cartpole dim (zeros)
            if mc_state_dim < cartpole_state_dim:
                pad = np.zeros(cartpole_state_dim - mc_state_dim, dtype=np.float32)
                s_pad = np.concatenate([state, pad], axis=0)
            else:
                s_pad = state[:cartpole_state_dim]
            s_pad_t = torch.tensor(s_pad, dtype=torch.float32).unsqueeze(0)
            skill_probs = cortex(s_pad_t).detach().numpy().flatten()
            skill_id = int(np.argmax(skill_probs))
            skill_vec = one_hot(skill_id, skill_dim)

            # Ejecutar skill como macroacción
            for _ in range(macro_len):
                # construir input para policy: state (padded to cartpole dim) + skill
                state_for_policy = s_pad_t  # 1 x cartpole_state_dim
                skill_t = torch.tensor(skill_vec, dtype=torch.float32).unsqueeze(0)
                # policy fue entrenada con input_dim = cartpole_state_dim + skill_dim
                with torch.no_grad():
                    probs = policy(state_for_policy, skill_t)
                action = torch.distributions.Categorical(probs).sample().item()

                next_obs = env.step(action)
                if len(next_obs) == 5:
                    next_state, reward, done, _, _ = next_obs
                else:
                    next_state, reward, done, _ = next_obs

                total_reward += reward
                steps += 1

                # preparar siguiente padded state
                if mc_state_dim < cartpole_state_dim:
                    pad = np.zeros(cartpole_state_dim - mc_state_dim, dtype=np.float32)
                    s_pad = np.concatenate([next_state, pad], axis=0)
                else:
                    s_pad = next_state[:cartpole_state_dim]
                s_pad_t = torch.tensor(s_pad, dtype=torch.float32).unsqueeze(0)
                state = next_state
                if done:
                    break

        print(f"[Transfer] ep {ep+1}/{episodes}  steps={steps}  total_reward={total_reward:.2f}")

    env.close()

# -----------------------
# Main flow: DIAYN -> Hipocampo -> Cortex consolidation -> Transfer
# -----------------------
if __name__ == "__main__":
    print("=== INICIO: DIAYN + CLS (toy) ===")

    # 1) Hipocampo
    hippo = EpisodicBuffer(capacity=HIPPO_BUFFER_SIZE)

    # 2) Pretrain DIAYN en CartPole y llenar hipocampo
    print("\n>> Preentrenando DIAYN en CartPole y almacenando episodios en hipocampo...")
    policy = diayn_pretrain_collect(env_name="CartPole-v1", skills=SKILLS, episodes=DIAYN_EPISODES, hippo_buffer=hippo)

    # 3) Inicializar cerebro cortical e consolidar
    cartpole_state_dim = 4  # fixeado por CartPole
    cortex = Cortex(state_dim=cartpole_state_dim, skill_dim=SKILLS)
    print("\n>> Consolidación cortical (lento) a partir del hipocampo (imitación over top episodes)...")
    consolidate_cortex(cortex, hippo, epochs=CONSOLIDATION_EPOCHS, batch=CORTEX_BATCH, top_k=40)

    # 4) Transferencia a MountainCar usando cortex -> skill_policy como macroacciones
    print("\n>> Transferencia a MountainCar: cortex selecciona skills, skill-policy ejecuta macroacciones")
    transfer_with_cortex(policy, cortex, env_name="MountainCar-v0", episodes=12, macro_len=MACRO_LEN,
                         skill_dim=SKILLS, cartpole_state_dim=cartpole_state_dim)

    print("\n=== FIN ===")
