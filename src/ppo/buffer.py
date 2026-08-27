import torch
import numpy as np
from typing import Generator, Tuple, Dict


class RolloutBuffer:
    """
    Buffer de mémoire pour stocker les trajectoires et calculer les avantages GAE (Generalized Advantage Estimation).
    """

    def __init__(self, buffer_size: int, state_dim: int, device: torch.device = None):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.device = device or torch.device("cpu")

        # Initialisation des structures de stockage
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        """Ajoute une transition dans le buffer."""
        if self.ptr >= self.buffer_size:
            raise IndexError("Le RolloutBuffer est plein. Effectuez un entraînement avant de rajouter des données.")

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Calcule les avantages GAE-lambda et les cibles de retour R_t = A_t + V(s_t).
        """
        last_gae_lam = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            # Delta d'erreur temporelle (TD Error)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            # Accumulation GAE
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

        # Cible de retour pour l'entraînement du Critic
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """
        Générateur de mini-batches mélangés aléatoirement pour les époques d'optimisation PPO.
        """
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)

        # Normalisation des avantages pour stabiliser l'apprentissage
        adv_slice = self.advantages[:self.ptr]
        adv_mean = np.mean(adv_slice)
        adv_std = np.std(adv_slice) + 1e-8
        normalized_advantages = (adv_slice - adv_mean) / adv_std

        for start_idx in range(0, self.ptr, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            b_states = torch.as_tensor(self.states[batch_indices], dtype=torch.float32, device=self.device)
            b_actions = torch.as_tensor(self.actions[batch_indices], dtype=torch.int64, device=self.device)
            b_log_probs = torch.as_tensor(self.log_probs[batch_indices], dtype=torch.float32, device=self.device)
            b_advantages = torch.as_tensor(normalized_advantages[batch_indices], dtype=torch.float32, device=self.device)
            b_returns = torch.as_tensor(self.returns[batch_indices], dtype=torch.float32, device=self.device)
            b_values = torch.as_tensor(self.values[batch_indices], dtype=torch.float32, device=self.device)

            yield b_states, b_actions, b_log_probs, b_advantages, b_returns, b_values

    def clear(self):
        """Réinitialise le pointeur du buffer."""
        self.ptr = 0
        self.full = False
