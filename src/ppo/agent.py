import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List

from src.ppo.networks import ActorCritic
from src.ppo.buffer import RolloutBuffer


class PPOAgent:
    """
    Agent Proximal Policy Optimization (PPO).
    Implémente la fonction de perte clippée, l'estimation Actor-Critic et les mises à jour par mini-batches.
    """

    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 2,
        hidden_dims: List[int] = [64, 64],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        k_epochs: int = 4,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.k_epochs = k_epochs
        self.batch_size = batch_size

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Réseau unifié Actor-Critic
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Sélectionne une action pour un état donné.
        Retourne (action_int, log_prob_float, value_float).
        """
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, value, _ = self.actor_critic.get_action_and_value(
                state_tensor, deterministic=deterministic
            )
            return action.item(), log_prob.item(), value.item()

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Met à jour les paramètres de l'Actor-Critic sur plusieurs époques avec les données du buffer.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        for _ in range(self.k_epochs):
            for b_states, b_actions, b_old_log_probs, b_advantages, b_returns, b_old_values in buffer.get_batches(
                self.batch_size
            ):
                # Évaluation de la politique courante sur le batch
                _, new_log_probs, new_values, entropy = self.actor_critic.get_action_and_value(
                    b_states, b_actions
                )

                # 1. Calcul du ratio de probabilité r_t(theta) = pi_theta(a|s) / pi_old(a|s)
                log_ratio = new_log_probs - b_old_log_probs
                ratio = torch.exp(log_ratio)

                # Approximation de divergence KL pour monitoring
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                # 2. Perte de politique clippée (Clipped Surrogate Loss)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 3. Perte du Critic (Value Function Loss avec clipping optionnel)
                v_loss_unclipped = (new_values - b_returns) ** 2
                v_clipped = b_old_values + torch.clamp(new_values - b_old_values, -self.clip_eps, self.clip_eps)
                v_loss_clipped = (v_clipped - b_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # 4. Bonus d'entropie
                entropy_loss = entropy.mean()

                # 5. Perte totale combinée
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

                # Rétropropagation et optimisation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
            "approx_kl": total_approx_kl / max(1, num_updates),
        }

    def save(self, filepath: str):
        """Sauvegarde les poids du modèle et de l'optimiseur."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "actor_critic_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        """Charge les poids sauvegardés."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint introuvable : {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
