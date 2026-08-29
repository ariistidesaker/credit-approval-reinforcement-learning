import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.dqn.networks import QNetwork

class DQNAgent:
    """
    Agent DQN avec :
    - Politique ε-greedy pour l'exploration.
    - Double réseau (Q et Target) avec mise à jour périodique (hard update).
    - Perte TD (MSE) et optimiseur Adam.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 10,  # Mise à jour de la target tous les N épisodes
        device: torch.device = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Réseau principal Q
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        # Réseau cible (copie des poids)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        # On désactive les gradients pour le réseau cible (optimisation séparée)
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Sélectionne une action selon la politique ε-greedy.
        Si deterministic=True, on prend l'action argmax (pas d'exploration).
        """
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, replay_buffer, batch_size: int) -> dict:
        """
        Effectue une étape de mise à jour du réseau Q en utilisant un mini-batch du buffer.
        Retourne le dictionnaire des métriques de perte.
        """
        if len(replay_buffer) < batch_size:
            return {"td_loss": 0.0, "max_q": 0.0}

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 1. Calcul des Q-valeurs actuelles pour les actions prises
        current_q_values = self.q_network(states).gather(1, actions)

        # 2. Calcul des Q-valeurs cibles (Double DQN optionnel)
        #    Ici on utilise la DQN standard : max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        # 3. Perte TD (Huber / Smooth L1 Loss recommandée pour DQN)
        loss = nn.SmoothL1Loss()(current_q_values, targets)

        # 4. Descente de gradient
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour stabiliser l'entraînement
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "td_loss": loss.item(),
            "max_q": current_q_values.mean().item(),
        }

    def update_target_network(self):
        """Copie les poids du réseau Q vers le réseau cible (hard update)."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Décroît l'epsilon d'exploration jusqu'à la valeur minimale."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """Sauvegarde les poids du modèle Q."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        """Charge les poids du modèle Q."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_start)