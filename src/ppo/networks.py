import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, List


def layer_init(layer: nn.Linear, std: float = 1.414, bias_const: float = 0.0) -> nn.Linear:
    """Initialisation orthogonale standard pour les réseaux de Reinforcement Learning."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    """
    Réseau de neurones de la Politique (Actor).
    Prend en entrée l'état continu (18 dimensions) et produit les logits pour chaque action discrète.
    """

    def __init__(self, state_dim: int = 18, action_dim: int = 2, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(prev_dim, h)))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(layer_init(nn.Linear(prev_dim, action_dim), std=0.01))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Retourne les logits non normalisés de chaque action."""
        return self.network(state)

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """Retourne la distribution catégorielle sur l'espace d'action."""
        logits = self.forward(state)
        return Categorical(logits=logits)


class CriticNetwork(nn.Module):
    """
    Réseau de neurones de la Valeur (Critic).
    Prend en entrée l'état continu et estime la fonction de valeur d'état V(s).
    """

    def __init__(self, state_dim: int = 18, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(prev_dim, h)))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(layer_init(nn.Linear(prev_dim, 1), std=1.0))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Retourne l'estimation scalaire de la valeur de l'état V(s)."""
        return self.network(state)


class ActorCritic(nn.Module):
    """
    Module unifié Actor-Critic combinant la politique et la fonction de valeur.
    """

    def __init__(self, state_dim: int = 18, action_dim: int = 2, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.critic = CriticNetwork(state_dim, hidden_dims)

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """Retourne la distribution des actions et la valeur d'état estimée."""
        dist = self.actor.get_distribution(state)
        value = self.critic(state)
        return dist, value

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor = None, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sélectionne une action et évalue son log-probabilité, la valeur d'état et l'entropie.
        """
        dist = self.actor.get_distribution(state)
        value = self.critic(state)

        if action is None:
            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value.squeeze(-1), entropy
