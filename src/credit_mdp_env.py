import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces

from src.preprocessing import CreditDataPreprocessor
from src.utils import calculate_financial_outcome


class CreditApprovalEnv(gym.Env):
    """
    Environnement MDP pour l'approbation de prêts bancaires (Reinforcement Learning).
    
    Conforme aux spécifications Gymnasium :
    - Espace d'état S : Caractéristiques normalisées du demandeur de prêt + Contexte macroéconomique
    - Espace d'action A : Discret {0: Refuser, 1: Approuver}
    - Fonction de récompense R : Gain financier net corrigé du risque (intérêts perçus - pertes par défaut)
    - Dynamique de transition P : Évolution séquentielle à travers le portefeuille de demandes
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_steps: Optional[int] = None,
        shuffle_on_reset: bool = True,
        initial_capital: float = 1_000_000.0,
        reward_scale: float = 1e-4,
        operational_cost_reject: float = 10.0,
        cost_of_funds_annual: float = 0.05,
        collateral_haircut: float = 0.8,
        seed: Optional[int] = 42,
    ):
        super().__init__()

        self.data_path = data_path or "data/synthetic_sadc_lgd_dataset.csv"
        self.shuffle_on_reset = shuffle_on_reset
        self.initial_capital = float(initial_capital)
        self.reward_scale = float(reward_scale)
        self.operational_cost_reject = float(operational_cost_reject)
        self.cost_of_funds_annual = float(cost_of_funds_annual)
        self.collateral_haircut = float(collateral_haircut)

        # 1. Chargement et prétraitement des données
        self.preprocessor = CreditDataPreprocessor(self.data_path)
        self.state_matrix, self.processed_df = self.preprocessor.fit_transform()

        self.num_samples = len(self.processed_df)
        self.max_steps = max_steps or self.num_samples
        self.feature_names = self.preprocessor.feature_names
        self.obs_dim = self.state_matrix.shape[1]

        # 2. Espaces Gymnasium
        # Espace d'action : 0 = Refuser, 1 = Approuver
        self.action_space = spaces.Discrete(2)

        # Espace d'observation : Vecteur de caractéristiques continues normalisées
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # 3. Générateur de nombres aléatoires
        self._np_random = np.random.default_rng(seed)

        # 4. Variables d'état interne
        self.current_step = 0
        self.sample_indices = np.arange(self.num_samples)
        self.current_capital = self.initial_capital
        self.total_profit = 0.0
        self.approved_count = 0
        self.rejected_count = 0
        self.default_count = 0
        self.total_volume_lent = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Réorganisation ou sélection aléatoire des dossiers pour l'épisode
        self.sample_indices = np.arange(self.num_samples)
        if self.shuffle_on_reset:
            self._np_random.shuffle(self.sample_indices)

        self.current_step = 0
        self.current_capital = self.initial_capital
        self.total_profit = 0.0
        self.approved_count = 0
        self.rejected_count = 0
        self.default_count = 0
        self.total_volume_lent = 0.0

        current_idx = self.sample_indices[self.current_step]
        obs = self.state_matrix[current_idx]

        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Exécute une décision d'approbation/refus sur le dossier courant.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide : {action}")

        current_idx = self.sample_indices[self.current_step]
        row = self.processed_df.iloc[current_idx]

        # Calcul financier de l'issue
        outcome = calculate_financial_outcome(
            row=row,
            action=action,
            cost_of_funds_annual=self.cost_of_funds_annual,
            operational_cost_reject=self.operational_cost_reject,
            collateral_haircut=self.collateral_haircut,
            rng=self._np_random,
        )

        net_profit = outcome["net_profit"]
        self.total_profit += net_profit
        self.current_capital += net_profit

        if action == 1:
            self.approved_count += 1
            self.total_volume_lent += outcome["ead"]
            if outcome["defaulted"]:
                self.default_count += 1
        else:
            self.rejected_count += 1

        # Calcul de la récompense pour l'agent RL (mise à l'échelle)
        reward = float(net_profit * self.reward_scale)

        self.current_step += 1

        # Conditions d'arrêt : fin des dossiers ou faillite du capital
        terminated = (self.current_step >= self.max_steps) or (self.current_capital <= 0)
        truncated = False

        if not terminated:
            next_idx = self.sample_indices[self.current_step]
            next_obs = self.state_matrix[next_idx]
        else:
            # Observation finale fictive ou identique à la dernière
            next_obs = np.zeros(self.obs_dim, dtype=np.float32)

        info = self._get_info()
        info.update({
            "last_action": action,
            "last_net_profit": net_profit,
            "last_p_default": outcome["p_default"],
            "last_defaulted": outcome["defaulted"],
            "last_ead": outcome["ead"],
        })

        return next_obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Retourne les métriques globales du portefeuille."""
        approval_rate = (
            self.approved_count / max(1, self.approved_count + self.rejected_count)
        )
        default_rate = (
            self.default_count / max(1, self.approved_count)
        )
        roi = (
            self.total_profit / max(1.0, self.total_volume_lent)
        ) if self.total_volume_lent > 0 else 0.0

        return {
            "step": self.current_step,
            "total_profit": self.total_profit,
            "current_capital": self.current_capital,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "default_count": self.default_count,
            "approval_rate": approval_rate,
            "default_rate": default_rate,
            "total_volume_lent": self.total_volume_lent,
            "roi": roi,
        }

    def render(self):
        """Affiche l'état courant du portefeuille."""
        info = self._get_info()
        print(
            f"[Step {info['step']}/{self.max_steps}] "
            f"Capital: {info['current_capital']:,.2f}$ | "
            f"Profit Net: {info['total_profit']:,.2f}$ | "
            f"Approbations: {info['approved_count']} (Taux: {info['approval_rate']:.1%}) | "
            f"Défauts: {info['default_count']} (Taux défaut: {info['default_rate']:.1%})"
        )
