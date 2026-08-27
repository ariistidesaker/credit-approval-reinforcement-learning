import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TrainingLogger:
    """
    Système de monitoring et logging unifié pour le Reinforcement Learning.
    Gère l'enregistrement TensorBoard, l'exportation tabulaire CSV et la génération de graphiques.
    """

    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
    ):
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.exp_name = experiment_name or f"ppo_credit_{timestamp}"
        self.log_path = os.path.join(log_dir, self.exp_name)

        self.writer: Optional[SummaryWriter] = None
        if self.use_tensorboard:
            os.makedirs(self.log_path, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_path)
            print(f"[MONITORING] TensorBoard active dans : {self.log_path}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            print("[MONITORING] Attention : TensorBoard n'est pas installe. Logging uniquement en memoire/CSV.")

        self.history: List[Dict[str, Any]] = []

    def log_episode(
        self,
        episode: int,
        ep_reward: float,
        total_profit: float,
        approval_rate: float,
        default_rate: float,
        volume_lent: float,
        roi: float,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        entropy: float = 0.0,
        approx_kl: float = 0.0,
        eval_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Enregistre toutes les métriques scalaires de l'épisode dans l'historique et dans TensorBoard.
        """
        # Calcul de la moyenne mobile sur les 10 derniers épisodes
        recent_rewards = [h["ep_reward"] for h in self.history[-9:]] + [ep_reward]
        rolling_mean_reward = float(np.mean(recent_rewards))

        record = {
            "episode": episode,
            "ep_reward": ep_reward,
            "rolling_mean_reward": rolling_mean_reward,
            "total_profit": total_profit,
            "approval_rate": approval_rate,
            "default_rate": default_rate,
            "volume_lent": volume_lent,
            "roi": roi,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
        }

        if eval_metrics:
            record.update(eval_metrics)

        self.history.append(record)

        # Écriture dans TensorBoard
        if self.writer is not None:
            # Pertes d'optimisation
            self.writer.add_scalar("Loss/Policy_Loss", policy_loss, episode)
            self.writer.add_scalar("Loss/Value_Loss", value_loss, episode)
            self.writer.add_scalar("Loss/Entropy", entropy, episode)
            self.writer.add_scalar("Loss/Approx_KL", approx_kl, episode)

            # Récompenses RL
            self.writer.add_scalar("Reward/Episode_Reward", ep_reward, episode)
            self.writer.add_scalar("Reward/Rolling_Mean_10_Reward", rolling_mean_reward, episode)

            # Indicateurs Financiers
            self.writer.add_scalar("Financial/Total_Profit", total_profit, episode)
            self.writer.add_scalar("Financial/Approval_Rate", approval_rate, episode)
            self.writer.add_scalar("Financial/Default_Rate", default_rate, episode)
            self.writer.add_scalar("Financial/ROI", roi, episode)
            self.writer.add_scalar("Financial/Volume_Lent", volume_lent, episode)

            # Métriques d'évaluation déterministe si présentes
            if eval_metrics:
                if "eval_mean_profit" in eval_metrics:
                    self.writer.add_scalar("Eval/Mean_Profit", eval_metrics["eval_mean_profit"], episode)
                if "eval_mean_reward" in eval_metrics:
                    self.writer.add_scalar("Eval/Mean_Reward", eval_metrics["eval_mean_reward"], episode)
                if "eval_mean_approval_rate" in eval_metrics:
                    self.writer.add_scalar("Eval/Approval_Rate", eval_metrics["eval_mean_approval_rate"], episode)
                if "eval_mean_default_rate" in eval_metrics:
                    self.writer.add_scalar("Eval/Default_Rate", eval_metrics["eval_mean_default_rate"], episode)

    def plot_learning_curves(self, save_path: str = "reports/learning_curves.png"):
        """
        Génère un tableau de bord à 4 graphiques retraçant la dynamique d'apprentissage.
        """
        if not self.history:
            print("[MONITORING] Aucun historique disponible pour tracer les courbes.")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame(self.history)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Pertes (Policy & Value Loss)
        axs[0, 0].plot(df["episode"], df["policy_loss"], label="Policy Loss ($L^{CLIP}$)", color="#1f77b4", lw=1.8)
        axs[0, 0].plot(df["episode"], df["value_loss"], label="Value Loss ($L^{VF}$)", color="#d62728", lw=1.8)
        axs[0, 0].set_title("1. Convergence des Pertes (Losses)", fontsize=12, fontweight="bold")
        axs[0, 0].set_xlabel("Épisode")
        axs[0, 0].set_ylabel("Perte")
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # 2. Récompenses (Episode & Rolling Mean)
        axs[0, 1].plot(df["episode"], df["ep_reward"], label="Récompense Brute", color="#aec7e8", alpha=0.6, lw=1.2)
        axs[0, 1].plot(df["episode"], df["rolling_mean_reward"], label="Moyenne Mobile (10 ep)", color="#2ca02c", lw=2.2)
        axs[0, 1].set_title("2. Évolution de la Récompense RL", fontsize=12, fontweight="bold")
        axs[0, 1].set_xlabel("Épisode")
        axs[0, 1].set_ylabel("Récompense Cumulée")
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)

        # 3. Profit Net Bancaire
        axs[1, 0].plot(df["episode"], df["total_profit"] / 1e6, label="Profit Net Entraînement (M$)", color="#9467bd", lw=1.8)
        if "eval_mean_profit" in df.columns:
            eval_mask = df["eval_mean_profit"].notna()
            axs[1, 0].scatter(
                df.loc[eval_mask, "episode"],
                df.loc[eval_mask, "eval_mean_profit"] / 1e6,
                color="#ff7f0e",
                s=40,
                label="Validation Déterministe (M$)",
                zorder=5,
            )
        axs[1, 0].set_title("3. Profit Net Cumulé de la Banque (Millions $)", fontsize=12, fontweight="bold")
        axs[1, 0].set_xlabel("Épisode")
        axs[1, 0].set_ylabel("Profit (M$)")
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        # 4. Taux d'Approbation vs Taux de Défaut
        axs[1, 1].plot(df["episode"], df["approval_rate"] * 100, label="Taux d'Approbation (%)", color="#17becf", lw=1.8)
        axs[1, 1].plot(df["episode"], df["default_rate"] * 100, label="Taux de Défaut (%)", color="#e377c2", lw=1.8)
        axs[1, 1].set_title("4. Gestion du Risque : Approbation vs Défaut", fontsize=12, fontweight="bold")
        axs[1, 1].set_xlabel("Épisode")
        axs[1, 1].set_ylabel("Pourcentage (%)")
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Tableau de Bord de Monitoring : Entraînement PPO - Approbation de Crédits", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[MONITORING] Courbes d'apprentissage sauvegardees dans : {save_path}")

    def export_csv(self, save_path: str = "reports/training_history.csv") -> str:
        """Exporte l'historique complet des métriques dans un fichier CSV."""
        if not self.history:
            return ""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame(self.history)
        df.to_csv(save_path, index=False)
        print(f"[MONITORING] Historique complet exporte en CSV : {save_path}")
        return save_path

    def close(self):
        """Ferme proprement le writer TensorBoard."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
