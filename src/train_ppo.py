import os
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple

from src.credit_mdp_env import CreditApprovalEnv
from src.ppo.agent import PPOAgent
from src.ppo.buffer import RolloutBuffer
from src.monitoring import TrainingLogger


def evaluate_agent(env: CreditApprovalEnv, agent: PPOAgent, num_episodes: int = 5) -> Dict[str, float]:
    """
    Évalue les performances de l'agent PPO en mode déterministe.
    """
    total_rewards = []
    total_profits = []
    approval_rates = []
    default_rates = []
    rois = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=1000 + ep)
        terminated = False
        ep_reward = 0.0

        while not terminated:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

        total_rewards.append(ep_reward)
        total_profits.append(info["total_profit"])
        approval_rates.append(info["approval_rate"])
        default_rates.append(info["default_rate"])
        rois.append(info["roi"])

    return {
        "eval_mean_reward": float(np.mean(total_rewards)),
        "eval_mean_profit": float(np.mean(total_profits)),
        "eval_mean_approval_rate": float(np.mean(approval_rates)),
        "eval_mean_default_rate": float(np.mean(default_rates)),
        "eval_mean_roi": float(np.mean(rois)),
    }


def train_ppo(
    env: Optional[CreditApprovalEnv] = None,
    agent: Optional[PPOAgent] = None,
    num_episodes: int = 150,
    buffer_size: int = 500,
    eval_interval: int = 15,
    use_tensorboard: bool = True,
    log_dir: str = "runs",
    experiment_name: Optional[str] = None,
    reports_dir: str = "reports",
    save_dir: str = "models",
    save_name: str = "best_ppo_agent.pt",
    seed: int = 42,
) -> Tuple[PPOAgent, pd.DataFrame]:
    """
    Fonction d'entraînement de l'agent PPO avec monitoring complet (TensorBoard, Losses, Rewards & Métriques Métier).
    """
    if env is None:
        env = CreditApprovalEnv(data_path="data/synthetic_sadc_lgd_dataset.csv", seed=seed)

    if agent is None:
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            k_epochs=4,
            batch_size=64,
            device=None,
        )

    buffer = RolloutBuffer(buffer_size=buffer_size, state_dim=env.observation_space.shape[0], device=agent.device)
    logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name, use_tensorboard=use_tensorboard)

    best_profit = -np.inf
    best_model_path = os.path.join(save_dir, save_name)

    print("=" * 75)
    print(f" [ENTRAINEMENT PPO & MONITORING] - {num_episodes} Épisodes | Buffer Size: {buffer_size}")
    print("=" * 75)

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        ep_reward = 0.0
        update_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        while not terminated:
            # 1. Sélection de l'action stochastique
            action, log_prob, value = agent.select_action(obs, deterministic=False)

            # 2. Exécution du step MDP
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # 3. Stockage dans le buffer
            buffer.add(
                state=obs,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=terminated,
                value=value,
            )

            obs = next_obs

            # 4. Si le buffer est plein ou si l'épisode est terminé, mise à jour PPO
            if buffer.full or (terminated and buffer.ptr > 0):
                if terminated:
                    last_val = 0.0
                else:
                    _, _, last_val = agent.select_action(next_obs, deterministic=True)

                buffer.compute_returns_and_advantages(
                    last_value=last_val,
                    last_done=terminated,
                    gamma=agent.gamma,
                    gae_lambda=agent.gae_lambda,
                )

                update_stats = agent.update(buffer)
                buffer.clear()

        # Évaluation périodique
        eval_metrics = None
        if episode % eval_interval == 0 or episode == num_episodes:
            eval_metrics = evaluate_agent(env, agent, num_episodes=5)

            print(
                f"Ep {episode:03d}/{num_episodes:03d} | "
                f"Reward: {ep_reward:+6.2f} | "
                f"Profit: {info['total_profit']:>11,.2f}$ | "
                f"Approb: {info['approval_rate']:5.1%} | "
                f"Defaut: {info['default_rate']:5.1%} | "
                f"Eval Profit: {eval_metrics['eval_mean_profit']:>11,.2f}$"
            )

            # Sauvegarde du meilleur modèle
            if eval_metrics["eval_mean_profit"] > best_profit:
                best_profit = eval_metrics["eval_mean_profit"]
                agent.save(best_model_path)
                print(f"  [+] Nouveau Meilleur Modèle Sauvegardé : {best_model_path} (Profit: {best_profit:,.2f}$)")

        # Enregistrement dans le Logger (TensorBoard + Historique interne)
        logger.log_episode(
            episode=episode,
            ep_reward=ep_reward,
            total_profit=info["total_profit"],
            approval_rate=info["approval_rate"],
            default_rate=info["default_rate"],
            volume_lent=info["total_volume_lent"],
            roi=info["roi"],
            policy_loss=update_stats.get("policy_loss", 0.0),
            value_loss=update_stats.get("value_loss", 0.0),
            entropy=update_stats.get("entropy", 0.0),
            approx_kl=update_stats.get("approx_kl", 0.0),
            eval_metrics=eval_metrics,
        )

    elapsed = time.time() - start_time
    print("-" * 75)
    print(f" [FIN ENTRAINEMENT] Temps : {elapsed:.2f}s | Meilleur Profit Évalué : {best_profit:,.2f}$")
    print("=" * 75)

    # Export des graphiques et du fichier CSV
    plot_path = os.path.join(reports_dir, "learning_curves.png")
    csv_path = os.path.join(reports_dir, "training_history.csv")
    logger.plot_learning_curves(save_path=plot_path)
    logger.export_csv(save_path=csv_path)
    logger.close()

    history_df = pd.DataFrame(logger.history)
    return agent, history_df


if __name__ == "__main__":
    train_ppo(num_episodes=100, eval_interval=10)
