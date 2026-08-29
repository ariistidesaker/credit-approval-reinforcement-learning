"""
Script d'entraînement DQN.
Utilise le même environnement MDP que PPO et le même système de monitoring.
"""
import os
import sys
import argparse
import numpy as np
import torch

# Ajout du chemin racine pour permettre les imports absolus
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.credit_mdp_env import CreditApprovalEnv
from src.dqn.agent import DQNAgent
from src.dqn.buffer import ReplayBuffer
from src.monitoring import TrainingLogger


def train_dqn(
    env: CreditApprovalEnv,
    agent: DQNAgent,
    replay_buffer: ReplayBuffer,
    num_episodes: int = 150,
    warmup_steps: int = 1000,
    batch_size: int = 64,
    update_freq: int = 4,
    target_update_freq_episodes: int = 10,
    use_tensorboard: bool = True,
    experiment_name: str = "DQN",
    reports_dir: str = "reports",
    save_dir: str = "models",
    save_name: str = "best_dqn_agent.pt",
) -> DQNAgent:

    # Initialisation du logger (sans reports_dir, car géré plus tard)
    logger = TrainingLogger(
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard,
        log_dir="runs",
    )

    best_profit = -float("inf")
    episode_rewards = []
    episode_profits = []
    episode_approvals = []
    episode_defaults = []

    # Remplissage initial du buffer avec des actions aléatoires
    print(f"[DQN] Remplissage du buffer avec {warmup_steps} étapes aléatoires...")
    obs, info = env.reset(seed=42)
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    step_counter = 0  # Compteur local pour les métriques pas-à-pas (si besoin)

    print(f"[DQN] Début de l'entraînement sur {num_episodes} épisodes...")
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0.0
        episode_steps = 0

        while True:
            # Sélection de l'action
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Stockage de la transition
            replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
            episode_reward += reward
            episode_steps += 1
            step_counter += 1

            # Mise à jour du réseau Q si le buffer est assez rempli
            if len(replay_buffer) >= batch_size and episode_steps % update_freq == 0:
                metrics = agent.update(replay_buffer, batch_size)
                # Optionnel : écrire les métriques pas-à-pas dans TensorBoard
                if logger.writer is not None:
                    logger.writer.add_scalar("Loss/TD_Loss", metrics["td_loss"], step_counter)
                    logger.writer.add_scalar("Q/Max_Q", metrics["max_q"], step_counter)

            # Décroissance de l'epsilon
            agent.decay_epsilon()

            if terminated or truncated:
                # Récupération des infos financières
                episode_profit = info["total_profit"]
                episode_approved = info["approved_count"]
                episode_default_count = info["default_count"]

                # Sauvegarde des métriques
                episode_rewards.append(episode_reward)
                episode_profits.append(episode_profit)
                episode_approvals.append(episode_approved)
                episode_defaults.append(episode_default_count)

                # Logging épisodique via le logger existant
                logger.log_episode(
                    episode=episode,
                    ep_reward=episode_reward,
                    total_profit=episode_profit,
                    approval_rate=info["approval_rate"] * 100,  # en pourcentage
                    default_rate=info["default_rate"] * 100,
                    volume_lent=info["total_volume_lent"],
                    roi=info["roi"] * 100,
                    # Les métriques PPO ne sont pas applicables, on met des zéros
                    policy_loss=0.0,
                    value_loss=0.0,
                    entropy=0.0,
                    approx_kl=0.0,
                )

                # Mise à jour du réseau cible
                if episode % target_update_freq_episodes == 0:
                    agent.update_target_network()

                # Sauvegarde du meilleur modèle
                if episode_profit > best_profit:
                    best_profit = episode_profit
                    os.makedirs(save_dir, exist_ok=True)
                    agent.save(os.path.join(save_dir, save_name))
                    print(f"[DQN] Nouveau meilleur modèle sauvegardé (épisode {episode}, profit {best_profit:.2f} $)")

                break

            obs = next_obs

        if episode % 10 == 0:
            print(f"[DQN] Épisode {episode:3d} | Récompense: {episode_reward:.2f} | Profit: {episode_profit:.2f} $ | Epsilon: {agent.epsilon:.3f}")

    # Génération des courbes finales à partir de l'historique du logger
    logger.plot_learning_curves(save_path=os.path.join(reports_dir, "learning_curves.png"))
    logger.export_csv(save_path=os.path.join(reports_dir, "training_history.csv"))
    logger.close()

    print(f"[DQN] Entraînement terminé. Meilleur profit: {best_profit:.2f} $")
    print(f"[DQN] Modèle sauvegardé dans: {os.path.join(save_dir, save_name)}")
    print(f"[DQN] Courbes disponibles dans: {os.path.join(reports_dir, 'learning_curves.png')}")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Entraînement DQN pour l'approbation de prêts")
    parser.add_argument("--episodes", type=int, default=150, help="Nombre d'épisodes")
    parser.add_argument("--warmup", type=int, default=1000, help="Étapes de remplissage du buffer")
    parser.add_argument("--no_tensorboard", action="store_true", help="Désactiver TensorBoard")
    parser.add_argument("--name", type=str, default="DQN", help="Nom de l'expérience")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    args = parser.parse_args()

    # Création de l'environnement
    env = CreditApprovalEnv(data_path="data/synthetic_sadc_lgd_dataset.csv", shuffle_on_reset=True, seed=42)

    # Dimension de l'état
    obs, info = env.reset()
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DQN] Utilisation du device: {device}")

    # Agent DQN
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=10,
        device=device,
    )

    # Buffer de rejouissance
    replay_buffer = ReplayBuffer(capacity=100_000, device=device)

    # Lancement de l'entraînement
    train_dqn(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        num_episodes=args.episodes,
        warmup_steps=args.warmup,
        batch_size=64,
        update_freq=4,
        target_update_freq_episodes=10,
        use_tensorboard=not args.no_tensorboard,
        experiment_name=args.name,
        reports_dir="reports",
        save_dir="models",
        save_name="best_dqn_agent.pt",
    )


if __name__ == "__main__":
    main()