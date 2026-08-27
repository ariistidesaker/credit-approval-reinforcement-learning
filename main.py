"""
Script principal de demonstration complet (Etapes 1, 2 & 3) :
- Etape 1 : Recuperation et pretraitement des donnees
- Etape 2 : Instanciation du MDP (CreditApprovalEnv)
- Etape 3 : Entrainement et Evaluation de l'agent PPO face aux strategies de reference (Baselines)
"""

import os
import argparse
import numpy as np
import pandas as pd

from data.download_data import download_and_save_data
from src.preprocessing import CreditDataPreprocessor
from src.credit_mdp_env import CreditApprovalEnv
from src.ppo.agent import PPOAgent
from src.train_ppo import train_ppo


def run_full_pipeline(num_episodes: int = 150, force_retrain: bool = False):
    print("=" * 75)
    print(" PROJET RL : OPTIMISATION DE L'APPROBATION DES PRETS BANCAIRES")
    print("=" * 75)

    # -------------------------------------------------------------
    # ETAPE 1 : Verification / Recuperation des donnees
    # -------------------------------------------------------------
    print("\n--- ETAPE 1 : Chargement et Pretraitement des Donnees ---")
    data_path = download_and_save_data("data")

    preprocessor = CreditDataPreprocessor(data_path)
    state_matrix, clean_df = preprocessor.fit_transform()

    print(f"[OK] Donnees chargees : {len(clean_df)} dossiers clients.")
    print(f"[OK] Dimensions de la matrice d'etats MDP : {state_matrix.shape}")
    print(f"[OK] Nombre de features normalisees : {len(preprocessor.feature_names)}")

    # -------------------------------------------------------------
    # ETAPE 2 : Construction et Test de l'Environnement MDP
    # -------------------------------------------------------------
    print("\n--- ETAPE 2 : Validation du MDP (Gymnasium Environment) ---")
    env = CreditApprovalEnv(data_path=data_path, shuffle_on_reset=False, seed=42)
    print(f"[OK] Environnement MDP cree : {env.__class__.__name__}")
    print(f"     - Espace d'etats (S)  : {env.observation_space}")
    print(f"     - Espace d'actions (A): {env.action_space} (0 = Rejeter, 1 = Approuver)")

    # -------------------------------------------------------------
    # ETAPE 3 : Entrainement / Chargement de l'Agent PPO
    # -------------------------------------------------------------
    print("\n--- ETAPE 3 : Agent Reinforcement Learning (PPO) ---")
    model_path = os.path.join("models", "best_ppo_agent.pt")
    
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
    )

    if os.path.exists(model_path) and not force_retrain:
        print(f"[INFO] Chargement du modele PPO pre-entraine : {model_path}")
        agent.load(model_path)
    else:
        print(f"[INFO] Lancement de l'entrainement PPO ({num_episodes} episodes)...")
        train_env = CreditApprovalEnv(data_path=data_path, shuffle_on_reset=True, seed=42)
        agent, _ = train_ppo(
            env=train_env,
            agent=agent,
            num_episodes=num_episodes,
            eval_interval=15,
            save_dir="models",
            save_name="best_ppo_agent.pt",
        )

    # -------------------------------------------------------------
    # Evaluation Comparative : Baselines vs Agent PPO
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" BENCHMARK COMPARATIF : BASELINES VS AGENT PPO (500 Dossiers)")
    print("=" * 75)

    def evaluate_policy(policy_name: str, policy_fn):
        obs, info = env.reset(seed=42)
        terminated = False
        step_count = 0

        while not terminated:
            current_idx = env.sample_indices[env.current_step]
            raw_row = env.processed_df.iloc[current_idx]

            action = policy_fn(obs, raw_row)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

        return {
            "Strategie": policy_name,
            "Approbations": f"{info['approved_count']}/{step_count} ({info['approval_rate']:.1%})",
            "Defauts": f"{info['default_count']} ({info['default_rate']:.1%})",
            "Volume Prete ($)": f"{info['total_volume_lent']:,.0f} $",
            "Profit Net ($)": f"{info['total_profit']:,.2f} $",
            "ROI (%)": f"{info['roi']:.2%}",
        }

    results = []

    # 1. Tout Approuver
    results.append(evaluate_policy(
        "1. Tout Approuver (Naif)",
        lambda obs, row: 1
    ))

    # 2. Aleatoire (50/50)
    rng = np.random.default_rng(42)
    results.append(evaluate_policy(
        "2. Aleatoire (50/50)",
        lambda obs, row: int(rng.integers(0, 2))
    ))

    # 3. Heuristique Experte
    results.append(evaluate_policy(
        "3. Heuristique Experte (Score>=580, DTI<0.45)",
        lambda obs, row: 1 if (row["Risk_Score"] >= 580 and row.get("Debt_to_Income_Ratio", 0.5) < 0.45) else 0
    ))

    # 4. Agent PPO Entraine (Deterministe)
    results.append(evaluate_policy(
        "4. Agent PPO (Reinforcement Learning)",
        lambda obs, row: agent.select_action(obs, deterministic=True)[0]
    ))

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    print("\n" + "=" * 75)
    print(" L'agent PPO a ete entraine et evalue avec succes !")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projet RL - Approbation de prêts")
    parser.add_argument("--episodes", type=int, default=150, help="Nombre d'episodes d'entrainement PPO")
    parser.add_argument("--retrain", action="store_true", help="Forcer le re-entrainement du modele PPO")
    args = parser.parse_args()

    run_full_pipeline(num_episodes=args.episodes, force_retrain=args.retrain)
