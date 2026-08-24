"""
Script principal de demonstration des Etapes 1 & 2 :
- Etape 1 : Recuperation et pretraitement des donnees
- Etape 2 : Instanciation du MDP (CreditApprovalEnv) et evaluation de strategies de reference (Baselines)
"""

import numpy as np
import pandas as pd
from data.download_data import download_and_save_data
from src.preprocessing import CreditDataPreprocessor
from src.credit_mdp_env import CreditApprovalEnv


def run_baseline_evaluation():
    print("=" * 70)
    print(" PROJET RL : OPTIMISATION DE L'APPROBATION DES PRETS BANCAIRES")
    print("=" * 70)

    # -------------------------------------------------------------
    # ETAPE 1 : Verification / Recuperation des donnees
    # -------------------------------------------------------------
    print("\n--- ETAPE 1 : Chargement et Pretraitement des Donnees ---")
    data_path = download_and_save_data("data")
    
    preprocessor = CreditDataPreprocessor(data_path)
    state_matrix, clean_df = preprocessor.fit_transform()
    
    print(f"[OK] Donnees chargees : {len(clean_df)} dossiers clients.")
    print(f"[OK] Dimensions de la matrice d'etats MDP : {state_matrix.shape}")
    print(f"[OK] Variables d'etat ({len(preprocessor.feature_names)}) :")
    for i, name in enumerate(preprocessor.feature_names, 1):
        print(f"     {i:02d}. {name}")

    # -------------------------------------------------------------
    # ETAPE 2 : Construction et Test de l'Environnement MDP
    # -------------------------------------------------------------
    print("\n--- ETAPE 2 : Validation du MDP (Gymnasium Environment) ---")
    env = CreditApprovalEnv(data_path=data_path, shuffle_on_reset=False, seed=42)
    print(f"[OK] Environnement MDP cree : {env.__class__.__name__}")
    print(f"     - Espace d'etats (S)  : {env.observation_space}")
    print(f"     - Espace d'actions (A): {env.action_space} (0 = Rejeter, 1 = Approuver)")

    # -------------------------------------------------------------
    # Evaluation de Strategies de Reference (Baselines)
    # -------------------------------------------------------------
    print("\n--- Evaluation des Politiques de Reference (Baselines) ---")

    def evaluate_policy(policy_name: str, policy_fn):
        obs, info = env.reset(seed=42)
        terminated = False
        step_count = 0

        while not terminated:
            # Recuperation des donnees brutes du demandeur courant
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

    # Definition des 3 politiques de reference
    results = []

    # 1. Toujours Approuver
    results.append(evaluate_policy(
        "1. Tout Approuver (Naif)",
        lambda obs, row: 1
    ))

    # 2. Politique Aleatoire (50/50)
    rng = np.random.default_rng(42)
    results.append(evaluate_policy(
        "2. Aleatoire (50/50)",
        lambda obs, row: int(rng.integers(0, 2))
    ))

    # 3. Regle Heuristique Experte (Score Credit >= 580 & Ratio DTI < 0.45)
    results.append(evaluate_policy(
        "3. Heuristique Experte (Score>=580, DTI<0.45)",
        lambda obs, row: 1 if (row["Risk_Score"] >= 580 and row.get("Debt_to_Income_Ratio", 0.5) < 0.45) else 0
    ))

    # Affichage des resultats
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    print("\n" + "=" * 70)
    print(" L'environnement MDP est pret pour l'entrainement DQN / PPO !")
    print("=" * 70)


if __name__ == "__main__":
    run_baseline_evaluation()
