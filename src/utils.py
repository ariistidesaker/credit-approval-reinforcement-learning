import numpy as np
import pandas as pd
from typing import Dict, Any


def calculate_default_probability(row: pd.Series) -> float:
    """
    Estime la probabilite de defaut P(Defaut) basee sur les caracteristiques du client,
    du pret et des conditions macroeconomiques.
    
    Modele logistique standard calibre :
    - Risk_Score eleve (700+) -> P(Defaut) faible (~2-8%)
    - Risk_Score moyen (500-650) -> P(Defaut) moderee (~15-30%)
    - Risk_Score bas (<450) -> P(Defaut) elevee (~50-80%)
    - Modulateurs : Ratio Dette/Revenu, Statut d'emploi, Taux d'inflation.
    """
    # 1. Base score logistique centre autour d'un score pivot (580)
    # Score standardise inversé (plus le score est élevé, plus le risque diminue)
    score_z = (float(row["Risk_Score"]) - 580.0) / 100.0
    
    # 2. Facteur de charge de la dette
    dti = float(row.get("Debt_to_Income_Ratio", 0.3))
    dti_penalty = max(0.0, dti - 0.4) * 1.5
    
    # 3. Statut professionnel
    employment = str(row.get("Employment_Status", "Employed"))
    emp_penalty = 0.4 if employment == "Unemployed" else (0.1 if employment == "Self-Employed" else -0.2)
    
    # 4. Facteur macroeconomique (Stress inflation)
    inflation = float(row.get("Inflation_Rate_Percent", 80.0))
    macro_penalty = (inflation - 80.0) / 100.0 * 0.3
    
    # Score logit combiné
    logit = -1.2 - (1.5 * score_z) + dti_penalty + emp_penalty + macro_penalty
    
    # Sigmoide
    prob = 1.0 / (1.0 + np.exp(-logit))
    return float(np.clip(prob, 0.01, 0.95))


def calculate_financial_outcome(
    row: pd.Series,
    action: int,
    cost_of_funds_annual: float = 0.05,
    operational_cost_reject: float = 10.0,
    collateral_haircut: float = 0.8,
    rng: np.random.Generator = None,
) -> Dict[str, Any]:
    """
    Calcule le résultat financier (gain/perte) et les métriques de risque associées pour une décision d'octroi.
    
    Actions :
    - 0 : Rejet du prêt
    - 1 : Approbation du prêt
    """
    if rng is None:
        rng = np.random.default_rng()

    ead = float(row["Exposure_at_Default"])
    lending_rate = float(row["Lending_Rate_Percent"]) / 100.0
    duration_years = float(row["Loan_Duration_Months"]) / 12.0
    lgd = float(row["LGD"])
    asset_coverage = float(row["Asset_Coverage_Value"])

    p_default = calculate_default_probability(row)

    if action == 0:
        # Rejet du prêt
        return {
            "action": 0,
            "action_name": "REJECT",
            "p_default": p_default,
            "defaulted": False,
            "net_profit": -operational_cost_reject,
            "ead": ead,
            "interest_earned": 0.0,
            "loss_suffered": 0.0,
        }

    # Approbation du prêt (action == 1)
    # Tirage aléatoire de l'occurrence du défaut
    is_default = rng.random() < p_default

    if not is_default:
        # Remboursement intégral avec intérêts
        net_interest_rate = max(0.0, lending_rate - cost_of_funds_annual)
        profit = ead * net_interest_rate * duration_years
        return {
            "action": 1,
            "action_name": "APPROVE",
            "p_default": p_default,
            "defaulted": False,
            "net_profit": profit,
            "ead": ead,
            "interest_earned": profit,
            "loss_suffered": 0.0,
        }
    else:
        # Défaut de paiement : perte basée sur LGD et récupération via collatéral
        gross_loss = ead * lgd
        recoverable = min(gross_loss, asset_coverage * collateral_haircut)
        net_loss = max(0.0, gross_loss - recoverable)
        return {
            "action": 1,
            "action_name": "APPROVE",
            "p_default": p_default,
            "defaulted": True,
            "net_profit": -net_loss,
            "ead": ead,
            "interest_earned": 0.0,
            "loss_suffered": net_loss,
        }
