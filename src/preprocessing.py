import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class CreditDataPreprocessor:
    """
    Gestionnaire de prétraitement des données de crédit.
    Transforme les données brutes en caractéristiques exploitables par l'environnement MDP.
    """

    NUMERICAL_COLS = [
        "Risk_Score",
        "Applicant_Age",
        "Household_Income",
        "Exposure_at_Default",
        "Asset_Coverage_Value",
        "Lending_Rate_Percent",
        "Loan_Duration_Months",
        "GDP_Growth_Percent",
        "Inflation_Rate_Percent",
        "Policy_Rate_Percent",
        "Debt_to_Income_Ratio",
        "Coverage_Ratio",
    ]

    CATEGORICAL_COLS = ["Loan_Category", "Employment_Status"]

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Charge le dataset brut depuis un fichier CSV."""
        path = data_path or self.data_path or "data/synthetic_sadc_lgd_dataset.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        self.raw_df = pd.read_csv(path)
        return self.raw_df

    def clean_and_feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les anomalies et ajoute les ratios financiers clés.
        """
        data = df.copy()

        # 1. Correction des valeurs d'exposition (EAD) négatives ou nulles
        data["Exposure_at_Default"] = data["Exposure_at_Default"].apply(lambda x: max(1000.0, abs(x)))
        data["Asset_Coverage_Value"] = data["Asset_Coverage_Value"].apply(lambda x: max(0.0, x))
        data["Household_Income"] = data["Household_Income"].apply(lambda x: max(500.0, x))
        data["Risk_Score"] = data["Risk_Score"].clip(300.0, 850.0)
        data["LGD"] = data["LGD"].clip(0.0, 1.0)

        # 2. Création de features financières dérivées
        # Estimation de la mensualité du prêt (amortissement simplifié)
        monthly_interest_rate = (data["Lending_Rate_Percent"] / 100.0) / 12.0
        n_months = data["Loan_Duration_Months"].clip(lower=1)
        estimated_monthly_payment = (
            data["Exposure_at_Default"] * (1 + monthly_interest_rate * n_months)
        ) / n_months

        # Ratio Dette/Revenu mensuel
        monthly_income = data["Household_Income"] / 12.0
        data["Debt_to_Income_Ratio"] = (estimated_monthly_payment / monthly_income).clip(0.0, 5.0)

        # Ratio de couverture des garanties (Collatéral / Montant du prêt)
        data["Coverage_Ratio"] = (
            data["Asset_Coverage_Value"] / data["Exposure_at_Default"]
        ).clip(0.0, 5.0)

        return data

    def fit(self, df: Optional[pd.DataFrame] = None) -> "CreditDataPreprocessor":
        """Calibre les transformateurs (scaler et one-hot encoder)."""
        if df is None:
            if self.raw_df is None:
                self.load_data()
            df = self.raw_df

        self.clean_df = self.clean_and_feature_engineer(df)

        # Fit Scaler pour les variables numériques
        self.scaler.fit(self.clean_df[self.NUMERICAL_COLS])

        # Fit One-Hot Encoder pour les variables catégorielles
        self.encoder.fit(self.clean_df[self.CATEGORICAL_COLS])

        # Récupération des noms de colonnes encodées
        encoded_cat_names = list(self.encoder.get_feature_names_out(self.CATEGORICAL_COLS))
        self.feature_names = self.NUMERICAL_COLS + encoded_cat_names
        self.is_fitted = True

        return self

    def transform(self, df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Transforme un DataFrame en matrice d'états normalisée pour le MDP.
        Retourne (state_matrix, processed_df).
        """
        if not self.is_fitted:
            raise RuntimeError("Le preprocessor doit d'abord être ajusté avec .fit()")

        if df is None:
            if self.clean_df is None:
                if self.raw_df is None:
                    self.load_data()
                self.clean_df = self.clean_and_feature_engineer(self.raw_df)
            clean_df = self.clean_df
        else:
            clean_df = self.clean_and_feature_engineer(df)

        num_scaled = self.scaler.transform(clean_df[self.NUMERICAL_COLS])
        cat_encoded = self.encoder.transform(clean_df[self.CATEGORICAL_COLS])

        state_matrix = np.hstack([num_scaled, cat_encoded]).astype(np.float32)
        return state_matrix, clean_df

    def fit_transform(self, df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """Ajuste et transforme les données en une seule étape."""
        self.fit(df)
        return self.transform(df)
