# 🏦 Optimisation de l'Approbation des Prêts Bancaires par Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ce projet développe un agent intelligent d'apprentissage par renforcement (**Reinforcement Learning - RL**) capable d'optimiser les décisions d'approbation ou de refus de prêts bancaires en fonction du profil financier des demandeurs et du contexte macroéconomique.

---

## 📌 Pourquoi le Reinforcement Learning ?

Contrairement à un modèle de scoring supervisé classique qui prédit uniquement si un client risque de faire défaut ($P(\text{Défaut})$), l'agent RL apprend une **stratégie séquentielle optimale** en simulant une institution financière :
- **Maximisation du profit net** : balance les gains d'intérêts rémunérateurs et les pertes nettes par défaut (*LGD - Loss Given Default*).
- **Gestion de portefeuille dynamique** : intègre les contraintes de capital, les garanties collatérales (*Asset Coverage*) et la charge de la dette (*Debt-to-Income*).

---

## 🗂️ Structure du Projet

```text
credit-approval-reinforcement-learning/
├── .gitignore                    # Fichiers ignorés par Git
├── README.md                     # Documentation principale
├── requirements.txt              # Dépendances Python
├── pytest.ini                    # Configuration pytest
├── main.py                       # Démonstration, rollout et benchmark des baselines
├── EXPLICATION_WORKFLOW_MDP.md   # Guide exhaustif du MDP et analyse du code
├── data/
│   ├── download_data.py          # Script de téléchargement automatique Kaggle
│   └── synthetic_sadc_lgd_dataset.csv # Dataset SADC LGD (500 dossiers)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Nettoyage, feature engineering & normalisation
│   ├── utils.py                  # Modélisation du défaut P(Défaut) et calculs financiers
│   └── credit_mdp_env.py         # Environnement MDP (Gymnasium Env)
└── tests/
    └── test_environment.py      # Tests unitaires et conformité Gymnasium
```

---

## ⚙️ Installation & Configuration

### 1. Cloner le Répertoire
```bash
git clone https://github.com/ariistidesaker/credit-approval-reinforcement-learning.git
cd credit-approval-reinforcement-learning
```

### 2. Créer un Environnement Virtuel
```bash
python -m venv .venv
# Sur Windows :
.venv\Scripts\activate
# Sur Linux/Mac :
source .venv/bin/activate
```

### 3. Installer les Dépendances
```bash
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### 1. Télécharger les Données
```bash
python data/download_data.py
```

### 2. Lancer la Démonstration et le Benchmark des Baselines
```bash
python main.py
```

### 3. Lancer la Suite de Tests Unitaires
```bash
python -m pytest tests/test_environment.py
```

---

## 🧠 Modélisation du MDP (*Markov Decision Process*)

L'environnement est implémenté sous la classe [`CreditApprovalEnv`](src/credit_mdp_env.py) conforme aux spécifications **Gymnasium** :

- **Espace d'État ($\mathcal{S}$)** : Vecteur continu normalisé de dimension **18** :
  - *Profil client* : Score de crédit FICO (300-850), Âge, Revenu, Statut d'emploi (Employed, Self-Employed, Unemployed).
  - *Caractéristiques du prêt* : Montant demandé (EAD), Couverture collatérale, Taux d'intérêt, Durée, Catégorie de prêt.
  - *Ratios financiers* : `Debt_to_Income_Ratio`, `Coverage_Ratio`.
  - *Macroéconomie* : Croissance PIB, Inflation, Taux directeur.
- **Espace d'Action ($\mathcal{A}$)** : `Discrete(2)` :
  - `0` : **Rejeter** le prêt
  - `1` : **Approuver** le prêt
- **Fonction de Récompense ($\mathcal{R}$)** :
  - *Rejet ($a=0$)* : Frais de dossier minime ($-10\$$).
  - *Approbation sans défaut ($a=1$)* : Gain net d'intérêts $=\text{EAD} \times (r_{\text{prêt}} - r_{\text{fonds}}) \times \frac{\text{Durée}}{12}$.
  - *Approbation avec défaut ($a=1$)* : Perte nette $= -(\text{EAD} \times \text{LGD} - \text{Garantie})_+$.

---

## 📊 Résultats du Benchmark Comparatif (500 Dossiers)

Évaluation sur un portefeuille de 500 dossiers de demandes de prêt :

| Stratégie | Approbations | Défauts | Volume Prêté ($) | Profit Net ($) | ROI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Tout Approuver (Naïf)** | 500/500 (100.0%) | 375 (75.0%) | 24,888,053 $ | 4,294,743.06 $ | 17.26% |
| **2. Aléatoire (50/50)** | 244/500 (48.8%) | 181 (74.2%) | 12,052,187 $ | 1,955,933.20 $ | 16.23% |
| **3. Heuristique Experte** | 3/500 (0.6%) | 0 (0.0%) | 19,878 $ | 19,570.85 $ | 98.45% |
| **4. Agent PPO (Reinforcement Learning)** | **386/500 (77.2%)** | **277 (71.8%)** | **19,174,444 $** | **4,357,467.39 $** | **22.73%** |

> **Analyse des Performances de l'Agent PPO :**
> - **Surpasse la politique naïve** en profit net ($+62\,724\$$ de bénéfice supplémentaire) tout en engageant **$5.7$ millions de dollars de capital en moins** ($19.17\text{M}\$$ contre $24.88\text{M}\$$).
> - **Augmentation du ROI** à **$22.73\%$** (+5.47 points de pourcentage par rapport au tout approuver).
> - L'agent a appris à rejeter de manière ciblée les dossiers à profil asymétrique défavorable (faibles taux / garanties insuffisantes / risque de défaut élevé).

---

## 👥 Auteur & Projet
- **Projet de Reinforcement Learning - Groupe 5**
- Dépôt GitHub : [credit-approval-reinforcement-learning](https://github.com/ariistidesaker/credit-approval-reinforcement-learning.git)
