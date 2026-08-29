import os
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import CreditDataPreprocessor
from src.credit_mdp_env import CreditApprovalEnv
from src.ppo.agent import PPOAgent
from src.dqn.agent import DQNAgent
from src.utils import calculate_default_probability, calculate_financial_outcome


# ==============================================================================
# Configuration de la page Streamlit
# ==============================================================================
st.set_page_config(
    page_title="CreditRL - Approbation de Prêts par RL",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS pour une interface soignée
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .decision-approved {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.25);
    }
    .decision-rejected {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        box-shadow: 0 4px 6px rgba(239, 68, 68, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Chargement des Modèles et Données (avec Cache)
# ==============================================================================
@st.cache_resource
def load_preprocessor(data_path: str = "data/synthetic_sadc_lgd_dataset.csv"):
    """Charge uniquement le préprocesseur et les données."""
    preprocessor = CreditDataPreprocessor(data_path)
    state_matrix, clean_df = preprocessor.fit_transform()
    return preprocessor, clean_df, state_matrix.shape[1]

@st.cache_resource
def load_ppo_agent(model_path: str = "models/best_ppo_agent.pt", state_dim: int = 18):
    """Charge l'agent PPO."""
    agent = PPOAgent(state_dim=state_dim, action_dim=2, lr=3e-4)
    loaded = os.path.exists(model_path)
    if loaded:
        agent.load(model_path)
    return agent, loaded

@st.cache_resource
def load_dqn_agent(model_path: str = "models/best_dqn_agent.pt", state_dim: int = 18):
    """Charge l'agent DQN."""
    agent = DQNAgent(state_dim=state_dim, action_dim=2)
    loaded = os.path.exists(model_path)
    if loaded:
        agent.load(model_path)
    return agent, loaded


# Chargement global
preprocessor, dataset_df, state_dim = load_preprocessor()
ppo_agent, ppo_loaded = load_ppo_agent(state_dim=state_dim)
dqn_agent, dqn_loaded = load_dqn_agent(state_dim=state_dim)


# ==============================================================================
# En-tête Principal
# ==============================================================================
st.markdown('<div class="main-title">🏦 CreditRL : Optimisation de l\'Approbation des Prêts</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Système Décisionnel Autonome basé sur le <b>Reinforcement Learning (PPO ou DQN)</b> et modélisé par <b>Processus de Décision Markovien (MDP)</b>.</div>',
    unsafe_allow_html=True,
)

# Sidebar : Informations système
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
    st.title("Paramètres & Statut")

    st.markdown("### 🤖 Sélection du Modèle")
    model_choice = st.selectbox(
        "Choisissez l'algorithme",
        ["PPO (Actor-Critic)", "DQN (Deep Q-Network)"],
        index=0
    )

    if model_choice == "PPO (Actor-Critic)":
        if ppo_loaded:
            st.success(" Modèle PPO chargé (`models/best_ppo_agent.pt`)")
        else:
            st.warning("⚠️ Poids PPO non trouvés. Agent initialisé par défaut.")
    else:
        if dqn_loaded:
            st.success(" Modèle DQN chargé (`models/best_dqn_agent.pt`)")
        else:
            st.warning("⚠️ Poids DQN non trouvés. Agent initialisé par défaut.")

    st.markdown("---")
    st.markdown("### 📊 Environnement MDP")
    st.write(f"• **Dimensions d'état** : {preprocessor.state_matrix.shape[1] if hasattr(preprocessor, 'state_matrix') else 18}")
    st.write("• **Actions** : {0: Rejeter, 1: Approuver}")
    st.write(f"• **Algorithme actif** : {model_choice}")
    st.write(f"• **Taille Portefeuille** : {len(dataset_df)} dossiers")
    st.markdown("---")
    st.caption("Groupe 5 - Projet Reinforcement Learning")


# ==============================================================================
# Onglets de Navigation
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "🎯 Simulateur Client Individuel",
    "📈 Benchmark & Portefeuille (500 Dossiers)",
    "🔍 Théorie MDP & Monitoring TensorBoard",
])


# ==============================================================================
# ONGLET 1 : Simulateur de Décision Individuelle
# ==============================================================================
with tab1:
    st.subheader("📋 Évaluation en Temps Réel d'un Demandeur de Crédit")
    st.write("Ajustez les curseurs ci-dessous pour simuler une nouvelle demande de prêt et observer la décision de l'agent sélectionné.")

    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        st.markdown("##### 👤 Profil de l'Emprunteur")
        risk_score = st.slider("Score de Crédit (FICO)", min_value=300, max_value=850, value=650, step=5)
        applicant_age = st.slider("Âge du Demandeur", min_value=18, max_value=75, value=38, step=1)
        household_income = st.number_input("Revenu Annuel du Foyer ($)", min_value=1000.0, max_value=100000.0, value=25000.0, step=1000.0)
        employment_status = st.selectbox("Statut Professionnel", ["Employed", "Self-Employed", "Unemployed"], index=0)

    with col_input2:
        st.markdown("##### 💳 Caractéristiques du Prêt")
        exposure_at_default = st.number_input("Montant Demandé EAD ($)", min_value=1000.0, max_value=150000.0, value=35000.0, step=1000.0)
        asset_coverage = st.number_input("Valeur de Garantie / Collatéral ($)", min_value=0.0, max_value=150000.0, value=28000.0, step=1000.0)
        lending_rate = st.slider("Taux d'Intérêt Annuel (%)", min_value=5.0, max_value=65.0, value=38.0, step=0.5)
        loan_duration = st.slider("Durée du Prêt (Mois)", min_value=6, max_value=60, value=36, step=3)
        loan_category = st.selectbox("Catégorie du Prêt", ["Personal", "Business", "Agricultural"], index=0)

    with col_input3:
        st.markdown("##### 🌍 Contexte Macroéconomique")
        gdp_growth = st.slider("Croissance du PIB (%)", min_value=-5.0, max_value=10.0, value=1.5, step=0.1)
        inflation_rate = st.slider("Taux d'Inflation (%)", min_value=5.0, max_value=150.0, value=85.0, step=1.0)
        policy_rate = st.slider("Taux Directeur Banque Centrale (%)", min_value=1.0, max_value=60.0, value=35.0, step=0.5)
        lgd = st.slider("Sévérité de Perte Attendue (LGD)", min_value=0.0, max_value=1.0, value=0.30, step=0.01)

    # Préparation du dataframe de l'échantillon
    sample_dict = {
        "Risk_Score": risk_score,
        "Applicant_Age": applicant_age,
        "Household_Income": household_income,
        "Exposure_at_Default": exposure_at_default,
        "Asset_Coverage_Value": asset_coverage,
        "Lending_Rate_Percent": lending_rate,
        "Loan_Duration_Months": loan_duration,
        "Loan_Category": loan_category,
        "Employment_Status": employment_status,
        "GDP_Growth_Percent": gdp_growth,
        "Inflation_Rate_Percent": inflation_rate,
        "Policy_Rate_Percent": policy_rate,
        "LGD": lgd,
    }
    sample_df = pd.DataFrame([sample_dict])

    # Transformation en vecteur d'état
    sample_state, clean_sample = preprocessor.transform(sample_df)

    # Inférence selon le modèle choisi
    with torch.no_grad():
        state_tensor = torch.as_tensor(sample_state, dtype=torch.float32)

        if model_choice == "PPO (Actor-Critic)":
            agent = ppo_agent
            state_tensor = state_tensor.to(agent.device)
            dist, value_est = agent.actor_critic(state_tensor)
            action_probs = dist.probs.cpu().numpy()[0]
            decision = int(np.argmax(action_probs))

            # Graphique des probabilités
            fig_data = go.Bar(
                x=["0: Rejeter", "1: Approuver"],
                y=[action_probs[0] * 100, action_probs[1] * 100],
                marker_color=["#EF4444", "#10B981"],
                text=[f"{action_probs[0]:.1%}", f"{action_probs[1]:.1%}"],
                textposition="auto",
            )
            fig_title = "Distribution de Probabilité de la Politique π(a|s)"
            yaxis_title = "Probabilité (%)"
            yaxis_range = [0, 100]
            value_display = f"Valeur d'état estimée : **$V(s) = {value_est.item():.2f} $**"

        else:  # DQN
            agent = dqn_agent
            state_tensor = state_tensor.to(agent.device)
            q_values = agent.q_network(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]
            decision = int(np.argmax(q_values_np))

            # Graphique des valeurs Q
            fig_data = go.Bar(
                x=["0: Rejeter", "1: Approuver"],
                y=[q_values_np[0], q_values_np[1]],
                marker_color=["#EF4444", "#10B981"],
                text=[f"{q_values_np[0]:.2f}", f"{q_values_np[1]:.2f}"],
                textposition="auto",
            )
            fig_title = "Valeurs Q(s, a) estimées par le réseau DQN"
            yaxis_title = "Valeur Q"
            yaxis_range = None  # pas de range fixe
            value_display = f"Meilleure valeur Q : **max_a Q(s,a) = {q_values_np[decision]:.2f} $**"

    # Calculs financiers et de risque
    p_default = calculate_default_probability(clean_sample.iloc[0])
    dti = clean_sample.iloc[0]["Debt_to_Income_Ratio"]
    coverage = clean_sample.iloc[0]["Coverage_Ratio"]

    # Gains et pertes projetés
    duration_years = loan_duration / 12.0
    net_interest_rate = max(0.0, (lending_rate - 5.0) / 100.0)
    potential_gain = exposure_at_default * net_interest_rate * duration_years
    gross_loss = exposure_at_default * lgd
    max_loss = max(0.0, gross_loss - min(gross_loss, asset_coverage * 0.8))

    st.markdown("---")
    st.markdown(f"### 🏆 Verdict de l'Agent {model_choice} & Analyse Financière")

    col_res1, col_res2 = st.columns([1, 1.2])

    with col_res1:
        if decision == 1:
            st.markdown('<div class="decision-approved">✅ PRÊT APPROUVÉ (ACCORD)</div>', unsafe_allow_html=True)
            st.caption("L'agent estime que le retour d'intérêts compense largement le risque de défaut.")
        else:
            st.markdown('<div class="decision-rejected">❌ PRÊT REJETÉ (REFUS)</div>', unsafe_allow_html=True)
            st.caption("L'agent estime que le risque ou l'insuffisance de garantie est défavorable au capital bancaire.")

        st.write("")
        # Graphique (probabilités ou Q-valeurs)
        fig_prob = go.Figure(fig_data)
        fig_prob.update_layout(
            title=fig_title,
            yaxis_title=yaxis_title,
            yaxis_range=yaxis_range,
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_res2:
        # Cartes des métriques clés
        m1, m2, m3 = st.columns(3)
        m1.metric("P(Défaut) Estimée", f"{p_default:.1%}", delta=f"{'Risque Élevé' if p_default > 0.35 else 'Risque Maîtrisé'}", delta_color="inverse")
        m2.metric("Ratio Dette/Revenu", f"{dti:.2f}", delta=f"{'DTI Critique' if dti > 0.45 else 'DTI Conforme'}", delta_color="inverse")
        m3.metric("Couverture Garantie", f"{coverage:.1%}", delta=f"{'Couvert' if coverage >= 0.8 else 'Faible Garantie'}")

        m4, m5 = st.columns(2)
        m4.metric("Gain d'Intérêts Espéré", f"+{potential_gain:,.2f} $", help="Marge nette d'intérêts perçue en cas de remboursement sans incident.")
        m5.metric("Perte Nette si Défaut", f"-{max_loss:,.2f} $", help="Perte nette après liquidation du collatéral avec décote de 20%.")

        st.info(
            f"💡 **Recommandation du Système** : Pour ce dossier ({loan_category}, FICO {risk_score}), {value_display}"
        )


# ==============================================================================
# ONGLET 2 : Benchmark Portefeuille & Comparaison (500 Dossiers)
# ==============================================================================
with tab2:
    st.subheader("📊 Évaluation Globale sur le Portefeuille de 500 Dossiers")
    st.write("Comparez les stratégies apprises (PPO et DQN) face aux politiques conventionnelles sur l'ensemble des données historiques.")

    if st.button("🚀 Exécuter la Simulation Complète du Portefeuille", type="primary"):
        with st.spinner("Simulation des 500 dossiers en cours..."):
            env = CreditApprovalEnv(data_path="data/synthetic_sadc_lgd_dataset.csv", shuffle_on_reset=False, seed=42)

            def run_policy(policy_name: str, policy_fn):
                obs, info = env.reset(seed=42)
                terminated = False
                while not terminated:
                    current_idx = env.sample_indices[env.current_step]
                    raw_row = env.processed_df.iloc[current_idx]
                    action = policy_fn(obs, raw_row)
                    obs, reward, terminated, truncated, info = env.step(action)
                return {
                    "Stratégie": policy_name,
                    "Approbations": info["approved_count"],
                    "Taux_Approbation": info["approval_rate"] * 100,
                    "Défauts": info["default_count"],
                    "Taux_Défaut": info["default_rate"] * 100,
                    "Volume_Prêté": info["total_volume_lent"],
                    "Profit_Net": info["total_profit"],
                    "ROI": info["roi"] * 100,
                }

            rng = np.random.default_rng(42)
            bench_results = [
                run_policy("1. Tout Approuver (Naïf)", lambda obs, r: 1),
                run_policy("2. Aléatoire (50/50)", lambda obs, r: int(rng.integers(0, 2))),
                run_policy("3. Heuristique Experte", lambda obs, r: 1 if (r["Risk_Score"] >= 580 and r.get("Debt_to_Income_Ratio", 0.5) < 0.45) else 0),
                run_policy("4. Agent PPO (RL)", lambda obs, r: ppo_agent.select_action(obs, deterministic=True)[0] if ppo_loaded else 0),
            ]
            # Ajouter DQN si le modèle existe
            if dqn_loaded:
                bench_results.append(
                    run_policy("5. Agent DQN (RL)", lambda obs, r: dqn_agent.select_action(obs, deterministic=True))
                )
            st.session_state["bench_df"] = pd.DataFrame(bench_results)

    if "bench_df" in st.session_state:
        df_res = st.session_state["bench_df"]

        # Indicateurs clés : on prend la meilleure stratégie en profit (généralement PPO ou DQN)
        best_row = df_res.loc[df_res["Profit_Net"].idxmax()]
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Meilleur Profit Net", f"{best_row['Profit_Net']:,.2f} $", delta=f"{best_row['Stratégie']}")
        kpi2.metric("ROI associé", f"{best_row['ROI']:.2f} %")
        kpi3.metric("Taux d'Approbation", f"{best_row['Taux_Approbation']:.1f} %")
        kpi4.metric("Taux de Défaut", f"{best_row['Taux_Défaut']:.1f} %")

        st.markdown("---")

        # Graphiques comparatifs Plotly
        g1, g2 = st.columns(2)

        with g1:
            fig_profit = px.bar(
                df_res,
                x="Stratégie",
                y="Profit_Net",
                color="Stratégie",
                title="Profit Net Cumulé par Stratégie ($)",
                text_auto=".2s",
            )
            fig_profit.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig_profit, use_container_width=True)

        with g2:
            fig_roi = px.bar(
                df_res,
                x="Stratégie",
                y="ROI",
                color="Stratégie",
                title="Rendement sur Capital Prêté - ROI (%)",
                text_auto=".1f",
            )
            fig_roi.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig_roi, use_container_width=True)

        # Tableau complet formaté
        st.markdown("##### 📋 Tableau Détaillé du Portefeuille")
        display_table = df_res.copy()
        display_table["Volume_Prêté"] = display_table["Volume_Prêté"].apply(lambda x: f"{x:,.0f} $")
        display_table["Profit_Net"] = display_table["Profit_Net"].apply(lambda x: f"{x:,.2f} $")
        display_table["Taux_Approbation"] = display_table["Taux_Approbation"].apply(lambda x: f"{x:.1f} %")
        display_table["Taux_Défaut"] = display_table["Taux_Défaut"].apply(lambda x: f"{x:.1f} %")
        display_table["ROI"] = display_table["ROI"].apply(lambda x: f"{x:.2f} %")
        st.dataframe(display_table, use_container_width=True)


# ==============================================================================
# ONGLET 3 : Théorie MDP & Monitoring TensorBoard
# ==============================================================================
with tab3:
    st.subheader("🔍 Architecture Mathématique du MDP & Monitoring")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### 📐 Formalisation du MDP")
        st.markdown(
            r"""
            - **Espace d'État $\mathcal{S}$** : Vecteur continu à 18 dimensions normalisées ($\mu=0, \sigma=1$).
            - **Espace d'Action $\mathcal{A}$** : Discret $\{0: \text{Rejeter}, 1: \text{Approuver}\}$.
            - **Fonction de Récompense $\mathcal{R}(s, a)$** :
              $$r_t = 10^{-4} \times \begin{cases} -10\$ & \text{si } a=0 \\ \text{EAD} \cdot (r_{\text{prêt}} - r_{\text{fonds}}) \cdot \frac{\text{Durée}}{12} & \text{si } a=1 \text{ (sans défaut)} \\ -(\text{EAD}\cdot\text{LGD} - \text{Garantie})_+ & \text{si } a=1 \text{ (avec défaut)} \end{cases}$$
            - **Objectif PPO Clippé** :
              $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
            - **Perte TD de DQN** :
              $$\mathcal{L} = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\text{target}}(s', a') - Q(s, a) \right)^2 \right]$$
            """
        )

    with col_t2:
        st.markdown("#### 🖥️ Accès au Dashboard TensorBoard")
        st.info("Pour inspecter les courbes de perte, récompenses, profits et taux :")
        st.code("tensorboard --logdir runs", language="bash")
        st.write("Puis ouvrez [http://localhost:6006](http://localhost:6006) dans votre navigateur.")

    st.markdown("---")
    st.markdown("#### 📈 Courbes d'Entraînement Générées (`reports/learning_curves.png`)")
    if os.path.exists("reports/learning_curves.png"):
        st.image("reports/learning_curves.png", use_container_width=True, caption="Courbes d'apprentissage : Pertes, Récompenses, Profits et Gestion du Risque.")
    else:
        st.warning("Graphique `reports/learning_curves.png` non trouvé. Lancez un entraînement pour le générer.")