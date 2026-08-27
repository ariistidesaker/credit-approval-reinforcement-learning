import os
import pytest
import pandas as pd
import numpy as np
import torch
from app import load_system


def test_load_system():
    preprocessor, agent, df, is_loaded = load_system()
    assert preprocessor is not None
    assert agent is not None
    assert len(df) == 500
    assert is_loaded is True


def test_app_inference_pipeline():
    preprocessor, agent, _, _ = load_system()

    sample_dict = {
        "Risk_Score": 720,
        "Applicant_Age": 35,
        "Household_Income": 30000.0,
        "Exposure_at_Default": 25000.0,
        "Asset_Coverage_Value": 20000.0,
        "Lending_Rate_Percent": 35.0,
        "Loan_Duration_Months": 24,
        "Loan_Category": "Personal",
        "Employment_Status": "Employed",
        "GDP_Growth_Percent": 1.5,
        "Inflation_Rate_Percent": 80.0,
        "Policy_Rate_Percent": 35.0,
        "LGD": 0.25,
    }
    sample_df = pd.DataFrame([sample_dict])
    sample_state, clean_sample = preprocessor.transform(sample_df)

    assert sample_state.shape == (1, 18)
    assert not np.isnan(sample_state).any()

    with torch.no_grad():
        state_tensor = torch.as_tensor(sample_state, dtype=torch.float32, device=agent.device)
        dist, value = agent.actor_critic(state_tensor)
        probs = dist.probs.cpu().numpy()[0]
        decision = int(np.argmax(probs))

    assert decision in [0, 1]
    assert np.isclose(np.sum(probs), 1.0)
