import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env

from src.preprocessing import CreditDataPreprocessor
from src.credit_mdp_env import CreditApprovalEnv
from src.utils import calculate_default_probability


def test_preprocessor():
    preprocessor = CreditDataPreprocessor("data/synthetic_sadc_lgd_dataset.csv")
    state_matrix, df = preprocessor.fit_transform()

    assert state_matrix is not None
    assert len(df) == 500
    assert state_matrix.shape[0] == 500
    assert state_matrix.shape[1] == len(preprocessor.feature_names)
    assert not np.isnan(state_matrix).any()


def test_gymnasium_env_compliance():
    env = CreditApprovalEnv(data_path="data/synthetic_sadc_lgd_dataset.csv", max_steps=50)
    # Vérification standard des spécifications Gymnasium
    check_env(env.unwrapped)
    env.close()


def test_env_step_and_reset():
    env = CreditApprovalEnv(data_path="data/synthetic_sadc_lgd_dataset.csv", max_steps=20, seed=123)
    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert info["step"] == 0
    assert info["approved_count"] == 0
    assert info["rejected_count"] == 0

    # Test action 1 (Approve)
    next_obs, reward, terminated, truncated, step_info = env.step(1)
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert step_info["step"] == 1
    assert step_info["approved_count"] == 1
    assert not terminated

    # Test action 0 (Reject)
    next_obs, reward, terminated, truncated, step_info = env.step(0)
    assert step_info["step"] == 2
    assert step_info["rejected_count"] == 1

    # Rollout jusqu'à la fin
    for _ in range(18):
        next_obs, reward, terminated, truncated, step_info = env.step(1)

    assert terminated is True
    assert step_info["step"] == 20
    env.close()


def test_default_probability_bounds():
    env = CreditApprovalEnv(data_path="data/synthetic_sadc_lgd_dataset.csv")
    for _, row in env.processed_df.iterrows():
        p = calculate_default_probability(row)
        assert 0.0 <= p <= 1.0
