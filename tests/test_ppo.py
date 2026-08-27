import os
import torch
import numpy as np
import pytest

from src.ppo.networks import ActorNetwork, CriticNetwork, ActorCritic
from src.ppo.buffer import RolloutBuffer
from src.ppo.agent import PPOAgent
from src.credit_mdp_env import CreditApprovalEnv
from src.train_ppo import train_ppo, evaluate_agent


def test_networks_shapes():
    state_dim = 18
    action_dim = 2
    batch_size = 8

    actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim)
    critic = CriticNetwork(state_dim=state_dim)
    ac = ActorCritic(state_dim=state_dim, action_dim=action_dim)

    dummy_state = torch.randn(batch_size, state_dim)

    logits = actor(dummy_state)
    assert logits.shape == (batch_size, action_dim)

    values = critic(dummy_state)
    assert values.shape == (batch_size, 1)

    dist, val = ac(dummy_state)
    assert val.shape == (batch_size, 1)

    action, log_prob, val_sq, entropy = ac.get_action_and_value(dummy_state)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert val_sq.shape == (batch_size,)
    assert entropy.shape == (batch_size,)


def test_rollout_buffer_and_gae():
    buffer_size = 50
    state_dim = 18
    buffer = RolloutBuffer(buffer_size=buffer_size, state_dim=state_dim)

    for i in range(buffer_size):
        dummy_state = np.random.randn(state_dim).astype(np.float32)
        buffer.add(
            state=dummy_state,
            action=np.random.randint(0, 2),
            log_prob=-0.69,
            reward=1.0,
            done=(i == buffer_size - 1),
            value=0.5,
        )

    assert buffer.full is True

    # Calcul GAE
    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True, gamma=0.99, gae_lambda=0.95)

    assert not np.isnan(buffer.advantages).any()
    assert not np.isnan(buffer.returns).any()

    # Vérification des batches
    batch_count = 0
    total_samples = 0
    for b_states, b_actions, b_log_probs, b_advantages, b_returns, b_values in buffer.get_batches(batch_size=16):
        batch_count += 1
        total_samples += len(b_states)
        assert b_states.shape[1] == state_dim
        assert not torch.isnan(b_advantages).any()

    assert total_samples == buffer_size
    buffer.clear()
    assert buffer.ptr == 0
    assert buffer.full is False


def test_ppo_agent_update_and_save_load(tmp_path):
    state_dim = 18
    action_dim = 2
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, k_epochs=2, batch_size=16)

    dummy_state = np.random.randn(state_dim).astype(np.float32)
    action, log_prob, value = agent.select_action(dummy_state)

    assert action in [0, 1]
    assert isinstance(log_prob, float)
    assert isinstance(value, float)

    # Remplir un buffer et tester l'update
    buffer = RolloutBuffer(buffer_size=32, state_dim=state_dim, device=agent.device)
    for i in range(32):
        buffer.add(dummy_state, action=1, log_prob=-0.5, reward=0.2, done=(i == 31), value=0.1)

    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True)
    update_metrics = agent.update(buffer)

    assert "policy_loss" in update_metrics
    assert "value_loss" in update_metrics
    assert not np.isnan(update_metrics["policy_loss"])

    # Test Save & Load
    save_path = os.path.join(tmp_path, "test_ppo.pt")
    agent.save(save_path)
    assert os.path.exists(save_path)

    new_agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    new_agent.load(save_path)
    action2, _, _ = new_agent.select_action(dummy_state, deterministic=True)
    assert action2 in [0, 1]


def test_train_ppo_short_run(tmp_path):
    env = CreditApprovalEnv(data_path="data/synthetic_sadc_lgd_dataset.csv", max_steps=20)
    agent = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

    trained_agent, history_df = train_ppo(
        env=env,
        agent=agent,
        num_episodes=2,
        buffer_size=20,
        eval_interval=1,
        save_dir=str(tmp_path),
        save_name="test_agent.pt",
    )

    assert len(history_df) == 2
    assert "total_profit" in history_df.columns
    eval_res = evaluate_agent(env, trained_agent, num_episodes=1)
    assert "eval_mean_profit" in eval_res
