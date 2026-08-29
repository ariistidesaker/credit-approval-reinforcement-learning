import os
import torch
import numpy as np
import pytest

from src.dqn.networks import QNetwork
from src.dqn.buffer import ReplayBuffer
from src.dqn.agent import DQNAgent
from src.credit_mdp_env import CreditApprovalEnv
from src.dqn.train_dqn import train_dqn


def test_dqn_networks_shapes():
    state_dim = 18
    action_dim = 2
    batch_size = 8

    q_net = QNetwork(state_dim=state_dim, action_dim=action_dim)
    dummy_state = torch.randn(batch_size, state_dim)

    q_vals = q_net(dummy_state)
    assert q_vals.shape == (batch_size, action_dim)
    assert not torch.isnan(q_vals).any()


def test_replay_buffer():
    capacity = 100
    state_dim = 18
    device = torch.device("cpu")
    buffer = ReplayBuffer(capacity=capacity, device=device)

    for i in range(50):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randint(0, 2)
        r = float(np.random.randn())
        s_next = np.random.randn(state_dim).astype(np.float32)
        done = (i == 49)
        buffer.push(s, a, r, s_next, done)

    assert len(buffer) == 50
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=16)

    assert states.shape == (16, state_dim)
    assert actions.shape == (16, 1)
    assert rewards.shape == (16, 1)
    assert next_states.shape == (16, state_dim)
    assert dones.shape == (16, 1)


def test_dqn_agent_update_and_save_load(tmp_path):
    state_dim = 18
    action_dim = 2
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)

    dummy_state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(dummy_state, deterministic=True)
    assert action in [0, 1]

    # Test ReplayBuffer & Update
    buffer = ReplayBuffer(capacity=100, device=agent.device)
    for i in range(40):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randint(0, 2)
        r = 1.0
        s_next = np.random.randn(state_dim).astype(np.float32)
        buffer.push(s, a, r, s_next, False)

    metrics = agent.update(buffer, batch_size=16)
    assert "td_loss" in metrics
    assert "max_q" in metrics
    assert not np.isnan(metrics["td_loss"])

    # Test Save & Load
    save_path = os.path.join(tmp_path, "test_dqn.pt")
    agent.save(save_path)
    assert os.path.exists(save_path)

    new_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    new_agent.load(save_path)
    action2 = new_agent.select_action(dummy_state, deterministic=True)
    assert action2 in [0, 1]
