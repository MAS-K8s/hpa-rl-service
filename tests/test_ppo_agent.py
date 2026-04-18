"""
Unit tests for agents/ppo_agent.py

Run with:
    pytest tests/test_ppo_agent.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.ppo_agent import PPOAgent, RunningNormaliser, RolloutBuffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return PPOAgent(
        deployment_name="test-deployment",
        namespace="default",
        state_size=18,
        action_size=3
    )


def make_metrics(cpu=0.4, memory=2.0, request_rate=50.0,
                 latency_p95=0.3, replicas=2, error_rate=0.0):
    return {
        'cpu_usage':    cpu,
        'memory_usage': memory,
        'request_rate': request_rate,
        'latency_p50':  latency_p95 * 0.7,
        'latency_p95':  latency_p95,
        'latency_p99':  latency_p95 * 1.3,
        'replicas':     replicas,
        'error_rate':   error_rate,
        'pod_pending':  0,
        'pod_ready':    replicas,
        'cpu_trend_1m': 0.0,
        'cpu_trend_5m': 0.0,
        'request_trend':0.0,
        'hour':         12,
        'day_of_week':  2,
        'is_weekend':   False,
        'is_peak_hour': False,
    }


# ---------------------------------------------------------------------------
# TC-UT-21: get_state returns correctly shaped array
# ---------------------------------------------------------------------------

def test_get_state_shape(agent):
    m = make_metrics()
    state = agent.get_state(m)
    assert state.shape == (18,), f"expected (18,), got {state.shape}"
    assert state.dtype == np.float32


# ---------------------------------------------------------------------------
# TC-UT-22: get_state normalises replicas by dividing by 10
# ---------------------------------------------------------------------------

def test_get_state_normalises_replicas(agent):
    m = make_metrics(replicas=5)
    state = agent.get_state(m)
    # Index 6 = replicas / 10
    assert abs(state[6] - 0.5) < 1e-5, f"expected 0.5, got {state[6]}"


# ---------------------------------------------------------------------------
# TC-UT-23: rule prior returns scale_up for single stressed pod
# ---------------------------------------------------------------------------

def test_rule_prior_scale_up_single_pod(agent):
    # 1 pod, latency = 0.8s (above SLA 0.5s)
    m = make_metrics(replicas=1, latency_p95=0.8)
    state = agent.get_state(m)
    action = agent._rule_prior(state)
    assert action == 2, f"expected scale_up (2), got {action}"


# ---------------------------------------------------------------------------
# TC-UT-24: rule prior returns scale_down when clearly idle
# ---------------------------------------------------------------------------

def test_rule_prior_scale_down_idle(agent):
    # 3 pods, latency = 0.1s (well below SLA 0.5s * 0.5 = 0.25s)
    m = make_metrics(replicas=3, latency_p95=0.1)
    state = agent.get_state(m)
    action = agent._rule_prior(state)
    assert action == 0, f"expected scale_down (0), got {action}"


# ---------------------------------------------------------------------------
# TC-UT-25: rule prior returns None for ambiguous state
# ---------------------------------------------------------------------------

def test_rule_prior_ambiguous_returns_none(agent):
    # 2 pods, latency = 0.4s (below SLA but not far below)
    m = make_metrics(replicas=2, latency_p95=0.4)
    state = agent.get_state(m)
    action = agent._rule_prior(state)
    assert action is None, f"expected None for ambiguous state, got {action}"


# ---------------------------------------------------------------------------
# TC-UT-26: select_action returns a valid action in {0, 1, 2}
# ---------------------------------------------------------------------------

def test_select_action_valid_range(agent):
    m = make_metrics()
    state = agent.get_state(m)
    action, log_prob, value = agent.select_action(state)
    assert action in {0, 1, 2}, f"action {action} not in valid range"
    assert isinstance(value, float)


# ---------------------------------------------------------------------------
# TC-UT-27: calculate_confidence returns value in [0, 1]
# ---------------------------------------------------------------------------

def test_confidence_in_range(agent):
    m = make_metrics()
    state = agent.get_state(m)
    confidence, probs = agent.calculate_confidence(state)
    assert 0.0 <= confidence <= 1.0, f"confidence {confidence} out of [0,1]"
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-4, "probabilities must sum to 1"


# ---------------------------------------------------------------------------
# TC-UT-28: RunningNormaliser clips output to [-3, +3]
# ---------------------------------------------------------------------------

def test_running_normaliser_clips():
    n = RunningNormaliser()
    for _ in range(20):
        n.update(1.0)
    clipped = n.normalise(1000.0)
    assert clipped <= 3.0, f"expected clip at 3.0, got {clipped}"
    clipped_neg = n.normalise(-1000.0)
    assert clipped_neg >= -3.0, f"expected clip at -3.0, got {clipped_neg}"


# ---------------------------------------------------------------------------
# TC-UT-29: reward is zero when all metrics are zero
# ---------------------------------------------------------------------------

def test_reward_zero_when_metrics_zero(agent):
    zero_metrics = make_metrics(cpu=0.0, memory=0.0, latency_p95=0.0)
    reward = agent._raw_reward(zero_metrics, action=1, prev_metrics=None)
    assert reward == 0.0, f"expected 0.0 reward for zero metrics, got {reward}"


# ---------------------------------------------------------------------------
# TC-UT-30: scale_up under stress with few replicas receives positive reward
# ---------------------------------------------------------------------------

def test_reward_positive_scaleup_under_stress(agent):
    stressed = make_metrics(replicas=1, latency_p95=0.8)
    prev     = make_metrics(replicas=1, latency_p95=0.8)
    reward   = agent._raw_reward(stressed, action=2, prev_metrics=prev)
    assert reward > 0, f"expected positive reward for scale_up under stress, got {reward}"


# ---------------------------------------------------------------------------
# TC-UT-31: scale_down at minimum replicas receives large negative reward
# ---------------------------------------------------------------------------

def test_reward_negative_scaledown_at_min(agent):
    m    = make_metrics(replicas=1, latency_p95=0.3)
    prev = make_metrics(replicas=1, latency_p95=0.3)
    reward = agent._raw_reward(m, action=0, prev_metrics=prev)
    assert reward < -20, f"expected large negative reward for scale_down at min, got {reward}"


# ---------------------------------------------------------------------------
# TC-UT-32: store_transition adds to buffer
# ---------------------------------------------------------------------------

def test_store_transition_adds_to_buffer(agent):
    m = make_metrics()
    state = agent.get_state(m)
    initial_size = len(agent.buffer)
    agent.store_transition(state, action=1, reward=0.5,
                           next_state=state, done=False,
                           log_prob=0.0, value=0.0)
    assert len(agent.buffer) == initial_size + 1


# ---------------------------------------------------------------------------
# TC-UT-33: pretrain returns positive number of training calls
# ---------------------------------------------------------------------------

def test_pretrain_runs_successfully(agent):
    calls = agent.pretrain(n_steps=64)
    assert calls > 0, f"expected at least one training call, got {calls}"
    assert agent.training_steps > 0


# ---------------------------------------------------------------------------
# TC-UT-34: save and load model preserves training_steps
# ---------------------------------------------------------------------------

def test_save_load_model(agent, tmp_path):
    m = make_metrics()
    state = agent.get_state(m)
    agent.store_transition(state, 1, 0.5, state, False, 0.0, 0.0)
    agent.training_steps = 42

    path = str(tmp_path / "test_model.pt")
    agent.save_model(path)

    new_agent = PPOAgent("test-deployment", "default", 18, 3)
    new_agent.load_model(path)
    assert new_agent.training_steps == 42, \
        f"expected training_steps=42 after reload, got {new_agent.training_steps}"
