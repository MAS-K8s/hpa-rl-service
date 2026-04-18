"""
Integration tests for app.py Flask endpoints.

Run with:
    pytest tests/test_app.py -v
"""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import app as flask_app


@pytest.fixture
def client():
    flask_app.app.config['TESTING'] = True
    # Clear all agents between tests so state does not leak
    flask_app.agents.clear()
    flask_app.experience_store.experiences.clear()
    with flask_app.app.test_client() as c:
        yield c


def predict_payload(deployment="test-app", latency=0.3,
                    replicas=2, training=False):
    return {
        "deployment_name": deployment,
        "namespace": "default",
        "training_mode": training,
        "metrics": {
            "cpu_usage":    0.4,
            "memory_usage": 2.0,
            "request_rate": 50.0,
            "latency_p50":  latency * 0.7,
            "latency_p95":  latency,
            "latency_p99":  latency * 1.3,
            "replicas":     replicas,
            "error_rate":   0.0,
            "pod_pending":  0,
            "pod_ready":    replicas,
            "cpu_trend_1m": 0.0,
            "cpu_trend_5m": 0.0,
            "request_trend":0.0,
            "hour":         12,
            "day_of_week":  2,
            "is_weekend":   False,
            "is_peak_hour": False,
        }
    }


# ---------------------------------------------------------------------------
# TC-IT-01: GET /health returns 200 and healthy status
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'healthy'
    assert 'active_agents' in data
    assert 'timestamp' in data


# ---------------------------------------------------------------------------
# TC-IT-02: POST /predict with missing deployment_name returns 400
# ---------------------------------------------------------------------------

def test_predict_missing_deployment_name(client):
    resp = client.post('/predict',
                       data=json.dumps({"namespace": "default", "metrics": {}}),
                       content_type='application/json')
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['success'] is False


# ---------------------------------------------------------------------------
# TC-IT-03: POST /predict warmup probe returns 200 without creating agent
# ---------------------------------------------------------------------------

def test_predict_warmup_probe(client):
    payload = {
        "deployment_name": "__warmup__",
        "namespace": "default",
        "metrics": {},
        "training_mode": False
    }
    resp = client.post('/predict',
                       data=json.dumps(payload),
                       content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    assert data['action'] == 1
    # Warmup should not create an agent
    assert len(flask_app.agents) == 0


# ---------------------------------------------------------------------------
# TC-IT-04: POST /predict returns valid action and confidence
# ---------------------------------------------------------------------------

def test_predict_returns_valid_action(client):
    payload = predict_payload()
    resp = client.post('/predict',
                       data=json.dumps(payload),
                       content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    assert data['action'] in {0, 1, 2}
    assert data['action_name'] in {'scale_down', 'no_action', 'scale_up'}
    assert 0.0 <= data['confidence'] <= 1.0
    assert len(data['action_probabilities']) == 3


# ---------------------------------------------------------------------------
# TC-IT-05: POST /predict creates new agent for previously unseen deployment
# ---------------------------------------------------------------------------

def test_predict_creates_agent(client):
    assert len(flask_app.agents) == 0
    payload = predict_payload(deployment="new-app")
    client.post('/predict',
                data=json.dumps(payload),
                content_type='application/json')
    assert "default/new-app" in flask_app.agents


# ---------------------------------------------------------------------------
# TC-IT-06: POST /predict with stressed single pod returns scale_up
# ---------------------------------------------------------------------------

def test_predict_stressed_single_pod_scales_up(client):
    # Single pod with latency=0.9s — rule prior must fire scale_up
    payload = predict_payload(latency=0.9, replicas=1)
    resp = client.post('/predict',
                       data=json.dumps(payload),
                       content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['action'] == 2, \
        f"expected scale_up for stressed single pod, got {data['action_name']}"


# ---------------------------------------------------------------------------
# TC-IT-07: POST /predict at minimum replicas suppresses scale_down
# ---------------------------------------------------------------------------

def test_predict_min_replicas_suppresses_scaledown(client):
    # 1 replica, low latency — even if policy wants scale_down, guard must block
    for _ in range(3):
        payload = predict_payload(latency=0.1, replicas=1)
        resp = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
        data = resp.get_json()
        assert data['action'] != 0, \
            "scale_down must be suppressed when replicas=1"


# ---------------------------------------------------------------------------
# TC-IT-08: GET /stats returns agent training stats
# ---------------------------------------------------------------------------

def test_stats_returns_agent_data(client):
    # Create an agent first
    client.post('/predict',
                data=json.dumps(predict_payload()),
                content_type='application/json')
    resp = client.get('/stats')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'agents' in data
    assert data['total_agents'] >= 1


# ---------------------------------------------------------------------------
# TC-IT-09: POST /pretrain returns pretrained status
# ---------------------------------------------------------------------------

def test_pretrain_endpoint(client):
    payload = {
        "deployment_name": "test-app",
        "namespace": "default",
        "n_steps": 32
    }
    resp = client.post('/pretrain',
                       data=json.dumps(payload),
                       content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'pretrained'
    assert 'results' in data


# ---------------------------------------------------------------------------
# TC-IT-10: POST /pretrain without deployment_name returns 400
# ---------------------------------------------------------------------------

def test_pretrain_missing_name(client):
    resp = client.post('/pretrain',
                       data=json.dumps({}),
                       content_type='application/json')
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# TC-IT-11: POST /reset_agent clears agent state
# ---------------------------------------------------------------------------

def test_reset_agent(client):
    client.post('/predict',
                data=json.dumps(predict_payload(deployment="app-to-reset")),
                content_type='application/json')
    assert "default/app-to-reset" in flask_app.agents

    resp = client.post('/reset_agent',
                       data=json.dumps({
                           "deployment_name": "app-to-reset",
                           "namespace": "default"
                       }),
                       content_type='application/json')
    assert resp.status_code == 200
    assert "default/app-to-reset" not in flask_app.agents


# ---------------------------------------------------------------------------
# TC-IT-12: GET /dashboard returns agent snapshot
# ---------------------------------------------------------------------------

def test_dashboard_returns_data(client):
    client.post('/predict',
                data=json.dumps(predict_payload()),
                content_type='application/json')
    resp = client.get('/dashboard')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'agents' in data
    assert 'timestamp' in data
