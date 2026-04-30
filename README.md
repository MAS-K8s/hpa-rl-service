# 🧠 HPA-RL-SERVICE – Flask PPO RL Agent

> **Proximal Policy Optimization (PPO)** reinforcement learning agent service for intelligent, multi-objective Kubernetes autoscaling — with centralized critic (CTDE), Redis-backed multi-agent coordination, and a real-time dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Redis](https://img.shields.io/badge/Redis-7.0+-DC382D.svg)](https://redis.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Reference](#api-reference)
- [Training & Learning](#training--learning)
- [Multi-Agent Coordination (CTDE)](#multi-agent-coordination-ctde)
- [Prometheus Metrics](#prometheus-metrics)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The **HPA-RL-SERVICE** is the intelligence layer of the MARLS autoscaling system. It is a Flask microservice that hosts one **PPO agent per Kubernetes Deployment**, each learning independently while cooperating through a shared Redis coordination layer.

Each agent:

- **Observes** an 18-dimensional state vector derived from Prometheus metrics.
- **Acts** by returning one of three decisions: `scale_down` (0), `no_action` (1), or `scale_up` (2).
- **Learns online** — a PPO update fires every 32 collected transitions without any offline training phase.
- **Coordinates** with peer agents via Redis to respect cluster-wide capacity limits and share experiences.
- **Falls back safely** via hard-coded rule priors that override the neural network when it has not yet converged.

The Go controller (`go-controller-agent`) calls `POST /predict` on every reconciliation cycle. If this service is unreachable the controller activates its own rule-based fallback scaler; this service is therefore not on the critical path for cluster stability.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     HPA-RL-SERVICE  (app.py)                         │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Flask REST API  (threaded, port 5000)                       │    │
│  │  POST /predict  GET /health  GET /stats  GET /metrics        │    │
│  │  POST /save_model  POST /pretrain  POST /reset_agent         │    │
│  │  POST /reset_all   GET /cluster_status  GET /dashboard       │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                              │                                       │
│           one PPO agent per deployment (lazy creation)               │
│                              │                                       │
│  ┌───────────────────────────▼──────────────────────────────────┐   │
│  │  PPOAgent  (agents/ppo_agent.py)                             │   │
│  │  ┌─────────────────┐   ┌──────────────────┐                 │   │
│  │  │  ActorCritic    │   │  RolloutBuffer   │                 │   │
│  │  │  trunk(18→128→  │   │  capacity = 32   │                 │   │
│  │  │  128) + Tanh    │   │  circular window │                 │   │
│  │  │  actor  →  3    │   └──────────────────┘                 │   │
│  │  │  critic →  1    │   ┌──────────────────┐                 │   │
│  │  └─────────────────┘   │ RunningNormaliser│                 │   │
│  │  GAE  │  PPO clip  │   │  EMA mean/var    │                 │   │
│  │  2 epochs, mini-batch 8│  clip [-3, 3]    │                 │   │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Coordination Layer                                          │    │
│  │  MultiAgentCoordinator (app.py)  ──► Redis                  │    │
│  │    agent_state:{key}  TTL 60 s                               │    │
│  │    scaling_intent:{key}  TTL 30 s                            │    │
│  │  MultiAgentCoordinatorCTDE (coordination/coordinator.py)     │    │
│  │    CentralizedCritic (256-dim, max 10 agents)               │    │
│  │    SharedExperienceBuffer (Redis list, cap 10 000)           │    │
│  │    TeamReward  (α=0.3 local, β=0.7 global)                  │    │
│  │  MessageBus / AgentMessenger  (pub/sub channels)            │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
           │                              │
           ▼                              ▼
  Go Controller                       Redis 7
  POST /predict                  pub/sub + KV store
```

### PPO Actor-Critic Network

```
State vector (18 features)
          │
    Linear(18 → 128) + Tanh
    Linear(128 → 128) + Tanh
          │
    ┌─────┴─────┐
    ▼           ▼
Linear(128→3)  Linear(128→1)
   (Actor)       (Critic)
    │              │
  logits        V(s)
    │
  Categorical distribution
    │
  action ∈ {0=scale_down, 1=no_action, 2=scale_up}
```

---

## 📁 Project Structure

```
.
├── app.py                          # Flask service — routes, coordinator, agent lifecycle
├── rl_agent.py                     # Thin wrapper / legacy entry point
├── rl_service.py                   # Service helpers
├── requirements.txt
├── Dockerfile / .dockerignore
├── .env                            # Local env vars (not committed)
├── .gitignore
├── LICENSE
├── README.md
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py               # Abstract base class
│   ├── dqn_agent.py                # Legacy DQN agent (not active)
│   └── ppo_agent.py                # PPO agent — ActorCritic, RolloutBuffer, RunningNormaliser
│
├── coordination/
│   ├── coordinator.py              # MultiAgentCoordinatorCTDE — centralized critic training
│   ├── centralized_critic.py       # CentralizedCritic neural network (global value function)
│   ├── redis_buffer.py             # SharedExperienceBuffer backed by Redis list
│   └── team_reward.py              # TeamReward — global latency/cost/error reward
│
├── communication/
│   └── message_bus.py              # MessageBus (Redis pub/sub) + AgentMessenger
│
├── evaluation/                     # Offline evaluation scripts
├── models/                         # Saved .pt checkpoints (auto-created)
├── templates/
│   └── index.html                  # Web dashboard UI
└── tests/
```

---

## ⚙️ How It Works

### Per-Request Lifecycle (`POST /predict`)

Each call to `/predict` executes five steps in sequence:

```
Step 1  Build 18-dim state vector from incoming metrics
        │
Step 2  Compute reward for the PREVIOUS step's action
        Store transition (s, a, r, s', log_prob, value) → RolloutBuffer
        If buffer full (32 transitions) → PPO update (2 epochs, mini-batch 8)
        Auto-checkpoint every 50 training steps → models/<key>.pt
        │
Step 3  Select action for THIS step
        Training mode  → sample from Categorical(logits) + rule prior check
        Inference mode → argmax (deterministic)
        │
Step 4  Safety guards (in priority order)
        • Min-replica guard: suppress scale_down when replicas ≤ 1
        • Soft-cap guard:    suppress scale_up when replicas ≥ SOFT_REPLICA_CAP
                             and latency still above SLA after scaling
        • Coordinator check: request_scaling_approval() via Redis
        │
Step 5  Publish agent state to Redis (TTL 60 s)
        Return JSON response
```

### 18-Dimensional State Vector

| Index | Feature | Normalisation |
|---|---|---|
| 0 | `cpu_usage` | raw (0–1) |
| 1 | `memory_usage` | raw (GiB) |
| 2 | `request_rate` | ÷ 100 |
| 3 | `latency_p50` | raw (s) |
| 4 | `latency_p95` | raw (s) |
| 5 | `latency_p99` | raw (s) |
| 6 | `replicas` | ÷ 10 |
| 7 | `error_rate` | raw (0–1) |
| 8 | `pod_pending` | ÷ 10 |
| 9 | `pod_ready` | ÷ 10 |
| 10 | `cpu_trend_1m` | raw |
| 11 | `cpu_trend_5m` | raw |
| 12 | `request_trend` | ÷ 50 |
| 13 | `hour` | ÷ 24 |
| 14 | `day_of_week` | ÷ 7 |
| 15 | `is_weekend` | 0 / 1 |
| 16 | `is_peak_hour` | 0 / 1 |
| 17 | `max(0, latency_p95 − 0.5)` | SLA violation signal |

### Multi-Objective Reward Function

The raw reward has the following components (normalised to `[-3, 3]` via `RunningNormaliser` before storage):

| Component | Formula | Intent |
|---|---|---|
| SLA compliance | `+5 × (0.5 − latency)` when under SLA | reward headroom |
| SLA violation | `−50 × (latency − 0.5)²` when over SLA | penalise latency |
| Cost | `−0.5 × replicas` | minimise resource spend |
| Errors | `−100 × error_rate` | penalise failures |
| Over-scaling | `−5 × (replicas − 3)` when many replicas and still violating SLA | break runaway scaling |
| Idle excess | `−3 × (replicas − 1)` when replicas > 1 and latency well within SLA | reward efficiency |
| Scale-up shaping | `−25` if scaling up under low load; `−15` if replicas > 1 and latency fine | prevent unnecessary scale-ups |
| Scale-down shaping | `−30` at min replicas; `−20` under high load; `+20` when over-provisioned | safe scale-down |
| No-action shaping | `+8` idle and correct; `−5` under load; `−20` when SLA breach exists | reward correct inaction |
| Scaling incentive | `+30` when `replicas == 1` and `request_rate > 30 rps` | bootstrap scale-up behaviour |
| Transition bonus | `+15` successful scale-up; `+10` safe scale-down | reward measured transitions |

### Rule Prior (Safety Net)

Before the neural network output is used, a deterministic safety override is applied:

| Condition | Override | Reason |
|---|---|---|
| `replicas == 1` and `request_rate > 50 rps` | `scale_up` | single pod under load |
| `request_rate == 0` and `replicas == 1` | `no_action` | idle at minimum |
| `replicas ≤ 1` and `latency > 0.5 s` | `scale_up` | SLA breach at minimum |
| SLA violation `> 0.2` and `replicas ≤ 2` | `scale_up` | persistent SLA breach |
| `latency < 0.25 s` and `replicas > 1` | `scale_down` | idle and over-provisioned |

---

## 📦 Prerequisites

| Software | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required for PyTorch 2.x |
| Redis | 7.0+ | Optional — coordination disabled if absent |
| pip | latest | |
| CUDA | 11.8+ | Optional — CPU fallback is automatic |

---

## 🚀 Installation

### 1. Clone and enter the project

```bash
git clone https://github.com/MAS-K8s/hpa-rl-service.git
cd hpa-rl-service
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Redis (for multi-agent coordination)

```bash
# Docker (recommended)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# macOS
brew install redis && brew services start redis

# Ubuntu / Debian
sudo apt install redis-server && sudo systemctl start redis
```

### 5. Create the models directory

```bash
mkdir -p models
```

---

## 🔧 Configuration

All values are read from environment variables. Create a `.env` file for local development (loaded automatically at startup):

```bash
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Agent
MODEL_DIR=./models
LOG_LEVEL=INFO
CLUSTER_CAPACITY=50        # max total replicas across all agents
SOFT_REPLICA_CAP=5         # per-deployment soft cap for scale-up

# PPO Hyperparameters (all optional — defaults shown)
PPO_GAMMA=0.99
PPO_GAE_LAMBDA=0.95
PPO_CLIP_EPSILON=0.2
PPO_VALUE_LOSS_COEF=0.1
PPO_ENTROPY_COEF=0.05
PPO_LEARNING_RATE=1e-4
PPO_EPOCHS=2
PPO_MINI_BATCH=8
PPO_BUFFER_CAPACITY=32
```

### PPO Hyperparameter Reference

| Variable | Default | Description |
|---|---|---|
| `PPO_GAMMA` | `0.99` | Discount factor |
| `PPO_GAE_LAMBDA` | `0.95` | GAE smoothing parameter λ |
| `PPO_CLIP_EPSILON` | `0.2` | PPO clipping range ε |
| `PPO_VALUE_LOSS_COEF` | `0.1` | Value loss coefficient c₁ |
| `PPO_ENTROPY_COEF` | `0.05` | Entropy bonus coefficient c₂ |
| `PPO_LEARNING_RATE` | `1e-4` | Adam learning rate |
| `PPO_EPOCHS` | `2` | PPO update epochs per buffer flush |
| `PPO_MINI_BATCH` | `8` | Mini-batch size |
| `PPO_BUFFER_CAPACITY` | `32` | Rollout buffer capacity (triggers update when full) |

---

## ▶️ Running the Service

```bash
source venv/bin/activate
python app.py
```

Expected startup output:

```
============================================================
🚀 Starting Advanced RL Agent API Service (Enhanced)
============================================================
📁 Models directory: ./models
🔌 Redis: localhost:6379 (✅ Connected)
🤖 Algorithm: PPO (Proximal Policy Optimization)
📊 State size: 18 dimensions
🎯 Action size: 3 actions
🔄 Multi-agent coordination: ✅ Enabled
📈 Prometheus metrics available at /metrics
============================================================
 * Running on http://0.0.0.0:5000
```

The service is ready when the last line appears. The first `POST /predict` call for a new deployment triggers 500 steps of synthetic pre-training (a few seconds); send a `__warmup__` probe to absorb PyTorch JIT compile time before real traffic arrives.

---

## 📚 API Reference

### `GET /health`

Liveness and readiness probe.

```json
{
  "status": "healthy",
  "active_agents": 2,
  "cluster_status": { "total_agents": 2, "total_replicas": 5, "available_capacity": 45 },
  "redis_available": true,
  "timestamp": "2026-04-30T12:00:00.000Z"
}
```

---

### `POST /predict`

Main endpoint — receive metrics, return a scaling decision, and optionally train.

**Request body:**

```json
{
  "deployment_name": "my-app",
  "namespace": "production",
  "metrics": {
    "cpu_usage": 0.65,
    "memory_usage": 2.1,
    "request_rate": 120.5,
    "latency_p50": 0.18,
    "latency_p95": 0.32,
    "latency_p99": 0.41,
    "replicas": 2,
    "error_rate": 0.001,
    "pod_pending": 0,
    "pod_ready": 2,
    "cpu_trend_1m": 0.02,
    "cpu_trend_5m": 0.01,
    "request_trend": 5.0,
    "hour": 14,
    "day_of_week": 2,
    "is_weekend": false,
    "is_peak_hour": true
  },
  "training_mode": true
}
```

Set `deployment_name: "__warmup__"` to send a warm-up probe (PyTorch JIT compile) without creating a real agent.

**Response:**

```json
{
  "success": true,
  "action": 2,
  "action_name": "scale_up",
  "confidence": 0.78,
  "reward": 12.3,
  "value_estimate": 2.45,
  "action_probabilities": [0.08, 0.14, 0.78],
  "coordination_approved": true,
  "coordination_message": "Approved",
  "buffer_size": 32,
  "training_steps": 127
}
```

| Field | Description |
|---|---|
| `action` | `0` = scale_down, `1` = no_action, `2` = scale_up |
| `action_name` | Human-readable action string |
| `confidence` | Max action probability from the policy distribution |
| `reward` | Raw reward computed for the **previous** step's action |
| `value_estimate` | Critic's V(s) for the current state |
| `coordination_approved` | Whether cluster capacity check passed |
| `coordination_message` | Reason if action was modified by a safety guard |

---

### `GET /stats`

Per-agent summary: `training_steps`, `avg_reward_100`, `buffer_size`, `device`, plus cluster and Redis status.

---

### `POST /save_model`

Manually persist the current agent policy to disk.

```json
{ "deployment_name": "my-app", "namespace": "production" }
```

---

### `POST /pretrain`

Run synthetic pre-training for one or all agents.

```json
{ "deployment_name": "my-app", "namespace": "production", "n_steps": 1000 }
```

Set `"all_agents": true` to pre-train all currently active agents in one call.

---

### `GET /metrics`

Prometheus text-format metrics endpoint. See [Prometheus Metrics](#prometheus-metrics) below.

---

### `GET /dashboard`

Full aggregate snapshot for the web UI — per-agent metrics, last action, last 50 decisions, training history, and cluster status.

---

### `GET /cluster_status`

Redis-backed cluster-wide replica count, utilisation, and active scaling intents (TTL 30 s).

---

### `POST /reset_agent`

Delete a single agent, its in-memory state, its model file, and its Redis keys.

```json
{ "deployment_name": "my-app", "namespace": "production" }
```

---

### `POST /reset_all`

Delete all agents, all `.pt` model files, and all `agent_state:*` Redis keys.

---

## 🎓 Training & Learning

### Automatic Pre-Training

When a new agent is created and no `.pt` checkpoint exists, it automatically runs **500 steps of synthetic pre-training** across 8 representative scenarios (stressed, idle, over-scaled, SLA-breaching) before serving its first real prediction. The resulting model is saved immediately.

Trigger additional pre-training at any time:

```bash
curl -X POST http://localhost:5000/pretrain \
  -H "Content-Type: application/json" \
  -d '{"deployment_name": "my-app", "namespace": "production", "n_steps": 2000}'
```

### Online Learning

Set `training_mode: true` in each `/predict` request (the Go controller does this when `TRAINING_MODE=true`).

1. A **transition** `(s, a, r, s', log_prob, value)` is stored in the 32-slot `RolloutBuffer` on every call.
2. When the buffer is full, a **PPO update** fires:
   - GAE advantage estimation (`γ = 0.99`, `λ = 0.95`)
   - Advantage normalisation
   - **2 epochs** of clipped surrogate loss with mini-batch size **8**
   - Gradient clipping at norm `0.5`
3. The buffer is cleared after each update (rolling window, not episodic).
4. Models are **auto-checkpointed every 50 training steps** to `models/<namespace>_<deployment>.pt`.

### Monitoring Training

Watch logs for update lines:

```
🎓 Training | Agent: production/my-app | Step: 214 |
  Policy Loss: 0.0234 | Value Loss: 0.1712 | Avg Reward: 0.36 | Entropy: 0.74
```

Training history (last 50 entries) is also exposed in `GET /dashboard` under `training_history` per agent.

---

## 🤝 Multi-Agent Coordination (CTDE)

The system implements **Centralized Training with Decentralized Execution (CTDE)** at two levels.

### In-Request Coordination (`app.py → MultiAgentCoordinator`)

- After every `/predict` call each agent **publishes its state** (`replicas`, `cpu_usage`, `last_action`, `confidence`) to Redis with a 60-second TTL.
- Before applying `scale_up`, `request_scaling_approval()` checks that available cluster capacity covers the delta. If not, the action is overridden to `no_action`.
- **Scaling intents** are written to Redis with a 30-second TTL for auditability via `GET /cluster_status`.

### CTDE Critic (`coordination/`)

| Component | Description |
|---|---|
| `CentralizedCritic` | 256-hidden-dim network; accepts padded states and one-hot actions from up to 10 agents simultaneously; returns a global value V(s₁,a₁,…,sₙ,aₙ); agent masks handle variable-size agent sets |
| `SharedExperienceBuffer` | Redis list (cap 10,000) where all agents deposit transitions; the centralized critic trains on random mini-batches from this pool |
| `TeamReward` | Computes a global reward from cluster-wide avg P95 latency, total replicas, and total error rate; combined with local reward as `(1−α) × local + α × global` with `α=0.3` |

### Message Bus (`communication/message_bus.py`)

`MessageBus` wraps Redis pub/sub with a background listener thread. `AgentMessenger` provides structured high-level messaging:

| Method | Channel | Purpose |
|---|---|---|
| `send_status_update()` | `agent_status` | Broadcast agent state |
| `send_scaling_intent()` | `scaling_intents` | Notify peers of planned scaling |
| `send_experience()` | `experiences` | Share transitions |
| `send_alert()` | `alerts` | Propagate warnings |
| `send_direct_message()` | `agent:<key>` | Agent-to-agent messaging |
| `broadcast()` | `broadcast` | All-agents announcement |

---

## 📊 Prometheus Metrics

Configure Prometheus to scrape `http://<service>:5000/metrics`.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ppo_actions_total` | Counter | `action`, `deployment` | Total actions taken per type |
| `ppo_last_reward` | Gauge | `deployment` | Last raw reward value |
| `ppo_action_confidence` | Gauge | `deployment` | Max action probability |
| `ppo_value_estimate` | Gauge | `deployment` | Critic V(s) for current state |
| `ppo_buffer_size` | Gauge | `deployment` | Current rollout buffer fill level |
| `ppo_training_steps` | Gauge | `deployment` | Number of PPO update steps completed |

Example `scrape_config`:

```yaml
scrape_configs:
  - job_name: hpa-rl-service
    static_configs:
      - targets: ['hpa-rl-service.marls-system:5000']
```

---

## 🐳 Deployment

### Docker

```bash
docker build -t hpa-rl-service:latest .

docker run -d \
  --name hpa-rl-service \
  -p 5000:5000 \
  -e REDIS_HOST=redis \
  -e CLUSTER_CAPACITY=50 \
  -v $(pwd)/models:/app/models \
  hpa-rl-service:latest
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hpa-rl-service
  namespace: marls-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hpa-rl-service
  template:
    metadata:
      labels:
        app: hpa-rl-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "5000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: rl-service
          image: your-registry/hpa-rl-service:latest
          ports:
            - containerPort: 5000
          env:
            - name: REDIS_HOST
              value: "redis.marls-system.svc.cluster.local"
            - name: CLUSTER_CAPACITY
              value: "50"
            - name: SOFT_REPLICA_CAP
              value: "5"
            - name: MODEL_DIR
              value: "/models"
          volumeMounts:
            - name: models
              mountPath: /models
          livenessProbe:
            httpGet:
              path: /health
              port: 5000
            initialDelaySeconds: 30
            periodSeconds: 15
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: 2000m
              memory: 2Gi
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: marls-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: hpa-rl-service
  namespace: marls-system
spec:
  selector:
    app: hpa-rl-service
  ports:
    - port: 5000
      targetPort: 5000
```

> **Model persistence:** Mount a `PersistentVolumeClaim` at `/models` so checkpoints survive pod restarts. Without it, agents re-run 500-step pre-training on every restart.

---

## 🐛 Troubleshooting

**First `/predict` call takes 6–10 seconds**
PyTorch JIT compiles on the first forward pass. Send a warmup probe immediately after the pod becomes ready:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"deployment_name": "__warmup__", "metrics": {}}'
```

**`⚠️ Redis unavailable` on startup**
Coordination is disabled but the service still works — each agent operates independently. Start Redis and set `REDIS_HOST` / `REDIS_PORT` correctly to re-enable coordination and shared experience.

**High entropy / low confidence after many steps**
The agent is still exploring. Run `POST /pretrain` with a higher `n_steps`, or reduce `PPO_ENTROPY_COEF` to lower exploration pressure.

**`KeyError` when loading a checkpoint**
The checkpoint uses an older key name. The loader tries `policy`, `policy_net`, and `policy_state_dict` in sequence for backward compatibility. If all three fail, delete the `.pt` file and let the agent pre-train fresh.

**Actions are always `no_action`**
Verify `training_mode: true` is set in `/predict` requests. In deterministic (inference) mode `argmax` defaults to `no_action` before training converges. Also check that `CONFIDENCE_MIN` on the Go controller side is not set too high for the current training stage.

**Coordination always blocks `scale_up`**
The cluster is at capacity. Check `GET /cluster_status` — if `available_capacity` is 0, raise `CLUSTER_CAPACITY` to match your actual node pool size, or wait for other deployments to scale down.

**`scale_up` suppressed by soft-cap guard**
The agent reached `SOFT_REPLICA_CAP` replicas but latency is still above SLA. This means adding more replicas is not helping. Investigate the root cause (CPU throttling, slow DB, external dependency) rather than raising the cap blindly.

---

## 🔗 Integration with the Go Controller

The Go controller (`go-controller-agent`) is the sole client of this service. It calls `POST /predict` on every reconciliation cycle and expects a response within `RL_TIMEOUT` (default 2 s). If this service is unreachable the controller falls back to its rule-based scaler and records a circuit breaker failure — no cluster stability impact.

```
Go controller  →  POST /predict  →  RL service  (every INTERVAL seconds per deployment)
RL service     →  JSON response  →  Go controller applies scaling
Go controller  →  GET /health    →  startup warm-up check only
```

---

## 📚 Related

- [MARLS Go Controller README](./README_go_controller.md) — the Kubernetes controller that calls this service.
- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (Lowe et al., 2017)](https://arxiv.org/abs/1706.02275)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Redis Documentation](https://redis.io/docs/)