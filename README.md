# 🧠 Flask RL Agent Service

> Deep Q-Network (DQN) based Reinforcement Learning agent for intelligent Kubernetes autoscaling decisions.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Training](#training)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The Flask RL Agent Service is the "brain" of the autoscaling system. It uses a Deep Q-Network (DQN) to learn optimal scaling policies by:

- **Observing** system metrics (CPU, memory, latency, etc.)
- **Learning** from rewards (performance vs. cost trade-offs)
- **Deciding** when to scale up, scale down, or maintain current state
- **Improving** over time through continuous training

### What This Service Does

```
┌─────────────────────────────────────────────────────────┐
│                  Flask RL Agent Service                  │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  REST API (Flask)                              │    │
│  │  - /predict: Get scaling decisions             │    │
│  │  - /health: Health check                       │    │
│  │  - /stats: Agent statistics                    │    │
│  │  - /save_model: Save trained model             │    │
│  └────────────┬───────────────────────────────────┘    │
│               │                                          │
│               ▼                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  RL Agent (rl_agent.py)                        │    │
│  │  - Deep Q-Network (Neural Network)             │    │
│  │  - Experience Replay Memory                    │    │
│  │  - Epsilon-Greedy Exploration                  │    │
│  │  - Training & Optimization                     │    │
│  └────────────┬───────────────────────────────────┘    │
│               │                                          │
│               ▼                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  Redis (State Coordination)                    │    │
│  │  - Multi-agent communication                   │    │
│  │  - Shared knowledge store                      │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

### Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    System Integration                         │
└──────────────────────────────────────────────────────────────┘

    Go Controller                Flask RL Service
    (Kubernetes)                 (Decision Making)
         │                              │
         │ 1. Send Metrics              │
         │ ───────────────────────────► │
         │   POST /predict              │
         │   {                          │
         │     metrics: {...}           │
         │   }                          │
         │                              │
         │                         2. Process
         │                         ┌─────────┐
         │                         │ DQN     │
         │                         │ Network │
         │                         └─────────┘
         │                              │
         │                         3. Choose Action
         │                         - scale_up
         │                         - scale_down
         │                         - no_action
         │                              │
         │ 4. Return Decision           │
         │ ◄─────────────────────────── │
         │   {                          │
         │     action: 2,               │
         │     action_name: "scale_up"  │
         │   }                          │
         │                              │
         │                         5. Train (if enabled)
         │                         - Calculate reward
         │                         - Store experience
         │                         - Update weights
         │                              │
         ▼                              ▼
```

### Deep Q-Network Architecture

```
Input State (8 features)
    │
    │ [CPU, Memory, Latency, Request Rate,
    │  Replicas, Error Rate, Pending Pods, Time]
    ▼
┌─────────────────┐
│  Input Layer    │  8 neurons
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hidden Layer 1 │  128 neurons + ReLU
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hidden Layer 2 │  128 neurons + ReLU
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hidden Layer 3 │  64 neurons + ReLU
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Layer   │  3 neurons (Q-values)
└────────┬────────┘
         │
         ▼
    [Q(down), Q(stay), Q(up)]
    [-2.3,    1.5,     8.2]
              ▲
              │
         Pick Max: scale_up
```

### Learning Process

```
1. Experience Collection
   ┌──────────────────────────────────────┐
   │ State → Action → Reward → Next State│
   │ Stored in Replay Memory (2000 max)  │
   └──────────────────────────────────────┘

2. Training (Every Decision)
   ┌──────────────────────────────────────┐
   │ Sample 32 random experiences         │
   │ ↓                                    │
   │ Compute Q-values (Policy Network)   │
   │ ↓                                    │
   │ Compute Targets (Target Network)    │
   │ ↓                                    │
   │ Calculate Loss (MSE)                │
   │ ↓                                    │
   │ Backpropagation                     │
   │ ↓                                    │
   │ Update Weights (Adam Optimizer)     │
   └──────────────────────────────────────┘

3. Exploration → Exploitation
   ┌──────────────────────────────────────┐
   │ Epsilon (ε) starts at 1.0 (100%)    │
   │ ↓                                    │
   │ Gradually decays to 0.01 (1%)       │
   │ ↓                                    │
   │ More exploitation, less exploration  │
   └──────────────────────────────────────┘
```

---

## ✨ Features

- 🧠 **Deep Q-Network (DQN)** - State-of-the-art RL algorithm
- 🔄 **Experience Replay** - Learn from past experiences
- 🎲 **Epsilon-Greedy** - Balance exploration vs exploitation
- 📊 **Multi-Objective Reward** - Optimize latency, cost, and reliability
- 🤝 **Multi-Agent Support** - Manage multiple deployments
- 💾 **Model Persistence** - Save and load trained models
- 📈 **Training Modes** - Offline pre-training and online learning
- 🔍 **Real-time Monitoring** - Agent statistics and performance metrics

---

## 📦 Prerequisites

### Software Requirements

- **Python** 3.10 or higher
- **Redis** 7.0 or higher
- **pip** (Python package manager)

### System Requirements

- **Memory**: 2GB RAM minimum (4GB recommended)
- **CPU**: 2 cores minimum
- **Storage**: 1GB for models and dependencies
- **GPU**: Optional (for faster training)

---

## 🚀 Installation

### Step 1: Clone or Create Project Structure

```bash
mkdir -p flask-app
cd flask-app
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# Verify activation
which python  # Should point to venv/bin/python
```

### Step 3: Create Requirements File

```bash
cat > requirements.txt << 'EOF'
torch==2.0.1
numpy==1.24.3
redis==4.5.5
flask==2.3.2
flask-cors==4.0.0
EOF
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import redis; import flask; print('✅ All packages installed successfully')"
```

### Step 5: Install Redis

**On macOS:**
```bash
brew install redis
brew services start redis
redis-cli ping  # Should return: PONG
```

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install redis-server -y
sudo systemctl start redis
sudo systemctl enable redis
redis-cli ping  # Should return: PONG
```

**Using Docker:**
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
redis-cli ping  # Should return: PONG
```

### Step 6: Copy Source Files

Copy these files to `flask-app/`:
- `rl_agent.py` - RL agent implementation
- `rl_service.py` - Flask API service

### Step 7: Create Models Directory

```bash
mkdir -p models
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file (optional):

```bash
cat > .env << 'EOF'
REDIS_HOST=localhost
REDIS_PORT=6379
FLASK_ENV=development
MODEL_DIR=./models
LOG_LEVEL=INFO
FLASK_PORT=5000
EOF
```

### Configuration Parameters

Edit `rl_service.py` if needed:

```python
# Agent Configuration
STATE_SIZE = 8          # Number of state features
ACTION_SIZE = 3         # Number of actions (down, stay, up)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
MODEL_DIR = "./models"

# Flask Configuration
FLASK_HOST = "0.0.0.0"  # Listen on all interfaces
FLASK_PORT = 5000       # Default port
DEBUG = False           # Set to True for development
```

### Hyperparameters (in rl_agent.py)

```python
# Learning Parameters
self.gamma = 0.95           # Discount factor
self.epsilon = 1.0          # Initial exploration rate
self.epsilon_min = 0.01     # Minimum exploration
self.epsilon_decay = 0.995  # Decay rate
self.learning_rate = 0.001  # Adam optimizer learning rate
self.batch_size = 32        # Training batch size
self.memory_size = 2000     # Replay memory size
```

---

## 🎮 Usage

### Start the Service

```bash
# Activate virtual environment
source venv/bin/activate

# Start Flask service
python rl_service.py
```

**Expected Output:**
```
🚀 Starting RL Agent API Service
📁 Models directory: ./models
🔌 Redis host: localhost
 * Serving Flask app 'rl_service'
 * Debug mode: off
 * Running on http://0.0.0.0:5000
Press CTRL+C to quit
```

### Verify Service is Running

```bash
# Health check
curl http://localhost:5000/health

# Expected response:
# {
#   "status": "healthy",
#   "active_agents": 0,
#   "timestamp": "2025-10-21T..."
# }
```

### Test Prediction Endpoint

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_name": "test-app",
    "namespace": "default",
    "metrics": {
      "cpu_usage": 0.75,
      "memory_usage": 1.2,
      "request_rate": 120.0,
      "latency_p95": 0.45,
      "replicas": 3,
      "error_rate": 0.01,
      "pod_pending": 0,
      "timestamp": 1697654400
    },
    "training_mode": true
  }'
```

**Expected Response:**
```json
{
  "action": 2,
  "action_name": "scale_up",
  "confidence": 0.15,
  "epsilon": 0.995,
  "reward": 3.2
}
```

---

## 📚 API Reference

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "active_agents": 1,
  "timestamp": "2025-10-21T10:30:00"
}
```

### POST /predict

Get scaling decision from RL agent.

**Request Body:**
```json
{
  "deployment_name": "myapp",
  "namespace": "default",
  "metrics": {
    "cpu_usage": 0.75,
    "memory_usage": 1.2,
    "request_rate": 120.0,
    "latency_p95": 0.45,
    "replicas": 3,
    "error_rate": 0.01,
    "pod_pending": 0,
    "timestamp": 1697654400
  },
  "training_mode": true
}
```

**Response:**
```json
{
  "action": 2,
  "action_name": "scale_up",
  "confidence": 0.85,
  "epsilon": 0.15,
  "reward": 4.5
}
```

**Actions:**
- `0`: scale_down (decrease replicas)
- `1`: no_action (maintain current state)
- `2`: scale_up (increase replicas)

### GET /stats

Get agent statistics.

**Response:**
```json
{
  "agents": {
    "default/myapp": {
      "epsilon": 0.15,
      "memory_size": 523,
      "device": "cpu"
    }
  },
  "total_agents": 1
}
```

### POST /save_model

Save trained model to disk.

**Request Body:**
```json
{
  "deployment_name": "myapp",
  "namespace": "default"
}
```

**Response:**
```json
{
  "status": "saved",
  "model_path": "./models/default_myapp.pt",
  "epsilon": 0.15,
  "memory_size": 523
}
```

### POST /update_target

Update target network (called periodically).

**Request Body:**
```json
{
  "deployment_name": "myapp",
  "namespace": "default"
}
```

**Response:**
```json
{
  "status": "target_network_updated"
}
```

---

## 🎓 Training

### Offline Training

Pre-train the agent before deployment:

```bash
# Generate synthetic data
python train.py generate

# Train for 1000 episodes
python train.py train

# Model saved to: models/pretrained_agent.pt
```

### Online Training

Enable training mode in production:

```bash
# Training mode is controlled by Go Controller
# Use --training=true flag when starting Go agent
```

### Monitor Training Progress

```bash
# Check agent stats
curl http://localhost:5000/stats | python -m json.tool

# Look for:
# - epsilon decreasing (1.0 → 0.01)
# - memory_size increasing
# - In logs: "🎓 Training loss: X, Reward: Y"
```

### Save Model Periodically

```bash
# Save every hour or after good performance
curl -X POST http://localhost:5000/save_model \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_name": "myapp",
    "namespace": "default"
  }'
```

### Load Pre-trained Model

Models are automatically loaded on agent creation if they exist in `./models/` directory.

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to Redis"

**Solution:**
```bash
# Check Redis is running
redis-cli ping

# If not running, start it:
# Mac: brew services start redis
# Linux: sudo systemctl start redis
# Docker: docker start redis

# Check Redis port
netstat -an | grep 6379
```

### Issue: "Module 'torch' not found"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify
python -c "import torch; print(torch.__version__)"
```

### Issue: "Port 5000 already in use"

**Solution:**
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or change Flask port
export FLASK_PORT=5001
python rl_service.py
```

### Issue: "Agent not learning"

**Check:**
```bash
# 1. Verify training mode is enabled
curl http://localhost:5000/stats

# 2. Check memory is growing
# memory_size should increase over time

# 3. Check logs for training messages
# Look for: "🎓 Training loss: X"

# 4. Verify rewards are being calculated
# Check Go Controller logs for reward values
```

### Issue: "High memory usage"

**Solution:**
```python
# Reduce memory size in rl_agent.py
self.memory = deque(maxlen=1000)  # Was 2000

# Or clear old agents
curl -X POST http://localhost:5000/reset_agent \
  -d '{"deployment_name": "old-app"}'
```

---

## 📊 Monitoring

### View Logs

```bash
# Real-time logs
tail -f flask.log

# Search for specific patterns
grep "Action:" flask.log
grep "Training loss" flask.log
grep "ERROR" flask.log
```

### Agent Statistics

```bash
# Get detailed stats
curl -s http://localhost:5000/stats | jq

# Monitor continuously
watch -n 5 'curl -s http://localhost:5000/stats | jq'
```

### Performance Metrics

```bash
# Check response time
time curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @test_metrics.json

# Should be < 100ms
```

---

## 🔐 Security Notes

- Service binds to `0.0.0.0` (all interfaces) - be careful in production
- No authentication by default - add API keys for production
- Redis should be password-protected in production
- Use environment variables for sensitive config

---

## 📄 Files Structure

```
flask-app/
├── rl_agent.py           # RL agent implementation
├── rl_service.py         # Flask API service
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (optional)
├── README.md            # This file
├── models/              # Trained models directory
│   └── default_myapp.pt
├── venv/                # Virtual environment (created)
└── logs/                # Log files (auto-created)
```

---

## 🚀 Quick Commands Reference

```bash
# Start service
source venv/bin/activate && python rl_service.py

# Health check
curl http://localhost:5000/health

# Get prediction
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @metrics.json

# Get stats
curl http://localhost:5000/stats | jq

# Save model
curl -X POST http://localhost:5000/save_model -d '{"deployment_name":"myapp"}'

# Stop service
# Press Ctrl+C
```

---

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Redis Documentation](https://redis.io/docs/)
- [DQN Paper](https://www.nature.com/articles/nature14236)
- [Reinforcement Learning Book](http://incompleteideas.net/book/the-book-2nd.html)

---

## 📧 Support

For issues or questions:
- Check logs: `tail -f flask.log`
- Review API responses for error messages
- Verify Redis connection
- Check Python version compatibility

---

**⭐ Flask RL Agent Service - The Brain of Intelligent Autoscaling**