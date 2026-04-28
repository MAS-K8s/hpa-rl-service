from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
    _CORS_AVAILABLE = True
except ImportError:
    _CORS_AVAILABLE = False
import logging
import os
import json
from datetime import datetime
import numpy as np
from threading import Lock
import torch

# Prometheus client for exporting metrics
from prometheus_client import Counter, Gauge, generate_latest, REGISTRY

from agents.ppo_agent import PPOAgent, action_to_string

app = Flask(__name__)
if _CORS_AVAILABLE:
    CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== PROMETHEUS METRICS ========================

ppo_actions = Counter('ppo_actions_total', 'Total actions taken by PPO agent', ['action', 'deployment'])
ppo_reward = Gauge('ppo_last_reward', 'Last raw reward received', ['deployment'])
ppo_confidence = Gauge('ppo_action_confidence', 'Confidence of chosen action', ['deployment'])
ppo_value = Gauge('ppo_value_estimate', 'Value estimate from critic', ['deployment'])
ppo_buffer_size = Gauge('ppo_buffer_size', 'Current buffer size', ['deployment'])
ppo_training_steps = Gauge('ppo_training_steps', 'Number of training updates', ['deployment'])

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain'}

# ======================== CONFIGURATION ========================

STATE_SIZE = 18
ACTION_SIZE = 3
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
CHECKPOINT_INTERVAL = 50

os.makedirs(MODEL_DIR, exist_ok=True)

# ======================== REDIS (OPTIONAL) ========================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_AVAILABLE = False
redis_client = None

try:
    import redis as redis_lib
    redis_client = redis_lib.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=False,
        socket_timeout=2
    )
    redis_client.ping()
    logger.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    REDIS_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ Redis unavailable: {e}. Multi-agent coordination disabled.")

# ======================== GLOBAL STATE ========================

agents = {}
agents_lock = Lock()


class ExperienceStore:
    def __init__(self):
        self.experiences = {}
        self.lock = Lock()

    def set_last(self, agent_key, state, action, log_prob, value, metrics):
        with self.lock:
            self.experiences[agent_key] = {
                'state': state,
                'action': action,
                'log_prob': log_prob,
                'value': value,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

    def get_last(self, agent_key):
        with self.lock:
            return self.experiences.get(agent_key)

    def delete(self, agent_key):
        with self.lock:
            self.experiences.pop(agent_key, None)


experience_store = ExperienceStore()


class DecisionHistory:
    MAX = 50

    def __init__(self):
        self.history = {}
        self.lock = Lock()

    def push(self, agent_key, entry):
        with self.lock:
            if agent_key not in self.history:
                self.history[agent_key] = []
            self.history[agent_key].append(entry)
            if len(self.history[agent_key]) > self.MAX:
                self.history[agent_key].pop(0)

    def get(self, agent_key):
        with self.lock:
            return list(self.history.get(agent_key, []))


decision_history = DecisionHistory()


# ======================== MULTI-AGENT COORDINATION ========================

class MultiAgentCoordinator:
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.enabled = redis_client is not None
        # Make cluster capacity configurable via environment variable
        self.cluster_capacity = int(os.getenv("CLUSTER_CAPACITY", "50"))

    def publish_agent_state(self, agent_key, state_info):
        if not self.enabled:
            return
        try:
            self.redis_client.setex(
                f"agent_state:{agent_key}",
                60,
                json.dumps({**state_info, 'timestamp': datetime.now().isoformat()})
            )
        except Exception as e:
            logger.warning(f"Failed to publish state: {e}")

    def get_cluster_status(self):
        if not self.enabled:
            return {'total_agents': len(agents), 'total_replicas': 0, 'available_capacity': self.cluster_capacity}
        try:
            agent_keys = self.redis_client.keys("agent_state:*")
            total_replicas = 0
            for k in agent_keys:
                data = self.redis_client.get(k)
                if data:
                    try:
                        parsed = json.loads(data)
                        total_replicas += parsed.get('replicas', 0)
                    except:
                        pass
            return {
                'total_agents': len(agent_keys),
                'total_replicas': total_replicas,
                'available_capacity': max(0, self.cluster_capacity - total_replicas)
            }
        except Exception as e:
            logger.warning(f"Failed to get cluster status: {e}")
            return {'total_agents': 0, 'total_replicas': 0, 'available_capacity': self.cluster_capacity}

    def request_scaling_approval(self, agent_key, current_replicas, desired_replicas):
        if not self.enabled:
            return True, "Redis disabled"
        try:
            cluster_status = self.get_cluster_status()
            delta = desired_replicas - current_replicas
            if delta > 0 and cluster_status['available_capacity'] < delta:
                return False, f"Insufficient capacity: need {delta}, have {cluster_status['available_capacity']}"
            self.redis_client.setex(
                f"scaling_intent:{agent_key}",
                30,
                json.dumps({
                    'agent': agent_key,
                    'from': current_replicas,
                    'to': desired_replicas,
                    'timestamp': datetime.now().isoformat()
                })
            )
            return True, "Approved"
        except Exception as e:
            logger.warning(f"Approval check failed: {e}")
            return True, "Fallback approval"


coordinator = MultiAgentCoordinator(redis_client if REDIS_AVAILABLE else None)


# ======================== AGENT MANAGEMENT ========================

def get_or_create_agent(deployment_name, namespace="default"):
    agent_key = f"{namespace}/{deployment_name}"
    with agents_lock:
        if agent_key not in agents:
            logger.info(f"🆕 Creating new PPO agent for {agent_key}")
            agent = PPOAgent(
                deployment_name=deployment_name,
                namespace=namespace,
                state_size=STATE_SIZE,
                action_size=ACTION_SIZE
            )
            model_path = os.path.join(MODEL_DIR, f"{agent_key.replace('/', '_')}.pt")
            if os.path.exists(model_path):
                try:
                    agent.load_model(model_path)
                    logger.info(f"📂 Loaded existing model for {agent_key}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load model: {e}")
            else:
                logger.info(f"🏋️ No saved model found — running synthetic pre-training for {agent_key}")
                agent.pretrain(n_steps=500)
                agent.save_model(model_path)
                logger.info(f"✅ Pre-training complete and model saved for {agent_key}")
            agents[agent_key] = agent
        return agents[agent_key]


# ======================== API ENDPOINTS ========================

@app.route('/health', methods=['GET'])
def health():
    cluster_status = coordinator.get_cluster_status()
    return jsonify({
        "status": "healthy",
        "active_agents": len(agents),
        "cluster_status": cluster_status,
        "redis_available": REDIS_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        deployment_name = data.get('deployment_name')
        namespace = data.get('namespace', 'default')
        metrics = data.get('metrics', {})
        training_mode = data.get('training_mode', False)

        if not deployment_name:
            return jsonify({"success": False, "error": "deployment_name required"}), 400

        # Warmup probe
        if deployment_name == "__warmup__":
            logger.info("🔥 Warmup probe received — PyTorch is ready")
            return jsonify({
                "success": True,
                "action": 1,
                "action_name": "no_action",
                "confidence": 0.0,
                "epsilon": 0.0,
                "reward": 0.0,
                "value_estimate": 0.0,
                "action_probabilities": [0.333, 0.334, 0.333],
                "coordination_approved": True,
                "coordination_message": "warmup",
                "buffer_size": 0,
                "training_steps": 0
            })

        agent = get_or_create_agent(deployment_name, namespace)
        agent_key = f"{namespace}/{deployment_name}"

        # Step 1: Get current state
        current_state = agent.get_state(metrics)

        # Step 2: Compute reward for PREVIOUS step and train
        reward = 0.0
        train_stats = None
        last_exp = experience_store.get_last(agent_key)

        if training_mode and last_exp is not None:
            reward = agent.calculate_reward(
                metrics,
                last_exp['action'],
                last_exp['metrics']
            )
            agent.store_transition(
                last_exp['state'],
                last_exp['action'],
                reward,
                current_state,
                done=False,
                log_prob=last_exp['log_prob'],
                value=last_exp['value']
            )

            if len(agent.buffer) >= agent.batch_size:
                train_stats = agent.update()
                if train_stats:
                    logger.info(
                        f"🎓 Training | Agent: {agent_key} | "
                        f"Step: {train_stats['training_steps']} | "
                        f"Policy Loss: {train_stats['policy_loss']:.4f} | "
                        f"Value Loss: {train_stats['value_loss']:.4f} | "
                        f"Avg Reward: {train_stats['avg_reward']:.2f} | "
                        f"Entropy: {train_stats['entropy']:.3f}"
                    )
                    decision_history.push(agent_key + "/__training__", {
                        "timestamp": datetime.now().isoformat(),
                        "step": train_stats["training_steps"],
                        "policy_loss": round(float(train_stats["policy_loss"]), 6),
                        "value_loss": round(float(train_stats["value_loss"]), 4),
                        "avg_reward": round(float(train_stats["avg_reward"]), 4),
                        "entropy": round(float(train_stats["entropy"]), 4),
                    })
                    if train_stats['training_steps'] % CHECKPOINT_INTERVAL == 0:
                        model_path = os.path.join(MODEL_DIR, f"{agent_key.replace('/', '_')}.pt")
                        agent.save_model(model_path)
                        logger.info(f"💾 Checkpoint saved at step {train_stats['training_steps']}")

        # Step 3: Select action for THIS step
        if training_mode:
            action, log_prob, value = agent.select_action(current_state, deterministic=False)
        else:
            action, log_prob, value = agent.select_action(current_state, deterministic=True)

        action_name = action_to_string(action)
        confidence, action_probs = agent.calculate_confidence(current_state)

        # Update Prometheus metrics (export)
        ppo_actions.labels(action=action_name, deployment=agent_key).inc()
        ppo_reward.labels(deployment=agent_key).set(reward if reward is not None else 0.0)
        ppo_confidence.labels(deployment=agent_key).set(confidence)
        ppo_value.labels(deployment=agent_key).set(value)
        ppo_buffer_size.labels(deployment=agent_key).set(len(agent.buffer))
        ppo_training_steps.labels(deployment=agent_key).set(agent.training_steps)

        # Step 4: Multi-agent coordination & safety guards
        current_replicas = metrics.get('replicas', 1)
        desired_replicas = current_replicas

        if action == 0:
            desired_replicas = current_replicas - 1
        elif action == 2:
            desired_replicas = current_replicas + 1

        coordination_approved = True
        coordination_message = ""

        # Guard 1: never scale down below 1
        if action == 0 and current_replicas <= 1:
            action = 1
            action_name = "no_action"
            desired_replicas = current_replicas
            coordination_message = "already at min replicas (1), scale-down suppressed"
            logger.info(f"🛡️ Min-replica guard | Agent: {agent_key} | scale_down suppressed")

        # Guard 2: soft replica cap
        elif action == 2:
            latency_now = metrics.get('latency_p95', 0.0)
            SOFT_CAP = int(os.getenv("SOFT_REPLICA_CAP", "5"))
            if current_replicas >= SOFT_CAP and latency_now > 0.5:
                action = 1
                action_name = "no_action"
                desired_replicas = current_replicas
                coordination_message = (
                    f"soft replica cap ({SOFT_CAP}) reached with no latency improvement "
                    f"(latency={latency_now:.2f}s > SLA=0.5s) — scale_up suppressed"
                )
                logger.info(f"🧢 Soft-cap guard | Agent: {agent_key} | {coordination_message}")

        if desired_replicas != current_replicas and action != 1:
            coordination_approved, coordination_message = coordinator.request_scaling_approval(
                agent_key, current_replicas, desired_replicas
            )
            if not coordination_approved:
                logger.info(f"🚫 Coordination blocked scaling: {coordination_message}")
                action = 1
                action_name = "no_action"

        # Step 5: Store THIS step's final decided action
        experience_store.set_last(
            agent_key,
            current_state,
            action,
            log_prob,
            value,
            metrics
        )

        # Publish state for multi-agent awareness
        coordinator.publish_agent_state(agent_key, {
            'replicas': current_replicas,
            'cpu_usage': metrics.get('cpu_usage', 0),
            'last_action': action_name,
            'confidence': confidence,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })

        response = {
            "success": True,
            "action": action,
            "action_name": action_name,
            "confidence": confidence,
            "epsilon": 0.0,
            "reward": reward,
            "value_estimate": value,
            "action_probabilities": action_probs.tolist(),
            "coordination_approved": coordination_approved,
            "coordination_message": coordination_message,
            "buffer_size": len(agent.buffer),
            "training_steps": agent.training_steps
        }

        logger.info(
            f"🎯 Prediction | Agent: {agent_key} | "
            f"Action: {action_name} | "
            f"Reward: {reward:.2f} | "
            f"Value: {value:.2f} | "
            f"Confidence: {confidence:.2%} | "
            f"Buffer: {len(agent.buffer)}/{agent.batch_size}"
        )

        decision_history.push(agent_key, {
            "timestamp": datetime.now().isoformat(),
            "action": action_name,
            "action_id": action,
            "reward": round(reward, 4),
            "confidence": round(float(confidence), 6),
            "value_estimate": round(float(value), 4),
            "replicas": int(metrics.get("replicas", 0)),
            "buffer_size": len(agent.buffer),
            "training_steps": agent.training_steps,
        })

        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Error in predict: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/save_model', methods=['POST'])
def save_model():
    try:
        data = request.json
        agent_key = f"{data.get('namespace', 'default')}/{data.get('deployment_name')}"
        if agent_key not in agents:
            return jsonify({"error": "Agent not found"}), 404
        agent = agents[agent_key]
        model_path = os.path.join(MODEL_DIR, f"{agent_key.replace('/', '_')}.pt")
        agent.save_model(model_path)
        return jsonify({
            "status": "saved",
            "model_path": model_path,
            "training_steps": agent.training_steps,
            "buffer_size": len(agent.buffer)
        })
    except Exception as e:
        logger.error(f"❌ Error saving model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        stats = {}
        for agent_key, agent in agents.items():
            stats[agent_key] = agent.get_metrics()
            stats[agent_key]['buffer_size'] = len(agent.buffer)
            stats[agent_key]['training_steps'] = agent.training_steps
        return jsonify({
            "agents": stats,
            "total_agents": len(agents),
            "cluster_status": coordinator.get_cluster_status(),
            "redis_available": REDIS_AVAILABLE
        })
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/pretrain', methods=['POST'])
def pretrain():
    try:
        data = request.json or {}
        n_steps = int(data.get('n_steps', 500))
        all_agents_flag = data.get('all_agents', False)

        if all_agents_flag:
            targets = list(agents.values())
            if not targets:
                return jsonify({"error": "No active agents — call /predict first to create one"}), 400
        else:
            deployment_name = data.get('deployment_name')
            if not deployment_name:
                return jsonify({"error": "deployment_name required (or set all_agents: true)"}), 400
            namespace = data.get('namespace', 'default')
            targets = [get_or_create_agent(deployment_name, namespace)]

        results = {}
        for agent in targets:
            agent_key = f"{agent.namespace}/{agent.deployment_name}"
            logger.info(f"🏋️ Pre-training {agent_key} for {n_steps} steps")
            train_calls = agent.pretrain(n_steps=n_steps)
            model_path = os.path.join(MODEL_DIR, f"{agent_key.replace('/', '_')}.pt")
            agent.save_model(model_path)
            results[agent_key] = {
                "training_updates": train_calls,
                "training_steps": agent.training_steps,
                "n_steps": n_steps,
            }
            logger.info(f"✅ Pre-training done for {agent_key}: {train_calls} updates")

        return jsonify({"status": "pretrained", "results": results})

    except Exception as e:
        logger.error(f"❌ Error in pretrain: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/reset_agent', methods=['POST'])
def reset_agent():
    try:
        data = request.json
        agent_key = f"{data.get('namespace', 'default')}/{data.get('deployment_name')}"
        with agents_lock:
            if agent_key not in agents:
                return jsonify({"error": "Agent not found"}), 404
            del agents[agent_key]
            experience_store.delete(agent_key)
            model_path = os.path.join(MODEL_DIR, f"{agent_key.replace('/', '_')}.pt")
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"🗑️ Deleted model file: {model_path}")
            if REDIS_AVAILABLE:
                try:
                    redis_client.delete(f"agent_state:{agent_key}")
                except Exception as e:
                    logger.warning(f"Failed to clear Redis: {e}")
            logger.info(f"♻️ Agent reset: {agent_key}")
            return jsonify({"status": "agent_reset", "agent_key": agent_key})
    except Exception as e:
        logger.error(f"❌ Error resetting agent: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/reset_all', methods=['POST'])
def reset_all():
    with agents_lock:
        count = len(agents)
        agents.clear()
        with experience_store.lock:
            experience_store.experiences.clear()
        deleted = []
        for fname in os.listdir(MODEL_DIR):
            if fname.endswith('.pt'):
                path = os.path.join(MODEL_DIR, fname)
                os.remove(path)
                deleted.append(fname)
        if REDIS_AVAILABLE:
            try:
                for key in redis_client.keys("agent_state:*"):
                    redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Failed to clear Redis: {e}")
        logger.info(f"♻️ All {count} agents reset, deleted {len(deleted)} model files")
        return jsonify({
            "status": "all_reset",
            "agents_cleared": count,
            "models_deleted": deleted
        })


@app.route('/cluster_status', methods=['GET'])
def cluster_status():
    try:
        status = coordinator.get_cluster_status()
        intents = []
        if REDIS_AVAILABLE:
            try:
                for key in redis_client.keys("scaling_intent:*"):
                    data = redis_client.get(key)
                    if data:
                        intents.append(json.loads(data))
            except Exception as e:
                logger.warning(f"Failed to get intents: {e}")
        return jsonify({
            "cluster_status": status,
            "active_scaling_intents": intents,
            "coordination_enabled": REDIS_AVAILABLE
        })
    except Exception as e:
        logger.error(f"❌ Error getting cluster status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/dashboard', methods=['GET'])
def dashboard():
    try:
        result = {}

        for agent_key, agent in agents.items():
            exp = experience_store.get_last(agent_key)
            metrics_snapshot = exp['metrics'] if exp else {}
            last_action_id = exp['action'] if exp else 1
            value_estimate = float(exp['value']) if exp else 0.0
            timestamp = exp['timestamp'] if exp else None

            action_name = action_to_string(last_action_id)

            confidence = 0.0
            action_probs = [0.333, 0.334, 0.333]
            if exp is not None:
                try:
                    state = exp['state']
                    conf, probs = agent.calculate_confidence(state)
                    confidence = float(conf)
                    action_probs = [float(p) for p in probs]
                except Exception:
                    pass

            training = agent.get_metrics()

            result[agent_key] = {
                "last_action": action_name,
                "last_action_id": last_action_id,
                "confidence": confidence,
                "action_probabilities": action_probs,
                "value_estimate": value_estimate,
                "last_decision_ts": timestamp,
                "replicas": metrics_snapshot.get('replicas', 0),
                "pod_ready": metrics_snapshot.get('pod_ready', 0),
                "pod_pending": metrics_snapshot.get('pod_pending', 0),
                "cpu_usage": metrics_snapshot.get('cpu_usage', 0.0),
                "memory_gib": metrics_snapshot.get('memory_usage', 0.0),
                "request_rate": metrics_snapshot.get('request_rate', 0.0),
                "latency_p95": metrics_snapshot.get('latency_p95', 0.0),
                "error_rate": metrics_snapshot.get('error_rate', 0.0),
                "training_steps": training['training_steps'],
                "avg_reward_100": float(training['avg_reward_100']),
                "buffer_size": training['buffer_size'],
                "device": training['device'],
                "decision_history": decision_history.get(agent_key),
                "training_history": decision_history.get(agent_key + "/__training__"),
            }

        return jsonify({
            "agents": result,
            "total_agents": len(agents),
            "redis_available": REDIS_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error in /dashboard: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ======================== STARTUP ========================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Starting Advanced RL Agent API Service (Enhanced)")
    logger.info("=" * 60)
    logger.info(f"📁 Models directory: {MODEL_DIR}")
    logger.info(f"🔌 Redis: {REDIS_HOST}:{REDIS_PORT} ({'✅ Connected' if REDIS_AVAILABLE else '❌ Unavailable'})")
    logger.info(f"🤖 Algorithm: PPO (Proximal Policy Optimization)")
    logger.info(f"📊 State size: {STATE_SIZE} dimensions")
    logger.info(f"🎯 Action size: {ACTION_SIZE} actions")
    logger.info(f"🔄 Multi-agent coordination: {'✅ Enabled' if REDIS_AVAILABLE else '❌ Disabled'}")
    logger.info(f"📈 Prometheus metrics available at /metrics")
    logger.info("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)