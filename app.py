from flask import Flask, request, jsonify
import logging
import os
import json
from datetime import datetime
import numpy as np
import redis
from threading import Lock
import pickle
import torch

# Import our enhanced agents
from agents.ppo_agent import PPOAgent, action_to_string

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== CONFIGURATION ========================

STATE_SIZE = 18
ACTION_SIZE = 3
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MODEL_DIR = "./models"
CHECKPOINT_INTERVAL = 50  # FIXED: Save model every 50 updates (was 100)

os.makedirs(MODEL_DIR, exist_ok=True)

# ======================== GLOBAL STATE ========================

agents = {}
agents_lock = Lock()

# FIXED: Better experience storage
class ExperienceStore:
    """Store experiences for each agent"""
    def __init__(self):
        self.experiences = {}
        self.lock = Lock()
    
    def add(self, agent_key, state, action, reward, next_state, log_prob, value, prev_metrics):
        with self.lock:
            if agent_key not in self.experiences:
                self.experiences[agent_key] = []
            
            self.experiences[agent_key].append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'log_prob': log_prob,
                'value': value,
                'prev_metrics': prev_metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 1000 experiences per agent
            if len(self.experiences[agent_key]) > 1000:
                self.experiences[agent_key] = self.experiences[agent_key][-1000:]
    
    def get_last_experience(self, agent_key):
        with self.lock:
            if agent_key in self.experiences and len(self.experiences[agent_key]) > 0:
                return self.experiences[agent_key][-1]
            return None
    
    def get_count(self, agent_key):
        with self.lock:
            return len(self.experiences.get(agent_key, []))

experience_store = ExperienceStore()

# Redis connection
try:
    redis_client = redis.Redis(
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
    REDIS_AVAILABLE = False

# ======================== MULTI-AGENT COORDINATION ========================

class MultiAgentCoordinator:
    """Coordinates multiple agents"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.enabled = redis_client is not None
    
    def publish_agent_state(self, agent_key, state_info):
        if not self.enabled:
            return
        
        try:
            channel = f"agent_state:{agent_key}"
            self.redis_client.setex(
                channel,
                60,
                json.dumps({**state_info, 'timestamp': datetime.now().isoformat()})
            )
        except Exception as e:
            logger.warning(f"Failed to publish state: {e}")
    
    def get_cluster_status(self):
        if not self.enabled:
            return {
                'total_agents': len(agents),
                'total_replicas': 0,
                'available_capacity': 100
            }
        
        try:
            agent_keys = self.redis_client.keys("agent_state:*")
            total_replicas = 0
            
            for key in agent_keys:
                state_data = self.redis_client.get(key)
                if state_data:
                    state = json.loads(state_data)
                    total_replicas += state.get('replicas', 0)
            
            cluster_capacity = 50
            available_capacity = max(0, cluster_capacity - total_replicas)
            
            return {
                'total_agents': len(agent_keys),
                'total_replicas': total_replicas,
                'available_capacity': available_capacity
            }
        except Exception as e:
            logger.warning(f"Failed to get cluster status: {e}")
            return {'total_agents': 0, 'total_replicas': 0, 'available_capacity': 100}
    
    def request_scaling_approval(self, agent_key, current_replicas, desired_replicas):
        if not self.enabled:
            return True, "Redis disabled"
        
        try:
            cluster_status = self.get_cluster_status()
            delta = desired_replicas - current_replicas
            
            if delta > 0:
                if cluster_status['available_capacity'] < delta:
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
    """Get existing agent or create new one"""
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
            
            agents[agent_key] = agent
            
            if REDIS_AVAILABLE:
                try:
                    redis_client.setex(
                        f"agent_state:{agent_key}",
                        60,
                        json.dumps({
                            'deployment': deployment_name,
                            'namespace': namespace,
                            'created': datetime.now().isoformat()
                        })
                    )
                except Exception as e:
                    logger.warning(f"Failed to register in Redis: {e}")
        
        return agents[agent_key]

# ======================== API ENDPOINTS ========================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
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
    """FIXED: Main prediction endpoint with proper training"""
    try:
        data = request.json
        deployment_name = data.get('deployment_name')
        namespace = data.get('namespace', 'default')
        metrics = data.get('metrics', {})
        training_mode = data.get('training_mode', False)
        
        if not deployment_name:
            return jsonify({
                "success": False,
                "error": "deployment_name required"
            }), 400
        
        agent = get_or_create_agent(deployment_name, namespace)
        agent_key = f"{namespace}/{deployment_name}"
        
        # Convert metrics to state
        current_state = agent.get_state(metrics)
        
        # Get last experience for training
        last_exp = experience_store.get_last_experience(agent_key)
        
        # Select action
        if training_mode:
            action, log_prob, value = agent.select_action(current_state, deterministic=False)
        else:
            action, log_prob, value = agent.select_action(current_state, deterministic=True)
        
        action_name = action_to_string(action)
        
        # FIXED: Calculate real confidence
        confidence, action_probs = agent.calculate_confidence(current_state)
        
        # Calculate reward and train
        reward = 0.0
        if training_mode and last_exp is not None:
            prev_state = last_exp['state']
            prev_action = last_exp['action']
            prev_log_prob = last_exp['log_prob']
            prev_value = last_exp['value']
            prev_metrics = last_exp.get('prev_metrics', {})
            
            # Calculate reward
            reward = agent.calculate_reward(metrics, prev_action, prev_metrics)
            
            # Store transition in agent's buffer
            agent.store_transition(
                prev_state,
                prev_action,
                reward,
                current_state,
                done=False,
                log_prob=prev_log_prob,
                value=prev_value
            )
            
            # FIXED: Train when buffer reaches threshold
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
                    
                    # Save checkpoint periodically
                    if train_stats['training_steps'] % CHECKPOINT_INTERVAL == 0:
                        model_path = os.path.join(MODEL_DIR, f"{agent_key.replace('/', '_')}.pt")
                        agent.save_model(model_path)
                        logger.info(f"💾 Checkpoint saved at step {train_stats['training_steps']}")
        
        # Multi-agent coordination check
        current_replicas = metrics.get('replicas', 1)
        desired_replicas = current_replicas
        
        if action == 0:
            desired_replicas = max(1, current_replicas - 1)
        elif action == 2:
            desired_replicas = current_replicas + 1
        
        coordination_approved = True
        coordination_message = ""
        
        if desired_replicas != current_replicas:
            coordination_approved, coordination_message = coordinator.request_scaling_approval(
                agent_key, current_replicas, desired_replicas
            )
            
            if not coordination_approved:
                logger.info(f"🚫 Coordination blocked scaling: {coordination_message}")
                action = 1
                action_name = "no_action"
        
        # Store current experience for next iteration
        experience_store.add(
            agent_key,
            current_state,
            action,
            reward,
            None,
            log_prob,
            value,
            metrics  # Store metrics as prev_metrics for next iteration
        )
        
        # Publish agent state
        coordinator.publish_agent_state(agent_key, {
            'replicas': current_replicas,
            'cpu_usage': metrics.get('cpu_usage', 0),
            'last_action': action_name,
            'confidence': confidence,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        
        # FIXED: Enhanced response with detailed info
        response = {
            "success": True,
            "action": action,
            "action_name": action_name,
            "confidence": confidence,  # FIXED: Real confidence
            "epsilon": 0.0,
            "reward": reward,
            "value_estimate": value,
            "action_probabilities": action_probs.tolist(),  # NEW
            "coordination_approved": coordination_approved,
            "coordination_message": coordination_message,
            "buffer_size": len(agent.buffer),
            "experience_count": experience_store.get_count(agent_key),
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
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error in predict: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/save_model', methods=['POST'])
def save_model():
    """Save agent model to disk"""
    try:
        data = request.json
        deployment_name = data.get('deployment_name')
        namespace = data.get('namespace', 'default')
        
        agent_key = f"{namespace}/{deployment_name}"
        
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
    """Get statistics for all agents"""
    try:
        stats = {}
        cluster_status = coordinator.get_cluster_status()
        
        for agent_key, agent in agents.items():
            stats[agent_key] = agent.get_metrics()
            stats[agent_key]['experience_count'] = experience_store.get_count(agent_key)
            stats[agent_key]['buffer_size'] = len(agent.buffer)
            stats[agent_key]['training_steps'] = agent.training_steps
        
        return jsonify({
            "agents": stats,
            "total_agents": len(agents),
            "cluster_status": cluster_status,
            "redis_available": REDIS_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/reset_agent', methods=['POST'])
def reset_agent():
    """Reset an agent"""
    try:
        data = request.json
        deployment_name = data.get('deployment_name')
        namespace = data.get('namespace', 'default')
        
        agent_key = f"{namespace}/{deployment_name}"
        
        with agents_lock:
            if agent_key in agents:
                del agents[agent_key]
                
                if agent_key in experience_store.experiences:
                    del experience_store.experiences[agent_key]
                
                if REDIS_AVAILABLE:
                    try:
                        redis_client.delete(f"agent_state:{agent_key}")
                    except Exception as e:
                        logger.warning(f"Failed to clear Redis: {e}")
                
                logger.info(f"♻️ Agent reset: {agent_key}")
                
                return jsonify({"status": "agent_reset", "agent_key": agent_key})
            else:
                return jsonify({"error": "Agent not found"}), 404
                
    except Exception as e:
        logger.error(f"❌ Error resetting agent: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/cluster_status', methods=['GET'])
def cluster_status():
    """Get cluster-wide coordination status"""
    try:
        status = coordinator.get_cluster_status()
        
        intents = []
        if REDIS_AVAILABLE:
            try:
                intent_keys = redis_client.keys("scaling_intent:*")
                for key in intent_keys:
                    intent_data = redis_client.get(key)
                    if intent_data:
                        intents.append(json.loads(intent_data))
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


# ======================== STARTUP ========================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Starting Advanced RL Agent API Service")
    logger.info("=" * 60)
    logger.info(f"📁 Models directory: {MODEL_DIR}")
    logger.info(f"🔌 Redis: {REDIS_HOST}:{REDIS_PORT} ({'✅ Connected' if REDIS_AVAILABLE else '❌ Unavailable'})")
    logger.info(f"🤖 Algorithm: PPO (Proximal Policy Optimization)")
    logger.info(f"📊 State size: {STATE_SIZE} dimensions")
    logger.info(f"🎯 Action size: {ACTION_SIZE} actions")
    logger.info(f"🔄 Multi-agent coordination: {'✅ Enabled' if REDIS_AVAILABLE else '❌ Disabled'}")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)