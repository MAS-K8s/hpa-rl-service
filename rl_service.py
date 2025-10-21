from flask import Flask, request, jsonify
import logging
import os
import json
from datetime import datetime
import numpy as np

# Import our RL agent
from rl_agent import RLAgent, action_to_string

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store agents for different deployments
agents = {}
previous_metrics = {}

# Configuration
STATE_SIZE = 8
ACTION_SIZE = 3
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
MODEL_DIR = "./models"

os.makedirs(MODEL_DIR, exist_ok=True)


def get_or_create_agent(deployment_name, namespace="default"):
    """Get existing agent or create new one"""
    agent_key = f"{namespace}/{deployment_name}"
    
    if agent_key not in agents:
        logger.info(f"🆕 Creating new RL agent for {agent_key}")
        agent = RLAgent(
            deployment_name=deployment_name,
            namespace=namespace,
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            redis_host=REDIS_HOST
        )
        
        # Try to load existing model
        model_path = os.path.join(MODEL_DIR, f"{agent_key.replace('/', '_')}.pt")
        if os.path.exists(model_path):
            try:
                agent.load_model(model_path)
                logger.info(f"📂 Loaded existing model for {agent_key}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load model: {e}")
        
        agents[agent_key] = agent
        previous_metrics[agent_key] = {}
    
    return agents[agent_key]


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "active_agents": len(agents),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Receives metrics from Go controller and returns action
    """
    try:
        data = request.json
        deployment_name = data.get('deployment_name')
        namespace = data.get('namespace', 'default')
        metrics = data.get('metrics', {})
        training_mode = data.get('training_mode', False)
        
        if not deployment_name:
            return jsonify({"error": "deployment_name required"}), 400
        
        # Get or create agent
        agent = get_or_create_agent(deployment_name, namespace)
        agent_key = f"{namespace}/{deployment_name}"
        
        # Convert metrics to state
        state = agent.get_state(metrics)
        
        # Choose action
        action = agent.choose_action(state, training=training_mode)
        action_name = action_to_string(action)
        
        # Calculate confidence (inverse of epsilon for exploitation)
        confidence = 1.0 - agent.epsilon if training_mode else 1.0
        
        response = {
            "action": action,
            "action_name": action_name,
            "confidence": confidence,
            "epsilon": agent.epsilon
        }
        
        # Training mode: calculate reward and train
        if training_mode and agent_key in previous_metrics:
            prev_metrics = previous_metrics[agent_key]
            
            if prev_metrics:  # If we have previous state
                prev_state = agent.get_state(prev_metrics)
                reward = agent.calculate_reward(metrics, action, prev_metrics)
                
                # Store experience
                done = False  # Episode never really ends in continuous control
                agent.remember(prev_state, action, reward, state, done)
                
                # Train
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.replay()
                    if loss:
                        logger.info(f"🎓 Training loss: {loss:.4f}, Reward: {reward:.2f}")
                
                response["reward"] = reward
        
        # Store current metrics for next iteration
        previous_metrics[agent_key] = metrics
        
        logger.info(f"🎯 Action: {action_name} (confidence: {confidence:.2f}) for {agent_key}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error in predict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/train', methods=['POST'])
def train_episode():
    """
    Manual training endpoint for batch training
    """
    try:
        data = request.json
        deployment_name = data.get('deployment_name')
        namespace = data.get('namespace', 'default')
        episodes = data.get('episodes', 10)
        
        agent = get_or_create_agent(deployment_name, namespace)
        
        losses = []
        for _ in range(episodes):
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                if loss:
                    losses.append(loss)
        
        avg_loss = np.mean(losses) if losses else 0
        
        return jsonify({
            "status": "training_complete",
            "episodes": episodes,
            "average_loss": float(avg_loss),
            "memory_size": len(agent.memory)
        })
        
    except Exception as e:
        logger.error(f"❌ Error in training: {e}")
        return jsonify({"error": str(e)}), 500


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
            "epsilon": agent.epsilon,
            "memory_size": len(agent.memory)
        })
        
    except Exception as e:
        logger.error(f"❌ Error saving model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/update_target', methods=['POST'])
def update_target():
    """Update target network (should be called periodically)"""
    try:
        data = request.json
        deployment_name = data.get('deployment_name')
        namespace = data.get('namespace', 'default')
        
        agent_key = f"{namespace}/{deployment_name}"
        
        if agent_key not in agents:
            return jsonify({"error": "Agent not found"}), 404
        
        agent = agents[agent_key]
        agent.update_target_network()
        
        logger.info(f"🔄 Target network updated for {agent_key}")
        
        return jsonify({"status": "target_network_updated"})
        
    except Exception as e:
        logger.error(f"❌ Error updating target: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics for all agents"""
    try:
        stats = {}
        
        for agent_key, agent in agents.items():
            stats[agent_key] = {
                "epsilon": agent.epsilon,
                "memory_size": len(agent.memory),
                "device": str(agent.device)
            }
        
        return jsonify({
            "agents": stats,
            "total_agents": len(agents)
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/reset_agent', methods=['POST'])
def reset_agent():
    """Reset an agent (useful for retraining)"""
    try:
        data = request.json
        deployment_name = data.get('deployment_name')
        namespace = data.get('namespace', 'default')
        
        agent_key = f"{namespace}/{deployment_name}"
        
        if agent_key in agents:
            del agents[agent_key]
            if agent_key in previous_metrics:
                del previous_metrics[agent_key]
            
            logger.info(f"♻️ Agent reset: {agent_key}")
            
            return jsonify({"status": "agent_reset", "agent_key": agent_key})
        else:
            return jsonify({"error": "Agent not found"}), 404
            
    except Exception as e:
        logger.error(f"❌ Error resetting agent: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    logger.info("🚀 Starting RL Agent API Service")
    logger.info(f"📁 Models directory: {MODEL_DIR}")
    logger.info(f"🔌 Redis host: {REDIS_HOST}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)