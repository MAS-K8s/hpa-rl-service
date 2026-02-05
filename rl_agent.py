# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# import random
# import json
# import redis
# from datetime import datetime
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class DQNetwork(nn.Module):
#     """Deep Q-Network for decision making"""
#     def __init__(self, state_size, action_size):
#         super(DQNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, action_size)
        
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         return self.fc4(x)


# class RLAgent:
#     """Reinforcement Learning Agent for Kubernetes Autoscaling"""
    
#     def __init__(self, deployment_name, namespace="default", 
#                  state_size=8, action_size=3, redis_host="localhost"):
#         self.deployment_name = deployment_name
#         self.namespace = namespace
#         self.state_size = state_size
#         self.action_size = action_size  # 0: scale_down, 1: no_action, 2: scale_up
        
#         # Hyperparameters
#         self.gamma = 0.95  # Discount factor
#         self.epsilon = 1.0  # Exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.batch_size = 32
        
#         # Memory
#         self.memory = deque(maxlen=2000)
        
#         # Networks
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.policy_net = DQNetwork(state_size, action_size).to(self.device)
#         self.target_net = DQNetwork(state_size, action_size).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
#         # Redis for state sharing
#         self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        
#         logger.info(f"✅ RL Agent initialized for {deployment_name} on {self.device}")
    
#     def get_state(self, metrics):
#         """
#         Convert metrics to state vector
#         Metrics should contain: cpu_usage, memory_usage, request_rate, 
#                                 latency_p95, replicas, error_rate, pod_pending, time_of_day
#         """
#         state = np.array([
#             metrics.get('cpu_usage', 0.0),
#             metrics.get('memory_usage', 0.0),
#             metrics.get('request_rate', 0.0),
#             metrics.get('latency_p95', 0.0),
#             metrics.get('replicas', 1) / 10.0,  # Normalize
#             metrics.get('error_rate', 0.0),
#             metrics.get('pod_pending', 0),
#             (datetime.now().hour / 24.0)  # Time of day normalized
#         ], dtype=np.float32)
#         return state
    
#     def choose_action(self, state, training=True):
#         """Epsilon-greedy action selection"""
#         if training and random.random() < self.epsilon:
#             return random.randrange(self.action_size)
        
#         with torch.no_grad():
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             q_values = self.policy_net(state_tensor)
#             return q_values.argmax().item()
    
#     def calculate_reward(self, metrics, action, prev_metrics):
#         """
#         Reward function balancing performance and cost
        
#         Positive rewards:
#         - Low latency
#         - Low error rate
#         - Resource efficiency
        
#         Negative rewards:
#         - High latency
#         - High error rate
#         - Unnecessary scaling
#         """
#         latency = metrics.get('latency_p95', 0)
#         error_rate = metrics.get('error_rate', 0)
#         replicas = metrics.get('replicas', 1)
#         cpu_usage = metrics.get('cpu_usage', 0)
        
#         # Latency penalty (exponential if > 500ms)
#         if latency > 0.5:
#             latency_penalty = -10 * (latency - 0.5) ** 2
#         else:
#             latency_penalty = 1
        
#         # Error rate penalty
#         error_penalty = -50 * error_rate
        
#         # Cost penalty (encourage fewer replicas)
#         cost_penalty = -0.5 * replicas
        
#         # Efficiency bonus (high utilization but not overloaded)
#         if 0.5 <= cpu_usage <= 0.75:
#             efficiency_bonus = 5
#         elif cpu_usage > 0.9:
#             efficiency_bonus = -5  # Overloaded
#         else:
#             efficiency_bonus = 0
        
#         # Action stability (penalize frequent changes)
#         prev_replicas = prev_metrics.get('replicas', replicas)
#         if action != 1 and replicas == prev_replicas:  # Tried to scale but didn't change
#             stability_penalty = -2
#         else:
#             stability_penalty = 0
        
#         total_reward = (
#             latency_penalty + 
#             error_penalty + 
#             cost_penalty + 
#             efficiency_bonus + 
#             stability_penalty
#         )
        
#         return total_reward
    
#     def remember(self, state, action, reward, next_state, done):
#         """Store experience in memory"""
#         self.memory.append((state, action, reward, next_state, done))
    
#     def replay(self):
#         """Train the network using experience replay"""
#         if len(self.memory) < self.batch_size:
#             return
        
#         batch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
        
#         states = torch.FloatTensor(np.array(states)).to(self.device)
#         actions = torch.LongTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)
        
#         # Current Q values
#         current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
#         # Target Q values
#         next_q = self.target_net(next_states).max(1)[0].detach()
#         target_q = rewards + (1 - dones) * self.gamma * next_q
        
#         # Loss and optimization
#         loss = nn.MSELoss()(current_q.squeeze(), target_q)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
        
#         # Decay epsilon
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
        
#         return loss.item()
    
#     def update_target_network(self):
#         """Update target network with policy network weights"""
#         self.target_net.load_state_dict(self.policy_net.state_dict())
    
#     def save_model(self, filepath):
#         """Save model weights"""
#         torch.save({
#             'policy_net': self.policy_net.state_dict(),
#             'target_net': self.target_net.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#             'epsilon': self.epsilon
#         }, filepath)
#         logger.info(f"💾 Model saved to {filepath}")
    
#     def load_model(self, filepath):
#         """Load model weights"""
#         checkpoint = torch.load(filepath, map_location=self.device)
#         self.policy_net.load_state_dict(checkpoint['policy_net'])
#         self.target_net.load_state_dict(checkpoint['target_net'])
#         self.optimizer.load_state_dict(checkpoint['optimizer'])
#         self.epsilon = checkpoint['epsilon']
#         logger.info(f"📂 Model loaded from {filepath}")
    
#     def share_knowledge(self, other_agents_data):
#         """
#         Multi-agent coordination: Share and learn from other agents
#         """
#         # Store own metrics in Redis
#         own_data = {
#             'epsilon': self.epsilon,
#             'memory_size': len(self.memory),
#             'timestamp': datetime.now().isoformat()
#         }
#         self.redis_client.set(
#             f"agent:{self.deployment_name}", 
#             json.dumps(own_data)
#         )
        
#         # Learn from other agents (optional: federated learning approach)
#         # This is a simple example - can be enhanced with more sophisticated sharing
#         for agent_name, data in other_agents_data.items():
#             if agent_name != self.deployment_name:
#                 # Could implement model averaging or experience sharing here
#                 pass


# # Action mapping
# ACTION_MAP = {
#     0: "scale_down",
#     1: "no_action", 
#     2: "scale_up"
# }

# def action_to_string(action):
#     return ACTION_MAP.get(action, "unknown")