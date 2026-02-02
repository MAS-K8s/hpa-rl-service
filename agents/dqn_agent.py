import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DQNetwork(nn.Module):
    """Deep Q-Network for decision making"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Fixed DQN Agent for Kubernetes Autoscaling
    Simpler alternative to PPO, good for discrete actions
    """
    
    def __init__(self, deployment_name, namespace="default",
                 state_size=18, action_size=3):
        
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.state_size = state_size
        self.action_size = action_size
        
        # DQN Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 10  # Update target network every N steps
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Training metrics
        self.training_steps = 0
        self.episode_rewards = []
        self.losses = []
        
        # Reward parameters
        self.sla_target = 0.5
        self.cost_per_replica = 1.0
        
        logger.info(f"✅ DQN Agent initialized for {deployment_name} on {self.device}")
    
    def get_state(self, metrics):
        """Convert metrics to state vector (same as PPO)"""
        state = np.array([
            metrics.get('cpu_usage', 0.0),
            metrics.get('memory_usage', 0.0),
            metrics.get('request_rate', 0.0) / 1000.0,
            metrics.get('latency_p50', 0.0),
            metrics.get('latency_p95', 0.0),
            metrics.get('latency_p99', 0.0),
            metrics.get('replicas', 1) / 10.0,
            metrics.get('error_rate', 0.0),
            metrics.get('pod_pending', 0) / 10.0,
            metrics.get('pod_ready', 1) / 10.0,
            metrics.get('cpu_trend_1m', 0.0),
            metrics.get('cpu_trend_5m', 0.0),
            metrics.get('request_trend', 0.0) / 100.0,
            metrics.get('hour', 0) / 24.0,
            metrics.get('day_of_week', 0) / 7.0,
            float(metrics.get('is_weekend', False)),
            float(metrics.get('is_peak_hour', False)),
            max(0, metrics.get('latency_p95', 0.0) - self.sla_target),
        ], dtype=np.float32)
        
        return state
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state vector
            training: If True, use epsilon-greedy; if False, always exploit
        
        Returns:
            action: Selected action (0, 1, or 2)
            q_values: Q-values for all actions
        """
        # Exploitation (use learned policy)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]
        
        if training and random.random() < self.epsilon:
            # Exploration (random action)
            action = random.randrange(self.action_size)
        else:
            # Exploitation (best action)
            action = np.argmax(q_values_np)
        
        return action, q_values_np.tolist()
    
    def calculate_reward(self, metrics, action, prev_metrics):
        """
        Reward function (same as PPO for consistency)
        """
        latency = metrics.get('latency_p95', 0.0)
        error_rate = metrics.get('error_rate', 0.0)
        replicas = metrics.get('replicas', 1)
        cpu_usage = metrics.get('cpu_usage', 0.0)
        pod_pending = metrics.get('pod_pending', 0)
        
        reward = 0.0
        
        # SLA compliance
        if latency > self.sla_target:
            sla_violation = latency - self.sla_target
            reward -= 100.0 * (sla_violation ** 2)
        else:
            margin = self.sla_target - latency
            reward += 10.0 * margin
        
        # Cost optimization
        reward -= self.cost_per_replica * replicas
        
        # Error penalty
        reward -= 50.0 * error_rate
        
        # Resource efficiency
        if 0.6 <= cpu_usage <= 0.75:
            reward += 10.0
        elif cpu_usage > 0.9:
            reward -= 15.0 * (cpu_usage - 0.9)
        elif cpu_usage < 0.3:
            reward -= 5.0 * (0.3 - cpu_usage)
        
        # Pending pods penalty
        if pod_pending > 0:
            reward -= 10.0 * pod_pending
        
        # Stability
        if prev_metrics:
            prev_replicas = prev_metrics.get('replicas', replicas)
            if action != 1:
                if replicas == prev_replicas:
                    reward -= 2.0
                else:
                    reward -= 1.0
            else:
                if 0.5 <= cpu_usage <= 0.8 and latency < self.sla_target:
                    reward += 2.0
        
        # Proactive scaling
        cpu_trend = metrics.get('cpu_trend_1m', 0.0)
        if cpu_trend > 0.1 and action == 2:
            reward += 5.0
        elif cpu_trend < -0.1 and action == 0:
            reward += 3.0
        
        return reward
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self):
        """
        Train the network using experience replay
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()
        
        self.losses.append(loss.item())
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.info(f"🔄 Target network updated at step {self.training_steps}")
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
        }, filepath)
        logger.info(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint.get('training_steps', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        logger.info(f"📂 Model loaded from {filepath}")
    
    def get_metrics(self):
        """Get training metrics"""
        return {
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'avg_loss_100': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_reward_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'buffer_size': len(self.replay_buffer),
            'device': str(self.device)
        }


# Action mapping
ACTION_MAP = {
    0: "scale_down",
    1: "no_action",
    2: "scale_up"
}

def action_to_string(action):
    return ACTION_MAP.get(action, "unknown")