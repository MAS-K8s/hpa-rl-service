import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StateEncoder(nn.Module):
    """Advanced state encoder with LSTM for temporal patterns"""
    
    def __init__(self, input_size=18, hidden_size=128, num_layers=2):
        super(StateEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.fc = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out[:, -1, :])
        return torch.relu(out), hidden


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_size=18, action_size=3, hidden_size=128):
        super(ActorCritic, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        features = self.encoder(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, state_value, entropy


class PPOAgent:
    """Fixed PPO Agent with proper training"""
    
    def __init__(self, deployment_name, namespace="default", 
                 state_size=18, action_size=3):
        
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.state_size = state_size
        self.action_size = action_size
        
        # PPO Hyperparameters (FIXED)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.c1 = 1.0
        self.c2 = 0.01
        self.lr = 3e-4
        self.epochs = 4  # Reduced from 10 for faster training
        self.batch_size = 32  # FIXED: Reduced from 64
        self.buffer_size = 256  # FIXED: Reduced from 2048
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Experience buffer
        self.buffer = PPOBuffer(self.buffer_size, state_size)
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_steps = 0
        
        # Reward shaping parameters (FIXED)
        self.sla_target = 0.5
        self.cost_per_replica = 0.5  # FIXED: Reduced from 1.0
        self.sla_violation_penalty = 100.0
        
        logger.info(f"✅ PPO Agent initialized for {deployment_name} on {self.device}")
        logger.info(f"📊 Batch size: {self.batch_size}, Buffer size: {self.buffer_size}")
    
    def get_state(self, metrics):
        """Convert metrics to enhanced state vector"""
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
    
    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                action_probs, value = self.policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()
                log_prob = None
            else:
                action, log_prob = self.policy.act(state_tensor)
                _, value = self.policy(state_tensor)
        
        return action, log_prob, value.item()
    
    def calculate_confidence(self, state):
        """Calculate policy confidence from entropy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = self.policy(state_tensor)
            action_probs = action_probs.squeeze()
            
            # Calculate entropy
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
            
            # Normalize to 0-1 (0 = random, 1 = deterministic)
            max_entropy = np.log(self.action_size)
            confidence = 1.0 - (entropy.item() / max_entropy)
            
        return confidence, action_probs.cpu().numpy()
    
    def calculate_reward(self, metrics, action, prev_metrics):
        """FIXED: Improved reward function with balanced penalties"""
        
        latency = metrics.get('latency_p95', 0.0)
        error_rate = metrics.get('error_rate', 0.0)
        replicas = metrics.get('replicas', 1)
        cpu_usage = metrics.get('cpu_usage', 0.0)
        pod_pending = metrics.get('pod_pending', 0)
        pod_ready = metrics.get('pod_ready', 1)
        
        reward = 0.0
        
        # 1. SLA Compliance (most important)
        if latency > self.sla_target:
            violation = latency - self.sla_target
            reward -= self.sla_violation_penalty * (violation ** 2)
        else:
            margin = self.sla_target - latency
            reward += 20.0 * margin  # FIXED: Increased bonus
        
        # 2. Cost optimization (FIXED: Less aggressive)
        reward -= self.cost_per_replica * replicas
        
        # 3. Error penalty
        reward -= 50.0 * error_rate
        
        # 4. Resource efficiency
        if 0.6 <= cpu_usage <= 0.75:
            reward += 15.0  # FIXED: Increased bonus
        elif cpu_usage > 0.9:
            reward -= 15.0 * (cpu_usage - 0.9)
        elif cpu_usage < 0.3:
            reward -= 5.0 * (0.3 - cpu_usage)
        
        # 5. Pending pods penalty (FIXED: MUCH LESS HARSH)
        if pod_pending > 0:
            if pod_pending > replicas * 0.3:  # Only if >30% pending
                reward -= 3.0 * pod_pending  # FIXED: Reduced from 10.0
        
        # 6. Readiness bonus (NEW)
        ready_ratio = pod_ready / max(replicas, 1)
        if ready_ratio > 0.8:
            reward += 5.0
        
        # 7. Stability
        if prev_metrics:
            prev_replicas = prev_metrics.get('replicas', replicas)
            if action != 1:
                if replicas == prev_replicas:
                    reward -= 2.0
                else:
                    reward -= 0.5  # FIXED: Small penalty
            else:
                if 0.5 <= cpu_usage <= 0.8 and latency < self.sla_target:
                    reward += 5.0  # FIXED: Increased stability bonus
        
        # 8. Proactive scaling
        cpu_trend = metrics.get('cpu_trend_1m', 0.0)
        if cpu_trend > 0.1 and action == 2:
            reward += 8.0  # FIXED: Increased from 5.0
        elif cpu_trend < -0.1 and action == 0:
            reward += 5.0  # FIXED: Increased from 3.0
        
        return reward
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store experience in buffer"""
        self.buffer.add(state, action, reward, next_state, done, log_prob, value)
        
        # Log buffer status periodically
        if len(self.buffer) % 10 == 0:
            logger.debug(f"📦 Buffer: {len(self.buffer)}/{self.buffer_size}")
    
    def update(self):
        """FIXED: Update policy with better logging"""
        if len(self.buffer) < self.batch_size:
            logger.warning(f"⚠️ Buffer too small: {len(self.buffer)}/{self.batch_size}")
            return None
        
        # Get all experiences from buffer
        states, actions, rewards, next_states, dones, old_log_probs, old_values = self.buffer.get()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        old_values = torch.FloatTensor(old_values).to(self.device)
        
        # Compute advantages using GAE
        advantages, returns = self.compute_gae(rewards, old_values.cpu().numpy(), dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for K epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.epochs):
            # Evaluate actions
            log_probs, state_values, entropy = self.policy.evaluate(states, actions)
            state_values = state_values.squeeze()
            
            # Compute ratio
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Total loss
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.c1 * nn.MSELoss()(state_values, returns)
            entropy_loss = -self.c2 * entropy.mean()
            
            loss = policy_loss + value_loss + entropy_loss
            
            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # Clear buffer
        self.buffer.clear()
        
        self.training_steps += 1
        
        # Calculate average reward from recent experiences
        avg_reward = np.mean(rewards)
        
        stats = {
            'policy_loss': total_policy_loss / self.epochs,
            'value_loss': total_value_loss / self.epochs,
            'entropy': total_entropy / self.epochs,
            'training_steps': self.training_steps,
            'avg_reward': avg_reward,
            'buffer_size': len(self.buffer)
        }
        
        # FIXED: Better logging
        logger.info(
            f"🎓 Training Step {self.training_steps} | "
            f"Policy Loss: {stats['policy_loss']:.4f} | "
            f"Value Loss: {stats['value_loss']:.4f} | "
            f"Avg Reward: {stats['avg_reward']:.2f} | "
            f"Entropy: {stats['entropy']:.3f}"
        )
        
        return stats
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
        }, filepath)
        logger.info(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint.get('training_steps', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        logger.info(f"📂 Model loaded from {filepath}")
    
    def get_metrics(self):
        """Get training metrics"""
        return {
            'training_steps': self.training_steps,
            'avg_reward_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'buffer_size': len(self.buffer),
            'device': str(self.device)
        }


class PPOBuffer:
    """Experience replay buffer for PPO"""
    
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done, log_prob, value):
        """Add experience to buffer"""
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.log_probs[idx] = log_prob if log_prob is not None else 0
        self.values[idx] = value
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get(self):
        """Get all experiences"""
        idx = slice(0, self.size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.log_probs[idx],
            self.values[idx]
        )
    
    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size


# Action mapping
ACTION_MAP = {
    0: "scale_down",
    1: "no_action",
    2: "scale_up"
}

def action_to_string(action):
    return ACTION_MAP.get(action, "unknown")