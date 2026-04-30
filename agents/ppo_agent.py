import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import logging
import os

logger = logging.getLogger(__name__)

# Action space mapping: 0 = scale down, 1 = no action, 2 = scale up
ACTION_MAP = {0: "scale_down", 1: "no_action", 2: "scale_up"}

def action_to_string(action):
    return ACTION_MAP.get(action, "unknown")


class RunningNormaliser:
    """
    Online reward normaliser using exponential moving average.
    Keeps rewards in a stable range [-3, 3] to prevent value loss explosion.
    """
    def __init__(self, alpha=0.05):
        self.mean  = 0.0      # running mean
        self.var   = 1.0      # running variance
        self.alpha = alpha    # update rate (0.05 = smooth)

    def update(self, r):
        # Update mean and variance with exponential moving average
        self.mean = (1 - self.alpha) * self.mean + self.alpha * r
        self.var  = (1 - self.alpha) * self.var  + self.alpha * (r - self.mean) ** 2

    def normalise(self, r):
        # Standardise and clip to [-3, 3]
        std = max(float(np.sqrt(self.var)), 1.0)      # avoid division by zero
        return float(np.clip((r - self.mean) / std, -3.0, 3.0))


class ActorCritic(nn.Module):
    """
    Shared trunk Actor‑Critic network.
    - Trunk: two hidden layers of 128 neurons with Tanh activation.
    - Actor head: outputs logits for 3 actions (scale down, no action, scale up).
    - Critic head: outputs state value V(s).
    """
    def __init__(self, state_size=18, action_size=3, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, action_size)   # policy head
        self.critic = nn.Linear(hidden, 1)             # value head

        # Orthogonal initialisation (standard PPO practice)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, state_t):
        logits, value = self.forward(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(self, states_t, actions_t):
        logits, values = self.forward(states_t)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions_t)
        entropy = dist.entropy()          # exploration bonus
        return log_prob, values, entropy


class RolloutBuffer:
    """
    Circular buffer that stores the last 'capacity' transitions.
    Used for online PPO updates (rolling window, not episodic).
    """
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.state_size = state_size
        self.clear()

    def add(self, state, action, norm_reward, log_prob, value, done):
        i = self.ptr % self.capacity
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = norm_reward
        self.log_probs[i] = float(log_prob) if hasattr(log_prob, 'item') else log_prob
        self.values[i] = float(value) if hasattr(value, 'item') else value
        self.dones[i] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def clear(self):
        s = self.capacity
        self.states = np.zeros((s, self.state_size), dtype=np.float32)
        self.actions = np.zeros(s, dtype=np.int64)
        self.rewards = np.zeros(s, dtype=np.float32)
        self.log_probs = np.zeros(s, dtype=np.float32)
        self.values = np.zeros(s, dtype=np.float32)
        self.dones = np.zeros(s, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def get(self):
        idx = slice(0, self.size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.log_probs[idx], self.values[idx], self.dones[idx])

    def __len__(self):
        return self.size


class PPOAgent:
    """
    Proximal Policy Optimisation (PPO) agent for Kubernetes autoscaling.
    Features:
    - Online learning: updates every 32 steps (buffer full)
    - GAE advantage estimation
    - Clipped surrogate objective
    - Rule prior for safe initial behaviour
    - Multi‑objective reward (latency, cost, action penalties, incentives)
    """
    def __init__(self, deployment_name, namespace="default",
                 state_size=18, action_size=3):

        self.deployment_name = deployment_name
        self.namespace = namespace
        self.state_size = state_size
        self.action_size = action_size
        self.sla_target = 0.5      # 500 ms latency SLA

        # PPO hyperparameters – can be overridden via environment variables
        self.gamma = float(os.getenv("PPO_GAMMA", "0.99"))                     # discount factor
        self.gae_lambda = float(os.getenv("PPO_GAE_LAMBDA", "0.95"))           # GAE smoothing
        self.clip_epsilon = float(os.getenv("PPO_CLIP_EPSILON", "0.2"))        # PPO clip range
        self.c1 = float(os.getenv("PPO_VALUE_LOSS_COEF", "0.1"))               # value loss coefficient
        self.c2 = float(os.getenv("PPO_ENTROPY_COEF", "0.05"))                 # entropy bonus
        self.lr = float(os.getenv("PPO_LEARNING_RATE", "1e-4"))                # Adam learning rate
        self.epochs = int(os.getenv("PPO_EPOCHS", "2"))                        # PPO update epochs
        self.mini_batch = int(os.getenv("PPO_MINI_BATCH", "8"))                # mini‑batch size
        self.buffer_capacity = int(os.getenv("PPO_BUFFER_CAPACITY", "32"))     # rollout buffer size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimiser = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        self.buffer = RolloutBuffer(self.buffer_capacity, state_size)
        self.normaliser = RunningNormaliser()
        self.training_steps = 0
        self.episode_rewards = []
        self._step_counter = 0
        self.batch_size = self.buffer_capacity   # alias used by app.py

        logger.info(f"✅ PPO Agent initialised for {deployment_name} on {self.device}")
        logger.info(f"📊 Online PPO | buffer={self.buffer_capacity} | lr={self.lr} | "
                    f"c1={self.c1} | c2={self.c2} | epochs={self.epochs} | gamma={self.gamma}")

    # ----------------------------------------------------------------------
    # State construction
    # ----------------------------------------------------------------------
    def get_state(self, metrics):
        """
        Convert Prometheus metrics into an 18‑dimensional state vector.
        Normalisation applied: request_rate/100, replicas/10, pod counts/10,
        hour/24, day_of_week/7, request_trend/50.
        The last dimension is latency violation (max(0, latency - SLA)).
        """
        return np.array([
            metrics.get('cpu_usage', 0.0),
            metrics.get('memory_usage', 0.0),
            metrics.get('request_rate', 0.0) / 100.0,
            metrics.get('latency_p50', 0.0),
            metrics.get('latency_p95', 0.0),
            metrics.get('latency_p99', 0.0),
            metrics.get('replicas', 1) / 10.0,
            metrics.get('error_rate', 0.0),
            metrics.get('pod_pending', 0) / 10.0,
            metrics.get('pod_ready', 1) / 10.0,
            metrics.get('cpu_trend_1m', 0.0),
            metrics.get('cpu_trend_5m', 0.0),
            metrics.get('request_trend', 0.0) / 50.0,
            metrics.get('hour', 0) / 24.0,
            metrics.get('day_of_week', 0) / 7.0,
            float(metrics.get('is_weekend', False)),
            float(metrics.get('is_peak_hour', False)),
            max(0.0, metrics.get('latency_p95', 0.0) - self.sla_target),
        ], dtype=np.float32)

    # ----------------------------------------------------------------------
    # Action selection with rule prior
    # ----------------------------------------------------------------------
    def select_action(self, state, deterministic=False):
        """
        Choose an action.
        - If rule prior condition is met, override neural network output.
        - Otherwise, sample from the policy (stochastic during training,
          deterministic during evaluation).
        Returns (action, log_prob, value_estimate).
        """
        rule_action = self._rule_prior(state)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.policy(t)
            dist = Categorical(logits=logits)

            if rule_action is not None:
                action = rule_action
                act_t = torch.tensor([action], device=self.device)
                log_prob = dist.log_prob(act_t)
            elif deterministic:
                action = int(logits.argmax(dim=1).item())
                act_t = torch.tensor([action], device=self.device)
                log_prob = dist.log_prob(act_t)
            else:
                act_t = dist.sample()
                action = int(act_t.item())
                log_prob = dist.log_prob(act_t)

        return action, log_prob, value.item()

    def calculate_confidence(self, state):
        """
        Compute confidence = 1 - (entropy / max_entropy).
        Higher confidence means the policy is more deterministic.
        """
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, _ = self.policy(t)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
        max_entropy = np.log(self.action_size)
        confidence = float(1.0 - entropy / max_entropy)
        return max(0.0, min(1.0, confidence)), probs

    # ----------------------------------------------------------------------
    # Reward and learning loop
    # ----------------------------------------------------------------------
    def calculate_reward(self, metrics, action, prev_metrics):
        raw = self._raw_reward(metrics, action, prev_metrics)
        self.normaliser.update(raw)    # update running statistics
        return raw                      # return raw reward for logging

    def store_transition(self, state, action, reward, next_state,
                         done, log_prob, value):
        norm_r = self.normaliser.normalise(reward)   # normalise before buffer
        lp = float(log_prob) if hasattr(log_prob, 'item') else (log_prob or 0.0)
        self.buffer.add(state, action, norm_r, lp, value, done)
        self._step_counter += 1

    def update(self):
        """
        PPO update on the current buffer contents.
        Steps:
        1. Compute GAE advantages and returns.
        2. Normalise advantages.
        3. Perform multiple epochs of mini‑batch SGD on the clipped surrogate loss.
        """
        if len(self.buffer) < self.mini_batch:
            return None

        states, actions, rewards, old_log_probs, old_values, dones = self.buffer.get()

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_lp_t = torch.FloatTensor(old_log_probs).to(self.device)
        old_val_t = torch.FloatTensor(old_values).to(self.device)

        # GAE advantages
        advantages, returns = self._gae(rewards, old_values, dones)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)

        # Normalise advantages (standard PPO trick)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(states)
        total_pl = total_vl = total_ent = 0.0
        n_updates = 0

        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.mini_batch):
                idx = perm[start:start + self.mini_batch]
                if len(idx) < 2:
                    continue

                log_probs, values, entropy = self.policy.evaluate(states_t[idx], actions_t[idx])

                # Clipped surrogate objective
                ratios = torch.exp(log_probs - old_lp_t[idx])
                surr1 = ratios * adv_t[idx]
                surr2 = torch.clamp(ratios,
                                    1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * adv_t[idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.c1 * nn.MSELoss()(values, ret_t[idx])
                entropy_loss = -self.c2 * entropy.mean()

                loss = policy_loss + value_loss + entropy_loss
                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimiser.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        self.training_steps += 1

        stats = {
            'policy_loss': total_pl / max(n_updates, 1),
            'value_loss': total_vl / max(n_updates, 1),
            'entropy': total_ent / max(n_updates, 1),
            'training_steps': self.training_steps,
            'avg_reward': float(np.mean(rewards)),
            'buffer_size': len(self.buffer),
        }
        logger.info(
            f"🎓 Training Step {self.training_steps} | "
            f"Policy Loss: {stats['policy_loss']:.4f} | "
            f"Value Loss: {stats['value_loss']:.4f} | "
            f"Avg Reward: {stats['avg_reward']:.2f} | "
            f"Entropy: {stats['entropy']:.3f}"
        )
        return stats

    # ----------------------------------------------------------------------
    # Pre‑training with synthetic data
    # ----------------------------------------------------------------------
    def pretrain(self, n_steps=300):
        """
        Synthetic pre‑training to bootstrap the policy before live deployment.
        Uses 8 scenarios that represent typical Kubernetes scaling situations.
        """
        logger.info(f"🏋️ Starting PPO pre-training for {n_steps} steps...")
        scenarios = [
            (0.02, 0.01, 50.0, 0.80, 1),   # stressed 1 pod → scale_up
            (0.02, 0.01, 50.0, 0.75, 2),   # stressed 2 pods → scale_up
            (0.02, 0.01, 50.0, 0.70, 3),   # stressed 3 pods → no_action
            (0.02, 0.01, 50.0, 0.72, 4),   # over-scaled → no_action
            (0.02, 0.01, 50.0, 0.72, 5),   # over-scaled → scale_down
            (0.01, 0.01, 10.0, 0.10, 3),   # idle 3 pods → scale_down
            (0.01, 0.01, 10.0, 0.08, 2),   # idle 2 pods → scale_down
            (0.02, 0.01, 30.0, 0.30, 1),   # stable → no_action
        ]
        train_calls = 0
        for step in range(n_steps):
            cpu, mem, rr, lat, reps = scenarios[step % len(scenarios)]
            metrics = {
                'cpu_usage': float(np.clip(cpu + np.random.normal(0, 0.003), 0, 1)),
                'memory_usage': float(np.clip(mem + np.random.normal(0, 0.001), 0, 1)),
                'request_rate': float(max(0, rr + np.random.normal(0, 5))),
                'latency_p50': float(max(0, lat * 0.7)),
                'latency_p95': float(max(0, lat + np.random.normal(0, 0.03))),
                'latency_p99': float(max(0, lat * 1.3)),
                'replicas': reps,
                'error_rate': 0.0,
                'pod_pending': 0,
                'pod_ready': reps,
                'cpu_trend_1m': float(np.random.normal(0, 0.001)),
                'cpu_trend_5m': float(np.random.normal(0, 0.001)),
                'request_trend': 0.0,
                'hour': 12,
                'day_of_week': 2,
                'is_weekend': False,
                'is_peak_hour': False,
            }
            state = self.get_state(metrics)
            action, log_prob, value = self.select_action(state)
            next_metrics = dict(metrics)
            # Simulate effect of action
            if action == 2:   # scale_up
                next_metrics['replicas'] = reps + 1
                next_metrics['latency_p95'] = max(0, lat - 0.05)
            elif action == 0: # scale_down
                next_metrics['replicas'] = max(1, reps - 1)
                next_metrics['latency_p95'] = lat + 0.03
            reward = self._raw_reward(next_metrics, action, metrics)
            self.normaliser.update(reward)
            norm_r = self.normaliser.normalise(reward)
            lp = float(log_prob) if hasattr(log_prob, 'item') else (log_prob or 0.0)
            self.buffer.add(state, action, norm_r, lp, value, done=False)
            if len(self.buffer) >= self.mini_batch:
                if self.update():
                    train_calls += 1
        logger.info(f"✅ PPO pre-training complete — {train_calls} updates over {n_steps} steps")
        return train_calls

    # ----------------------------------------------------------------------
    # Model persistence
    # ----------------------------------------------------------------------
    def save_model(self, filepath):
        torch.save({
            'policy': self.policy.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
            'norm_mean': self.normaliser.mean,
            'norm_var': self.normaliser.var,
        }, filepath)
        logger.info(f"💾 Model saved to {filepath}")

    def load_model(self, filepath):
        ckpt = torch.load(filepath, map_location=self.device)
        # Backward compatibility for older checkpoints
        policy_state = (ckpt.get('policy')
                        or ckpt.get('policy_net')
                        or ckpt.get('policy_state_dict'))
        if policy_state is None:
            raise KeyError("No recognised policy key in checkpoint")
        self.policy.load_state_dict(policy_state)
        opt_state = ckpt.get('optimiser') or ckpt.get('optimizer')
        if opt_state:
            try:
                self.optimiser.load_state_dict(opt_state)
            except Exception:
                pass   # non‑fatal
        self.training_steps = ckpt.get('training_steps', 0)
        self.episode_rewards = ckpt.get('episode_rewards', [])
        self.normaliser.mean = ckpt.get('norm_mean', 0.0)
        self.normaliser.var = ckpt.get('norm_var', 1.0)
        logger.info(f"📂 Model loaded from {filepath}")

    def get_metrics(self):
        return {
            'training_steps': self.training_steps,
            'avg_reward_100': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
            'buffer_size': len(self.buffer),
            'device': str(self.device),
        }

    # ---------- Internal helpers ----------

    def _rule_prior(self, state):
        """
        Hard‑coded safety rules that override the neural network.
        These guarantee correct behaviour from day 1, before the policy has converged.
        """
        latency = float(state[4])
        replicas = round(float(state[6]) * 10)
        violation = float(state[17])
        request_rate = float(state[2]) * 100

        # Scale up when single pod and high traffic
        if replicas == 1 and request_rate > 50:
            logger.info(f"⚡ Rule prior: forcing scale_up (request_rate={request_rate:.1f} > 50)")
            return 2
        # Do nothing when idle and at min replicas (prevents scale‑down spam)
        if request_rate == 0 and replicas == 1:
            logger.info("⚡ Rule prior: forcing no_action (idle, no traffic)")
            return 1
        # Scale up when latency exceeds SLA with few replicas
        if replicas <= 1 and latency > self.sla_target:
            logger.info(f"⚡ Rule prior: forcing scale_up (latency={latency:.3f} > SLA)")
            return 2
        # Scale up when SLA violation is high and few replicas
        if violation > 0.2 and replicas <= 2:
            logger.info(f"⚡ Rule prior: forcing scale_up (SLA violation={violation:.3f})")
            return 2
        # Scale down when idle and over‑provisioned
        if latency < self.sla_target * 0.5 and replicas > 1:
            logger.info(f"⚡ Rule prior: forcing scale_down (idle, replicas={replicas})")
            return 0
        return None

    def _raw_reward(self, metrics, action, prev_metrics):
        """
        Multi‑objective reward function (normalised later, raw value logged).
        Components:
        - SLA compliance: +5 per 0.1s under target, -50 per squared violation above SLA.
        - Cost penalty: -0.5 per replica.
        - Error penalty: -100 per error rate.
        - Over‑scaling penalties (latency > SLA with many replicas, or idle with extra replicas).
        - Action‑specific shaping:
            * scale_up under low load: -25
            * scale_up when already >1 pod and low latency: -15
            * scale_down at min replicas: -30
            * scale_down under high load: -20
            * scale_down when over‑provisioned: +20
            * no_action penalty when traffic exists but no action: -5
        - Scaling incentive: +30 when single pod and request_rate > 30 RPS.
        - Transition bonuses: reward if scaling improved latency or reduced cost without harming latency.
        """
        latency = metrics.get('latency_p95', 0.0)
        replicas = metrics.get('replicas', 1)
        error_rate = metrics.get('error_rate', 0.0)
        cpu_usage = metrics.get('cpu_usage', 0.0)
        memory = metrics.get('memory_usage', 0.0)
        request_rate = metrics.get('request_rate', 0.0)

        if cpu_usage == 0.0 and memory == 0.0:
            return 0.0

        reward = 0.0

        # SLA compliance
        if latency > self.sla_target:
            reward -= 50.0 * ((latency - self.sla_target) ** 2)
        elif latency > 0:
            reward += 5.0 * (self.sla_target - latency)

        # Cost
        reward -= 0.5 * replicas

        # Errors
        reward -= 100.0 * error_rate

        # Over‑scaling penalties
        if replicas > 3 and latency > self.sla_target:
            reward -= 5.0 * (replicas - 3)
        if replicas > 1 and latency < self.sla_target * 0.8:
            reward -= 3.0 * (replicas - 1)

        # Action‑specific shaping
        if action == 2:  # scale_up
            if request_rate < 30 and latency < self.sla_target * 0.8:
                reward -= 25.0
                logger.debug(f"Penalty: scale_up under low load (req={request_rate:.1f})")
            elif replicas > 1 and latency < self.sla_target:
                reward -= 15.0
        elif action == 0:  # scale_down
            if replicas <= 1:
                reward -= 30.0
            elif latency > self.sla_target and replicas <= 3:
                reward -= 20.0
            elif replicas > 2 and latency < self.sla_target * 0.5:
                reward += 20.0
        elif action == 1:  # no_action
            if not (latency > self.sla_target) and replicas == 1:
                if request_rate < 30:
                    reward += 8.0
                else:
                    reward -= 5.0   # penalise inaction under load
            elif latency > self.sla_target and replicas <= 2:
                reward -= 20.0
            elif replicas > 2 and latency < self.sla_target * 0.6:
                reward -= 8.0

        # Extra incentive to scale up under real load
        if replicas == 1 and request_rate > 30:
            reward += 30.0
            logger.debug(f"Scaling incentive: replicas=1, request_rate={request_rate:.1f} -> +30")

        # Transition‑based rewards (if previous metrics exist)
        if prev_metrics:
            prev_latency = prev_metrics.get('latency_p95', 0.0)
            prev_replicas = prev_metrics.get('replicas', 1)
            if action == 2 and replicas > prev_replicas and latency < prev_latency:
                reward += 15.0   # successful scale‑up
            if action == 0 and replicas < prev_replicas and latency <= prev_latency + 0.05:
                reward += 10.0   # safe scale‑down

        return reward

    def _gae(self, rewards, values, dones):
        """
        Generalised Advantage Estimation (GAE).
        Returns (advantages, returns).
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(n)):
            next_val = 0.0 if t == n - 1 else values[t + 1]
            delta = (rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t])
            advantages[t] = last_adv = (delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_adv)
        returns = advantages + values
        return advantages, returns