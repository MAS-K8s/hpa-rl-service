import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)

ACTION_MAP = {0: "scale_down", 1: "no_action", 2: "scale_up"}

def action_to_string(action):
    return ACTION_MAP.get(action, "unknown")


class RunningNormaliser:
    """Online reward normaliser — keeps rewards in a stable range."""
    def __init__(self, alpha=0.05):
        self.mean  = 0.0
        self.var   = 1.0
        self.alpha = alpha

    def update(self, r):
        self.mean = (1 - self.alpha) * self.mean + self.alpha * r
        self.var  = (1 - self.alpha) * self.var  + self.alpha * (r - self.mean) ** 2

    def normalise(self, r):
        std = max(float(np.sqrt(self.var)), 1.0)
        return float(np.clip((r - self.mean) / std, -3.0, 3.0))


class ActorCritic(nn.Module):
    """Shared-trunk Actor-Critic for PPO."""

    def __init__(self, state_size=18, action_size=3, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, action_size)
        self.critic = nn.Linear(hidden, 1)

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
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(self, states_t, actions_t):
        logits, values = self.forward(states_t)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(actions_t)
        entropy  = dist.entropy()
        return log_prob, values, entropy


class RolloutBuffer:
    def __init__(self, capacity, state_size):
        self.capacity   = capacity
        self.state_size = state_size
        self.clear()

    def add(self, state, action, norm_reward, log_prob, value, done):
        i = self.ptr % self.capacity
        self.states[i]   = state
        self.actions[i]  = action
        self.rewards[i]  = norm_reward
        self.log_probs[i]= float(log_prob) if hasattr(log_prob, 'item') else log_prob
        self.values[i]   = float(value)    if hasattr(value,    'item') else value
        self.dones[i]    = done
        self.ptr        += 1
        self.size        = min(self.size + 1, self.capacity)

    def clear(self):
        s = self.capacity
        self.states   = np.zeros((s, self.state_size), dtype=np.float32)
        self.actions  = np.zeros(s, dtype=np.int64)
        self.rewards  = np.zeros(s, dtype=np.float32)
        self.log_probs= np.zeros(s, dtype=np.float32)
        self.values   = np.zeros(s, dtype=np.float32)
        self.dones    = np.zeros(s, dtype=np.float32)
        self.ptr      = 0
        self.size     = 0

    def get(self):
        idx = slice(0, self.size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.log_probs[idx], self.values[idx], self.dones[idx])

    def __len__(self):
        return self.size


class PPOAgent:
    def __init__(self, deployment_name, namespace="default",
                 state_size=18, action_size=3):

        self.deployment_name = deployment_name
        self.namespace       = namespace
        self.state_size      = state_size
        self.action_size     = action_size
        self.sla_target      = 0.5   # 500 ms

        # PPO hyper-parameters
        self.gamma          = 0.99
        self.gae_lambda     = 0.95
        self.clip_epsilon   = 0.2
        self.c1             = 0.1
        self.c2             = 0.05
        self.lr             = 1e-4
        self.epochs         = 2
        self.mini_batch     = 8
        self.train_every    = 1
        self.buffer_capacity= 32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimiser = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        self.buffer          = RolloutBuffer(self.buffer_capacity, state_size)
        self.normaliser      = RunningNormaliser()
        self.training_steps  = 0
        self.episode_rewards = []
        self._step_counter   = 0
        self.batch_size      = self.buffer_capacity

        logger.info(f"✅ PPO Agent initialised for {deployment_name} on {self.device}")
        logger.info(f"📊 Online PPO | buffer={self.buffer_capacity} | "
                    f"lr={self.lr} | c1={self.c1} | epochs={self.epochs}")

    def get_state(self, metrics):
        return np.array([
            metrics.get('cpu_usage',      0.0),
            metrics.get('memory_usage',   0.0),
            metrics.get('request_rate',   0.0) / 100.0,
            metrics.get('latency_p50',    0.0),
            metrics.get('latency_p95',    0.0),
            metrics.get('latency_p99',    0.0),
            metrics.get('replicas',       1)   / 10.0,
            metrics.get('error_rate',     0.0),
            metrics.get('pod_pending',    0)   / 10.0,
            metrics.get('pod_ready',      1)   / 10.0,
            metrics.get('cpu_trend_1m',   0.0),
            metrics.get('cpu_trend_5m',   0.0),
            metrics.get('request_trend',  0.0) / 50.0,
            metrics.get('hour',           0)   / 24.0,
            metrics.get('day_of_week',    0)   / 7.0,
            float(metrics.get('is_weekend',   False)),
            float(metrics.get('is_peak_hour', False)),
            max(0.0, metrics.get('latency_p95', 0.0) - self.sla_target),
        ], dtype=np.float32)

    def select_action(self, state, deterministic=False):
        rule_action = self._rule_prior(state)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.policy(t)
            dist = Categorical(logits=logits)

            if rule_action is not None:
                action   = rule_action
                act_t    = torch.tensor([action], device=self.device)
                log_prob = dist.log_prob(act_t)
            elif deterministic:
                action   = int(logits.argmax(dim=1).item())
                act_t    = torch.tensor([action], device=self.device)
                log_prob = dist.log_prob(act_t)
            else:
                act_t    = dist.sample()
                action   = int(act_t.item())
                log_prob = dist.log_prob(act_t)

        return action, log_prob, value.item()

    def calculate_confidence(self, state):
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, _ = self.policy(t)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        entropy    = -float(np.sum(probs * np.log(probs + 1e-8)))
        max_entropy = np.log(self.action_size)
        confidence  = float(1.0 - entropy / max_entropy)
        return max(0.0, min(1.0, confidence)), probs

    def calculate_reward(self, metrics, action, prev_metrics):
        raw = self._raw_reward(metrics, action, prev_metrics)
        self.normaliser.update(raw)
        return raw

    def store_transition(self, state, action, reward, next_state,
                         done, log_prob, value):
        norm_r = self.normaliser.normalise(reward)
        lp = float(log_prob) if hasattr(log_prob, 'item') else (log_prob or 0.0)
        self.buffer.add(state, action, norm_r, lp, value, done)
        self._step_counter += 1

    def update(self):
        if len(self.buffer) < self.mini_batch:
            return None

        states, actions, rewards, old_log_probs, old_values, dones = self.buffer.get()

        states_t     = torch.FloatTensor(states).to(self.device)
        actions_t    = torch.LongTensor(actions).to(self.device)
        old_lp_t     = torch.FloatTensor(old_log_probs).to(self.device)
        old_val_t    = torch.FloatTensor(old_values).to(self.device)

        advantages, returns = self._gae(rewards, old_values, dones)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)

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

                ratios  = torch.exp(log_probs - old_lp_t[idx])
                surr1   = ratios * adv_t[idx]
                surr2   = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_t[idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = self.c1 * nn.MSELoss()(values, ret_t[idx])
                entropy_loss= -self.c2 * entropy.mean()

                loss = policy_loss + value_loss + entropy_loss
                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimiser.step()

                total_pl  += policy_loss.item()
                total_vl  += value_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        self.training_steps += 1

        stats = {
            'policy_loss':    total_pl  / max(n_updates, 1),
            'value_loss':     total_vl  / max(n_updates, 1),
            'entropy':        total_ent / max(n_updates, 1),
            'training_steps': self.training_steps,
            'avg_reward':     float(np.mean(rewards)),
            'buffer_size':    len(self.buffer),
        }
        logger.info(
            f"🎓 Training Step {self.training_steps} | "
            f"Policy Loss: {stats['policy_loss']:.4f} | "
            f"Value Loss: {stats['value_loss']:.4f} | "
            f"Avg Reward: {stats['avg_reward']:.2f} | "
            f"Entropy: {stats['entropy']:.3f}"
        )
        return stats

    def pretrain(self, n_steps=300):
        logger.info(f"🏋️  Starting PPO pre-training for {n_steps} steps...")
        scenarios = [
            (0.02, 0.01, 0.0, 0.80, 1),
            (0.02, 0.01, 0.0, 0.75, 2),
            (0.02, 0.01, 0.0, 0.70, 3),
            (0.02, 0.01, 0.0, 0.72, 4),
            (0.02, 0.01, 0.0, 0.72, 5),
            (0.01, 0.01, 0.0, 0.10, 3),
            (0.01, 0.01, 0.0, 0.08, 2),
            (0.02, 0.01, 0.0, 0.30, 1),
        ]
        train_calls = 0
        for step in range(n_steps):
            cpu, mem, rr, lat, reps = scenarios[step % len(scenarios)]
            metrics = {
                'cpu_usage':    float(np.clip(cpu + np.random.normal(0, 0.003), 0, 1)),
                'memory_usage': float(np.clip(mem + np.random.normal(0, 0.001), 0, 1)),
                'request_rate': float(max(0, rr  + np.random.normal(0, 0.5))),
                'latency_p50':  float(max(0, lat * 0.7)),
                'latency_p95':  float(max(0, lat + np.random.normal(0, 0.03))),
                'latency_p99':  float(max(0, lat * 1.3)),
                'replicas':     reps,
                'error_rate':   0.0,
                'pod_pending':  0,
                'pod_ready':    reps,
                'cpu_trend_1m': float(np.random.normal(0, 0.001)),
                'cpu_trend_5m': float(np.random.normal(0, 0.001)),
                'request_trend':0.0,
                'hour':         12,
                'day_of_week':  2,
                'is_weekend':   False,
                'is_peak_hour': False,
            }
            state = self.get_state(metrics)
            action, log_prob, value = self.select_action(state)
            next_metrics = dict(metrics)
            if action == 2:
                next_metrics['replicas']    = reps + 1
                next_metrics['latency_p95'] = max(0, lat - 0.05)
            elif action == 0:
                next_metrics['replicas']    = max(1, reps - 1)
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

    def save_model(self, filepath):
        torch.save({
            'policy':          self.policy.state_dict(),
            'optimiser':       self.optimiser.state_dict(),
            'training_steps':  self.training_steps,
            'episode_rewards': self.episode_rewards,
            'norm_mean':       self.normaliser.mean,
            'norm_var':        self.normaliser.var,
        }, filepath)
        logger.info(f"💾 Model saved to {filepath}")

    def load_model(self, filepath):
        ckpt = torch.load(filepath, map_location=self.device)
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
                pass
        self.training_steps  = ckpt.get('training_steps', 0)
        self.episode_rewards = ckpt.get('episode_rewards', [])
        self.normaliser.mean = ckpt.get('norm_mean', 0.0)
        self.normaliser.var  = ckpt.get('norm_var',  1.0)
        logger.info(f"📂 Model loaded from {filepath}")

    def get_metrics(self):
        return {
            'training_steps': self.training_steps,
            'avg_reward_100': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
            'buffer_size':    len(self.buffer),
            'device':         str(self.device),
        }

    # ---------- Internal helpers ----------

    def _rule_prior(self, state):
        latency  = float(state[4])
        replicas = round(float(state[6]) * 10)
        violation= float(state[17])
        request_rate = float(state[2]) * 100

        # Force scale up when single pod and moderate traffic
        if replicas == 1 and request_rate > 30:
            logger.info(f"⚡ Rule prior: forcing scale_up (request_rate={request_rate:.1f} > 50)")
            return 2

        if replicas <= 1 and latency > self.sla_target:
            return 2
        if violation > 0.2 and replicas <= 2:
            return 2
        if latency < self.sla_target * 0.5 and replicas > 1:
            return 0
        return None

    def _raw_reward(self, metrics, action, prev_metrics):
        latency    = metrics.get('latency_p95', 0.0)
        replicas   = metrics.get('replicas', 1)
        error_rate = metrics.get('error_rate', 0.0)
        cpu_usage  = metrics.get('cpu_usage', 0.0)
        memory     = metrics.get('memory_usage', 0.0)
        request_rate = metrics.get('request_rate', 0.0)

        if cpu_usage == 0.0 and memory == 0.0:
            return 0.0

        reward = 0.0

        # SLA compliance
        if latency > self.sla_target:
            reward -= 50.0 * ((latency - self.sla_target) ** 2)
        elif latency > 0:
            reward += 5.0 * (self.sla_target - latency)

        # Cost (gentle)
        reward -= 0.5 * replicas

        # Errors
        reward -= 100.0 * error_rate

        # Over-scaling penalty (stops runaway scale-up)
        if replicas > 3 and latency > self.sla_target:
            reward -= 5.0 * (replicas - 3)
        if replicas > 1 and latency < self.sla_target * 0.8:
            reward -= 3.0 * (replicas - 1)

        # ---------- FIX: Action-specific penalties that always apply ----------
        # Penalise scale_up when there is no need (idle or very low traffic)
        if action == 2:   # scale_up
            # Strong penalty if load is low (request_rate < 30) and latency is fine
            if request_rate < 30 and latency < self.sla_target * 0.8:
                reward -= 25.0
                logger.debug(f"Penalty: scale_up under low load (req={request_rate:.1f}, lat={latency:.3f}) -> -25")
            # Additional penalty if already above min replicas and no stress
            elif replicas > 1 and latency < self.sla_target:
                reward -= 15.0
        elif action == 0:   # scale_down
            # Penalise scale_down when at min replicas or under high latency
            if replicas <= 1:
                reward -= 30.0
            elif latency > self.sla_target and replicas <= 3:
                reward -= 20.0
            else:
                # Reward scale_down if over-provisioned and idle
                if replicas > 2 and latency < self.sla_target * 0.5:
                    reward += 20.0
        elif action == 1:   # no_action
            # Reward no_action when perfectly sized
            if latency < self.sla_target * 0.6 and replicas == 1:
                reward += 8.0
            elif latency > self.sla_target and replicas <= 2:
                reward -= 12.0
            elif replicas > 2 and latency < self.sla_target * 0.6:
                reward -= 8.0

        # ---------- Additional incentive to scale up under real load ----------
        if replicas == 1 and request_rate > 50:
            reward += 30.0
            logger.debug(f"Scaling incentive: replicas=1, request_rate={request_rate:.1f} -> +30")

        # ---------- Transition-based rewards (if prev_metrics exists) ----------
        if prev_metrics:
            prev_latency = prev_metrics.get('latency_p95', 0.0)
            prev_replicas = prev_metrics.get('replicas', 1)
            # Reward if scaling reduced latency
            if action == 2 and replicas > prev_replicas and latency < prev_latency:
                reward += 15.0
            # Reward if scaling down reduced cost without hurting latency
            if action == 0 and replicas < prev_replicas and latency <= prev_latency + 0.05:
                reward += 10.0

        return reward

    def _gae(self, rewards, values, dones):
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(n)):
            next_val = 0.0 if t == n - 1 else values[t + 1]
            delta = (rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t])
            advantages[t] = last_adv = (delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_adv)
        returns = advantages + values
        return advantages, returns