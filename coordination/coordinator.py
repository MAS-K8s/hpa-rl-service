import logging
import threading
import time
from typing import Dict, List, Optional
import numpy as np
import torch

from coordination.redis_buffer import SharedExperienceBuffer
from coordination.centralized_critic import CentralizedCritic
from coordination.team_reward import TeamReward

logger = logging.getLogger(__name__)


class MultiAgentCoordinatorCTDE:
    def __init__(self, redis_client, state_size: int, action_size: int,
                 max_agents: int = 10, hidden_dim: int = 256,
                 lr: float = 1e-4, update_interval: int = 10):
        self.redis = redis_client
        self.state_size = state_size
        self.action_size = action_size
        self.max_agents = max_agents
        self.update_interval = update_interval
        self.step_counter = 0

        self.shared_buffer = SharedExperienceBuffer(redis_client)
        self.critic = CentralizedCritic(state_size, action_size, max_agents, hidden_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic.to(self.device)
        self.team_reward = TeamReward(alpha=0.3, beta=0.7)
        self.lock = threading.Lock()
        logger.info(f"✅ CTDE Coordinator initialised | critic device: {self.device} | max_agents: {max_agents}")

    def add_experience(self, agent_key: str, state: np.ndarray, action: int, reward: float,
                       next_state: np.ndarray, done: bool, log_prob: float, value: float):
        try:
            self.shared_buffer.add(agent_key, state, action, reward, next_state, done, log_prob, value)
        except Exception as e:
            logger.error(f"Failed to add experience to shared buffer: {e}")

    def compute_team_reward(self, all_metrics: List[Dict]) -> float:
        try:
            if not all_metrics:
                return 0.0
            return self.team_reward.compute_global_reward(all_metrics)
        except Exception as e:
            logger.warning(f"Team reward computation failed: {e}")
            return 0.0

    def update_centralized_critic(self, batch_size: int = 32) -> Optional[float]:
        """
        Update the centralized critic using the shared experience buffer.
        Returns loss if successful, else None.
        """
        batch = self.shared_buffer.sample(batch_size)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones, log_probs, values, agent_keys = batch

        # Convert to tensors with explicit dtypes
        try:
            states_t = torch.from_numpy(states.astype(np.float32)).to(self.device)
            actions_t = torch.from_numpy(actions.astype(np.int64)).to(self.device)
            rewards_t = torch.from_numpy(rewards.astype(np.float32)).to(self.device)

            # Compute discounted returns
            returns = []
            cumulative = 0.0
            for r in reversed(rewards):
                cumulative = r + 0.99 * cumulative
                returns.insert(0, cumulative)
            returns_t = torch.from_numpy(np.array(returns, dtype=np.float32)).to(self.device)

            batch_size_actual = states_t.shape[0]
            padded_states = torch.zeros(batch_size_actual, self.max_agents, self.state_size,
                                        device=self.device, dtype=torch.float32)
            padded_actions = torch.zeros(batch_size_actual, self.max_agents,
                                         device=self.device, dtype=torch.long)
            mask = torch.zeros(batch_size_actual, self.max_agents,
                               device=self.device, dtype=torch.float32)

            for i in range(batch_size_actual):
                # Place each agent's data in the first slot (simplified)
                padded_states[i, 0] = states_t[i]
                padded_actions[i, 0] = actions_t[i]
                mask[i, 0] = 1.0

            # Forward pass through the centralized critic
            values_pred = self.critic(padded_states, padded_actions, mask)
            loss = torch.nn.MSELoss()(values_pred, returns_t)

            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            return loss.item()

        except Exception as e:
            logger.error(f"Centralized critic update error: {type(e).__name__}: {e}")
            return None

    def get_shared_buffer_size(self) -> int:
        return self.shared_buffer.size()

    def clear_shared_buffer(self):
        self.shared_buffer.clear()
        logger.info("Shared experience buffer cleared")