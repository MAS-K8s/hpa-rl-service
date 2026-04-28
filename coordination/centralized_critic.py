import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

class CentralizedCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int, max_agents: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.max_agents = max_agents
        input_dim = max_agents * (state_size + action_size)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor, agent_mask: torch.Tensor = None):
        """
        states: (batch, max_agents, state_size) – float32
        actions: (batch, max_agents) – long (int64)
        agent_mask: (batch, max_agents) – float32, 1 for active agents, 0 for padding
        """
        # Ensure correct dtypes
        states = states.float()
        actions = actions.long()          # critical: convert to int64
        if agent_mask is not None:
            agent_mask = agent_mask.float()

        # One-hot encode actions
        actions_onehot = F.one_hot(actions, num_classes=self.action_size).float()

        # Concatenate states and actions
        x = torch.cat([states, actions_onehot], dim=-1)

        if agent_mask is not None:
            x = x * agent_mask.unsqueeze(-1)

        x = x.view(x.size(0), -1)
        value = self.net(x).squeeze(-1)
        return value

    def get_global_value(self, all_agent_states: List[np.ndarray], all_agent_actions: List[int],
                         max_agents: int = None) -> float:
        if max_agents is None:
            max_agents = self.max_agents
        n = len(all_agent_states)
        padded_states = np.zeros((max_agents, self.state_size), dtype=np.float32)
        padded_actions = np.zeros(max_agents, dtype=np.int64)
        for i in range(min(n, max_agents)):
            padded_states[i] = all_agent_states[i]
            padded_actions[i] = all_agent_actions[i]
        state_t = torch.FloatTensor(padded_states).unsqueeze(0)
        action_t = torch.LongTensor(padded_actions).unsqueeze(0)
        mask = torch.zeros(1, max_agents, dtype=torch.float32)
        mask[0, :n] = 1.0
        with torch.no_grad():
            value = self.forward(state_t, action_t, mask)
        return value.item()