import numpy as np
from typing import Dict, List

class TeamReward:
    """
    Computes a global (team) reward based on cluster-wide metrics.
    This reward is added to each agent's local reward to encourage cooperation.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7):
        """
        alpha: weight for global reward in final reward (1-alpha for local)
        beta: weight for efficiency vs performance in global reward
        """
        self.alpha = alpha
        self.beta = beta

    def compute_global_reward(self, all_metrics: List[Dict]) -> float:
        """
        all_metrics: list of metric dicts for each agent/deployment.
        Returns a scalar global reward.
        """
        if not all_metrics:
            return 0.0
        # Average latency across all services
        avg_latency = np.mean([m.get('latency_p95', 0.0) for m in all_metrics])
        # Total replicas (cost)
        total_replicas = sum([m.get('replicas', 1) for m in all_metrics])
        # Total error rate
        total_errors = sum([m.get('error_rate', 0.0) for m in all_metrics])

        # Global reward components
        # Penalize high average latency
        latency_penalty = 0.0
        if avg_latency > 0.5:
            latency_penalty = -20.0 * (avg_latency - 0.5) ** 2
        # Penalize many replicas (cost)
        cost_penalty = -0.5 * total_replicas
        # Penalize errors
        error_penalty = -50.0 * total_errors

        # Combine
        global_reward = self.beta * (latency_penalty + error_penalty) + (1 - self.beta) * cost_penalty
        # Normalize to reasonable range
        return np.clip(global_reward, -30.0, 30.0)

    def combine_rewards(self, local_reward: float, global_reward: float) -> float:
        """Combine local and global reward using alpha."""
        return (1 - self.alpha) * local_reward + self.alpha * global_reward