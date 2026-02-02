import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RewardFunction:
    """
    Sophisticated reward function for autoscaling
    Balances multiple objectives: performance, cost, stability
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize reward function with configuration
        
        Args:
            config: Optional dictionary with reward parameters
        """
        config = config or {}
        
        # SLA parameters
        self.sla_target = config.get('sla_target', 0.5)  # 500ms
        self.sla_weight = config.get('sla_weight', 100.0)
        
        # Cost parameters
        self.cost_per_replica = config.get('cost_per_replica', 1.0)
        self.cost_weight = config.get('cost_weight', 1.0)
        
        # Performance parameters
        self.optimal_cpu_min = config.get('optimal_cpu_min', 0.6)
        self.optimal_cpu_max = config.get('optimal_cpu_max', 0.75)
        self.efficiency_weight = config.get('efficiency_weight', 10.0)
        
        # Reliability parameters
        self.error_weight = config.get('error_weight', 50.0)
        self.pending_weight = config.get('pending_weight', 10.0)
        
        # Stability parameters
        self.stability_weight = config.get('stability_weight', 2.0)
        self.change_penalty = config.get('change_penalty', 1.0)
        
        # Proactive scaling parameters
        self.proactive_weight = config.get('proactive_weight', 5.0)
        
    def calculate(self, metrics: Dict, action: int, prev_metrics: Optional[Dict] = None) -> float:
        """
        Calculate reward for current state and action
        
        Args:
            metrics: Current metrics dictionary
            action: Action taken (0=down, 1=stay, 2=up)
            prev_metrics: Previous metrics for comparison
        
        Returns:
            reward: Calculated reward value
        """
        reward = 0.0
        
        # Component 1: SLA Compliance (highest priority)
        sla_reward = self._calculate_sla_reward(metrics)
        reward += sla_reward
        
        # Component 2: Cost Optimization
        cost_penalty = self._calculate_cost_penalty(metrics)
        reward += cost_penalty
        
        # Component 3: Resource Efficiency
        efficiency_reward = self._calculate_efficiency_reward(metrics)
        reward += efficiency_reward
        
        # Component 4: Reliability
        reliability_penalty = self._calculate_reliability_penalty(metrics)
        reward += reliability_penalty
        
        # Component 5: Stability
        if prev_metrics:
            stability_reward = self._calculate_stability_reward(
                metrics, action, prev_metrics
            )
            reward += stability_reward
        
        # Component 6: Proactive Scaling
        proactive_reward = self._calculate_proactive_reward(metrics, action)
        reward += proactive_reward
        
        return reward
    
    def _calculate_sla_reward(self, metrics: Dict) -> float:
        """
        Reward/penalize based on SLA compliance
        
        Critical: Meeting SLA is the primary objective
        """
        latency = metrics.get('latency_p95', 0.0)
        
        if latency > self.sla_target:
            # SLA violation - severe quadratic penalty
            violation = latency - self.sla_target
            penalty = -self.sla_weight * (violation ** 2)
            return penalty
        else:
            # Meeting SLA - reward proportional to margin
            margin = self.sla_target - latency
            bonus = self.sla_weight * 0.1 * margin  # 10% of penalty weight
            return bonus
    
    def _calculate_cost_penalty(self, metrics: Dict) -> float:
        """
        Penalize resource cost (number of replicas)
        
        Linear penalty to encourage efficiency
        """
        replicas = metrics.get('replicas', 1)
        return -self.cost_weight * self.cost_per_replica * replicas
    
    def _calculate_efficiency_reward(self, metrics: Dict) -> float:
        """
        Reward optimal CPU utilization
        
        Target: 60-75% CPU (sweet spot)
        Penalty: >90% (overloaded) or <30% (underutilized)
        """
        cpu_usage = metrics.get('cpu_usage', 0.0)
        
        if self.optimal_cpu_min <= cpu_usage <= self.optimal_cpu_max:
            # Optimal zone - give reward
            return self.efficiency_weight
        
        elif cpu_usage > 0.9:
            # Overloaded - severe penalty
            overload = cpu_usage - 0.9
            return -self.efficiency_weight * 1.5 * (overload / 0.1)
        
        elif cpu_usage < 0.3:
            # Underutilized - moderate penalty
            waste = 0.3 - cpu_usage
            return -self.efficiency_weight * 0.5 * (waste / 0.3)
        
        return 0.0
    
    def _calculate_reliability_penalty(self, metrics: Dict) -> float:
        """
        Penalize errors and pending pods
        
        Errors indicate system stress
        Pending pods indicate scaling lag
        """
        penalty = 0.0
        
        # Error penalty
        error_rate = metrics.get('error_rate', 0.0)
        penalty -= self.error_weight * error_rate
        
        # Pending pods penalty
        pod_pending = metrics.get('pod_pending', 0)
        if pod_pending > 0:
            penalty -= self.pending_weight * pod_pending
        
        return penalty
    
    def _calculate_stability_reward(self, metrics: Dict, action: int, 
                                   prev_metrics: Dict) -> float:
        """
        Reward stability and penalize thrashing
        
        Frequent changes indicate poor policy
        """
        reward = 0.0
        
        replicas = metrics.get('replicas', 1)
        prev_replicas = prev_metrics.get('replicas', replicas)
        cpu_usage = metrics.get('cpu_usage', 0.0)
        latency = metrics.get('latency_p95', 0.0)
        
        if action != 1:  # Not "no_action"
            if replicas == prev_replicas:
                # Tried to scale but nothing happened
                # This suggests poor timing or policy
                reward -= self.stability_weight
            else:
                # Successfully scaled - small penalty for change
                reward -= self.change_penalty
        else:
            # Maintained stability
            # Reward if system is in good state
            if (self.optimal_cpu_min <= cpu_usage <= self.optimal_cpu_max and 
                latency < self.sla_target):
                reward += self.stability_weight
        
        return reward
    
    def _calculate_proactive_reward(self, metrics: Dict, action: int) -> float:
        """
        Reward proactive scaling based on trends
        
        Scaling up when load is increasing = good anticipation
        Scaling down when load is decreasing = good optimization
        """
        cpu_trend = metrics.get('cpu_trend_1m', 0.0)
        
        # Threshold for significant trend
        trend_threshold = 0.1
        
        if cpu_trend > trend_threshold and action == 2:
            # Scaling up during increasing load
            return self.proactive_weight
        
        elif cpu_trend < -trend_threshold and action == 0:
            # Scaling down during decreasing load
            return self.proactive_weight * 0.6  # Slightly less reward
        
        return 0.0
    
    def explain_reward(self, reward: float, metrics: Dict, action: int, 
                       prev_metrics: Optional[Dict] = None) -> Dict:
        """
        Break down reward into components for debugging
        
        Returns:
            components: Dictionary with each reward component
        """
        components = {
            'total': reward,
            'sla': self._calculate_sla_reward(metrics),
            'cost': self._calculate_cost_penalty(metrics),
            'efficiency': self._calculate_efficiency_reward(metrics),
            'reliability': self._calculate_reliability_penalty(metrics),
            'proactive': self._calculate_proactive_reward(metrics, action),
        }
        
        if prev_metrics:
            components['stability'] = self._calculate_stability_reward(
                metrics, action, prev_metrics
            )
        
        return components


class AdaptiveRewardFunction(RewardFunction):
    """
    Adaptive reward function that adjusts weights based on performance
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # Track SLA violations
        self.sla_violations = []
        self.max_history = 100
        
        # Adaptation parameters
        self.adapt_rate = 0.05
        self.min_sla_weight = 50.0
        self.max_sla_weight = 200.0
    
    def calculate(self, metrics: Dict, action: int, prev_metrics: Optional[Dict] = None) -> float:
        """Calculate reward with adaptive weights"""
        
        # Track SLA violations
        latency = metrics.get('latency_p95', 0.0)
        self.sla_violations.append(1.0 if latency > self.sla_target else 0.0)
        if len(self.sla_violations) > self.max_history:
            self.sla_violations.pop(0)
        
        # Adapt SLA weight based on violation rate
        if len(self.sla_violations) >= 10:
            violation_rate = np.mean(self.sla_violations[-10:])
            
            if violation_rate > 0.1:  # More than 10% violations
                # Increase SLA weight to emphasize meeting SLA
                self.sla_weight = min(
                    self.max_sla_weight,
                    self.sla_weight * (1 + self.adapt_rate)
                )
            elif violation_rate < 0.02:  # Less than 2% violations
                # Decrease SLA weight to focus on cost
                self.sla_weight = max(
                    self.min_sla_weight,
                    self.sla_weight * (1 - self.adapt_rate)
                )
        
        # Calculate reward with adapted weights
        return super().calculate(metrics, action, prev_metrics)
    
    def get_adaptation_stats(self) -> Dict:
        """Get statistics about weight adaptation"""
        return {
            'current_sla_weight': self.sla_weight,
            'recent_violation_rate': np.mean(self.sla_violations[-10:]) if self.sla_violations else 0,
            'total_samples': len(self.sla_violations)
        }