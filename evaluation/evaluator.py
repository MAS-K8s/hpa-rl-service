import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class BaselineAutoscaler:
    """Base class for baseline autoscalers"""
    
    def __init__(self, name: str):
        self.name = name
    
    def decide(self, metrics: Dict) -> int:
        """
        Decide scaling action based on metrics
        Returns: 0 (scale_down), 1 (no_action), 2 (scale_up)
        """
        raise NotImplementedError


class KubernetesHPA(BaselineAutoscaler):
    """
    Simulates Kubernetes Horizontal Pod Autoscaler
    Simple CPU-based scaling
    """
    
    def __init__(self, target_cpu=0.7, scale_down_threshold=0.3):
        super().__init__("Kubernetes HPA")
        self.target_cpu = target_cpu
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_steps = 5  # Cooldown period
        self.last_scale_step = -999
        self.current_step = 0
    
    def decide(self, metrics: Dict) -> int:
        cpu_usage = metrics.get('cpu_usage', 0.5)
        
        self.current_step += 1
        
        # Cooldown check
        if self.current_step - self.last_scale_step < self.cooldown_steps:
            return 1  # no_action during cooldown
        
        # Scale up if CPU > target
        if cpu_usage > self.target_cpu:
            self.last_scale_step = self.current_step
            return 2  # scale_up
        
        # Scale down if CPU < threshold
        elif cpu_usage < self.scale_down_threshold:
            self.last_scale_step = self.current_step
            return 0  # scale_down
        
        return 1  # no_action


class RuleBasedAutoscaler(BaselineAutoscaler):
    """
    Rule-based autoscaler using multiple metrics
    More sophisticated than HPA
    """
    
    def __init__(self):
        super().__init__("Rule-Based")
    
    def decide(self, metrics: Dict) -> int:
        cpu = metrics.get('cpu_usage', 0.5)
        latency = metrics.get('latency_p95', 0.3)
        error_rate = metrics.get('error_rate', 0.0)
        
        # Emergency scale up
        if latency > 0.5 or error_rate > 0.05:
            return 2  # scale_up
        
        # High load
        if cpu > 0.75:
            return 2  # scale_up
        
        # Low load
        if cpu < 0.3 and latency < 0.2:
            return 0  # scale_down
        
        return 1  # no_action


class ReactiveAutoscaler(BaselineAutoscaler):
    """
    Purely reactive autoscaler
    Only scales when there's a problem
    """
    
    def __init__(self, sla_target=0.5):
        super().__init__("Reactive")
        self.sla_target = sla_target
    
    def decide(self, metrics: Dict) -> int:
        latency = metrics.get('latency_p95', 0.3)
        cpu = metrics.get('cpu_usage', 0.5)
        
        # SLA violation
        if latency > self.sla_target:
            return 2  # scale_up
        
        # Very underutilized
        if cpu < 0.2 and latency < self.sla_target * 0.5:
            return 0  # scale_down
        
        return 1  # no_action


class PredictiveAutoscaler(BaselineAutoscaler):
    """
    Simple predictive autoscaler using trends
    """
    
    def __init__(self):
        super().__init__("Predictive (Trend-Based)")
        self.history = []
        self.window_size = 5
    
    def decide(self, metrics: Dict) -> int:
        cpu = metrics.get('cpu_usage', 0.5)
        cpu_trend = metrics.get('cpu_trend_1m', 0.0)
        
        self.history.append(cpu)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Calculate trend
        if len(self.history) >= 3:
            recent_trend = np.mean(np.diff(self.history[-3:]))
        else:
            recent_trend = cpu_trend
        
        # Proactive scaling
        if cpu > 0.6 and recent_trend > 0.05:
            return 2  # scale_up proactively
        
        if cpu < 0.4 and recent_trend < -0.05:
            return 0  # scale_down proactively
        
        # Reactive
        if cpu > 0.8:
            return 2
        if cpu < 0.3:
            return 0
        
        return 1


class WorkloadSimulator:
    """
    Simulates realistic workload patterns
    """
    
    def __init__(self, pattern='daily', noise_level=0.1):
        self.pattern = pattern
        self.noise_level = noise_level
        self.step = 0
    
    def get_load(self) -> float:
        """
        Returns normalized load (0-1)
        """
        self.step += 1
        t = self.step
        
        if self.pattern == 'daily':
            # Daily pattern: low at night, high during day
            base_load = 0.3 + 0.4 * np.sin(2 * np.pi * t / 288)  # 24h cycle at 5min intervals
        
        elif self.pattern == 'spiky':
            # Spiky pattern with sudden bursts
            base_load = 0.4
            if t % 50 == 0:  # Spike every ~4 hours
                base_load = 0.9
        
        elif self.pattern == 'gradual':
            # Gradual increase then decrease
            cycle_position = (t % 200) / 200.0
            base_load = 0.3 + 0.6 * np.sin(np.pi * cycle_position)
        
        elif self.pattern == 'constant':
            base_load = 0.6
        
        else:
            base_load = 0.5
        
        # Add noise
        noise = np.random.normal(0, self.noise_level)
        load = np.clip(base_load + noise, 0.05, 0.95)
        
        return load
    
    def simulate_metrics(self, replicas: int, load: float) -> Dict:
        """
        Simulate metrics given current replicas and load
        """
        # CPU usage depends on load and replicas
        cpu_per_replica = load / max(replicas, 1)
        cpu_usage = min(cpu_per_replica, 1.0)
        
        # Latency increases with high CPU
        base_latency = 0.1
        if cpu_usage > 0.8:
            latency_p95 = base_latency + 0.5 * ((cpu_usage - 0.8) / 0.2) ** 2
        else:
            latency_p95 = base_latency + 0.1 * cpu_usage
        
        # Error rate increases when overloaded
        if cpu_usage > 0.9:
            error_rate = 0.01 * ((cpu_usage - 0.9) / 0.1)
        else:
            error_rate = 0.0001
        
        # Request rate
        request_rate = load * 1000  # Requests per second
        
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': 2.0 + np.random.uniform(-0.2, 0.2),
            'request_rate': request_rate,
            'latency_p50': latency_p95 * 0.6,
            'latency_p95': latency_p95,
            'latency_p99': latency_p95 * 1.3,
            'replicas': replicas,
            'error_rate': error_rate,
            'pod_pending': 0,
            'pod_ready': replicas,
            'cpu_trend_1m': np.random.uniform(-0.05, 0.05),
            'cpu_trend_5m': np.random.uniform(-0.03, 0.03),
            'request_trend': np.random.uniform(-10, 10),
            'hour': (self.step // 12) % 24,
            'day_of_week': (self.step // 288) % 7,
            'is_weekend': ((self.step // 288) % 7) >= 5,
            'is_peak_hour': 9 <= ((self.step // 12) % 24) <= 17
        }


class PerformanceMetrics:
    """Track and calculate performance metrics"""
    
    def __init__(self, sla_target=0.5, cost_per_replica=1.0):
        self.sla_target = sla_target
        self.cost_per_replica = cost_per_replica
        self.reset()
    
    def reset(self):
        self.latencies = []
        self.replica_counts = []
        self.sla_violations = []
        self.error_rates = []
        self.cpu_utilizations = []
    
    def record(self, metrics: Dict):
        self.latencies.append(metrics.get('latency_p95', 0))
        self.replica_counts.append(metrics.get('replicas', 1))
        self.error_rates.append(metrics.get('error_rate', 0))
        self.cpu_utilizations.append(metrics.get('cpu_usage', 0))
        
        # Check SLA violation
        violation = 1 if metrics.get('latency_p95', 0) > self.sla_target else 0
        self.sla_violations.append(violation)
    
    def get_summary(self) -> Dict:
        """Calculate summary statistics"""
        return {
            'avg_latency': np.mean(self.latencies),
            'p95_latency': np.percentile(self.latencies, 95),
            'p99_latency': np.percentile(self.latencies, 99),
            'max_latency': np.max(self.latencies),
            'avg_replicas': np.mean(self.replica_counts),
            'total_cost': np.sum(self.replica_counts) * self.cost_per_replica,
            'sla_violation_rate': np.mean(self.sla_violations),
            'sla_violations_count': np.sum(self.sla_violations),
            'avg_error_rate': np.mean(self.error_rates),
            'avg_cpu_utilization': np.mean(self.cpu_utilizations),
            'resource_efficiency': np.mean(self.cpu_utilizations) / max(np.mean(self.replica_counts), 1)
        }


class AutoscalerEvaluator:
    """
    Comprehensive evaluator for comparing autoscaling strategies
    """
    
    def __init__(self, sla_target=0.5, min_replicas=1, max_replicas=10):
        self.sla_target = sla_target
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
    
    def run_episode(self, autoscaler: BaselineAutoscaler, 
                    workload_pattern: str = 'daily',
                    steps: int = 288) -> Tuple[PerformanceMetrics, List[Dict]]:
        """
        Run one episode with given autoscaler
        
        Args:
            autoscaler: The autoscaler to evaluate
            workload_pattern: Workload pattern to simulate
            steps: Number of time steps
        
        Returns:
            metrics: Performance metrics
            history: List of state transitions
        """
        simulator = WorkloadSimulator(pattern=workload_pattern)
        metrics_tracker = PerformanceMetrics(sla_target=self.sla_target)
        
        current_replicas = 3  # Start with 3 replicas
        history = []
        
        for step in range(steps):
            # Get current load
            load = simulator.get_load()
            
            # Simulate metrics
            metrics = simulator.simulate_metrics(current_replicas, load)
            
            # Record metrics
            metrics_tracker.record(metrics)
            
            # Get action from autoscaler
            action = autoscaler.decide(metrics)
            
            # Execute action
            if action == 0:  # scale_down
                current_replicas = max(self.min_replicas, current_replicas - 1)
            elif action == 2:  # scale_up
                current_replicas = min(self.max_replicas, current_replicas + 1)
            
            # Record history
            history.append({
                'step': step,
                'load': load,
                'replicas': current_replicas,
                'action': action,
                'latency': metrics['latency_p95'],
                'cpu': metrics['cpu_usage'],
                'cost': current_replicas
            })
        
        return metrics_tracker, history
    
    def compare_autoscalers(self, patterns: List[str] = None, 
                           steps_per_pattern: int = 288) -> pd.DataFrame:
        """
        Compare all autoscalers across different workload patterns
        
        Returns:
            DataFrame with comparison results
        """
        if patterns is None:
            patterns = ['daily', 'spiky', 'gradual', 'constant']
        
        autoscalers = [
            KubernetesHPA(),
            RuleBasedAutoscaler(),
            ReactiveAutoscaler(),
            PredictiveAutoscaler()
        ]
        
        results = []
        
        for pattern in patterns:
            logger.info(f"Evaluating pattern: {pattern}")
            
            for autoscaler in autoscalers:
                # Run episode
                metrics, history = self.run_episode(
                    autoscaler, 
                    pattern, 
                    steps_per_pattern
                )
                
                summary = metrics.get_summary()
                summary['autoscaler'] = autoscaler.name
                summary['pattern'] = pattern
                
                results.append(summary)
                
                logger.info(
                    f"  {autoscaler.name:20s} | "
                    f"Avg Latency: {summary['avg_latency']:.3f}s | "
                    f"SLA Violations: {summary['sla_violations_count']:3.0f} | "
                    f"Avg Replicas: {summary['avg_replicas']:.2f} | "
                    f"Cost: {summary['total_cost']:.0f}"
                )
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, results_df: pd.DataFrame, output_path: str = None):
        """
        Create visualization of comparison results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        patterns = results_df['pattern'].unique()
        
        # Plot 1: Average Latency
        ax = axes[0, 0]
        for pattern in patterns:
            data = results_df[results_df['pattern'] == pattern]
            ax.bar(
                data['autoscaler'], 
                data['avg_latency'],
                alpha=0.7,
                label=pattern
            )
        ax.set_ylabel('Average Latency (s)')
        ax.set_title('Average Latency by Autoscaler')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: SLA Violations
        ax = axes[0, 1]
        for pattern in patterns:
            data = results_df[results_df['pattern'] == pattern]
            ax.bar(
                data['autoscaler'],
                data['sla_violation_rate'] * 100,
                alpha=0.7,
                label=pattern
            )
        ax.set_ylabel('SLA Violation Rate (%)')
        ax.set_title('SLA Violations by Autoscaler')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Cost
        ax = axes[1, 0]
        for pattern in patterns:
            data = results_df[results_df['pattern'] == pattern]
            ax.bar(
                data['autoscaler'],
                data['total_cost'],
                alpha=0.7,
                label=pattern
            )
        ax.set_ylabel('Total Cost')
        ax.set_title('Cost by Autoscaler')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Resource Efficiency
        ax = axes[1, 1]
        for pattern in patterns:
            data = results_df[results_df['pattern'] == pattern]
            ax.bar(
                data['autoscaler'],
                data['resource_efficiency'],
                alpha=0.7,
                label=pattern
            )
        ax.set_ylabel('Resource Efficiency')
        ax.set_title('Resource Efficiency by Autoscaler')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")
        
        return fig
    
    def plot_episode_details(self, history: List[Dict], title: str = "Episode Details"):
        """
        Plot detailed timeline of a single episode
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        
        steps = [h['step'] for h in history]
        
        # Plot 1: Load and Replicas
        ax = axes[0]
        ax.plot(steps, [h['load'] for h in history], label='Load', color='blue', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(steps, [h['replicas'] for h in history], label='Replicas', color='green', linewidth=2)
        ax.set_ylabel('Load', color='blue')
        ax2.set_ylabel('Replicas', color='green')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Latency
        ax = axes[1]
        ax.plot(steps, [h['latency'] for h in history], label='Latency P95', color='orange')
        ax.axhline(y=0.5, color='red', linestyle='--', label='SLA Target')
        ax.set_ylabel('Latency (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: CPU Usage
        ax = axes[2]
        ax.plot(steps, [h['cpu'] for h in history], label='CPU Usage', color='purple')
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Target')
        ax.set_ylabel('CPU Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Actions
        ax = axes[3]
        actions = [h['action'] for h in history]
        action_colors = ['red' if a == 0 else 'green' if a == 2 else 'gray' for a in actions]
        ax.scatter(steps, actions, c=action_colors, alpha=0.5, s=10)
        ax.set_ylabel('Action')
        ax.set_xlabel('Time Step')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Scale Down', 'No Action', 'Scale Up'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    evaluator = AutoscalerEvaluator()
    
    # Run comparison
    logger.info("Starting autoscaler comparison...")
    results = evaluator.compare_autoscalers(
        patterns=['daily', 'spiky'],
        steps_per_pattern=288  # 24 hours at 5-min intervals
    )
    
    # Save results
    results.to_csv('autoscaler_comparison.csv', index=False)
    logger.info("Results saved to autoscaler_comparison.csv")
    
    # Plot comparison
    evaluator.plot_comparison(results, 'comparison_plot.png')
    
    # Plot detailed episode
    hpa = KubernetesHPA()
    _, history = evaluator.run_episode(hpa, 'daily', 288)
    evaluator.plot_episode_details(history, "Kubernetes HPA - Daily Pattern")
    plt.savefig('episode_details.png', dpi=300, bbox_inches='tight')
    
    logger.info("Evaluation complete!")