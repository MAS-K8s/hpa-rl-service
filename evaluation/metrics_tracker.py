import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks and analyzes performance metrics over time
    Used for evaluation and monitoring
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize metrics tracker
        
        Args:
            agent_name: Name of the agent being tracked
        """
        self.agent_name = agent_name
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.episodes = []
        self.current_episode = {
            'rewards': [],
            'actions': [],
            'states': [],
            'latencies': [],
            'replicas': [],
            'cpu_usage': [],
            'error_rates': [],
            'sla_violations': [],
            'costs': [],
            'timestamps': []
        }
    
    def record_step(self, metrics: Dict, action: int, reward: float, state: np.ndarray):
        """
        Record a single step
        
        Args:
            metrics: Current metrics dictionary
            action: Action taken
            reward: Reward received
            state: State vector
        """
        self.current_episode['rewards'].append(reward)
        self.current_episode['actions'].append(action)
        self.current_episode['states'].append(state.tolist() if isinstance(state, np.ndarray) else state)
        self.current_episode['latencies'].append(metrics.get('latency_p95', 0))
        self.current_episode['replicas'].append(metrics.get('replicas', 1))
        self.current_episode['cpu_usage'].append(metrics.get('cpu_usage', 0))
        self.current_episode['error_rates'].append(metrics.get('error_rate', 0))
        
        # Track SLA violations
        sla_violated = 1 if metrics.get('latency_p95', 0) > 0.5 else 0
        self.current_episode['sla_violations'].append(sla_violated)
        
        # Track cost (replica-minutes)
        cost = metrics.get('replicas', 1)  # Simple cost model
        self.current_episode['costs'].append(cost)
        
        self.current_episode['timestamps'].append(datetime.now().isoformat())
    
    def end_episode(self):
        """Mark end of current episode and start new one"""
        if self.current_episode['rewards']:
            self.episodes.append(self.current_episode.copy())
            logger.info(f"Episode {len(self.episodes)} completed: "
                       f"Total reward: {sum(self.current_episode['rewards']):.2f}, "
                       f"Steps: {len(self.current_episode['rewards'])}")
        self.reset()
    
    def get_episode_summary(self, episode_idx: int = -1) -> Dict:
        """
        Get summary statistics for an episode
        
        Args:
            episode_idx: Episode index (-1 for current)
        
        Returns:
            summary: Dictionary with episode statistics
        """
        if episode_idx == -1:
            episode = self.current_episode
        else:
            if episode_idx >= len(self.episodes):
                return {}
            episode = self.episodes[episode_idx]
        
        if not episode['rewards']:
            return {}
        
        return {
            'total_reward': sum(episode['rewards']),
            'avg_reward': np.mean(episode['rewards']),
            'steps': len(episode['rewards']),
            'avg_latency': np.mean(episode['latencies']),
            'p95_latency': np.percentile(episode['latencies'], 95),
            'avg_replicas': np.mean(episode['replicas']),
            'total_cost': sum(episode['costs']),
            'sla_violation_rate': np.mean(episode['sla_violations']),
            'sla_violations': sum(episode['sla_violations']),
            'avg_cpu': np.mean(episode['cpu_usage']),
            'avg_error_rate': np.mean(episode['error_rates']),
            'action_distribution': {
                'scale_down': episode['actions'].count(0),
                'no_action': episode['actions'].count(1),
                'scale_up': episode['actions'].count(2)
            }
        }
    
    def get_all_summaries(self) -> List[Dict]:
        """Get summaries for all episodes"""
        return [
            self.get_episode_summary(i)
            for i in range(len(self.episodes))
        ]
    
    def get_learning_curve(self) -> Dict:
        """
        Get learning curve data
        
        Returns:
            curve: Dictionary with learning metrics over time
        """
        if not self.episodes:
            return {}
        
        episode_rewards = [sum(ep['rewards']) for ep in self.episodes]
        episode_lengths = [len(ep['rewards']) for ep in self.episodes]
        
        # Calculate moving average
        window = min(10, len(episode_rewards))
        if window > 0:
            rewards_ma = pd.Series(episode_rewards).rolling(window).mean().tolist()
        else:
            rewards_ma = episode_rewards
        
        return {
            'episodes': list(range(len(episode_rewards))),
            'rewards': episode_rewards,
            'rewards_ma': rewards_ma,
            'lengths': episode_lengths,
            'avg_reward': np.mean(episode_rewards),
            'best_reward': max(episode_rewards) if episode_rewards else 0
        }
    
    def get_performance_metrics(self) -> Dict:
        """
        Get overall performance metrics across all episodes
        
        Returns:
            metrics: Dictionary with aggregated metrics
        """
        if not self.episodes:
            return {}
        
        all_latencies = []
        all_sla_violations = []
        all_costs = []
        all_rewards = []
        
        for episode in self.episodes:
            all_latencies.extend(episode['latencies'])
            all_sla_violations.extend(episode['sla_violations'])
            all_costs.extend(episode['costs'])
            all_rewards.extend(episode['rewards'])
        
        return {
            'total_episodes': len(self.episodes),
            'total_steps': sum(len(ep['rewards']) for ep in self.episodes),
            'avg_latency': np.mean(all_latencies),
            'p95_latency': np.percentile(all_latencies, 95),
            'p99_latency': np.percentile(all_latencies, 99),
            'sla_violation_rate': np.mean(all_sla_violations),
            'total_sla_violations': sum(all_sla_violations),
            'avg_cost_per_step': np.mean(all_costs),
            'total_cost': sum(all_costs),
            'avg_reward': np.mean(all_rewards),
            'total_reward': sum(all_rewards)
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all episodes to pandas DataFrame
        
        Returns:
            df: DataFrame with all recorded data
        """
        rows = []
        
        for ep_idx, episode in enumerate(self.episodes):
            for step_idx in range(len(episode['rewards'])):
                rows.append({
                    'episode': ep_idx,
                    'step': step_idx,
                    'reward': episode['rewards'][step_idx],
                    'action': episode['actions'][step_idx],
                    'latency': episode['latencies'][step_idx],
                    'replicas': episode['replicas'][step_idx],
                    'cpu_usage': episode['cpu_usage'][step_idx],
                    'error_rate': episode['error_rates'][step_idx],
                    'sla_violated': episode['sla_violations'][step_idx],
                    'cost': episode['costs'][step_idx],
                    'timestamp': episode['timestamps'][step_idx]
                })
        
        return pd.DataFrame(rows)
    
    def save_to_file(self, filepath: str):
        """
        Save metrics to JSON file
        
        Args:
            filepath: Path to save file
        """
        data = {
            'agent_name': self.agent_name,
            'episodes': self.episodes,
            'summary': self.get_performance_metrics(),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Metrics saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """
        Load metrics from JSON file
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.agent_name = data['agent_name']
        self.episodes = data['episodes']
        
        logger.info(f"📂 Metrics loaded from {filepath}")
    
    def plot_learning_curve(self, save_path: Optional[str] = None):
        """
        Plot learning curve
        
        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        curve = self.get_learning_curve()
        
        if not curve:
            logger.warning("No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rewards
        ax1.plot(curve['episodes'], curve['rewards'], alpha=0.3, label='Episode Reward')
        ax1.plot(curve['episodes'], curve['rewards_ma'], label='Moving Average (10 ep)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title(f'Learning Curve - {self.agent_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot episode lengths
        ax2.plot(curve['episodes'], curve['lengths'])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length (steps)')
        ax2.set_title('Episode Lengths')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Learning curve saved to {save_path}")
        
        return fig
    
    def compare_with_baseline(self, baseline_metrics: Dict) -> Dict:
        """
        Compare performance with baseline
        
        Args:
            baseline_metrics: Dictionary with baseline performance
        
        Returns:
            comparison: Dictionary with comparison results
        """
        current = self.get_performance_metrics()
        
        if not current or not baseline_metrics:
            return {}
        
        comparison = {
            'latency_improvement': (
                (baseline_metrics['avg_latency'] - current['avg_latency']) /
                baseline_metrics['avg_latency'] * 100
            ),
            'sla_improvement': (
                (baseline_metrics['sla_violation_rate'] - current['sla_violation_rate']) /
                baseline_metrics['sla_violation_rate'] * 100
            ) if baseline_metrics['sla_violation_rate'] > 0 else 0,
            'cost_reduction': (
                (baseline_metrics['avg_cost_per_step'] - current['avg_cost_per_step']) /
                baseline_metrics['avg_cost_per_step'] * 100
            ),
            'current': current,
            'baseline': baseline_metrics
        }
        
        return comparison