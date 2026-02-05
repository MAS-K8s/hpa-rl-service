import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class MultiAgentCoordinator:
    """
    Coordinates multiple RL agents for cluster-wide optimization
    
    Features:
    - Resource negotiation
    - Experience sharing
    - Consensus-based decisions
    - Cluster capacity management
    """
    
    def __init__(self, redis_client=None, cluster_capacity: int = 50):
        """
        Initialize coordinator
        
        Args:
            redis_client: Redis client for distributed coordination
            cluster_capacity: Total cluster capacity (max replicas)
        """
        self.redis_client = redis_client
        self.cluster_capacity = cluster_capacity
        self.enabled = redis_client is not None
        
        if not self.enabled:
            logger.warning("⚠️ Redis not available - multi-agent coordination disabled")
    
    def publish_agent_state(self, agent_key: str, state_info: Dict) -> bool:
        """
        Publish agent state to Redis for other agents to see
        
        Args:
            agent_key: Unique agent identifier (namespace/deployment)
            state_info: Dictionary with agent state
        
        Returns:
            success: True if published successfully
        """
        if not self.enabled:
            return False
        
        try:
            # Store in Redis with 60s TTL
            self.redis_client.setex(
                f"agent_state:{agent_key}",
                60,  # Expires after 60 seconds
                json.dumps({
                    **state_info,
                    'timestamp': datetime.now().isoformat(),
                    'ttl': 60
                })
            )
            
            # Also publish to channel for real-time updates
            self.redis_client.publish(
                f"agent_updates",
                json.dumps({
                    'agent': agent_key,
                    'state': state_info
                })
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish agent state: {e}")
            return False
    
    def get_cluster_status(self) -> Dict:
        """
        Get cluster-wide resource status
        
        Returns:
            status: Dictionary with cluster metrics
        """
        if not self.enabled:
            return {
                'total_agents': 0,
                'total_replicas': 0,
                'available_capacity': self.cluster_capacity,
                'utilization': 0.0,
                'agents': []
            }
        
        try:
            # Get all agent states
            agent_keys = self.redis_client.keys("agent_state:*")
            
            total_replicas = 0
            agents_info = []
            
            for key in agent_keys:
                try:
                    state_data = self.redis_client.get(key)
                    if state_data:
                        state = json.loads(state_data)
                        replicas = state.get('replicas', 0)
                        total_replicas += replicas
                        
                        agents_info.append({
                            'agent': key.decode() if isinstance(key, bytes) else key,
                            'replicas': replicas,
                            'cpu': state.get('cpu_usage', 0),
                            'last_action': state.get('last_action', 'unknown')
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse agent state: {e}")
                    continue
            
            available_capacity = max(0, self.cluster_capacity - total_replicas)
            utilization = total_replicas / self.cluster_capacity if self.cluster_capacity > 0 else 0
            
            return {
                'total_agents': len(agents_info),
                'total_replicas': total_replicas,
                'available_capacity': available_capacity,
                'cluster_capacity': self.cluster_capacity,
                'utilization': utilization,
                'agents': agents_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {
                'total_agents': 0,
                'total_replicas': 0,
                'available_capacity': self.cluster_capacity,
                'error': str(e)
            }
    
    def request_scaling_approval(self, agent_key: str, current_replicas: int, 
                                 desired_replicas: int) -> Tuple[bool, str]:
        """
        Request approval for scaling action
        
        Implements consensus mechanism for resource allocation
        
        Args:
            agent_key: Agent requesting scaling
            current_replicas: Current replica count
            desired_replicas: Desired replica count
        
        Returns:
            (approved, message): Approval status and reason
        """
        if not self.enabled:
            return True, "Coordination disabled"
        
        try:
            cluster_status = self.get_cluster_status()
            
            # Calculate required capacity change
            delta = desired_replicas - current_replicas
            
            if delta > 0:
                # Scaling up - check capacity
                if cluster_status['available_capacity'] < delta:
                    return False, (
                        f"Insufficient cluster capacity: "
                        f"need {delta}, have {cluster_status['available_capacity']}"
                    )
                
                # Check if cluster is heavily utilized
                if cluster_status['utilization'] > 0.9:
                    # High utilization - need stronger justification
                    # Could implement priority system here
                    logger.info(f"⚠️ Cluster highly utilized ({cluster_status['utilization']:.1%}), "
                               f"but approving {agent_key} scale up")
            
            # Record scaling intent
            self.redis_client.setex(
                f"scaling_intent:{agent_key}",
                30,  # 30 second TTL
                json.dumps({
                    'agent': agent_key,
                    'from': current_replicas,
                    'to': desired_replicas,
                    'delta': delta,
                    'timestamp': datetime.now().isoformat()
                })
            )
            
            return True, "Approved"
            
        except Exception as e:
            logger.error(f"Approval check failed: {e}")
            # Fail open - allow scaling if coordination fails
            return True, f"Fallback approval ({str(e)})"
    
    def share_experience(self, agent_key: str, experience: Dict) -> bool:
        """
        Share successful experience with other agents
        
        Args:
            agent_key: Agent sharing experience
            experience: Experience dictionary with state, action, reward, etc.
        
        Returns:
            success: True if shared successfully
        """
        if not self.enabled:
            return False
        
        try:
            # Add metadata
            experience['agent'] = agent_key
            experience['shared_at'] = datetime.now().isoformat()
            
            # Store in shared experience pool
            self.redis_client.lpush(
                f"shared_experiences:{agent_key}",
                pickle.dumps(experience)
            )
            
            # Keep only last 1000 experiences per agent
            self.redis_client.ltrim(f"shared_experiences:{agent_key}", 0, 999)
            
            logger.debug(f"📤 Shared experience from {agent_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to share experience: {e}")
            return False
    
    def get_shared_experiences(self, agent_key: str, count: int = 10) -> List[Dict]:
        """
        Get shared experiences from similar agents
        
        Agents can learn from others' successes
        
        Args:
            agent_key: Agent requesting experiences
            count: Number of experiences to retrieve
        
        Returns:
            experiences: List of experience dictionaries
        """
        if not self.enabled:
            return []
        
        try:
            # Get experiences from this agent's pool
            experience_bytes = self.redis_client.lrange(
                f"shared_experiences:{agent_key}",
                0,
                count - 1
            )
            
            experiences = []
            for exp_bytes in experience_bytes:
                try:
                    exp = pickle.loads(exp_bytes)
                    experiences.append(exp)
                except Exception as e:
                    logger.warning(f"Failed to deserialize experience: {e}")
                    continue
            
            logger.debug(f"📥 Retrieved {len(experiences)} shared experiences for {agent_key}")
            return experiences
            
        except Exception as e:
            logger.error(f"Failed to get shared experiences: {e}")
            return []
    
    def get_scaling_intents(self) -> List[Dict]:
        """
        Get all active scaling intents from agents
        
        Returns:
            intents: List of scaling intent dictionaries
        """
        if not self.enabled:
            return []
        
        try:
            intent_keys = self.redis_client.keys("scaling_intent:*")
            intents = []
            
            for key in intent_keys:
                try:
                    intent_data = self.redis_client.get(key)
                    if intent_data:
                        intent = json.loads(intent_data)
                        intents.append(intent)
                except Exception as e:
                    logger.warning(f"Failed to parse intent: {e}")
                    continue
            
            return intents
            
        except Exception as e:
            logger.error(f"Failed to get scaling intents: {e}")
            return []
    
    def negotiate_resources(self, requests: List[Dict]) -> Dict[str, int]:
        """
        Negotiate resource allocation among multiple agents
        
        Implements fair allocation when resources are scarce
        
        Args:
            requests: List of {agent, current, desired} dictionaries
        
        Returns:
            allocation: Dictionary mapping agent_key to approved replicas
        """
        if not self.enabled or not requests:
            return {}
        
        cluster_status = self.get_cluster_status()
        available = cluster_status['available_capacity']
        
        # Calculate total requested increase
        total_requested = sum(
            max(0, req['desired'] - req['current'])
            for req in requests
        )
        
        allocation = {}
        
        if total_requested <= available:
            # Enough capacity for all
            for req in requests:
                allocation[req['agent']] = req['desired']
        else:
            # Need to allocate fairly
            # Priority: agents with higher load or SLA violations
            # For now, simple proportional allocation
            
            for req in requests:
                increase = max(0, req['desired'] - req['current'])
                if total_requested > 0:
                    proportion = increase / total_requested
                    allocated_increase = int(proportion * available)
                    allocation[req['agent']] = req['current'] + allocated_increase
                else:
                    allocation[req['agent']] = req['current']
        
        logger.info(f"🤝 Negotiated resources: {allocation}")
        return allocation
    
    def get_stats(self) -> Dict:
        """Get coordinator statistics"""
        return {
            'enabled': self.enabled,
            'cluster_status': self.get_cluster_status(),
            'active_intents': len(self.get_scaling_intents()),
        }