import json
import logging
import threading
from typing import Callable, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Publish-subscribe message bus for agent communication
    Uses Redis pub/sub for distributed messaging
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize message bus
        
        Args:
            redis_client: Redis client for pub/sub
        """
        self.redis_client = redis_client
        self.enabled = redis_client is not None
        self.subscribers = {}  # channel -> [callbacks]
        self.pubsub = None
        self.listen_thread = None
        self.running = False
        
        if self.enabled:
            self.pubsub = self.redis_client.pubsub()
            logger.info("✅ Message bus initialized")
        else:
            logger.warning("⚠️ Message bus disabled (no Redis)")
    
    def publish(self, channel: str, message: Dict) -> bool:
        """
        Publish message to channel
        
        Args:
            channel: Channel name
            message: Message dictionary
        
        Returns:
            success: True if published
        """
        if not self.enabled:
            return False
        
        try:
            message_with_meta = {
                **message,
                'timestamp': datetime.now().isoformat(),
                'channel': channel
            }
            
            self.redis_client.publish(
                channel,
                json.dumps(message_with_meta)
            )
            
            logger.debug(f"📤 Published to {channel}: {message.get('type', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    def subscribe(self, channel: str, callback: Callable) -> bool:
        """
        Subscribe to channel with callback
        
        Args:
            channel: Channel name
            callback: Function to call on message (takes message dict)
        
        Returns:
            success: True if subscribed
        """
        if not self.enabled:
            return False
        
        try:
            if channel not in self.subscribers:
                self.subscribers[channel] = []
                self.pubsub.subscribe(channel)
                logger.info(f"📥 Subscribed to channel: {channel}")
            
            self.subscribers[channel].append(callback)
            
            # Start listen thread if not running
            if not self.running:
                self.start_listening()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            return False
    
    def unsubscribe(self, channel: str, callback: Optional[Callable] = None) -> bool:
        """
        Unsubscribe from channel
        
        Args:
            channel: Channel name
            callback: Specific callback to remove (None = remove all)
        
        Returns:
            success: True if unsubscribed
        """
        if not self.enabled or channel not in self.subscribers:
            return False
        
        try:
            if callback:
                self.subscribers[channel].remove(callback)
            else:
                self.subscribers[channel] = []
            
            if not self.subscribers[channel]:
                self.pubsub.unsubscribe(channel)
                del self.subscribers[channel]
                logger.info(f"📤 Unsubscribed from channel: {channel}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")
            return False
    
    def start_listening(self):
        """Start background thread to listen for messages"""
        if not self.enabled or self.running:
            return
        
        self.running = True
        self.listen_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self.listen_thread.start()
        logger.info("🎧 Message bus listening thread started")
    
    def stop_listening(self):
        """Stop listening thread"""
        if not self.running:
            return
        
        self.running = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2.0)
        logger.info("🛑 Message bus listening thread stopped")
    
    def _listen_loop(self):
        """Background loop to receive messages"""
        logger.info("🎧 Started listening for messages")
        
        try:
            for message in self.pubsub.listen():
                if not self.running:
                    break
                
                if message['type'] == 'message':
                    self._handle_message(message)
                    
        except Exception as e:
            logger.error(f"Error in listen loop: {e}")
        finally:
            logger.info("🛑 Stopped listening for messages")
    
    def _handle_message(self, redis_message: Dict):
        """Handle incoming message"""
        try:
            channel = redis_message['channel']
            if isinstance(channel, bytes):
                channel = channel.decode()
            
            data = redis_message['data']
            if isinstance(data, bytes):
                data = data.decode()
            
            message = json.loads(data)
            
            # Call all subscribers for this channel
            if channel in self.subscribers:
                for callback in self.subscribers[channel]:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in callback for {channel}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")


class AgentMessenger:
    """
    High-level messaging interface for agents
    Provides common message types and patterns
    """
    
    def __init__(self, agent_key: str, message_bus: MessageBus):
        """
        Initialize messenger for an agent
        
        Args:
            agent_key: Unique agent identifier
            message_bus: MessageBus instance
        """
        self.agent_key = agent_key
        self.bus = message_bus
        
        # Subscribe to agent-specific channel
        if self.bus.enabled:
            self.bus.subscribe(f"agent:{agent_key}", self._on_direct_message)
            self.bus.subscribe("broadcast", self._on_broadcast)
    
    def send_status_update(self, status: Dict):
        """Broadcast agent status"""
        self.bus.publish("agent_status", {
            'type': 'status_update',
            'agent': self.agent_key,
            'status': status
        })
    
    def send_scaling_intent(self, from_replicas: int, to_replicas: int):
        """Notify about scaling intent"""
        self.bus.publish("scaling_intents", {
            'type': 'scaling_intent',
            'agent': self.agent_key,
            'from': from_replicas,
            'to': to_replicas,
            'delta': to_replicas - from_replicas
        })
    
    def send_experience(self, experience: Dict):
        """Share experience with other agents"""
        self.bus.publish("experiences", {
            'type': 'experience',
            'agent': self.agent_key,
            'experience': experience
        })
    
    def send_alert(self, level: str, message: str):
        """Send alert message"""
        self.bus.publish("alerts", {
            'type': 'alert',
            'agent': self.agent_key,
            'level': level,
            'message': message
        })
    
    def send_direct_message(self, target_agent: str, message: Dict):
        """Send message to specific agent"""
        self.bus.publish(f"agent:{target_agent}", {
            'type': 'direct_message',
            'from': self.agent_key,
            'message': message
        })
    
    def broadcast(self, message: Dict):
        """Broadcast to all agents"""
        self.bus.publish("broadcast", {
            'type': 'broadcast',
            'from': self.agent_key,
            'message': message
        })
    
    def _on_direct_message(self, message: Dict):
        """Handle direct message"""
        logger.info(f"📨 Direct message for {self.agent_key}: {message.get('type')}")
        # Could implement callbacks here
    
    def _on_broadcast(self, message: Dict):
        """Handle broadcast message"""
        if message.get('from') != self.agent_key:
            logger.debug(f"📻 Broadcast from {message.get('from')}: {message.get('type')}")