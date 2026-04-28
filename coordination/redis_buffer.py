import json
import logging
import numpy as np
from typing import List, Optional, Tuple
import redis
import torch

logger = logging.getLogger(__name__)

class SharedExperienceBuffer:
    def __init__(self, redis_client: redis.Redis, max_size: int = 10000, key_prefix: str = "shared_exp:"):
        self.redis = redis_client
        self.max_size = max_size
        self.prefix = key_prefix
        self.list_key = f"{self.prefix}list"

    def _to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    def add(self, agent_key: str, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool, log_prob: float = 0.0, value: float = 0.0):
        transition = {
            "agent_key": agent_key,
            "state": self._to_serializable(state),
            "action": int(action),
            "reward": float(reward),
            "next_state": self._to_serializable(next_state),
            "done": bool(done),
            "log_prob": float(log_prob),
            "value": float(value),
        }
        self.redis.lpush(self.list_key, json.dumps(transition))
        self.redis.ltrim(self.list_key, 0, self.max_size - 1)

    def sample(self, batch_size: int) -> Optional[Tuple]:
        length = self.redis.llen(self.list_key)
        if length < batch_size:
            return None
        indices = np.random.choice(length, batch_size, replace=False)
        batch = []
        for idx in indices:
            item = self.redis.lindex(self.list_key, int(idx))
            if item:
                batch.append(json.loads(item))
        if len(batch) < batch_size:
            return None
        # Convert to numpy arrays with proper types
        states = np.array([b["state"] for b in batch], dtype=np.float32)
        actions = np.array([b["action"] for b in batch], dtype=np.int64)
        rewards = np.array([b["reward"] for b in batch], dtype=np.float32)
        next_states = np.array([b["next_state"] for b in batch], dtype=np.float32)
        dones = np.array([b["done"] for b in batch], dtype=np.float32)
        log_probs = np.array([b["log_prob"] for b in batch], dtype=np.float32)
        values = np.array([b["value"] for b in batch], dtype=np.float32)
        agent_keys = [b["agent_key"] for b in batch]
        return (states, actions, rewards, next_states, dones, log_probs, values, agent_keys)

    def clear(self):
        self.redis.delete(self.list_key)

    def size(self) -> int:
        return self.redis.llen(self.list_key)