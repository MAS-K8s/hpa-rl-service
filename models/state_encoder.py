import numpy as np
import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    Advanced state encoder with LSTM for temporal patterns
    Converts raw metrics into meaningful state representations
    """
    
    def __init__(self, input_size=18, hidden_size=128, num_layers=2):
        super(StateEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=4,
            dropout=0.1
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the encoder
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional hidden state from previous step
        
        Returns:
            encoded: Encoded state representation
            hidden: Updated hidden state
        """
        batch_size = x.size(0)
        
        # LSTM encoding
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply self-attention
        # Transpose for attention: (seq_len, batch, hidden)
        lstm_out_t = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out_t, lstm_out_t, lstm_out_t)
        
        # Transpose back: (batch, seq_len, hidden)
        attn_out = attn_out.transpose(0, 1)
        
        # Take last timestep
        last_step = attn_out[:, -1, :]
        
        # Feature extraction
        features = self.feature_extractor(last_step)
        
        # Output projection
        encoded = self.output_projection(features)
        
        return encoded, hidden
    
    def encode_single_state(self, state):
        """
        Encode a single state vector
        
        Args:
            state: State vector of shape (input_size,)
        
        Returns:
            encoded: Encoded representation
        """
        # Add batch and sequence dimensions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            encoded, _ = self.forward(state_tensor)
        
        return encoded.squeeze(0)


class StatisicalFeatureExtractor:
    """
    Extracts statistical features from raw metrics
    Can be used to augment state representation
    """
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []
    
    def add_metrics(self, metrics):
        """Add metrics to history"""
        self.history.append(metrics)
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def extract_features(self):
        """
        Extract statistical features from metrics history
        
        Returns:
            features: Dictionary of statistical features
        """
        if len(self.history) < 2:
            return self._get_default_features()
        
        features = {}
        
        # CPU features
        cpu_values = [m.get('cpu_usage', 0) for m in self.history]
        features['cpu_mean'] = np.mean(cpu_values)
        features['cpu_std'] = np.std(cpu_values)
        features['cpu_min'] = np.min(cpu_values)
        features['cpu_max'] = np.max(cpu_values)
        features['cpu_trend'] = self._calculate_trend(cpu_values)
        
        # Latency features
        latency_values = [m.get('latency_p95', 0) for m in self.history]
        features['latency_mean'] = np.mean(latency_values)
        features['latency_std'] = np.std(latency_values)
        features['latency_min'] = np.min(latency_values)
        features['latency_max'] = np.max(latency_values)
        features['latency_trend'] = self._calculate_trend(latency_values)
        
        # Request rate features
        request_values = [m.get('request_rate', 0) for m in self.history]
        features['request_mean'] = np.mean(request_values)
        features['request_std'] = np.std(request_values)
        features['request_trend'] = self._calculate_trend(request_values)
        
        # Volatility (coefficient of variation)
        features['cpu_volatility'] = features['cpu_std'] / (features['cpu_mean'] + 1e-6)
        features['latency_volatility'] = features['latency_std'] / (features['latency_mean'] + 1e-6)
        
        return features
    
    def _calculate_trend(self, values):
        """Calculate linear trend of values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _get_default_features(self):
        """Return default features when history is insufficient"""
        return {
            'cpu_mean': 0.0,
            'cpu_std': 0.0,
            'cpu_min': 0.0,
            'cpu_max': 0.0,
            'cpu_trend': 0.0,
            'latency_mean': 0.0,
            'latency_std': 0.0,
            'latency_min': 0.0,
            'latency_max': 0.0,
            'latency_trend': 0.0,
            'request_mean': 0.0,
            'request_std': 0.0,
            'request_trend': 0.0,
            'cpu_volatility': 0.0,
            'latency_volatility': 0.0,
        }
    
    def get_feature_vector(self):
        """Get features as numpy array"""
        features = self.extract_features()
        return np.array(list(features.values()), dtype=np.float32)


class MetricsNormalizer:
    """
    Normalizes metrics to improve learning stability
    """
    
    def __init__(self):
        self.running_mean = {}
        self.running_std = {}
        self.n_samples = 0
        self.alpha = 0.01  # Exponential moving average factor
    
    def normalize(self, metrics):
        """
        Normalize metrics using running statistics
        
        Args:
            metrics: Dictionary of metrics
        
        Returns:
            normalized: Dictionary of normalized metrics
        """
        normalized = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Update running statistics
                if key not in self.running_mean:
                    self.running_mean[key] = value
                    self.running_std[key] = 1.0
                else:
                    self.running_mean[key] = (
                        (1 - self.alpha) * self.running_mean[key] +
                        self.alpha * value
                    )
                    
                    diff = value - self.running_mean[key]
                    self.running_std[key] = np.sqrt(
                        (1 - self.alpha) * self.running_std[key] ** 2 +
                        self.alpha * diff ** 2
                    )
                
                # Normalize
                normalized[key] = (
                    (value - self.running_mean[key]) /
                    (self.running_std[key] + 1e-6)
                )
            else:
                normalized[key] = value
        
        self.n_samples += 1
        return normalized
    
    def denormalize(self, key, normalized_value):
        """Denormalize a single value"""
        if key in self.running_mean:
            return (
                normalized_value * self.running_std[key] +
                self.running_mean[key]
            )
        return normalized_value