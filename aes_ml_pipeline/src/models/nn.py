"""Enhanced Feedforward Neural Network."""
import torch.nn as nn
from .base import BaseTemporalModel


class EnhancedNNModel(BaseTemporalModel):
    """Enhanced feedforward NN with experiment encoding."""
    
    def __init__(self, latent_dim: int, input_dim: int = 2, hidden_dim: int = 128, 
                 num_layers: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__(latent_dim)
        self.input_dim = input_dim
        
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)
    
    def get_input_dim(self) -> int:
        return self.input_dim
    
    def get_hyperparameter_space(self) -> dict:
        return {
            'lr': (1e-4, 5e-3),
            'hidden_dim': (32, 256),
            'num_layers': (2, 5),
            'dropout': (0.0, 0.3)
        }
    
    def forward(self, x):
        return self.net(x)






