"""Temporal Convolutional Network."""
import torch.nn as nn
from .base import BaseTemporalModel


class TCNTemporalModel(BaseTemporalModel):
    """Temporal Convolutional Network with causal convolutions."""
    
    def __init__(self, latent_dim: int, input_dim: int = 2, num_filters: int = 64, 
                 kernel_size: int = 3, num_layers: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__(latent_dim)
        self.input_dim = input_dim
        
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size, 
                                   dilation=dilation, padding=padding))
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = num_filters
        
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, latent_dim)
    
    def get_input_dim(self) -> int:
        return self.input_dim
    
    def get_hyperparameter_space(self) -> dict:
        return {
            'lr': (1e-4, 5e-3),
            'num_filters': (32, 128),
            'kernel_size': (2, 5),
            'num_layers': (2, 5),
            'dropout': (0.0, 0.3)
        }
    
    def forward(self, x):
        # x: (batch, input_dim) or (batch, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        else:
            x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        x = self.conv_layers(x)
        # Take last timestep (causal padding ensures no future leakage)
        x = x[:, :, -1]  # (batch, num_filters)
        return self.fc(x)






