"""Transformer Model with self-attention."""
import torch
import torch.nn as nn
from .base import BaseTemporalModel


class TransformerTemporalModel(BaseTemporalModel):
    """Transformer with self-attention for temporal dependencies."""
    
    def __init__(self, latent_dim: int, input_dim: int = 2, d_model: int = 128, 
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, 
                 dropout: float = 0.1, **kwargs):
        super().__init__(latent_dim)
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))  # Max seq len 100
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, latent_dim)
    
    def get_input_dim(self) -> int:
        return self.input_dim
    
    def get_hyperparameter_space(self) -> dict:
        return {
            'lr': (1e-4, 5e-3),
            'd_model': (64, 256),
            'nhead': (2, 8),
            'num_layers': (1, 4),
            'dim_feedforward': (128, 512),
            'dropout': (0.0, 0.3)
        }
    
    def forward(self, x):
        # x: (batch, input_dim) or (batch, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        x = self.transformer(x)
        # Take last timestep
        last_output = x[:, -1, :]
        return self.fc(last_output)






