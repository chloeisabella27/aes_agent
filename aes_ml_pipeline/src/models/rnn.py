"""Basic RNN Model."""
import torch.nn as nn
from .base import BaseTemporalModel


class RNNTemporalModel(BaseTemporalModel):
    """Basic RNN for sequence learning."""
    
    def __init__(self, latent_dim: int, input_dim: int = 2, hidden_dim: int = 128, 
                 num_layers: int = 2, **kwargs):
        super().__init__(latent_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
    
    def get_input_dim(self) -> int:
        return self.input_dim
    
    def get_hyperparameter_space(self) -> dict:
        return {
            'lr': (1e-4, 5e-3),
            'hidden_dim': (32, 256),
            'num_layers': (1, 3)
        }
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        return self.fc(last_output)






