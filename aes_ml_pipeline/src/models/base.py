"""Base model interface for temporal prediction."""
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseTemporalModel(nn.Module, ABC):
    """Abstract base class for all temporal prediction models."""
    
    def __init__(self, latent_dim: int, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
    
    @abstractmethod
    def get_input_dim(self) -> int:
        """Return the input feature dimension."""
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self) -> dict:
        """Return Bayesian optimization search space as dict."""
        pass
    
    @abstractmethod
    def forward(self, x):
        """Forward pass."""
        pass






