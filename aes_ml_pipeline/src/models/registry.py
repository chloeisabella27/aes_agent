"""Model registry for easy model access."""
from .nn import EnhancedNNModel
from .lstm import LSTMTemporalModel
from .rnn import RNNTemporalModel
from .gru import GRUTemporalModel
from .transformer import TransformerTemporalModel
from .tcn import TCNTemporalModel


MODEL_REGISTRY = {
    'nn': EnhancedNNModel,
    'lstm': LSTMTemporalModel,
    'rnn': RNNTemporalModel,
    'gru': GRUTemporalModel,
    'transformer': TransformerTemporalModel,
    'tcn': TCNTemporalModel,
}


def get_model_class(model_name: str):
    """Get model class from registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]






