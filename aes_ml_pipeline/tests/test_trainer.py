"""Unit test for trainer module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from src.trainer import train_model_with_bo
from src.models import get_model_class


def test_trainer_import():
    """Test that trainer imports correctly."""
    print("Testing trainer import...")
    from src.trainer import train_model_with_bo
    assert train_model_with_bo is not None
    print("✓ Trainer imports correctly")


def test_model_creation():
    """Test that models can be created."""
    print("\nTesting model creation...")
    for model_name in ['nn', 'lstm', 'gru']:
        model_class = get_model_class(model_name)
        model = model_class(latent_dim=5, input_dim=2)
        assert model is not None
        print(f"✓ {model_name} model created")
        
        # Test forward pass
        x = torch.randn(10, 2)
        y = model(x)
        assert y.shape == (10, 5), f"Wrong output shape: {y.shape}"
        print(f"  Forward pass: {x.shape} -> {y.shape}")


def test_trainer_with_dummy_data():
    """Test trainer with small dummy dataset."""
    print("\nTesting trainer with dummy data...")
    
    # Create dummy data
    n_train = 100
    n_val = 20
    input_dim = 2
    latent_dim = 5
    
    X_train = np.random.randn(n_train, input_dim).astype(np.float32)
    y_train = np.random.randn(n_train, latent_dim).astype(np.float32)
    X_val = np.random.randn(n_val, input_dim).astype(np.float32)
    y_val = np.random.randn(n_val, latent_dim).astype(np.float32)
    
    print(f"  Training data: {X_train.shape}, {y_train.shape}")
    print(f"  Validation data: {X_val.shape}, {y_val.shape}")
    
    try:
        model, best_params, bo_opt = train_model_with_bo(
            'nn',
            X_train, y_train,
            X_val, y_val,
            latent_dim=latent_dim,
            input_dim=input_dim,
            bo_iterations=2,  # Very small for testing
            bo_init_points=1,
            train_epochs=10,  # Very small for testing
            device='cpu',
            verbose=0
        )
        
        assert model is not None
        assert best_params is not None
        print("✓ Trainer completed successfully")
        print(f"  Best params: {best_params}")
        
        # Test prediction
        X_test = torch.tensor(X_val[:5])
        model.eval()
        with torch.no_grad():
            pred = model(X_test)
        assert pred.shape == (5, latent_dim)
        print(f"  Prediction test: {X_test.shape} -> {pred.shape}")
        
    except Exception as e:
        print(f"✗ Trainer failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    print("="*60)
    print("TRAINER TESTS")
    print("="*60)
    test_trainer_import()
    test_model_creation()
    test_trainer_with_dummy_data()
    print("\n" + "="*60)
    print("All trainer tests passed!")
    print("="*60)






