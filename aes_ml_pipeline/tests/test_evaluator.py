"""Unit test for evaluator module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from src.evaluator import compute_metrics, ModelComparison
from src.models import get_model_class


def test_metrics_computation():
    """Test metrics computation."""
    print("Testing metrics computation...")
    
    y_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_pred = y_true + 0.1  # Small error
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert 'MSE' in metrics
    assert 'MAE' in metrics
    assert 'RMSE' in metrics
    assert 'R²' in metrics
    assert 'Correlation' in metrics
    
    print("✓ Metrics computed successfully")
    print(f"  MSE: {metrics['MSE']:.6f}")
    print(f"  R²: {metrics['R²']:.6f}")


def test_model_comparison():
    """Test ModelComparison class."""
    print("\nTesting ModelComparison...")
    
    # Create dummy data
    n_train = 50
    n_val = 10
    n_test = 20
    input_dim = 2
    latent_dim = 5
    
    X_train = np.random.randn(n_train, input_dim).astype(np.float32)
    y_train = np.random.randn(n_train, latent_dim).astype(np.float32)
    X_val = np.random.randn(n_val, input_dim).astype(np.float32)
    y_val = np.random.randn(n_val, latent_dim).astype(np.float32)
    X_test = np.random.randn(n_test, input_dim).astype(np.float32)
    y_test = np.random.randn(n_test, latent_dim).astype(np.float32)
    
    comparison = ModelComparison(
        X_train, y_train, X_val, y_val,
        X_test, y_test,
        latent_dim, input_dim, device='cpu'
    )
    
    assert comparison is not None
    print("✓ ModelComparison created")
    
    # Test with minimal training
    try:
        results = comparison.train_and_evaluate(
            ['nn'],
            bo_iterations=1,
            bo_init_points=1,
            train_epochs=5
        )
        
        assert 'nn' in results
        assert 'metrics' in results['nn']
        print("✓ ModelComparison training works")
        
        df = comparison.get_comparison_dataframe()
        assert len(df) > 0
        print("✓ Comparison dataframe created")
        
    except Exception as e:
        print(f"✗ ModelComparison failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    print("="*60)
    print("EVALUATOR TESTS")
    print("="*60)
    test_metrics_computation()
    test_model_comparison()
    print("\n" + "="*60)
    print("All evaluator tests passed!")
    print("="*60)






