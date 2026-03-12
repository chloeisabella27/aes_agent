"""
Evaluation Module - Metrics and model comparison.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import torch
from .trainer import train_model_with_bo


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics: MSE, MAE, RMSE, R², Correlation
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Correlation
    if y_true.size > 1:
        corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    else:
        corr = 0.0
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Correlation': corr
    }


class ModelComparison:
    """Compare multiple models on the same test set."""
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                 X_test: np.ndarray, y_test: np.ndarray,
                 latent_dim: int, input_dim: int, device: str = 'cpu'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.results = {}
        self.models = {}
    
    def train_and_evaluate(self, model_names: List[str], 
                          bo_iterations: int = 10, 
                          bo_init_points: int = 3, 
                          train_epochs: int = 200) -> Dict[str, Any]:
        """Train multiple models and evaluate on test set."""
        for model_name in model_names:
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()} model")
            print(f"{'='*60}")
            
            # Train with BO
            model, best_params, bo_opt = train_model_with_bo(
                model_name, self.X_train, self.y_train, 
                self.X_val, self.y_val,
                self.latent_dim, self.input_dim,
                bo_iterations=bo_iterations, 
                bo_init_points=bo_init_points,
                train_epochs=train_epochs, 
                device=self.device
            )
            
            # Evaluate on test set
            X_test_tensor = torch.tensor(self.X_test).to(self.device)
            model.eval()
            with torch.no_grad():
                y_pred_test = model(X_test_tensor).cpu().numpy()
            
            # Compute metrics
            metrics = compute_metrics(self.y_test, y_pred_test)
            
            # Store results
            self.results[model_name] = {
                'metrics': metrics,
                'best_params': best_params,
                'bo_optimizer': bo_opt,
                'y_pred': y_pred_test
            }
            self.models[model_name] = model
            
            print(f"\n{model_name.upper()} Test Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")
        
        return self.results
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get comparison results as DataFrame."""
        data = {}
        for model_name, result in self.results.items():
            for metric, value in result['metrics'].items():
                if metric not in data:
                    data[metric] = {}
                data[metric][model_name] = value
        
        return pd.DataFrame(data)
    
    def get_best_model(self, metric: str = 'RMSE') -> str:
        """Get name of best model based on metric (lower is better for MSE/MAE/RMSE)."""
        df = self.get_comparison_dataframe()
        if metric in ['MSE', 'MAE', 'RMSE']:
            return df[metric].idxmin()
        else:  # R², Correlation (higher is better)
            return df[metric].idxmax()

