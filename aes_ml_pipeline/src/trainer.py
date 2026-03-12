"""
Training Module with Bayesian Optimization.

Handles model training with automatic hyperparameter optimization.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from bayes_opt import BayesianOptimization
from .models import get_model_class


def train_model_with_bo(model_name: str,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: Optional[np.ndarray],
                       y_val: Optional[np.ndarray],
                       latent_dim: int,
                       input_dim: int,
                       bo_iterations: int = 15,
                       bo_init_points: int = 5,
                       train_epochs: int = 300,
                       device: str = 'cpu',
                       verbose: int = 1) -> Tuple[torch.nn.Module, Dict[str, Any], BayesianOptimization]:
    """
    Train a model with Bayesian Optimization.
    
    Args:
        model_name: Name of model (from registry)
        X_train: Training inputs
        y_train: Training targets
        X_val: Validation inputs (if None, uses train set)
        y_val: Validation targets (if None, uses train set)
        latent_dim: Output dimension (PCA latent)
        input_dim: Input dimension
        bo_iterations: Number of BO iterations
        bo_init_points: Number of random initial points
        train_epochs: Epochs per BO trial
        device: Device to use ('cpu' or 'cuda')
        verbose: Verbosity level
        
    Returns:
        best_model: Trained model with best hyperparameters
        best_params: Best hyperparameters found
        bo_optimizer: BayesianOptimization object
    """
    model_class = get_model_class(model_name)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train).to(device)
    y_train_tensor = torch.tensor(y_train).to(device)
    
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.tensor(X_val).to(device)
        y_val_tensor = torch.tensor(y_val).to(device)
    else:
        # Use training set for validation if not provided
        X_val_tensor = X_train_tensor
        y_val_tensor = y_train_tensor
    
    # Get hyperparameter space
    temp_model = model_class(latent_dim=latent_dim, input_dim=input_dim)
    pbounds = temp_model.get_hyperparameter_space()
    del temp_model
    
    def objective(**params):
        """BO objective: train model and return negative validation loss."""
        # Get learning rate (remove from params for model init)
        params_copy = params.copy()
        lr = float(params_copy.pop('lr', 1e-3))
        
        # Convert integer parameters (BO returns floats)
        int_params = ['num_layers', 'hidden_dim', 'nhead', 'num_filters', 'kernel_size', 
                     'd_model', 'dim_feedforward']
        for key in int_params:
            if key in params_copy:
                params_copy[key] = int(params_copy[key])
        
        # Create model with current hyperparameters
        model = model_class(latent_dim=latent_dim, input_dim=input_dim, **params_copy)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        # Train
        model.train()
        for epoch in range(train_epochs):
            pred = model(X_train_tensor)
            loss = loss_fn(pred, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = loss_fn(val_pred, y_val_tensor).item()
        
        # BO maximizes, so return negative loss
        return -val_loss
    
    # Run Bayesian Optimization
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=verbose
    )
    
    optimizer.maximize(init_points=bo_init_points, n_iter=bo_iterations)
    
    # Get best parameters
    best_params = optimizer.max['params'].copy()
    best_lr = float(best_params.pop('lr', 1e-3))
    
    # Convert integer parameters
    int_params = ['num_layers', 'hidden_dim', 'nhead', 'num_filters', 'kernel_size',
                 'd_model', 'dim_feedforward']
    for key in int_params:
        if key in best_params:
            best_params[key] = int(best_params[key])
    
    # Train final model with best parameters
    best_model = model_class(latent_dim=latent_dim, input_dim=input_dim, **best_params)
    best_model = best_model.to(device)
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr)
    loss_fn = nn.MSELoss()
    
    if verbose > 0:
        print(f"\nTraining final model with best params: lr={best_lr:.6f}, {best_params}")
    
    best_model.train()
    final_epochs = train_epochs * 2  # Train longer for final model
    for epoch in range(final_epochs):
        pred = best_model(X_train_tensor)
        loss = loss_fn(pred, y_train_tensor)
        best_optimizer.zero_grad()
        loss.backward()
        best_optimizer.step()
        
        if verbose > 0 and (epoch + 1) % 100 == 0:
            best_model.eval()
            with torch.no_grad():
                val_pred = best_model(X_val_tensor)
                val_loss = loss_fn(val_pred, y_val_tensor).item()
            print(f"  Epoch {epoch+1}/{final_epochs}, Val Loss: {val_loss:.6f}")
            best_model.train()
    
    best_model.eval()
    return best_model, best_params, optimizer

