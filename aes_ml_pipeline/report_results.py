#!/usr/bin/env python3
"""
Generate a report on model performance from training results.
"""
import sys
import pickle
import pandas as pd
from pathlib import Path

def main():
    results_path = Path('outputs/comparison_results.pkl')
    metrics_path = Path('outputs/metrics.csv')
    
    print("="*60)
    print("MODEL PERFORMANCE REPORT")
    print("="*60)
    
    # Try to load from pickle, fall back to CSV
    results = None
    if results_path.exists():
        try:
            with open(results_path, 'rb') as f:
                data = pickle.load(f)
            results = data.get('results')
        except Exception as e:
            print(f"Note: Could not load pickle file: {e}")
            print("Using CSV file instead...\n")
    
    if results is None and metrics_path.exists():
        # Create results dict from CSV
        df = pd.read_csv(metrics_path, index_col=0)
        results = {}
        for model_name in df.index:
            results[model_name] = {
                'metrics': {
                    'MSE': df.loc[model_name, 'MSE'],
                    'MAE': df.loc[model_name, 'MAE'],
                    'RMSE': df.loc[model_name, 'RMSE'],
                    'R²': df.loc[model_name, 'R²'],
                    'Correlation': df.loc[model_name, 'Correlation'],
                },
                'best_params': {}  # Not available from CSV
            }
    
    if results is None:
        print(f"Error: No results found. Run training first.")
        return
    
    # Print metrics for each model
    print("\nModel Performance Summary:")
    print("-" * 60)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name.upper()} Model:")
        print(f"  MSE:  {metrics['MSE']:.6f}")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  R²:   {metrics['R²']:.6f}")
        print(f"  Correlation: {metrics['Correlation']:.6f}")
        print(f"  Best Params: {result['best_params']}")
    
    # Load and display metrics table
    if metrics_path.exists():
        print("\n" + "="*60)
        print("METRICS COMPARISON TABLE")
        print("="*60)
        df = pd.read_csv(metrics_path)
        print(df.to_string(index=True))
        
        # Find best model
        print("\n" + "="*60)
        print("BEST MODEL ANALYSIS")
        print("="*60)
        
        best_rmse = df['RMSE'].idxmin()
        best_r2 = df['R²'].idxmax()
        best_corr = df['Correlation'].idxmax()
        
        print(f"Best RMSE (lower is better): {best_rmse} ({df.loc[best_rmse, 'RMSE']:.6f})")
        print(f"Best R² (higher is better):   {best_r2} ({df.loc[best_r2, 'R²']:.6f})")
        print(f"Best Correlation:             {best_corr} ({df.loc[best_corr, 'Correlation']:.6f})")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # Check for negative R² (worse than baseline)
    negative_r2 = [name for name, r in results.items() if r['metrics']['R²'] < 0]
    if negative_r2:
        print(f"⚠️  Models with negative R² (worse than baseline): {', '.join(negative_r2)}")
        print("   This suggests the models are not generalizing well.")
        print("   Consider:")
        print("   - More training data")
        print("   - Different hyperparameters")
        print("   - Different model architectures")
    
    # Best performing model
    best_model = min(results.items(), key=lambda x: x[1]['metrics']['RMSE'])
    print(f"\n✓ Best performing model: {best_model[0].upper()}")
    print(f"  RMSE: {best_model[1]['metrics']['RMSE']:.6f}")
    print(f"  R²: {best_model[1]['metrics']['R²']:.6f}")
    print(f"  Correlation: {best_model[1]['metrics']['Correlation']:.6f}")

if __name__ == '__main__':
    main()

