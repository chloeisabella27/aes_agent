#!/usr/bin/env python3
"""
Simple CLI to train models.

Usage:
    python train.py --data-path "Files for Yash" --models nn lstm gru
"""
import argparse
import sys
import os
import torch
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_ti_scans
from src.preprocessing import preprocess_pipeline, create_temporal_split
from src.evaluator import ModelComparison


def main():
    parser = argparse.ArgumentParser(description='Train models with Bayesian Optimization')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data directory (e.g., "Files for Yash")')
    parser.add_argument('--models', nargs='+', default=['nn', 'lstm'],
                       choices=['nn', 'lstm', 'rnn', 'gru', 'transformer', 'tcn'],
                       help='Models to train')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for models and results')
    parser.add_argument('--train-scan-max', type=int, default=4,
                       help='Maximum scan number for training (default: 4)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2, set to 0 to disable)')
    parser.add_argument('--bo-iterations', type=int, default=10,
                       help='Bayesian optimization iterations (default: 10)')
    parser.add_argument('--bo-init-points', type=int, default=3,
                       help='BO initial random points (default: 3)')
    parser.add_argument('--train-epochs', type=int, default=200,
                       help='Training epochs per BO trial (default: 200)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Training models: {args.models}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    ti_scans = load_ti_scans(args.data_path)
    print(f"Loaded {len(ti_scans)} Ti scans")
    
    # Create temporal split
    print(f"\nCreating temporal split (train: scans 1-{args.train_scan_max}, test: scan {args.train_scan_max+1}+)...")
    train_indices, test_indices, train_groups, test_groups = create_temporal_split(
        ti_scans, train_scan_max=args.train_scan_max
    )
    
    # Preprocess
    print("\nPreprocessing data...")
    val_split = args.val_split if args.val_split > 0 else None
    preprocessed = preprocess_pipeline(
        ti_scans, train_indices, test_indices,
        val_split=val_split
    )
    
    # Train and compare models
    print("\nTraining models...")
    comparison = ModelComparison(
        preprocessed['X_train'], preprocessed['y_train'],
        preprocessed['X_val'], preprocessed['y_val'],
        preprocessed['X_test'], preprocessed['y_test'],
        preprocessed['latent_dim'], preprocessed['input_dim'],
        device=device
    )
    
    comparison.train_and_evaluate(
        args.models,
        bo_iterations=args.bo_iterations,
        bo_init_points=args.bo_init_points,
        train_epochs=args.train_epochs
    )
    
    # Save models and results
    print("\nSaving models and results...")
    for model_name, model in comparison.models.items():
        model_path = models_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  Saved {model_name} to {model_path}")
    
    # Save comparison results (remove BO optimizer as it can't be pickled)
    results_path = output_dir / 'comparison_results.pkl'
    results_to_save = {}
    for model_name, result in comparison.results.items():
        results_to_save[model_name] = {
            'metrics': result['metrics'],
            'best_params': result['best_params'],
            'y_pred': result['y_pred'],
            # Don't save bo_optimizer as it contains unpicklable local functions
        }
    
    # Save metrics table first (this always works)
    df = comparison.get_comparison_dataframe()
    csv_path = output_dir / 'metrics.csv'
    df.to_csv(csv_path)
    print(f"  Saved metrics to {csv_path}")
    
    # Save comparison results (may fail if objects can't be pickled)
    try:
        with open(results_path, 'wb') as f:
            pickle.dump({
                'results': results_to_save,
                'preprocessed': {
                    'pca': preprocessed['pca'],
                    'exp_encoder': preprocessed['exp_encoder'],
                    'common_energy': preprocessed['common_energy'],
                    'latent_dim': preprocessed['latent_dim'],
                    'input_dim': preprocessed['input_dim'],
                },
                'ti_scans': ti_scans,
                'train_indices': train_indices,
                'test_indices': test_indices,
            }, f)
        print(f"  Saved results to {results_path}")
    except Exception as e:
        print(f"  Warning: Could not save results pickle: {e}")
        print("  Metrics CSV is still available at:", csv_path)
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    print(df)
    
    best_model = comparison.get_best_model('RMSE')
    print(f"\nBest model (lowest RMSE): {best_model}")
    print(f"\nTraining complete! Models saved to {models_dir}")


if __name__ == '__main__':
    main()

