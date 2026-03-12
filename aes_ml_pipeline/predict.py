#!/usr/bin/env python3
"""
Simple CLI to make predictions.

Usage:
    python predict.py --experiment TF268 --scan 5 --model lstm --model-path outputs/models/lstm.pth
"""
import argparse
import sys
import pickle
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import get_model_class
from src.predictor import predict_scan, plot_prediction, plot_derivative


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (e.g., TF268)')
    parser.add_argument('--scan', type=int, default=5,
                       help='Scan number to predict (default: 5)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['nn', 'lstm', 'rnn', 'gru', 'transformer', 'tcn'],
                       help='Model name')
    parser.add_argument('--results-path', type=str, default='outputs/comparison_results.pkl',
                       help='Path to saved results pickle file')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model (if None, loads from results)')
    parser.add_argument('--output-dir', type=str, default='outputs/plots',
                       help='Output directory for plots')
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
    
    # Load results
    print(f"\nLoading results from {args.results_path}...")
    try:
        with open(args.results_path, 'rb') as f:
            data = pickle.load(f)
        preprocessed = data['preprocessed']
        ti_scans = data['ti_scans']
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        print("Attempting to regenerate from saved models...")
        print("Run: python regenerate_results.py")
        sys.exit(1)
    
    # Load model
    model_class = get_model_class(args.model)
    model = model_class(
        latent_dim=preprocessed['latent_dim'],
        input_dim=preprocessed['input_dim']
    )
    
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        # Try to load from results
        print("Loading model from results...")
        # Note: models aren't saved in results, need model_path
        raise ValueError("Must provide --model-path or models need to be saved in results")
    
    model = model.to(device)
    model.eval()
    
    # Make prediction
    print(f"\nPredicting {args.experiment} scan {args.scan}...")
    energy, actual, predicted = predict_scan(
        args.experiment, args.scan, model, ti_scans,
        preprocessed['pca'], preprocessed['common_energy'],
        preprocessed['exp_encoder'],
        device=device
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plots
    plot_path = output_dir / f"{args.experiment}_scan{args.scan}_{args.model}_prediction.png"
    plot_prediction(
        args.experiment, args.scan, energy, actual, predicted,
        model_name=args.model, save_path=str(plot_path)
    )
    
    deriv_path = output_dir / f"{args.experiment}_scan{args.scan}_{args.model}_derivative.png"
    plot_derivative(
        args.experiment, args.scan, energy, actual, predicted,
        model_name=args.model, save_path=str(deriv_path)
    )
    
    print(f"\nPredictions saved to {output_dir}")


if __name__ == '__main__':
    main()

