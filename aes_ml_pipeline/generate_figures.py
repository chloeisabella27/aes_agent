#!/usr/bin/env python3
"""
Generate Research Report Figures

Creates all visualizations needed for a formal research report on AES Ti MVV
spectral prediction using ML models.

Usage:
    python generate_figures.py --data-path "Files for Yash" --output-dir outputs/figures
    python generate_figures.py --category model_comparison
    python generate_figures.py --category temporal
    python generate_figures.py --category predictions
"""
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Use non-interactive backend (must be before importing pyplot)
import matplotlib
matplotlib.use('Agg')

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_ti_scans
from src.preprocessing import preprocess_pipeline, create_temporal_split
from src.models import get_model_class
from src.evaluator import compute_metrics
from src.visualization import (
    setup_style, save_figure, create_figure_index,
    # Data exploration
    plot_raw_spectra, plot_normalized_spectra, plot_pca_variance,
    plot_pca_components, plot_data_distribution,
    # Model comparison
    plot_metrics_comparison, plot_error_boxplot, plot_scatter_comparison,
    # Temporal analysis
    plot_error_vs_scan, plot_error_by_experiment, plot_error_heatmap,
    # Predictions
    plot_predictions_grid, plot_best_median_worst, plot_derivatives_grid,
    plot_residuals,
    # Diagnostics
    plot_error_distributions, plot_latent_correlation,
)

import matplotlib.pyplot as plt


def load_data_and_results(data_path: str, results_path: Path, models_dir: Path):
    """Load data, preprocessed results, and models."""
    print("\n" + "="*60)
    print("LOADING DATA AND RESULTS")
    print("="*60)
    
    # Load raw data
    print("\nLoading raw data...")
    ti_scans = load_ti_scans(data_path, verbose=True)
    print(f"  Loaded {len(ti_scans)} scans")
    
    # Create temporal split
    print("\nCreating temporal split...")
    train_indices, test_indices, _, _ = create_temporal_split(ti_scans, train_scan_max=4)
    print(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    # Preprocess
    print("\nPreprocessing data...")
    preprocessed = preprocess_pipeline(ti_scans, train_indices, test_indices, val_split=0.2)
    
    # Load results from pickle or compute
    results = {}
    if results_path.exists():
        try:
            with open(results_path, 'rb') as f:
                data = pickle.load(f)
            results = data.get('results', {})
            print(f"\nLoaded results for models: {list(results.keys())}")
        except Exception as e:
            print(f"\nWarning: Could not load results: {e}")
    
    # Load from CSV if pickle failed
    metrics_path = results_path.parent / 'metrics.csv'
    if not results and metrics_path.exists():
        print(f"\nLoading metrics from {metrics_path}...")
        df = pd.read_csv(metrics_path, index_col=0)
        for model_name in df.index:
            # We need to regenerate predictions for the figures
            model_path = models_dir / f'{model_name}.pth'
            if model_path.exists():
                model_class = get_model_class(model_name)
                model = model_class(
                    latent_dim=preprocessed['latent_dim'],
                    input_dim=preprocessed['input_dim']
                )
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                
                X_test_tensor = torch.tensor(preprocessed['X_test'])
                with torch.no_grad():
                    y_pred = model(X_test_tensor).numpy()
                
                results[model_name] = {
                    'metrics': {
                        'MSE': df.loc[model_name, 'MSE'],
                        'MAE': df.loc[model_name, 'MAE'],
                        'RMSE': df.loc[model_name, 'RMSE'],
                        'R²': df.loc[model_name, 'R²'],
                        'Correlation': df.loc[model_name, 'Correlation'],
                    },
                    'y_pred': y_pred,
                    'best_params': {}
                }
                print(f"  Loaded {model_name} model and predictions")
    
    return ti_scans, train_indices, test_indices, preprocessed, results


def generate_data_exploration_figures(ti_scans, preprocessed, output_dir):
    """Generate data exploration figures."""
    print("\n" + "="*60)
    print("1. DATA EXPLORATION FIGURES")
    print("="*60)
    
    figures = []
    
    # Get unique experiments (top 5 by count)
    exp_counts = defaultdict(int)
    for scan in ti_scans:
        exp_counts[scan['experiment']] += 1
    top_experiments = sorted(exp_counts.keys(), key=lambda x: exp_counts[x], reverse=True)[:5]
    
    # 1a. Raw spectra
    print("\n  Generating: Raw spectra...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_raw_spectra(ti_scans, top_experiments[:4], ax, n_per_experiment=3)
    save_figure(fig, output_dir / 'data_raw_spectra.png')
    figures.append(('data_raw_spectra.png', 'Raw Ti MVV spectra from different experiments'))
    
    # 1b. Normalized spectra
    print("  Generating: Normalized spectra...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_normalized_spectra(preprocessed['norm_spectra'], preprocessed['common_energy'],
                           ti_scans, top_experiments[:4], ax, n_per_experiment=3)
    save_figure(fig, output_dir / 'data_normalized_spectra.png')
    figures.append(('data_normalized_spectra.png', 'Normalized spectra on common energy grid'))
    
    # 1c. PCA variance
    print("  Generating: PCA variance...")
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_pca_variance(preprocessed['pca'], ax)
    save_figure(fig, output_dir / 'pca_variance.png')
    figures.append(('pca_variance.png', 'PCA explained variance (cumulative and per-component)'))
    
    # 1d. PCA components
    print("  Generating: PCA components...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pca_components(preprocessed['pca'], preprocessed['common_energy'], n_components=3, ax=ax)
    save_figure(fig, output_dir / 'pca_components.png')
    figures.append(('pca_components.png', 'First 3 PCA components visualization'))
    
    # 1e. Data distribution
    print("  Generating: Data distribution...")
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_data_distribution(ti_scans, ax)
    save_figure(fig, output_dir / 'data_distribution.png')
    figures.append(('data_distribution.png', 'Data distribution: scans per experiment'))
    
    return figures


def generate_model_comparison_figures(results, preprocessed, output_dir):
    """Generate model comparison figures."""
    print("\n" + "="*60)
    print("2. MODEL COMPARISON FIGURES")
    print("="*60)
    
    figures = []
    y_test = preprocessed['y_test']
    
    if not results:
        print("  No results available for model comparison figures")
        return figures
    
    # 2a. Metrics comparison
    print("\n  Generating: Metrics comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_metrics_comparison(results, ax)
    save_figure(fig, output_dir / 'model_metrics_comparison.png')
    figures.append(('model_metrics_comparison.png', 'Model metrics comparison (RMSE, MAE, R², Correlation)'))
    
    # 2b. Error boxplot
    print("  Generating: Error boxplot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_error_boxplot(results, y_test, ax)
    save_figure(fig, output_dir / 'model_error_boxplot.png')
    figures.append(('model_error_boxplot.png', 'Box plot of prediction errors per model'))
    
    # 2c. Scatter comparison
    print("  Generating: Scatter comparison...")
    fig = plot_scatter_comparison(results, y_test)
    save_figure(fig, output_dir / 'model_scatter_comparison.png')
    figures.append(('model_scatter_comparison.png', 'Scatter: Actual vs Predicted for each model'))
    
    return figures


def generate_temporal_figures(results, ti_scans, test_indices, preprocessed, output_dir):
    """Generate temporal analysis figures."""
    print("\n" + "="*60)
    print("3. TEMPORAL ANALYSIS FIGURES")
    print("="*60)
    
    figures = []
    y_test = preprocessed['y_test']
    
    if not results:
        print("  No results available for temporal figures")
        return figures
    
    # 3a. Error vs scan number
    print("\n  Generating: Error vs scan number...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_error_vs_scan(results, ti_scans, test_indices, y_test, ax)
    save_figure(fig, output_dir / 'temporal_error_vs_scan.png')
    figures.append(('temporal_error_vs_scan.png', 'Prediction error vs scan number (temporal degradation)'))
    
    # 3b. Performance by experiment
    print("  Generating: Performance by experiment...")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_error_by_experiment(results, ti_scans, test_indices, y_test, ax, top_n=10)
    save_figure(fig, output_dir / 'temporal_by_experiment.png')
    figures.append(('temporal_by_experiment.png', 'Model performance by experiment'))
    
    # 3c. Error heatmap (for best model)
    print("  Generating: Error heatmap...")
    best_model = min(results.keys(), key=lambda m: results[m]['metrics']['RMSE'])
    fig = plot_error_heatmap(results, ti_scans, test_indices, y_test, model_name=best_model)
    save_figure(fig, output_dir / 'temporal_error_heatmap.png')
    figures.append(('temporal_error_heatmap.png', f'Error heatmap by (experiment, scan_number) for {best_model.upper()}'))
    
    return figures


def generate_prediction_figures(results, ti_scans, test_indices, preprocessed, output_dir):
    """Generate prediction quality figures."""
    print("\n" + "="*60)
    print("4. PREDICTION QUALITY FIGURES")
    print("="*60)
    
    figures = []
    
    if not results:
        print("  No results available for prediction figures")
        return figures
    
    # Use best model for prediction figures
    best_model = min(results.keys(), key=lambda m: results[m]['metrics']['RMSE'])
    y_pred = results[best_model]['y_pred']
    y_test = preprocessed['y_test']
    
    # Reconstruct spectra from PCA
    pca = preprocessed['pca']
    common_energy = preprocessed['common_energy']
    
    pred_spectra = pca.inverse_transform(y_pred)
    actual_spectra = pca.inverse_transform(y_test)
    
    # 4a. Predictions grid
    print(f"\n  Generating: Predictions grid (using {best_model.upper()})...")
    # Select diverse samples
    n_grid = 16
    step = max(1, len(test_indices) // n_grid)
    grid_indices = list(range(0, len(test_indices), step))[:n_grid]
    
    fig = plot_predictions_grid(actual_spectra[grid_indices], pred_spectra[grid_indices],
                               common_energy, ti_scans, 
                               [test_indices[i] for i in grid_indices],
                               nrows=4, ncols=4)
    save_figure(fig, output_dir / 'predictions_grid.png')
    figures.append(('predictions_grid.png', f'Grid of actual vs predicted spectra ({best_model.upper()})'))
    
    # 4b. Best, median, worst
    print("  Generating: Best, median, worst predictions...")
    fig = plot_best_median_worst(actual_spectra, pred_spectra, common_energy, 
                                 ti_scans, test_indices)
    save_figure(fig, output_dir / 'predictions_best_median_worst.png')
    figures.append(('predictions_best_median_worst.png', 'Best, median, and worst predictions'))
    
    # 4c. Derivatives grid
    print("  Generating: Derivatives grid...")
    fig = plot_derivatives_grid(actual_spectra[grid_indices[:9]], pred_spectra[grid_indices[:9]],
                               common_energy, ti_scans, 
                               [test_indices[i] for i in grid_indices[:9]],
                               nrows=3, ncols=3)
    save_figure(fig, output_dir / 'derivatives_grid.png')
    figures.append(('derivatives_grid.png', 'Derivative (dN/dE) comparison grid'))
    
    # 4d. Residuals plot
    print("  Generating: Residuals plot...")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_residuals(actual_spectra, pred_spectra, common_energy, ax)
    save_figure(fig, output_dir / 'residuals_plot.png')
    figures.append(('residuals_plot.png', 'Residuals (predicted - actual) vs energy'))
    
    return figures


def generate_diagnostic_figures(results, preprocessed, output_dir):
    """Generate diagnostic figures."""
    print("\n" + "="*60)
    print("5. DIAGNOSTIC FIGURES")
    print("="*60)
    
    figures = []
    y_test = preprocessed['y_test']
    
    # 5a. Training curves (skip if no data - would need to save during training)
    print("\n  Skipping: Training curves (data not saved during training)")
    # TODO: Save training logs during training to enable this
    
    # 5b. Error distributions
    if results:
        print("  Generating: Error distributions...")
        fig = plot_error_distributions(results, y_test)
        save_figure(fig, output_dir / 'error_distributions.png')
        figures.append(('error_distributions.png', 'Error distribution histograms for all models'))
    
    # 5c. Latent correlation
    print("  Generating: Latent correlation...")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_latent_correlation(preprocessed['latent'], ax)
    save_figure(fig, output_dir / 'latent_correlation.png')
    figures.append(('latent_correlation.png', 'Correlation matrix of latent PCA components'))
    
    # 5d. BO convergence (skip - requires BO optimizer data)
    print("  Skipping: BO convergence (data not saved)")
    
    return figures


def main():
    parser = argparse.ArgumentParser(description='Generate research report figures')
    parser.add_argument('--data-path', type=str, 
                       default='/Users/chloeisabella/Desktop/Files for Yash',
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/figures',
                       help='Output directory for figures')
    parser.add_argument('--results-path', type=str, default='outputs/comparison_results.pkl',
                       help='Path to results pickle file')
    parser.add_argument('--models-dir', type=str, default='outputs/models',
                       help='Directory containing saved models')
    parser.add_argument('--category', type=str, default='all',
                       choices=['all', 'data', 'model_comparison', 'temporal', 
                               'predictions', 'diagnostics'],
                       help='Which category of figures to generate')
    
    args = parser.parse_args()
    
    # Setup
    setup_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_path)
    models_dir = Path(args.models_dir)
    
    print("\n" + "="*60)
    print("RESEARCH REPORT FIGURE GENERATOR")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {output_dir}")
    print(f"Category: {args.category}")
    
    # Load data
    ti_scans, train_indices, test_indices, preprocessed, results = \
        load_data_and_results(args.data_path, results_path, models_dir)
    
    # Generate figures
    all_figures = []
    
    if args.category in ['all', 'data']:
        all_figures.extend(generate_data_exploration_figures(ti_scans, preprocessed, output_dir))
    
    if args.category in ['all', 'model_comparison']:
        all_figures.extend(generate_model_comparison_figures(results, preprocessed, output_dir))
    
    if args.category in ['all', 'temporal']:
        all_figures.extend(generate_temporal_figures(results, ti_scans, test_indices, 
                                                     preprocessed, output_dir))
    
    if args.category in ['all', 'predictions']:
        all_figures.extend(generate_prediction_figures(results, ti_scans, test_indices,
                                                       preprocessed, output_dir))
    
    if args.category in ['all', 'diagnostics']:
        all_figures.extend(generate_diagnostic_figures(results, preprocessed, output_dir))
    
    # Create figure index
    print("\n" + "="*60)
    print("CREATING FIGURE INDEX")
    print("="*60)
    create_figure_index(all_figures, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nGenerated {len(all_figures)} figures:")
    for filename, desc in all_figures:
        print(f"  - {filename}: {desc[:50]}...")
    
    print(f"\nAll figures saved to: {output_dir.absolute()}")
    print(f"Figure index: {output_dir / 'figure_index.md'}")
    

if __name__ == '__main__':
    main()

