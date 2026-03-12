"""
Visualization Module - Research-quality figures for AES ML pipeline.

Generates all figures needed for a formal research report.
"""
import numpy as np
import pandas as pd

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict


# ============================================================================
# STYLE SETUP
# ============================================================================

def setup_style() -> None:
    """Configure matplotlib for research-quality figures."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        # Fallback if seaborn style not available
        plt.style.use('ggplot')
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (8, 6),
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'axes.prop_cycle': plt.cycler(color=[
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ])
    })


# Colorblind-friendly palette
COLORS = {
    'nn': '#1f77b4',      # Blue
    'lstm': '#ff7f0e',    # Orange
    'gru': '#2ca02c',     # Green
    'transformer': '#d62728',  # Red
    'tcn': '#9467bd',     # Purple
    'rnn': '#8c564b',     # Brown
    'actual': '#333333',  # Dark gray
    'predicted': '#e74c3c',  # Red
}


# ============================================================================
# DATA EXPLORATION FIGURES
# ============================================================================

def plot_raw_spectra(ti_scans: List[Dict], experiments: List[str], 
                     ax: Optional[plt.Axes] = None, 
                     n_per_experiment: int = 3) -> plt.Axes:
    """
    Plot raw spectra samples from different experiments.
    
    Args:
        ti_scans: List of scan records
        experiments: List of experiment names to plot
        ax: Matplotlib axes (optional)
        n_per_experiment: Number of spectra per experiment
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for exp_idx, exp_name in enumerate(experiments):
        # Find scans for this experiment
        exp_scans = [s for s in ti_scans if s['experiment'] == exp_name]
        
        # Plot first n scans
        for i, scan in enumerate(exp_scans[:n_per_experiment]):
            label = f"{exp_name}" if i == 0 else None
            alpha = 1.0 - (i * 0.2)
            ax.plot(scan['energy'], scan['signal'], 
                   color=colors[exp_idx], alpha=alpha, label=label, linewidth=1)
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title('Raw Ti MVV Spectra by Experiment')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(25, 60)
    
    return ax


def plot_normalized_spectra(norm_spectra: np.ndarray, common_energy: np.ndarray,
                           ti_scans: List[Dict], experiments: List[str],
                           ax: Optional[plt.Axes] = None,
                           n_per_experiment: int = 3) -> plt.Axes:
    """Plot normalized spectra on common energy grid."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for exp_idx, exp_name in enumerate(experiments):
        # Find indices for this experiment
        indices = [i for i, s in enumerate(ti_scans) if s['experiment'] == exp_name]
        
        for i, idx in enumerate(indices[:n_per_experiment]):
            label = f"{exp_name}" if i == 0 else None
            alpha = 1.0 - (i * 0.2)
            ax.plot(common_energy, norm_spectra[idx], 
                   color=colors[exp_idx], alpha=alpha, label=label, linewidth=1)
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Normalized Spectra on Common Energy Grid')
    ax.legend(loc='upper right', framealpha=0.9)
    
    return ax


def plot_pca_variance(pca, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot PCA explained variance (cumulative and per-component).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    n_components = len(pca.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)
    
    # Bar plot for individual variance
    bars = ax.bar(components - 0.2, pca.explained_variance_ratio_ * 100, 
                  width=0.4, label='Individual', color='#1f77b4', alpha=0.8)
    
    # Line plot for cumulative variance
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(components, cumulative, 'o-', color='#ff7f0e', 
            linewidth=2, markersize=8, label='Cumulative')
    
    # Add percentage labels
    for i, (bar, cum) in enumerate(zip(bars, cumulative)):
        ax.annotate(f'{pca.explained_variance_ratio_[i]*100:.1f}%', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Explained Variance')
    ax.set_xticks(components)
    ax.legend(loc='center right')
    ax.set_ylim(0, 105)
    
    return ax


def plot_pca_components(pca, common_energy: np.ndarray, 
                       n_components: int = 3,
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot first n PCA components."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_components))
    
    for i in range(min(n_components, len(pca.components_))):
        ax.plot(common_energy, pca.components_[i], 
               color=colors[i], linewidth=1.5,
               label=f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Component Loading')
    ax.set_title('PCA Components')
    ax.legend(loc='best')
    
    return ax


def plot_data_distribution(ti_scans: List[Dict], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot data distribution: scans per experiment."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    # Count scans per experiment
    exp_counts = defaultdict(int)
    for scan in ti_scans:
        exp_counts[scan['experiment']] += 1
    
    # Sort by count
    sorted_exps = sorted(exp_counts.items(), key=lambda x: x[1], reverse=True)
    experiments = [x[0] for x in sorted_exps]
    counts = [x[1] for x in sorted_exps]
    
    bars = ax.bar(experiments, counts, color='#1f77b4', alpha=0.8, edgecolor='white')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.annotate(str(count), xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Number of Scans')
    ax.set_title('Data Distribution by Experiment')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    return ax


# ============================================================================
# MODEL COMPARISON FIGURES
# ============================================================================

def plot_metrics_comparison(results: Dict[str, Dict], 
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot metrics comparison bar chart (RMSE, MAE, R², Correlation).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    metrics = ['RMSE', 'MAE', 'R²', 'Correlation']
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        values = [results[model]['metrics'].get(m, 0) for m in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        color = COLORS.get(model.lower(), f'C{i}')
        bars = ax.bar(x + offset, values, width, label=model.upper(), 
                     color=color, alpha=0.8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='best')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    return ax


def plot_error_boxplot(results: Dict[str, Dict], y_test: np.ndarray,
                      ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot box plot of prediction errors per model."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    errors_data = []
    labels = []
    colors_list = []
    
    for model_name, result in results.items():
        y_pred = result['y_pred']
        # Per-sample RMSE
        sample_errors = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=1))
        errors_data.append(sample_errors)
        labels.append(model_name.upper())
        colors_list.append(COLORS.get(model_name.lower(), '#1f77b4'))
    
    bp = ax.boxplot(errors_data, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Per-Sample RMSE')
    ax.set_title('Prediction Error Distribution by Model')
    
    return ax


def plot_scatter_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str,
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Scatter plot: Actual vs Predicted values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Flatten for scatter
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10, 
              color=COLORS.get(model_name.lower(), '#1f77b4'))
    
    # Perfect prediction line
    lims = [min(y_true_flat.min(), y_pred_flat.min()),
            max(y_true_flat.max(), y_pred_flat.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{model_name.upper()}: Actual vs Predicted')
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_scatter_comparison(results: Dict[str, Dict], y_test: np.ndarray) -> Figure:
    """Create multi-panel scatter plot for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, result) in zip(axes, results.items()):
        plot_scatter_actual_vs_pred(y_test, result['y_pred'], model_name, ax)
    
    plt.tight_layout()
    return fig


# ============================================================================
# TEMPORAL ANALYSIS FIGURES
# ============================================================================

def plot_error_vs_scan(results: Dict[str, Dict], ti_scans: List[Dict],
                       test_indices: List[int], y_test: np.ndarray,
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot prediction error vs scan number (temporal degradation)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get scan numbers for test set
    scan_numbers = [int(ti_scans[i]['scan_number']) for i in test_indices]
    unique_scans = sorted(set(scan_numbers))
    
    for model_name, result in results.items():
        y_pred = result['y_pred']
        # Per-sample errors
        sample_errors = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=1))
        
        # Average error per scan number
        avg_errors = []
        for scan_num in unique_scans:
            mask = np.array(scan_numbers) == scan_num
            if np.any(mask):
                avg_errors.append(np.mean(sample_errors[mask]))
            else:
                avg_errors.append(np.nan)
        
        color = COLORS.get(model_name.lower(), '#1f77b4')
        ax.plot(unique_scans, avg_errors, 'o-', label=model_name.upper(), 
               color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Scan Number')
    ax.set_ylabel('Mean RMSE')
    ax.set_title('Prediction Error vs Scan Number (Temporal Analysis)')
    ax.legend(loc='best')
    ax.set_xticks(unique_scans)
    
    return ax


def plot_error_by_experiment(results: Dict[str, Dict], ti_scans: List[Dict],
                             test_indices: List[int], y_test: np.ndarray,
                             ax: Optional[plt.Axes] = None,
                             top_n: int = 10) -> plt.Axes:
    """Plot performance by experiment (grouped bar chart)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get experiments for test set
    experiments = [ti_scans[i]['experiment'] for i in test_indices]
    unique_exps = list(set(experiments))
    
    # Limit to top N experiments by sample count
    exp_counts = {exp: experiments.count(exp) for exp in unique_exps}
    unique_exps = sorted(exp_counts.keys(), key=lambda x: exp_counts[x], reverse=True)[:top_n]
    
    x = np.arange(len(unique_exps))
    width = 0.8 / len(results)
    
    for i, (model_name, result) in enumerate(results.items()):
        y_pred = result['y_pred']
        sample_errors = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=1))
        
        # Average error per experiment
        avg_errors = []
        for exp in unique_exps:
            mask = np.array(experiments) == exp
            if np.any(mask):
                avg_errors.append(np.mean(sample_errors[mask]))
            else:
                avg_errors.append(0)
        
        offset = (i - len(results)/2 + 0.5) * width
        color = COLORS.get(model_name.lower(), f'C{i}')
        ax.bar(x + offset, avg_errors, width, label=model_name.upper(), 
              color=color, alpha=0.8)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Mean RMSE')
    ax.set_title('Model Performance by Experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_exps, rotation=45, ha='right')
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    return ax


def plot_error_heatmap(results: Dict[str, Dict], ti_scans: List[Dict],
                       test_indices: List[int], y_test: np.ndarray,
                       model_name: str = 'lstm') -> Figure:
    """Heatmap: Error by (experiment, scan_number)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    result = results.get(model_name, list(results.values())[0])
    y_pred = result['y_pred']
    sample_errors = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=1))
    
    # Get experiments and scan numbers
    experiments = [ti_scans[i]['experiment'] for i in test_indices]
    scan_numbers = [int(ti_scans[i]['scan_number']) for i in test_indices]
    
    unique_exps = sorted(set(experiments))
    unique_scans = sorted(set(scan_numbers))
    
    # Create heatmap data
    heatmap_data = np.full((len(unique_exps), len(unique_scans)), np.nan)
    
    for i, (exp, scan_num, error) in enumerate(zip(experiments, scan_numbers, sample_errors)):
        exp_idx = unique_exps.index(exp)
        scan_idx = unique_scans.index(scan_num)
        if np.isnan(heatmap_data[exp_idx, scan_idx]):
            heatmap_data[exp_idx, scan_idx] = error
        else:
            # Average if multiple samples
            heatmap_data[exp_idx, scan_idx] = (heatmap_data[exp_idx, scan_idx] + error) / 2
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    
    ax.set_xticks(np.arange(len(unique_scans)))
    ax.set_yticks(np.arange(len(unique_exps)))
    ax.set_xticklabels(unique_scans)
    ax.set_yticklabels(unique_exps)
    ax.set_xlabel('Scan Number')
    ax.set_ylabel('Experiment')
    ax.set_title(f'Prediction Error Heatmap ({model_name.upper()})')
    
    plt.colorbar(im, ax=ax, label='RMSE')
    plt.tight_layout()
    
    return fig


# ============================================================================
# PREDICTION QUALITY FIGURES
# ============================================================================

def plot_spectrum_comparison(actual: np.ndarray, predicted: np.ndarray,
                            energy: np.ndarray, title: str = '',
                            ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot single spectrum comparison: actual vs predicted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(energy, actual, color=COLORS['actual'], linewidth=1.5, label='Actual')
    ax.plot(energy, predicted, '--', color=COLORS['predicted'], 
            linewidth=1.5, label='Predicted')
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    return ax


def plot_predictions_grid(actuals: np.ndarray, predictions: np.ndarray,
                         energy: np.ndarray, ti_scans: List[Dict],
                         indices: List[int],
                         nrows: int = 4, ncols: int = 4) -> Figure:
    """Grid of actual vs predicted spectra."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()
    
    n_plots = min(nrows * ncols, len(indices))
    
    for i in range(n_plots):
        idx = indices[i]
        scan = ti_scans[idx]
        title = f"{scan['experiment']} Scan {scan['scan_number']}"
        
        ax = axes[i]
        ax.plot(energy, actuals[i], color=COLORS['actual'], linewidth=1, label='Actual')
        ax.plot(energy, predictions[i], '--', color=COLORS['predicted'], 
                linewidth=1, label='Pred')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Energy (eV)', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    # Hide empty subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_best_median_worst(actuals: np.ndarray, predictions: np.ndarray,
                           energy: np.ndarray, ti_scans: List[Dict],
                           test_indices: List[int]) -> Figure:
    """Plot best, median, and worst predictions (3-panel)."""
    # Calculate errors
    errors = np.sqrt(np.mean((actuals - predictions) ** 2, axis=1))
    
    # Find best, median, worst
    best_idx = np.argmin(errors)
    worst_idx = np.argmax(errors)
    median_idx = np.argsort(errors)[len(errors)//2]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for ax, (idx, label, error) in zip(axes, [
        (best_idx, 'Best', errors[best_idx]),
        (median_idx, 'Median', errors[median_idx]),
        (worst_idx, 'Worst', errors[worst_idx])
    ]):
        scan = ti_scans[test_indices[idx]]
        title = f'{label} Prediction (RMSE={error:.4f})\n{scan["experiment"]} Scan {scan["scan_number"]}'
        
        ax.plot(energy, actuals[idx], color=COLORS['actual'], 
                linewidth=1.5, label='Actual')
        ax.plot(energy, predictions[idx], '--', color=COLORS['predicted'],
                linewidth=1.5, label='Predicted')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(title)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_derivatives_grid(actuals: np.ndarray, predictions: np.ndarray,
                         energy: np.ndarray, ti_scans: List[Dict],
                         indices: List[int],
                         nrows: int = 3, ncols: int = 3) -> Figure:
    """Grid of derivative (dN/dE) comparisons."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()
    
    n_plots = min(nrows * ncols, len(indices))
    
    for i in range(n_plots):
        idx = indices[i]
        scan = ti_scans[idx]
        title = f"{scan['experiment']} Scan {scan['scan_number']}"
        
        # Compute derivatives
        d_actual = np.gradient(actuals[i], energy)
        d_pred = np.gradient(predictions[i], energy)
        
        ax = axes[i]
        ax.plot(energy, d_actual, color=COLORS['actual'], linewidth=1, label='Actual')
        ax.plot(energy, d_pred, '--', color=COLORS['predicted'], linewidth=1, label='Pred')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Energy (eV)', fontsize=8)
        ax.set_ylabel('dN/dE', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    # Hide empty subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_residuals(actuals: np.ndarray, predictions: np.ndarray,
                   energy: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot residuals (predicted - actual vs energy)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    residuals = predictions - actuals
    
    # Plot all residuals with low alpha
    for i in range(len(residuals)):
        ax.plot(energy, residuals[i], alpha=0.1, color='#1f77b4', linewidth=0.5)
    
    # Plot mean residual
    mean_residual = np.mean(residuals, axis=0)
    std_residual = np.std(residuals, axis=0)
    
    ax.plot(energy, mean_residual, color='#d62728', linewidth=2, label='Mean')
    ax.fill_between(energy, mean_residual - std_residual, mean_residual + std_residual,
                   color='#d62728', alpha=0.2, label='±1 Std')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Residual (Predicted - Actual)')
    ax.set_title('Residuals Analysis')
    ax.legend(loc='upper right')
    
    return ax


# ============================================================================
# DIAGNOSTIC FIGURES
# ============================================================================

def plot_training_curves(training_logs: Dict[str, List[float]], 
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot training loss curves (all models overlaid)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, losses in training_logs.items():
        epochs = np.arange(1, len(losses) + 1)
        color = COLORS.get(model_name.lower(), '#1f77b4')
        ax.plot(epochs, losses, label=model_name.upper(), color=color, linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Loss Curves')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    
    return ax


def plot_error_histogram(errors: np.ndarray, model_name: str,
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot error distribution histogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    color = COLORS.get(model_name.lower(), '#1f77b4')
    ax.hist(errors, bins=30, color=color, alpha=0.7, edgecolor='white')
    
    # Add mean and std lines
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_err:.4f}')
    ax.axvline(mean_err + std_err, color='red', linestyle=':', linewidth=1.5)
    ax.axvline(mean_err - std_err, color='red', linestyle=':', linewidth=1.5,
               label=f'Std: {std_err:.4f}')
    
    ax.set_xlabel('Per-Sample RMSE')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{model_name.upper()} Error Distribution')
    ax.legend(loc='upper right')
    
    return ax


def plot_error_distributions(results: Dict[str, Dict], y_test: np.ndarray) -> Figure:
    """Plot error distributions for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, result) in zip(axes, results.items()):
        y_pred = result['y_pred']
        errors = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=1))
        plot_error_histogram(errors, model_name, ax)
    
    plt.tight_layout()
    return fig


def plot_latent_correlation(latent: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot correlation matrix of latent PCA components."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    corr_matrix = np.corrcoef(latent.T)
    n_components = corr_matrix.shape[0]
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add text annotations
    for i in range(n_components):
        for j in range(n_components):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=10)
    
    ax.set_xticks(np.arange(n_components))
    ax.set_yticks(np.arange(n_components))
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_components)])
    ax.set_yticklabels([f'PC{i+1}' for i in range(n_components)])
    ax.set_title('Latent Space Correlation Matrix')
    
    plt.colorbar(im, ax=ax, label='Correlation')
    
    return ax


def plot_bo_convergence(bo_results: Dict[str, Any], 
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot Bayesian Optimization convergence."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # This requires the BO optimizer results
    # For now, plot if available
    if 'iterations' in bo_results:
        iterations = bo_results['iterations']
        values = bo_results['values']
        
        ax.plot(iterations, values, 'o-', color='#1f77b4', linewidth=1.5)
        ax.set_xlabel('BO Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Bayesian Optimization Convergence')
    else:
        ax.text(0.5, 0.5, 'BO data not available', transform=ax.transAxes,
               ha='center', va='center', fontsize=12)
    
    return ax


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_figure(fig: Figure, output_path: Path, dpi: int = 300) -> None:
    """Save figure to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_figure_index(figures: List[Tuple[str, str]], output_dir: Path) -> None:
    """Create markdown index of all generated figures."""
    index_path = output_dir / 'figure_index.md'
    
    with open(index_path, 'w') as f:
        f.write('# Figure Index\n\n')
        f.write('Generated figures for research report.\n\n')
        
        for filename, description in figures:
            f.write(f'## {filename}\n')
            f.write(f'{description}\n\n')
            f.write(f'![{filename}]({filename})\n\n')
    
    print(f"  Saved: {index_path}")

