"""
Preprocessing Module - Resampling, normalization, PCA, and temporal split.

Handles train/test split with optional validation split.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional
from collections import defaultdict


# ============================================================================
# RESAMPLING
# ============================================================================

def resample_spectrum(energy: np.ndarray, signal: np.ndarray, 
                     common_energy: np.ndarray) -> np.ndarray:
    """
    Resample spectrum to common energy grid.
    
    Args:
        energy: Original energy array
        signal: Original signal array
        common_energy: Target energy grid
        
    Returns:
        Resampled signal array
    """
    mask = (energy >= common_energy.min()) & (energy <= common_energy.max())
    energy_masked = energy[mask]
    signal_masked = signal[mask]
    return np.interp(common_energy, energy_masked, signal_masked)


# ============================================================================
# TEMPORAL SPLIT
# ============================================================================

def create_temporal_split(ti_scans: List[Dict[str, Any]], 
                         train_scan_max: int = 4) -> Tuple[List[int], List[int], Dict, Dict]:
    """
    Create temporal train/test split.
    
    For each (experiment, material) group:
    - Train: scans 1 to train_scan_max (default: 1-4)
    - Test: scans (train_scan_max+1)+ (default: 5+)
    
    Args:
        ti_scans: List of scan records
        train_scan_max: Maximum scan number for training (default: 4)
        
    Returns:
        train_indices: List of global indices for training
        test_indices: List of global indices for testing
        train_groups: Dict mapping (exp, material) to train scan numbers
        test_groups: Dict mapping (exp, material) to test scan numbers
    """
    # Group by (experiment, material)
    groups = defaultdict(lambda: defaultdict(list))
    
    for i, rec in enumerate(ti_scans):
        exp = rec["experiment"]
        material = rec["material"]
        scan_num = int(rec["scan_number"])
        groups[exp][material].append((i, scan_num, rec))
    
    # Sort each group by scan_number
    for exp in groups:
        for material in groups[exp]:
            groups[exp][material].sort(key=lambda x: x[1])
    
    train_indices = []
    test_indices = []
    train_groups = {}
    test_groups = {}
    
    for exp in groups:
        for material in groups[exp]:
            scans = groups[exp][material]
            key = (exp, material)
            
            train_group = []
            test_group = []
            
            for global_idx, scan_num, rec in scans:
                if scan_num <= train_scan_max:
                    train_indices.append(global_idx)
                    train_group.append(scan_num)
                else:
                    test_indices.append(global_idx)
                    test_group.append(scan_num)
            
            if train_group:
                train_groups[key] = sorted(train_group)
            if test_group:
                test_groups[key] = sorted(test_group)
    
    return train_indices, test_indices, train_groups, test_groups


# ============================================================================
# EXPERIMENT ENCODING
# ============================================================================

def create_experiment_encoder(ti_scans: List[Dict[str, Any]]) -> LabelEncoder:
    """Create label encoder for experiments."""
    experiments = [rec["experiment"] for rec in ti_scans]
    encoder = LabelEncoder()
    encoder.fit(experiments)
    return encoder


def prepare_temporal_inputs(ti_scans: List[Dict[str, Any]], 
                           train_indices: List[int], 
                           test_indices: List[int],
                           exp_encoder: LabelEncoder) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare temporal-aware inputs: (experiment_id, scan_number).
    
    Returns:
        X_train, X_test as numpy arrays
    """
    # Encode experiments
    exp_ids = exp_encoder.transform([ti_scans[i]["experiment"] for i in range(len(ti_scans))])
    scan_nums = np.array([int(ti_scans[i]["scan_number"]) for i in range(len(ti_scans))])
    
    # Create inputs: (experiment_id, scan_number)
    X = np.column_stack([exp_ids, scan_nums]).astype(np.float32)
    
    # Split
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    return X_train, X_test


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def preprocess_pipeline(ti_scans: List[Dict[str, Any]],
                       train_indices: List[int],
                       test_indices: List[int],
                       energy_min: float = 25.0,
                       energy_max: float = 60.0,
                       step: float = 0.1,
                       n_components: int = 5,
                       val_split: Optional[float] = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
    """
    Full preprocessing pipeline.
    
    Args:
        ti_scans: List of scan records
        train_indices: Training set indices
        test_indices: Test set indices
        energy_min: Minimum energy (eV)
        energy_max: Maximum energy (eV)
        step: Energy grid step (eV)
        n_components: Number of PCA components
        val_split: Validation split ratio (None to skip validation)
        random_state: Random seed
        
    Returns:
        Dictionary with all preprocessed data and objects
    """
    # Create common energy grid
    common_energy = np.arange(energy_min, energy_max + step, step)
    
    # Resample all spectra
    resampled_spectra = []
    for s in ti_scans:
        resampled = resample_spectrum(s["energy"], s["signal"], common_energy)
        resampled_spectra.append(resampled)
    resampled_spectra = np.array(resampled_spectra)
    
    # Normalize
    norm_spectra = resampled_spectra / np.max(resampled_spectra, axis=1, keepdims=True)
    
    # Fit PCA on training data only (no data leakage)
    pca = PCA(n_components=n_components)
    latent_train = pca.fit_transform(norm_spectra[train_indices])
    latent_test = pca.transform(norm_spectra[test_indices])
    
    # Create full latent array
    latent = np.zeros((len(norm_spectra), n_components))
    latent[train_indices] = latent_train
    latent[test_indices] = latent_test
    
    # Create experiment encoder
    exp_encoder = create_experiment_encoder(ti_scans)
    
    # Prepare temporal inputs
    X_train_full, X_test = prepare_temporal_inputs(ti_scans, train_indices, test_indices, exp_encoder)
    y_train_full = latent_train.astype(np.float32)
    y_test = latent_test.astype(np.float32)
    
    # Optional validation split
    if val_split is not None and val_split > 0:
        train_idx_subset, val_idx_subset = train_test_split(
            np.arange(len(train_indices)), 
            test_size=val_split, 
            random_state=random_state
        )
        X_train = X_train_full[train_idx_subset]
        X_val = X_train_full[val_idx_subset]
        y_train = y_train_full[train_idx_subset]
        y_val = y_train_full[val_idx_subset]
    else:
        X_train = X_train_full
        X_val = None
        y_train = y_train_full
        y_val = None
    
    return {
        'common_energy': common_energy,
        'norm_spectra': norm_spectra,
        'pca': pca,
        'latent': latent,
        'exp_encoder': exp_encoder,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'latent_dim': n_components,
        'input_dim': 2,
    }






