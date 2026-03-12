"""Unit test for preprocessing module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.preprocessing import (
    resample_spectrum, create_temporal_split,
    preprocess_pipeline, create_experiment_encoder
)


def test_resample_spectrum():
    """Test spectrum resampling."""
    print("Testing resample_spectrum...")
    
    energy = np.linspace(25, 60, 100)
    signal = np.random.rand(100)
    common_energy = np.arange(25, 60.1, 0.1)
    
    resampled = resample_spectrum(energy, signal, common_energy)
    
    assert len(resampled) == len(common_energy)
    assert not np.isnan(resampled).any()
    print("✓ Resampling works")


def test_temporal_split():
    """Test temporal split creation."""
    print("\nTesting temporal split...")
    
    dummy_scans = [
        {"experiment": "TF1", "material": "A", "scan_number": "1"},
        {"experiment": "TF1", "material": "A", "scan_number": "2"},
        {"experiment": "TF1", "material": "A", "scan_number": "4"},
        {"experiment": "TF1", "material": "A", "scan_number": "5"},
        {"experiment": "TF1", "material": "B", "scan_number": "1"},
        {"experiment": "TF1", "material": "B", "scan_number": "6"},
    ]
    
    train_idx, test_idx, train_groups, test_groups = create_temporal_split(
        dummy_scans, train_scan_max=4
    )
    
    assert len(train_idx) == 3  # scans 1, 2, 4 from A and scan 1 from B
    assert len(test_idx) == 2  # scan 5 from A and scan 6 from B
    assert len(set(train_idx) & set(test_idx)) == 0, "Overlap detected!"
    
    print("✓ Temporal split works")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")


def test_preprocess_pipeline():
    """Test full preprocessing pipeline with dummy data."""
    print("\nTesting preprocessing pipeline...")
    
    # Create dummy scans
    n_scans = 20
    dummy_scans = []
    for i in range(n_scans):
        exp = f"TF{i // 5 + 1}"
        material = f"Mat{i % 3}"
        scan_num = (i % 5) + 1
        
        energy = np.linspace(25, 60, 50)
        signal = np.random.rand(50)
        
        dummy_scans.append({
            "experiment": exp,
            "material": material,
            "scan_number": str(scan_num),
            "energy": energy,
            "signal": signal,
        })
    
    # Create split
    train_idx, test_idx, _, _ = create_temporal_split(dummy_scans, train_scan_max=4)
    
    # Run preprocessing
    try:
        result = preprocess_pipeline(
            dummy_scans, train_idx, test_idx,
            val_split=0.2,
            n_components=3
        )
        
        assert 'X_train' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_test' in result
        assert 'pca' in result
        assert 'exp_encoder' in result
        
        print("✓ Preprocessing pipeline works")
        print(f"  X_train: {result['X_train'].shape}")
        print(f"  X_test: {result['X_test'].shape}")
        print(f"  y_train: {result['y_train'].shape}")
        print(f"  y_test: {result['y_test'].shape}")
        
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    print("="*60)
    print("PREPROCESSING TESTS")
    print("="*60)
    test_resample_spectrum()
    test_temporal_split()
    test_preprocess_pipeline()
    print("\n" + "="*60)
    print("All preprocessing tests passed!")
    print("="*60)






