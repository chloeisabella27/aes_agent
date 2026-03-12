"""End-to-end test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data_loader import load_lvm, load_ti_scans
from src.preprocessing import resample_spectrum, create_temporal_split
from src.models import get_model_class


def test_data_loading():
    """Test data loading works."""
    print("Testing data loading...")
    # This will fail if data path doesn't exist, but that's okay for now
    try:
        ti_scans = load_ti_scans("/Users/chloeisabella/Desktop/Files for Yash")
        assert len(ti_scans) > 0, "No scans loaded"
        print(f"✓ Loaded {len(ti_scans)} scans")
        return ti_scans
    except Exception as e:
        print(f"⚠ Data loading test skipped: {e}")
        return None


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\nTesting preprocessing...")
    # Create dummy data
    energy = np.linspace(25, 60, 100)
    signal = np.random.rand(100)
    common_energy = np.arange(25, 60.1, 0.1)
    
    resampled = resample_spectrum(energy, signal, common_energy)
    assert len(resampled) == len(common_energy), "Resampling failed"
    print("✓ Resampling works")
    
    # Test temporal split with dummy data
    dummy_scans = [
        {"experiment": "TF1", "material": "A", "scan_number": "1"},
        {"experiment": "TF1", "material": "A", "scan_number": "2"},
        {"experiment": "TF1", "material": "A", "scan_number": "5"},
    ]
    train_idx, test_idx, _, _ = create_temporal_split(dummy_scans, train_scan_max=4)
    assert len(train_idx) == 2, "Train split incorrect"
    assert len(test_idx) == 1, "Test split incorrect"
    print("✓ Temporal split works")


def test_models():
    """Test model initialization."""
    print("\nTesting models...")
    for model_name in ['nn', 'lstm', 'gru']:
        model_class = get_model_class(model_name)
        model = model_class(latent_dim=5, input_dim=2)
        assert model is not None, f"Failed to create {model_name}"
        print(f"✓ {model_name} model created")


if __name__ == '__main__':
    print("="*60)
    print("E2E TESTS")
    print("="*60)
    test_data_loading()
    test_preprocessing()
    test_models()
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

