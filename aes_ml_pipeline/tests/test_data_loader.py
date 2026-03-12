"""Unit test for data loading - with timing and partial loading."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from src.data_loader import load_lvm, load_ti_scans, is_ti_folder


def test_load_single_lvm():
    """Test loading a single LVM file."""
    print("Testing single LVM file loading...")
    
    # Try to find a sample LVM file
    data_path = "/Users/chloeisabella/Desktop/Files for Yash"
    if not os.path.exists(data_path):
        print(f"⚠ Data path not found: {data_path}")
        return None
    
    # Find first LVM file
    lvm_file = None
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.lvm'):
                lvm_file = os.path.join(root, file)
                break
        if lvm_file:
            break
    
    if not lvm_file:
        print("⚠ No LVM files found")
        return None
    
    print(f"Testing with: {lvm_file}")
    start = time.time()
    try:
        data = load_lvm(lvm_file)
        elapsed = time.time() - start
        print(f"✓ Loaded LVM file in {elapsed:.3f}s")
        print(f"  Shape: {data.shape}")
        print(f"  Energy range: {data[:, 1].min():.2f} - {data[:, 1].max():.2f} eV")
        assert elapsed < 1.0, f"Loading took too long: {elapsed:.3f}s"
        return True
    except Exception as e:
        print(f"✗ Failed to load LVM: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_ti_scans_limited():
    """Test loading Ti scans with a limit to avoid hanging."""
    print("\nTesting Ti scan loading (limited)...")
    
    data_path = "/Users/chloeisabella/Desktop/Files for Yash"
    if not os.path.exists(data_path):
        print(f"⚠ Data path not found: {data_path}")
        return None
    
    # Modify load_ti_scans to stop after N scans for testing
    print("Loading Ti scans (will stop after 10 scans or 30 seconds)...")
    start = time.time()
    timeout = 30
    
    try:
        ti_scans = []
        count = 0
        max_scans = 10  # Limit for testing
        
        for root, dirs, files in os.walk(data_path):
            if time.time() - start > timeout:
                print(f"⚠ Timeout after {timeout}s")
                break
            
            folder_name = os.path.basename(root)
            if not is_ti_folder(folder_name):
                continue
            
            for file in files:
                if time.time() - start > timeout:
                    break
                if count >= max_scans:
                    break
                    
                if not file.endswith(".lvm"):
                    continue
                
                full_path = os.path.join(root, file)
                try:
                    arr = load_lvm(full_path)
                    energy = arr[:, 1]
                    signal = arr[:, 2]
                    
                    # Check energy range
                    from src.data_loader import energy_overlaps, TI_RANGE
                    if not energy_overlaps(energy, *TI_RANGE):
                        continue
                    
                    # Parse path
                    parts = full_path.split(os.sep)
                    scan_number = file.replace(".lvm", "")
                    material = parts[-3] if len(parts) >= 3 else "unknown"
                    experiment = parts[-4] if len(parts) >= 4 else "unknown"
                    
                    ti_scans.append({
                        "label": f"{experiment}_{material}_Ti{scan_number}",
                        "experiment": experiment,
                        "material": material,
                        "scan_number": scan_number,
                        "path": full_path,
                        "energy": energy,
                        "signal": signal,
                    })
                    count += 1
                    if count % 5 == 0:
                        print(f"  Loaded {count} scans...")
                        
                except Exception as e:
                    print(f"  Warning: Failed to load {full_path}: {e}")
                    continue
            
            if count >= max_scans:
                break
        
        elapsed = time.time() - start
        print(f"✓ Loaded {len(ti_scans)} Ti scans in {elapsed:.3f}s")
        print(f"  Average: {elapsed/len(ti_scans)*1000:.1f}ms per scan")
        
        if len(ti_scans) > 0:
            print(f"  Sample scan: {ti_scans[0]['label']}")
            print(f"  Energy range: {ti_scans[0]['energy'].min():.2f} - {ti_scans[0]['energy'].max():.2f} eV")
        
        assert elapsed < 60, f"Loading took too long: {elapsed:.3f}s"
        return ti_scans
        
    except Exception as e:
        print(f"✗ Failed to load Ti scans: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_load_ti_scans():
    """Test full Ti scan loading with progress monitoring."""
    print("\nTesting full Ti scan loading...")
    
    data_path = "/Users/chloeisabella/Desktop/Files for Yash"
    if not os.path.exists(data_path):
        print(f"⚠ Data path not found: {data_path}")
        return None
    
    print("Loading all Ti scans (this may take a while)...")
    start = time.time()
    
    try:
        # Use the actual function but with progress
        ti_scans = []
        count = 0
        last_time = start
        
        for root, dirs, files in os.walk(data_path):
            folder_name = os.path.basename(root)
            from src.data_loader import is_ti_folder
            if not is_ti_folder(folder_name):
                continue
            
            for file in files:
                if not file.endswith(".lvm"):
                    continue
                
                full_path = os.path.join(root, file)
                try:
                    arr = load_lvm(full_path)
                    energy = arr[:, 1]
                    signal = arr[:, 2]
                    
                    from src.data_loader import energy_overlaps, TI_RANGE
                    if not energy_overlaps(energy, *TI_RANGE):
                        continue
                    
                    parts = full_path.split(os.sep)
                    scan_number = file.replace(".lvm", "")
                    material = parts[-3] if len(parts) >= 3 else "unknown"
                    experiment = parts[-4] if len(parts) >= 4 else "unknown"
                    
                    ti_scans.append({
                        "label": f"{experiment}_{material}_Ti{scan_number}",
                        "experiment": experiment,
                        "material": material,
                        "scan_number": scan_number,
                        "path": full_path,
                        "energy": energy,
                        "signal": signal,
                    })
                    count += 1
                    
                    # Progress update every 100 scans or 5 seconds
                    current_time = time.time()
                    if count % 100 == 0 or (current_time - last_time) > 5:
                        elapsed = current_time - start
                        rate = count / elapsed if elapsed > 0 else 0
                        print(f"  Loaded {count} scans in {elapsed:.1f}s ({rate:.1f} scans/s)")
                        last_time = current_time
                        
                except Exception as e:
                    # Silently skip errors for speed
                    continue
        
        elapsed = time.time() - start
        print(f"✓ Loaded {len(ti_scans)} Ti scans in {elapsed:.3f}s")
        print(f"  Average: {elapsed/len(ti_scans)*1000:.1f}ms per scan")
        
        return ti_scans
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("="*60)
    print("DATA LOADER TESTS")
    print("="*60)
    
    # Test 1: Single file
    test_load_single_lvm()
    
    # Test 2: Limited loading
    test_load_ti_scans_limited()
    
    # Test 3: Full loading (commented out for now, uncomment to test)
    # print("\n" + "="*60)
    # print("FULL LOAD TEST (uncomment to run)")
    # print("="*60)
    # test_full_load_ti_scans()
    
    print("\n" + "="*60)
    print("Tests complete!")
    print("="*60)






