"""
Data Loading Module - Easy to modify for different data sources.

This module handles loading LVM files and discovering Ti scans.
Modify the constants and functions here to adapt to different data structures.
"""
import os
import numpy as np
from typing import List, Dict, Any


# ============================================================================
# CONFIGURATION - Easy to modify
# ============================================================================

# Ti MVV energy range (eV)
TI_RANGE = (25, 60)

# Keywords to identify Ti folders (case-insensitive)
TI_KEYWORDS = ["ti", "titanium", "mv", "mvv", "ti_mvv", "ti-scan"]


# ============================================================================
# LVM FILE LOADER
# ============================================================================

def _parse_lvm_lines(lines) -> np.ndarray:
    """Parse LVM content (list of lines or string). Returns numeric array."""
    if isinstance(lines, str):
        lines = lines.splitlines()
    raw_rows = []
    for line in lines:
        line = line.strip() if isinstance(line, str) else line
        if str(line).startswith("----------"):
            break
        parts = str(line).split()
        if len(parts) == 0:
            continue
        try:
            nums = [float(x) for x in parts]
            raw_rows.append(nums)
        except (ValueError, TypeError):
            continue
    if len(raw_rows) == 0:
        raise ValueError("No valid numeric rows in LVM content")
    lengths = [len(r) for r in raw_rows]
    target_len = max(set(lengths), key=lengths.count)
    clean = [r for r in raw_rows if len(r) == target_len]
    return np.array(clean, dtype=float)


def load_lvm(path: str) -> np.ndarray:
    """
    Load Staib .lvm file.
    
    Args:
        path: Path to .lvm file
        
    Returns:
        numpy array of shape (N_rows, N_cols) with numeric data
    """
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        content = f.read()
    return _parse_lvm_lines(content)


def load_lvm_from_content(content: str) -> np.ndarray:
    """
    Load Staib .lvm format from a string (e.g. downloaded from Drive).
    Same format as load_lvm but without touching disk.
    """
    if isinstance(content, bytes):
        content = content.decode("latin-1", errors="ignore")
    return _parse_lvm_lines(content)


# ============================================================================
# TI FOLDER DETECTION
# ============================================================================

def is_ti_folder(name: str) -> bool:
    """Check if folder name indicates Ti scans."""
    name_lower = name.lower()
    return any(kw in name_lower for kw in TI_KEYWORDS)


def energy_overlaps(energy: np.ndarray, low: float, high: float) -> bool:
    """Check if energy array overlaps with range [low, high]."""
    return (energy.min() <= high) and (energy.max() >= low)


# ============================================================================
# TI SCAN DISCOVERY
# ============================================================================

def _load_ti_scans_local(root_folder: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """Load Ti MVV scans from a local directory (used by load_ti_scans when not gdrive:)."""
    import time
    ti_scans = []
    count = 0
    errors = 0
    start_time = time.time()
    last_print_time = start_time
    
    for root, dirs, files in os.walk(root_folder):
        folder_name = os.path.basename(root)
        
        # Only consider folders that look like Ti
        if not is_ti_folder(folder_name):
            continue
        
        for file in files:
            if not file.endswith(".lvm"):
                continue
            
            full_path = os.path.join(root, file)
            
            try:
                # Load data to check energy window
                arr = load_lvm(full_path)
                energy = arr[:, 1]  # Column 2 = energy (eV)
                signal = arr[:, 2]  # Column 3 = intensity
                
                # Check if this scan is in the Ti MVV region
                if not energy_overlaps(energy, *TI_RANGE):
                    continue
                
                # Parse path pieces
                parts = full_path.split(os.sep)
                scan_number = file.replace(".lvm", "")
                element = "Ti"
                material = parts[-3] if len(parts) >= 3 else "unknown"
                experiment = parts[-4] if len(parts) >= 4 else "unknown"
                
                # Create label
                label = f"{experiment}_{material}_{element}{scan_number}"
                
                ti_scans.append({
                    "label": label,
                    "experiment": experiment,
                    "material": material,
                    "element": element,
                    "scan_number": scan_number,
                    "path": full_path,
                    "energy": energy,
                    "signal": signal,
                })
                count += 1
                
                # Progress update every 100 scans or every 5 seconds
                current_time = time.time()
                if verbose and (count % 100 == 0 or (current_time - last_print_time) > 5):
                    elapsed = current_time - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"  Loaded {count} scans in {elapsed:.1f}s ({rate:.1f} scans/s, {errors} errors)")
                    last_print_time = current_time
                    
            except Exception as e:
                errors += 1
                if verbose and errors <= 5:  # Only print first few errors
                    print(f"Warning: Failed to load {full_path}: {e}")
                continue
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"Loaded {count} Ti scans in {elapsed:.1f}s ({errors} errors)")
    
    return ti_scans


def group_scans_by_experiment(ti_scans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group scans by experiment name."""
    from collections import defaultdict
    
    scans_by_exp = defaultdict(list)
    for rec in ti_scans:
        exp = rec["experiment"]
        scan_num = int(rec["scan_number"])
        rec["scan_number"] = scan_num  # Convert to int
        scans_by_exp[exp].append(rec)
    
    # Sort each experiment's scans by scan_number
    for exp in scans_by_exp:
        scans_by_exp[exp] = sorted(scans_by_exp[exp], key=lambda x: x["scan_number"])
    
    return dict(scans_by_exp)


# ============================================================================
# GOOGLE DRIVE SOURCE (streaming, no full local copy)
# ============================================================================

GDRIVE_PREFIX = "gdrive:"


def _list_drive_folder_tree(service, folder_id: str, path_prefix: str = "") -> List[tuple]:
    """
    Recursively list all files under a Drive folder. Returns list of (file_id, logical_path).
    logical_path uses '/' and does not include a leading slash.
    """
    results = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=200,
                pageToken=page_token,
            )
            .execute()
        )
        for f in response.get("files", []):
            fid, name, mime = f["id"], f["name"], f.get("mimeType", "")
            prefix = f"{path_prefix}/{name}" if path_prefix else name
            if mime == "application/vnd.google-apps.folder":
                results.extend(_list_drive_folder_tree(service, fid, prefix))
            else:
                results.append((fid, prefix))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return results


def load_ti_scans_from_drive(
    folder_id: str,
    verbose: bool = True,
    credentials=None,
) -> List[Dict[str, Any]]:
    """
    Load Ti MVV scans from a Google Drive folder without copying the folder locally.
    Only fetches .lvm files that sit in folders matching TI_KEYWORDS and that
    fall in the Ti MVV energy range (25--60 eV). Each file is downloaded into memory,
    parsed, and discarded so minimal local disk is used.

    Requires Google Drive API credentials (Application Default Credentials or
    GOOGLE_APPLICATION_CREDENTIALS). Share the Drive folder with the service account
    or use a user OAuth token.

    Args:
        folder_id: Google Drive folder ID (e.g. from the folder URL).
        verbose: Print progress.
        credentials: Optional google.oauth2 credentials. If None, uses default.

    Returns:
        Same structure as load_ti_scans(): list of scan dicts with label, experiment,
        material, element, scan_number, path (logical), energy, signal.
    """
    import time
    from google.auth.default import default
    from google.auth.exceptions import DefaultCredentialsError
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io

    if credentials is None:
        try:
            credentials, _ = default(
                scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
        except DefaultCredentialsError:
            raise RuntimeError(
                "Google Drive credentials not found. Run:\n"
                "  gcloud auth application-default login\n"
                "or set GOOGLE_APPLICATION_CREDENTIALS to a service account key JSON path."
            ) from None

    service = build("drive", "v3", credentials=credentials)

    if verbose:
        print("  Listing Drive folder (only Ti-related .lvm files will be fetched)...")
    start = time.time()
    all_files = _list_drive_folder_tree(service, folder_id)
    # Filter: path must contain a segment that looks like a Ti folder, and file must be .lvm
    candidates = []
    for fid, logical_path in all_files:
        if not logical_path.endswith(".lvm"):
            continue
        parts = logical_path.replace("\\", "/").split("/")
        # Parent folder of the file is the folder name we check for Ti
        parent_folder = parts[-2] if len(parts) >= 2 else ""
        if not is_ti_folder(parent_folder):
            continue
        candidates.append((fid, logical_path))
    if verbose:
        print(f"  Found {len(candidates)} candidate .lvm files in Ti folders (of {len(all_files)} total files).")

    ti_scans = []
    errors = 0
    for i, (file_id, logical_path) in enumerate(candidates):
        try:
            request = service.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            content = buf.getvalue().decode("latin-1", errors="ignore")

            arr = load_lvm_from_content(content)
            energy = arr[:, 1]
            signal = arr[:, 2]
            if not energy_overlaps(energy, *TI_RANGE):
                continue

            parts = logical_path.replace("\\", "/").split("/")
            scan_number = parts[-1].replace(".lvm", "") if parts else "0"
            material = parts[-3] if len(parts) >= 3 else "unknown"
            experiment = parts[-4] if len(parts) >= 4 else "unknown"
            label = f"{experiment}_{material}_Ti{scan_number}"

            ti_scans.append({
                "label": label,
                "experiment": experiment,
                "material": material,
                "element": "Ti",
                "scan_number": scan_number,
                "path": logical_path,
                "energy": energy,
                "signal": signal,
            })
            if verbose and (len(ti_scans) % 100 == 0) and len(ti_scans) > 0:
                elapsed = time.time() - start
                print(f"  Loaded {len(ti_scans)} scans in {elapsed:.1f}s ({errors} skipped/errors)")
        except Exception as e:
            errors += 1
            if verbose and errors <= 5:
                print(f"  Warning: Skip {logical_path}: {e}")

    if verbose:
        elapsed = time.time() - start
        print(f"  Loaded {len(ti_scans)} Ti scans from Drive in {elapsed:.1f}s ({errors} errors).")
    return ti_scans


def load_ti_scans(root_folder: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Load all Ti MVV scans from a local path or from Google Drive.

    - If root_folder is a local path: walks the directory and loads .lvm files
      in folders matching TI_KEYWORDS with energy in TI_RANGE (25--60 eV).
    - If root_folder starts with "gdrive:" (e.g. "gdrive:1lYoklYJlYTB_XHCzs7l21bqk8PbmRPME"):
      uses the Google Drive API to list and stream only relevant .lvm files from that
      folder, without copying the whole folder to disk.

    Returns:
        List of scan records (label, experiment, material, element, scan_number, path, energy, signal).
    """
    if root_folder.strip().lower().startswith(GDRIVE_PREFIX):
        folder_id = root_folder.strip()[len(GDRIVE_PREFIX):].strip()
        if not folder_id:
            raise ValueError("gdrive: requires a folder ID, e.g. gdrive:1lYoklYJlYTB_XHCzs7l21bqk8PbmRPME")
        return load_ti_scans_from_drive(folder_id, verbose=verbose)
    return _load_ti_scans_local(root_folder, verbose)

