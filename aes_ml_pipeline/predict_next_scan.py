#!/usr/bin/env python3
"""
Inference wrapper for predicting the next AES Ti MVV scan for a single experiment.

This module is intentionally additive: it does NOT modify or replace any existing
training, preprocessing, PCA, or figure-generation logic. It simply reuses the
objects saved by the training pipeline (PCA, encoders, trained LSTM weights).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure the project root (parent of aes_ml_pipeline) is on sys.path so that
# imports like `from src.data_loader import ...` work regardless of CWD.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from aes_ml_pipeline.src.data_loader import load_ti_scans, group_scans_by_experiment
from aes_ml_pipeline.src.preprocessing import resample_spectrum
from aes_ml_pipeline.src.models import get_model_class

@dataclass
class PredictionRecord:
    experiment: str
    model: str
    energy_min: float
    energy_max: float
    n_points: int
    input_scans: List[int]
    predicted_scan_index: int
    rmse: Optional[float]
    model_path: str
    comparison_results_path: str
    timestamp: str
    prediction_file: str


def _find_latest_lstm_run(outputs_root: Path) -> Tuple[Path, Path]:
    """
    Locate the most recent outputs/* run that contains both:
      - models/lstm.pth
      - comparison_results.pkl

    Returns:
        (run_dir, model_path)
    """
    candidates: List[Tuple[float, Path, Path]] = []

    if not outputs_root.exists():
        raise FileNotFoundError(f"No outputs directory found at {outputs_root}")

    for run_dir in outputs_root.iterdir():
        if not run_dir.is_dir():
            continue
        model_path = run_dir / "models" / "lstm.pth"
        results_path = run_dir / "comparison_results.pkl"
        if model_path.exists() and results_path.exists():
            mtime = max(model_path.stat().st_mtime, results_path.stat().st_mtime)
            candidates.append((mtime, run_dir, model_path))

    if not candidates:
        raise FileNotFoundError(
            f"No LSTM run with comparison_results.pkl found under {outputs_root}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_run_dir, best_model_path = candidates[0]
    return best_run_dir, best_model_path


def _load_training_artifacts(run_dir: Path) -> Dict[str, Any]:
    """
    Load the training pickle saved by train.py for the given run directory.
    """
    import pickle

    results_path = run_dir / "comparison_results.pkl"
    if not results_path.exists():
        raise FileNotFoundError(f"comparison_results.pkl not found at {results_path}")

    with open(results_path, "rb") as f:
        data = pickle.load(f)

    # Expected structure (see train.py):
    # {
    #   'results': {...},
    #   'preprocessed': {
    #       'pca': ...,
    #       'exp_encoder': ...,
    #       'common_energy': ...,
    #       'latent_dim': ...,
    #       'input_dim': ...,
    #   },
    #   'ti_scans': [...],
    #   'train_indices': ...,
    #   'test_indices': ...,
    # }
    if "preprocessed" not in data or "ti_scans" not in data or "results" not in data:
        raise ValueError(
            f"comparison_results.pkl at {results_path} is missing expected keys"
        )

    return data


def _infer_experiment_name(experiment_folder: Path) -> str:
    """
    Derive experiment name from folder path. During training, the experiment
    name is taken from the directory structure (see src.data_loader._load_ti_scans_local),
    which typically matches the folder name for the experiment (e.g. TF268).
    """
    return experiment_folder.name


def _select_input_and_target_scans(
    scans_for_experiment: List[Dict[str, Any]],
    input_max_scan: int = 4,
) -> Tuple[List[int], int]:
    """
    Given all scan records for one experiment (already sorted by scan_number),
    choose input scans and the next scan index to predict.

    By default, uses scans 1–4 (if available) to predict scan 5, matching the
    temporal split used during training.
    """
    scan_numbers = [int(rec["scan_number"]) for rec in scans_for_experiment]
    # Use all scans <= input_max_scan as context (e.g., 1–4)
    input_scans = sorted([s for s in scan_numbers if s <= input_max_scan])
    if not input_scans:
        raise ValueError(
            f"No scans <= {input_max_scan} found for experiment; cannot form input sequence."
        )

    predicted_scan = max(input_scans) + 1
    return input_scans, predicted_scan


def predict_next_scan(experiment_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict the next AES Ti MVV spectrum for a single experiment.

    Args:
        experiment_folder: Path to the folder containing one experiment's scans
                           (e.g., .../Files for Yash/TF268).

    Returns:
        (energy_grid, predicted_spectrum) where:
          - energy_grid is a 1D numpy array (25–60 eV grid)
          - predicted_spectrum is a 1D numpy array of normalized intensity values
    """
    experiment_path = Path(experiment_folder).expanduser().resolve()
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment folder does not exist: {experiment_path}")

    experiment_name = _infer_experiment_name(experiment_path)

    # 1. Load training artifacts from the most recent LSTM run
    outputs_root = _PROJECT_ROOT / "outputs"  
    run_dir, model_path = _find_latest_lstm_run(outputs_root)
    artifacts = _load_training_artifacts(run_dir)
    preprocessed = artifacts["preprocessed"]
    results = artifacts["results"]
    ti_scans_all: List[Dict[str, Any]] = artifacts["ti_scans"]

    pca = preprocessed["pca"]
    exp_encoder = preprocessed["exp_encoder"]
    common_energy: np.ndarray = preprocessed["common_energy"]
    latent_dim: int = preprocessed["latent_dim"]
    input_dim: int = preprocessed["input_dim"]

    # Recover best hyperparameters used during training for the LSTM so that
    # the instantiated architecture matches the checkpoint exactly.
    lstm_result = results.get("lstm")
    if lstm_result is None:
        raise ValueError(
            "No LSTM entry found in saved results; cannot infer architecture."
        )
    best_params = lstm_result.get("best_params") or {}
    hidden_dim = int(best_params.get("hidden_dim", 128))
    num_layers = int(best_params.get("num_layers", 2))
    dropout = float(best_params.get("dropout", 0.1))

    # 2. Locate scans for the requested experiment using the same grouping logic
    scans_by_exp = group_scans_by_experiment(ti_scans_all)
    if experiment_name not in scans_by_exp:
        raise ValueError(
            f"Experiment '{experiment_name}' not found in training data. "
            f"Available experiments: {sorted(scans_by_exp.keys())}"
        )
    scans_for_experiment = scans_by_exp[experiment_name]

    # 3. Decide which scans are inputs and which scan to predict
    input_scans, predicted_scan = _select_input_and_target_scans(scans_for_experiment)

    # 4. Build model input using the same temporal encoding as training:
    #    X = [experiment_id, scan_number]
    try:
        exp_id = int(exp_encoder.transform([experiment_name])[0])
    except ValueError as exc:
        raise ValueError(
            f"Experiment '{experiment_name}' is unknown to the trained encoder."
        ) from exc

    X_input = np.array([[exp_id, float(predicted_scan)]], dtype=np.float32)

    # 5. Load the trained LSTM model
    model_class = get_model_class("lstm")
    model = model_class(
        latent_dim=latent_dim,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 6. Run the model to predict the latent PCA vector
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_input).to(device)
        latent_pred = model(x_tensor).cpu().numpy()[0]  # shape: (latent_dim,)

    # 7. Convert latent prediction back to normalized spectrum via PCA inverse transform
    predicted_norm_spectrum = pca.inverse_transform(latent_pred.reshape(1, -1))[0]

    # 8. Optionally compute RMSE against the actual next scan if it exists on disk
    rmse: Optional[float] = None
    # Try to find a record for the predicted scan number
    true_scan_recs = [
        rec for rec in scans_for_experiment if int(rec["scan_number"]) == predicted_scan
    ]
    if true_scan_recs:
        # Use the same energy window and normalization as in training
        true_rec = true_scan_recs[0]
        true_resampled = resample_spectrum(
            np.asarray(true_rec["energy"], dtype=float),
            np.asarray(true_rec["signal"], dtype=float),
            common_energy,
        )
        # Normalize by maximum intensity
        max_val = float(np.max(true_resampled)) if np.max(true_resampled) != 0 else 1.0
        true_norm = true_resampled / max_val

        # Ensure shapes match
        if true_norm.shape == predicted_norm_spectrum.shape:
            diff = true_norm - predicted_norm_spectrum
            rmse = float(np.sqrt(np.mean(diff ** 2)))

    # 9. Persist prediction metadata and spectrum to a dated inference directory
    today = date.today().isoformat()
    project_root = _PROJECT_ROOT
    inference_root = project_root / "outputs" / f"{today}-inference"
    inference_root.mkdir(parents=True, exist_ok=True)

    # One file per (experiment, predicted_scan)
    base_name = f"{experiment_name}_scan{predicted_scan}_lstm"
    spectrum_path = inference_root / f"{base_name}_spectrum.npz"
    np.savez(
        spectrum_path,
        energy=common_energy,
        intensity=predicted_norm_spectrum,
    )

    record = PredictionRecord(
        experiment=experiment_name,
        model="lstm",
        energy_min=float(common_energy.min()),
        energy_max=float(common_energy.max()),
        n_points=int(common_energy.size),
        input_scans=input_scans,
        predicted_scan_index=predicted_scan,
        rmse=rmse,
        model_path=str(model_path),
        comparison_results_path=str(run_dir / "comparison_results.pkl"),
        timestamp=datetime.utcnow().isoformat() + "Z",
        prediction_file=str(spectrum_path),
    )

    record_path = inference_root / f"{base_name}_prediction.json"
    with record_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2)

    return common_energy, predicted_norm_spectrum


__all__ = ["predict_next_scan"]

