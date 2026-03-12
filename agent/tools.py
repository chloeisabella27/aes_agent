"""
Agent tools: structured JSON-only functions for the AES analysis agent.
All tools dispatch to Python (no subprocess). DATA_ROOT + experiment name
is used to build the experiment folder path for the ML pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import difflib

# Project root = parent of agent/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data root: env override or default
DATA_ROOT = os.getenv("AES_DATA_ROOT", "/Users/chloeisabella/Desktop/Files for Yash")


def list_experiments() -> Dict[str, List[str]]:
    """List available experiment names from the dataset directory. Returns structured JSON only."""
    if not os.path.exists(DATA_ROOT):
        return {"experiments": []}
    experiments = [
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]
    experiments.sort()
    return {"experiments": experiments}


def resolve_experiment(query: str) -> Dict[str, Any]:
    """Resolve a possibly misspelled or partial experiment name. Returns match, score, alternatives."""
    data = list_experiments()
    experiments = data.get("experiments", [])
    if not experiments:
        return {"match": None, "score": 0.0, "alternatives": []}
    query_clean = query.strip()
    if not query_clean:
        return {"match": None, "score": 0.0, "alternatives": experiments[:5]}
    # Exact match (case-insensitive)
    for ex in experiments:
        if ex.upper() == query_clean.upper():
            return {"match": ex, "score": 1.0, "alternatives": [ex]}
    # Fuzzy match
    matches = difflib.get_close_matches(query_clean, experiments, n=5, cutoff=0.4)
    if not matches:
        return {"match": None, "score": 0.0, "alternatives": []}
    best = matches[0]
    # Approximate score from sequence matcher
    score = difflib.SequenceMatcher(None, query_clean.upper(), best.upper()).ratio()
    return {"match": best, "score": round(score, 2), "alternatives": matches}


def predict_next_scan(experiment: str) -> Dict[str, Any]:
    """
    Run the ML pipeline to predict the next Ti MVV scan for the given experiment.
    Builds experiment_folder = DATA_ROOT / experiment and calls the pipeline directly.
    Returns structured JSON from the written prediction record (no raw logs).
    """
    experiment_folder = os.path.join(DATA_ROOT, experiment.strip())
    if not os.path.isdir(experiment_folder):
        return {
            "error": f"Experiment folder not found: {experiment_folder}",
            "experiment": experiment,
            "energy_points": None,
            "rmse": None,
            "prediction_file": None,
        }
    try:
        # Import here so agent can run even if pipeline deps (e.g. torch) are heavy
        sys_path = list(__import__("sys").path)
        root = str(_PROJECT_ROOT)
        if root not in sys_path:
            __import__("sys").path.insert(0, root)
        from aes_ml_pipeline.predict_next_scan import predict_next_scan as _predict

        energy, predicted = _predict(experiment_folder)
        # Pipeline writes to outputs/YYYY-MM-DD-inference/{experiment}_scan{N}_lstm_prediction.json
        # Find the most recent inference dir and the matching JSON
        outputs_dir = _PROJECT_ROOT / "outputs"
        record = _load_latest_prediction_record(outputs_dir, experiment)
        if record:
            return {
                "experiment": record.get("experiment", experiment),
                "energy_points": record.get("n_points", len(energy)),
                "energy_min": record.get("energy_min"),
                "energy_max": record.get("energy_max"),
                "rmse": record.get("rmse"),
                "prediction_file": record.get("prediction_file"),
                "predicted_scan_index": record.get("predicted_scan_index"),
                "input_scans": record.get("input_scans", []),
            }
        return {
            "experiment": experiment,
            "energy_points": int(len(energy)),
            "energy_min": float(energy.min()) if hasattr(energy, "min") else None,
            "energy_max": float(energy.max()) if hasattr(energy, "max") else None,
            "rmse": None,
            "prediction_file": None,
            "predicted_scan_index": None,
            "input_scans": [],
        }
    except Exception as e:
        return {
            "error": str(e),
            "experiment": experiment,
            "energy_points": None,
            "rmse": None,
            "prediction_file": None,
        }


def _load_latest_prediction_record(outputs_dir: Path, experiment: str) -> Optional[Dict[str, Any]]:
    """Load the most recent prediction JSON for the given experiment from outputs/*-inference/."""
    if not outputs_dir.exists():
        return None
    experiment = experiment.strip()
    candidates = []
    for d in outputs_dir.iterdir():
        if not d.is_dir() or not d.name.endswith("-inference"):
            continue
        for f in d.glob("*_prediction.json"):
            if f.stem.startswith(experiment) or experiment in f.stem:
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    data["_path"] = str(f)
                    data["_mtime"] = f.stat().st_mtime
                    candidates.append(data)
                except Exception:
                    continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.get("_mtime", 0), reverse=True)
    out = candidates[0].copy()
    out.pop("_path", None)
    out.pop("_mtime", None)
    return out


def get_prediction_summary(experiment: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the latest prediction summary for an experiment, or the most recent run if experiment is omitted.
    Reads from inference output JSON only (no pipeline call).
    """
    outputs_dir = _PROJECT_ROOT / "outputs"
    if not outputs_dir.exists():
        return {"summary": None, "experiment": experiment, "message": "No outputs directory found."}
    all_records = []
    for d in outputs_dir.iterdir():
        if not d.is_dir() or not d.name.endswith("-inference"):
            continue
        for f in d.glob("*_prediction.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                data["_mtime"] = f.stat().st_mtime
                all_records.append(data)
            except Exception:
                continue
    if not all_records:
        return {"summary": None, "experiment": experiment, "message": "No prediction records found."}
    all_records.sort(key=lambda x: x.get("_mtime", 0), reverse=True)
    if experiment:
        filtered = [r for r in all_records if r.get("experiment") == experiment.strip()]
        if not filtered:
            return {"summary": None, "experiment": experiment, "message": f"No predictions found for experiment '{experiment}'."}
        rec = filtered[0]
    else:
        rec = all_records[0]
    rec = {k: v for k, v in rec.items() if not k.startswith("_")}
    return {"summary": rec, "experiment": rec.get("experiment"), "message": "OK"}
