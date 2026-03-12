# AES ML Pipeline

Simplified ML pipeline for AES Ti MVV spectral prediction with temporal sequence handling.

## Quick Start

### Option A: Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package and project manager. Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then:

```bash
# 1. Initialize project with Python 3.12 (creates/updates pyproject.toml)
uv init --python=3.12

# 2. Create virtual environment at .venv
uv venv

# 3. Add PyTorch (and sync the environment)
uv add torch
```

Run the app with: `uv run python train.py ...` or activate `.venv` and use `python` as usual.

### Option B: pip

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (local data)
python train.py --data-path "/path/to/local/data" --models nn lstm

# Train models using data from Google Drive (no full download; streams only Ti .lvm files)
python train.py --data-path "gdrive:1lYoklYJlYTB_XHCzs7l21bqk8PbmRPME" --models nn lstm

# Make predictions
python predict.py --experiment TF268 --scan 5 --model lstm --model-path outputs/models/lstm.pth
```

## Data source: Google Drive

You can point the pipeline at your Google Drive folder **without downloading the whole folder** to disk. Only **relevant data** is used:

- **Included:** `.lvm` files inside folders whose names match Ti keywords (`ti`, `titanium`, `mv`, `mvv`, etc.) and whose energy range overlaps **25–60 eV** (Ti MVV). These are streamed from Drive into memory and not written to disk.
- **Ignored:** Everything else in the folder (other file types, non‑Ti folders, scans outside 25–60 eV).

Use the folder ID from your Drive link as the data path with the `gdrive:` prefix:

```bash
python train.py --data-path "gdrive:1lYoklYJlYTB_XHCzs7l21bqk8PbmRPME" --models nn lstm
```

**One-time setup for Drive access:** use one of:

1. **Application Default Credentials (recommended):**  
   `gcloud auth application-default login`  
   Then open the Drive folder link in the browser and ensure the same Google account has access.

2. **Service account:**  
   Create a key in Google Cloud Console, share the Drive folder with the service account email, then set  
   `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`

Same `--data-path` works in `generate_figures.py`.

## Structure

- `src/` - Core modules (data loading, preprocessing, models, training, evaluation, prediction)
- `train.py` - Training CLI
- `predict.py` - Prediction CLI
- `tests/` - Test suite
- `outputs/` - Generated models, plots, reports

---

## Project status and “what was done” (to restart work)

- **Data:** Pipeline can use a **local path** or **Google Drive** via `--data-path`. For Drive, use `gdrive:<folder_id>` (e.g. `gdrive:1lYoklYJlYTB_XHCzs7l21bqk8PbmRPME`). Only Ti-related `.lvm` files in the 25–60 eV range are read; they are streamed from the API, so the Drive folder is not fully copied to disk.
- **Reports:** See `accuracy_report.md` and `SUMMARY.md` for metrics and best model (LSTM). `outputs/figures/research_report.tex` is the main report.
- **Best model:** LSTM (RMSE ~0.88); config in `accuracy_report.md`. Models are saved under `outputs/models/`.
- **CLI:** `train.py` (train), `predict.py` (predict), `generate_figures.py` (figures), `report_results.py` (accuracy report).
- **Tests:** `tests/`; data loader and Drive path are in `src/data_loader.py`.





