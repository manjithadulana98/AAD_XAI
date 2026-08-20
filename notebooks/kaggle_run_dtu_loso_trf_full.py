# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DTU LOSO TRF Baseline -- Full 18-Subject Sweep (Kaggle Notebook)
#
# Scales `kaggle_run_dtu_loso_trf_smoke.py`'s validated 2-fold smoke test
# (S1, S10 held out in turn, mean_acc=0.5500) to the full 18-subject LOSO
# sweep -- same pipeline, same window (5s), same seed, no other changes.
#
# Cost note: `aad_xai.evaluation.loso_runner` loads and preprocesses ALL 18
# subjects upfront regardless of `--max-folds` (confirmed by reading
# `loso_runner.py` directly) -- the smoke test's ~13-minute wall-clock was
# almost entirely this one-time load, with each fold's actual TRF fit+eval
# taking only ~7s. Expected total here: ~13 min load + 18x7s per-fold ~=
# ~15 minutes, not 9x the smoke test's wall-clock.
#
# **Kaggle setup requirements**
# - Enable Internet in notebook settings (for git clone + pip install)
# - Attach dataset `dulanamanjitha/aad-xai-artifacts` (holds DTU EEG + Audio)
# - No GPU needed -- TRF is a CPU-only ridge regression
#
# Outputs land in `/kaggle/working/results_dtu_loso_trf_full/`.

# %% [markdown]
# ## 1. Clone repository and install dependencies

# %%
import os
import subprocess
import sys

REPO_DIR = "/kaggle/working/AAD_XAI"

if not os.path.exists(REPO_DIR):
    subprocess.run(
        ["git", "clone", "https://github.com/manjithadulana98/AAD_XAI.git", REPO_DIR],
        check=True,
    )
else:
    print(f"Repository already cloned at {REPO_DIR}")

os.chdir(REPO_DIR)

try:
    import torch as _torch_preinstalled
    print(f"Pre-installed torch {_torch_preinstalled.__version__} found -- "
          "keeping it; installing the rest of requirements.txt without touching torch.")
    with open("requirements.txt") as _f:
        _reqs_no_torch = [ln for ln in _f if ln.strip() and not ln.strip().lower().startswith("torch")]
    with open("/tmp/requirements_no_torch.txt", "w") as _f:
        _f.writelines(_reqs_no_torch)
    subprocess.run(["pip", "install", "-q", "-r", "/tmp/requirements_no_torch.txt"], check=True)
except ImportError:
    print("No pre-installed torch found -- installing requirements.txt as-is.")
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

subprocess.run(["pip", "install", "-q", "-e", "."], check=True)
print("Setup done.")

# %% [markdown]
# ## 2. Resolve the DTU dataset path

# %%
DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]

DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, (
    "DTU dataset not found. Attach the 'dulanamanjitha/aad-xai-artifacts' dataset "
    "to this notebook. Tried: " + ", ".join(DTU_KAGGLE_ROOT_CANDIDATES)
)
print(f"DTU dataset: {DTU_ROOT}")
print("eeg_new/ subjects found:", sorted(os.listdir(os.path.join(DTU_ROOT, "eeg_new"))))

OUTPUT_DIR = "/kaggle/working/results_dtu_loso_trf_full"

# %% [markdown]
# ## 3. Run the full 18-subject DTU LOSO TRF sweep
#
# No `--max-folds` -- every subject held out in turn, same 5s window and
# seed=42 as the validated smoke test.

# %%
cmd = [
    sys.executable, "-m", "aad_xai.evaluation.loso_runner",
    "--data-dir", DTU_ROOT,
    "--output-dir", OUTPUT_DIR,
    "--window-s", "5",
    "--seed", "42",
]

print("Command:", " ".join(cmd))
print("=" * 70)

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in process.stdout:
    print(line, end="")
process.wait()
print("=" * 70)
print(f"Exit code: {process.returncode}")

# %% [markdown]
# ## 4. Display results

# %%
import csv
import json
from pathlib import Path

run_dirs = sorted(Path(OUTPUT_DIR).glob("DTU_TRF_LOSO_*"))
if not run_dirs:
    print(f"No run directory found under {OUTPUT_DIR} -- check for errors above.")
else:
    run_dir = run_dirs[-1]
    print(f"Run directory: {run_dir}\n")

    summary_path = run_dir / "baseline_performance_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("Baseline summary:", json.dumps(summary, indent=2))

    subj_path = run_dir / "subject_predictions.csv"
    if subj_path.exists():
        with subj_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"\nPer-subject results ({len(rows)} subjects):")
        for r in rows:
            print(f"  {r['subject_id']}: n_windows={r['n_windows']} accuracy={float(r['accuracy']):.4f}")

    win_path = run_dir / "window_predictions.csv"
    if win_path.exists():
        with win_path.open(newline="", encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1
        print(f"\nwindow_predictions.csv: {n_rows} rows")

# %% [markdown]
# ## 5. Next steps
#
# This establishes the full-population accuracy baseline. Explainability
# (Haufe-transformed TRF weight patterns + full statistical parity with
# the AADNet/VLAAI XAI pipelines) is a separate, larger build -- tracked
# and planned independently, not part of this notebook.
