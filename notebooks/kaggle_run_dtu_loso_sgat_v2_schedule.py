# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT -- DTU LOSO v2: schedule-only retry (Kaggle)
#
# The v1 LOSO run (`kaggle_run_dtu_loso_sgat_full.ipynb`) returned mean
# accuracy 0.4887 +/- 0.0247 across 18 folds -- indistinguishable from
# chance. The stuck-at-chance diagnostic
# (`kaggle_diag_sgat_symmetry_and_overfit.ipynb`) then showed decisively that
# the architecture CAN fit the data: a tiny fixed batch of 16 real DTU
# windows (8 attended + 8 unattended) reached 100% train accuracy by epoch
# 150 at a 10x higher learning rate (lr=1e-2). That rules out structural
# failure and points at the training *schedule* (lr/epochs/patience) as the
# cause of v1's stuck-at-chance result.
#
# This v2 notebook re-runs the full 18-fold LOSO sweep with exactly those
# two things changed, nothing else:
#
# |              | v1 (stuck at chance) | v2 (this notebook) |
# | ---          | ---                  | ---                |
# | `--lr`       | 1e-3 (default)       | 1e-2 (10x, matches the overfit-check LR that worked) |
# | `--epochs`   | 15                   | 60                 |
# | `--patience` | 5                    | 15                 |
# | window       | 5s                   | 5s (unchanged)     |
# | seed         | 42                   | 42 (unchanged)     |
#
# No code was changed between v1 and v2 -- the model, data pipeline, split
# logic, and FiLM mechanism are all exactly the same. Just the three CLI
# flags above. This is deliberately the cheapest possible intervention: if
# v2 still returns ~chance, THEN we have real evidence to justify discussing
# a code-level change (e.g. FiLM bias init), per CLAUDE.md's "do not change
# the architecture without discussing it first" rule.
#
# **Kaggle setup requirements**
# - Enable Internet (for git clone + pip install)
# - Attach dataset `dulanamanjitha/aad-xai-artifacts`
# - GPU strongly recommended -- 18 independent trainings at 4x the v1 epoch
#   count, so expect ~4x v1's wall-clock (v1 was ~44 min; v2 ~3 hours).

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
    print(f"Repository already present at {REPO_DIR} -- pulling latest.")
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)

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

_src_dir = os.path.join(REPO_DIR, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import aad_xai  # noqa: F401
print(f"aad_xai importable from: {aad_xai.__file__}")
print("Setup done.")

# %% [markdown]
# ## 2. Resolve the DTU dataset path

# %%
from pathlib import Path

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]

DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, (
    "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'. "
    "Tried: " + ", ".join(DTU_KAGGLE_ROOT_CANDIDATES)
)
print(f"DTU dataset: {DTU_ROOT}")

# %% [markdown]
# ## 3. GPU sanity check

# %%
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 4. Dry run: time 2 folds first
#
# Same dry-run discipline as v1: time 2 folds before committing to all 18.
# v1 fold time was ~145s/fold at 15 epochs; this is 4x epochs so expect
# ~10 min/fold => ~3 hours total.

# %%
OUTPUT_DIR_DRYRUN = "/kaggle/working/results_dtu_loso_sgat_v2_dryrun"

cmd_dryrun = [
    sys.executable, "-m", "aad_xai.run_experiments",
    "--dataset", "dtu",
    "--data-dir", DTU_ROOT,
    "--cv", "loso",
    "--model", "sgat",
    "--window", "5",
    "--seed", "42",
    "--lr", "1e-2",
    "--epochs", "60",
    "--patience", "15",
    "--max-folds", "2",
    "--output", OUTPUT_DIR_DRYRUN,
]
print("Command:", " ".join(cmd_dryrun))
print("=" * 70)

import time as _time
_t0 = _time.time()
process = subprocess.Popen(cmd_dryrun, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="")
process.wait()
_dryrun_elapsed = _time.time() - _t0
print("=" * 70)
print(f"Exit code: {process.returncode}  |  2-fold wall-clock: {_dryrun_elapsed:.0f}s "
      f"(~{_dryrun_elapsed / 2:.0f}s/fold -> ~{_dryrun_elapsed / 2 * 18 / 60:.0f} min for all 18)")

# %% [markdown]
# ## 5. Full 18-subject LOSO run (v2)

# %%
OUTPUT_DIR = "/kaggle/working/results_dtu_loso_sgat_v2"

cmd = [
    sys.executable, "-m", "aad_xai.run_experiments",
    "--dataset", "dtu",
    "--data-dir", DTU_ROOT,
    "--cv", "loso",
    "--model", "sgat",
    "--window", "5",
    "--seed", "42",
    "--lr", "1e-2",
    "--epochs", "60",
    "--patience", "15",
    "--output", OUTPUT_DIR,
]
print("Command:", " ".join(cmd))
print("=" * 70)

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="")
process.wait()
print("=" * 70)
print(f"Exit code: {process.returncode}")

# %% [markdown]
# ## 6. Display v2 results side-by-side with v1

# %%
import json

summary_path = Path(OUTPUT_DIR) / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("v2 master summary (mean +/- std across folds):")
    for s in summary:
        print(f"  {s['cv']:<20s} {s['model'].upper():<8s} "
              f"{s['mean_acc']:.4f} +/- {s['std_acc']:.4f}  "
              f"({s['n_folds']} folds, {s['time_s']:.0f}s)")
else:
    print(f"No v2 summary found at {summary_path} -- check for errors above.")

fold_detail_path = Path(OUTPUT_DIR) / "loso_sgat_w5p0s.json"
if fold_detail_path.exists():
    fold_detail = json.loads(fold_detail_path.read_text(encoding="utf-8"))
    folds = fold_detail["per_fold"]
    print(f"\nPer-fold detail ({len(folds)} folds):")
    for r in folds:
        test_subj = r.get("meta", {}).get("test_subject", r.get("fold_id"))
        print(f"  {test_subj}: acc={r['test_accuracy']:.4f}")

print("\nFor reference, v1 (lr=1e-3, epochs=15) was: 0.4887 +/- 0.0247 across 18 folds.")

# %% [markdown]
# ## 7. What to look at
#
# - **If v2 mean accuracy is above ~0.55 with per-fold spread similar to
#   v1's 0.44-0.56**: the diagnosis was right -- schedule was the only
#   issue, no code change needed. Pick a representative fold (e.g. median
#   accuracy) and train that fold's checkpoint standalone via
#   `aad_xai.train --dataset dtu --model sgat` so a real trained checkpoint
#   exists; then re-point the smoke notebook's interpretability dry run at
#   that checkpoint for actual (not noise) interpretability findings.
#
# - **If v2 is still ~0.5**: schedule alone isn't enough. That's real
#   evidence to justify a CLAUDE.md-compliant discussion of a minimal code
#   change (most likely: a positive bias init on `film.bias` so FiLM's
#   gamma/beta start non-zero at epoch 1, breaking the two-pass symmetry
#   mechanically rather than hoping gradient flow does it).
#
# - **If v2 is somewhere in between (e.g. 0.52 - 0.54)**: schedule
#   helped but isn't enough. Options then: longer window (10s), more epochs
#   still, or the FiLM-init tweak above.
