# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT -- DTU LOSO v3: FiLM-init fix (Kaggle)
#
# v1 (lr=1e-3, epochs=15):      0.4887 +/- 0.0247 across 18 folds -- stuck at chance
# v2 (lr=1e-2, epochs=60):      0.4814 +/- 0.0192 across 18 folds -- still stuck at chance
#
# The schedule-only v2 retry falsified the "undertrained" hypothesis. The
# diagnostic notebook (Oct 17) showed the architecture CAN fit 16 real DTU
# windows to 100% train accuracy at lr=1e-2 by epoch 150 -- so the issue
# was gradient flow at init on the full dataset, not model capacity.
#
# After CLAUDE.md-compliant discussion (Oct 18), a two-line parameter-init
# change was accepted (commit `5f49991`): `CandidateConditionedGAT.__init__`
# now sets `film.weight = 0` and `film.bias` such that at init, FiLM output
# is `gamma = 0.1, beta = 0` -- so `film.weight` receives real gradient
# signal from the first backward pass instead of relying on noise to break
# the two-pass symmetry. Local verification showed overfit now reaches
# 100% train accuracy by epoch 20 (vs. epoch 150 at the same lr=1e-2 before
# the fix), confirming the gradient signal is now immediate.
#
# This v3 notebook re-runs the full 18-fold LOSO sweep against the new init
# at v2's schedule (lr=1e-2, epochs=60). If the FiLM-init hypothesis is
# right, v3 should land meaningfully above chance (>~0.55 mean). If it's
# still ~0.5, the hypothesis is falsified and we have real evidence to go
# look at the other candidates (envelope informativeness, adjacency
# correctness) rather than more code changes.
#
# |              | v1 (stuck)   | v2 (stuck)   | v3 (this run)        |
# | ---          | ---          | ---          | ---                  |
# | FiLM init    | default      | default      | zeros + 0.1 bias     |
# | `--lr`       | 1e-3         | 1e-2         | 1e-2                 |
# | `--epochs`   | 15           | 60           | 60                   |
# | `--patience` | 5            | 15           | 15                   |
# | window       | 5s           | 5s           | 5s                   |
# | seed         | 42           | 42           | 42                   |
#
# **Kaggle setup requirements**
# - Enable Internet (git clone + pip install)
# - Attach dataset `dulanamanjitha/aad-xai-artifacts`
# - GPU strongly recommended -- same 18 folds, 60 epochs; expect ~3 hours

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

# %% [markdown]
# ## 2. Confirm the FiLM-init fix is live
#
# Explicit guard against running v3 against an unupdated checkout -- if the
# repo somehow didn't pull commit `5f49991`, v3 would quietly reproduce v2's
# null result and we'd have no idea why.

# %%
import torch
from aad_xai.models.sgat_gat_layer import CandidateConditionedGAT

_probe = CandidateConditionedGAT(
    n_channels=4, in_features=6, speech_dim=4, out_features=8,
    adjacency=torch.eye(4), n_heads=2,
)
assert torch.all(_probe.film.weight == 0.0), (
    "FiLM-init fix is NOT applied -- film.weight is non-zero at init. "
    "git pull or re-clone the repo. v3 would otherwise reproduce v2's null result."
)
assert torch.all(_probe.film.bias[:6] == 0.1), "FiLM gamma bias != 0.1 at init"
assert torch.all(_probe.film.bias[6:] == 0.0), "FiLM beta bias != 0.0 at init"
print("FiLM-init fix confirmed live (film.weight=0, gamma bias=0.1, beta bias=0).")

# %% [markdown]
# ## 3. Resolve the DTU dataset path

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
# ## 4. GPU sanity check

# %%
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 5. Dry run: 2 folds first

# %%
OUTPUT_DIR_DRYRUN = "/kaggle/working/results_dtu_loso_sgat_v3_dryrun"

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
# ## 6. Full 18-subject LOSO run (v3)

# %%
OUTPUT_DIR = "/kaggle/working/results_dtu_loso_sgat_v3"

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
# ## 7. Results, side-by-side with v1 and v2

# %%
import json

summary_path = Path(OUTPUT_DIR) / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("v3 master summary (mean +/- std across folds):")
    for s in summary:
        print(f"  {s['cv']:<20s} {s['model'].upper():<8s} "
              f"{s['mean_acc']:.4f} +/- {s['std_acc']:.4f}  "
              f"({s['n_folds']} folds, {s['time_s']:.0f}s)")
else:
    print(f"No v3 summary found at {summary_path} -- check for errors above.")

fold_detail_path = Path(OUTPUT_DIR) / "loso_sgat_w5p0s.json"
if fold_detail_path.exists():
    fold_detail = json.loads(fold_detail_path.read_text(encoding="utf-8"))
    folds = fold_detail["per_fold"]
    print(f"\nPer-fold detail ({len(folds)} folds):")
    for r in folds:
        test_subj = r.get("meta", {}).get("test_subject", r.get("fold_id"))
        print(f"  {test_subj}: acc={r['test_accuracy']:.4f}")

print("\nFor reference:")
print("  v1 (lr=1e-3, epochs=15, default FiLM init):  0.4887 +/- 0.0247")
print("  v2 (lr=1e-2, epochs=60, default FiLM init):  0.4814 +/- 0.0192")
print("  v3 (lr=1e-2, epochs=60, FiLM-init fix):      ... (this run)")

# %% [markdown]
# ## 8. What to look at
#
# - **v3 mean > ~0.55**: the FiLM-init hypothesis is confirmed. Pick a
#   representative fold, train that fold's checkpoint standalone via
#   `aad_xai.train --dataset dtu --model sgat`, and re-point the smoke
#   notebook's interpretability dry run at that real checkpoint for actual
#   interpretability findings (as opposed to the current smoke-checkpoint
#   noise).
#
# - **v3 is still ~0.5**: FiLM-init hypothesis falsified. Next moves then
#   go to the two other candidates from the Oct 18 discussion:
#   (a) verify the DTU envelopes carry the attended vs. unattended signal at
#       all (check `data/dtu_dataset.py` and the raw .mat metadata),
#   (b) run the stgcn/channel_order_check.py-style check against a real
#       S1.mat to verify the adjacency prior isn't scrambled.
#
# - **v3 is 0.52 - 0.54**: FiLM init helped but something else is also
#   leaving performance on the table. Longer window (10s), different seed,
#   or address envelope/adjacency as above.
