# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT — DTU Full LOSO Evaluation (Kaggle)
#
# A proper subject-independent evaluation of `StimulusGAT`: true
# leave-one-subject-out over all 18 DTU subjects, using
# `aad_xai.run_experiments --dataset dtu --cv loso --model sgat` -- the
# generic CV/model harness (previously KUL-only), now generalized to accept
# DTU as a data source (dataset-agnostic CV strategies only: `loso` and
# `within_subject_5fold` -- the others parse KUL-specific story/condition
# metadata that DTU's `group_id` doesn't have; the CLI raises a clear error
# if you pick an incompatible one).
#
# Unlike `kaggle_train_sgat_dtu_smoke.py` (a handful of subjects, a few
# epochs, to validate plumbing), this trains **one independent StimulusGAT
# per held-out subject** (18 folds total) for the harness's real defaults
# (15 epochs, early-stopping patience 5) -- a genuine accuracy estimate,
# not a smoke test.
#
# **Kaggle setup requirements**
# - Enable Internet in notebook settings (for git clone + pip install)
# - Attach dataset `dulanamanjitha/aad-xai-artifacts` (holds DTU EEG + Audio)
# - GPU strongly recommended -- 18 independent trainings, not 1
#
# **Known open item** (see the project plan): the default electrode
# adjacency `StimulusGAT` builds (`config/dtu_channel_montage.csv`,
# distance-based k=6) has not yet been verified against DTU's true native
# channel order. Treat channel-level interpretability from this run's
# checkpoints as provisional until that's verified -- it does not affect
# whether training/accuracy here is valid, only channel-identity claims.

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
    # A kernel restart (as opposed to a fresh session) can leave REPO_DIR
    # sitting on disk from a previous run -- pull instead of silently
    # training against a stale checkout.
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

# Belt-and-suspenders: `pip install -e .` should put `aad_xai` on sys.path,
# but a running kernel doesn't always pick up a package installed mid-session.
_src_dir = os.path.join(REPO_DIR, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import aad_xai  # noqa: F401 -- fail fast here, not several cells later
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
    "DTU dataset not found. Attach the 'dulanamanjitha/aad-xai-artifacts' dataset "
    "to this notebook. Tried: " + ", ".join(DTU_KAGGLE_ROOT_CANDIDATES)
)
print(f"DTU dataset: {DTU_ROOT}")

_all_subject_ids = sorted(p.stem for p in (Path(DTU_ROOT) / "eeg_new").glob("S*.mat"))
print(f"Subjects found ({len(_all_subject_ids)}):", _all_subject_ids)

# %% [markdown]
# ## 3. GPU sanity check

# %%
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 4. Dry run: time 2 folds before committing to all 18
#
# `StimulusGAT` trains from scratch per fold -- unlike the TRF LOSO runner
# (a ridge regression, seconds per fold), 18 independent deep-model
# trainings could plausibly take anywhere from minutes to hours depending
# on the assigned GPU. Time 2 folds first rather than guessing.

# %%
OUTPUT_DIR_DRYRUN = "/kaggle/working/results_dtu_loso_sgat_dryrun"

cmd_dryrun = [
    sys.executable, "-m", "aad_xai.run_experiments",
    "--dataset", "dtu",
    "--data-dir", DTU_ROOT,
    "--cv", "loso",
    "--model", "sgat",
    "--window", "5",
    "--seed", "42",
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
# ## 5. Full 18-subject LOSO run
#
# No `--max-folds` -- every subject held out in turn. Same window (5s),
# seed, and harness defaults (15 epochs, patience 5) as the dry run above.

# %%
OUTPUT_DIR = "/kaggle/working/results_dtu_loso_sgat_full"

cmd = [
    sys.executable, "-m", "aad_xai.run_experiments",
    "--dataset", "dtu",
    "--data-dir", DTU_ROOT,
    "--cv", "loso",
    "--model", "sgat",
    "--window", "5",
    "--seed", "42",
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
# ## 6. Display results

# %%
import json

summary_path = Path(OUTPUT_DIR) / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("Master summary (mean +/- std across folds):")
    for s in summary:
        print(f"  {s['cv']:<20s} {s['model'].upper():<8s} "
              f"{s['mean_acc']:.4f} +/- {s['std_acc']:.4f}  "
              f"({s['n_folds']} folds, {s['time_s']:.0f}s)")
else:
    print(f"No summary found at {summary_path} -- check for errors above.")

fold_detail_path = Path(OUTPUT_DIR) / "loso_sgat_w5p0s.json"
if fold_detail_path.exists():
    fold_detail = json.loads(fold_detail_path.read_text(encoding="utf-8"))
    folds = fold_detail["per_fold"]
    print(f"\nPer-fold detail ({len(folds)} folds):")
    for r in folds:
        test_subj = r.get("meta", {}).get("test_subject", r.get("fold_id"))
        print(f"  {test_subj}: acc={r['test_accuracy']:.4f}")

# %% [markdown]
# ## 7. Next steps
#
# - This is a real subject-independent accuracy estimate (18 held-out-
#   subject folds), not a smoke test -- but still a single seed/window.
#   Re-run with a different `--seed` and/or `--window` (1, 2, or 10s) to
#   check how sensitive the result is before treating any one number as
#   definitive.
# - Per-fold checkpoints are NOT saved by this harness (unlike
#   `aad_xai.train`) -- it reports accuracy per fold, not model weights.
#   If you need a trained checkpoint for interpretability (attention
#   weights, channel importance), train that specific fold separately via
#   `aad_xai.train --dataset dtu --model sgat`.
# - Resolve the channel-order-vs-montage open item (see cell 1's note)
#   before treating any channel-level interpretability as trustworthy.
