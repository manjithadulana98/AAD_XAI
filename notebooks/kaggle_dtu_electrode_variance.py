# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DTU electrode variance per subject-fold (reference map for the explanation-transfer analysis; CPU)
#
# The same data, preprocessing, folds and 5 s test windows as the X1 run
# (notebooks/kaggle_run_dtu_xai_round9.py). For every fold and test subject it stores the
# per-electrode variance of the raw (pre-alignment) windows, averaged over windows, i.e. the
# record "variance" that notebooks/kaggle_run_kul_xai_transfer.py stores for KUL. No model is used.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - CPU is enough

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
GIT_COMMIT = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
print(f"Commit: {GIT_COMMIT}")

# %% [markdown]
# ## 2. DTU trials and folds (as in X1), electrode variance of each subject's test windows

# %%
import json
import time
from pathlib import Path

import numpy as np
import torch

from aad_xai.config import PreprocessConfig
from aad_xai.data.cv_splits import subject_group_story_out
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset

DTU_ROOT = next(p for p in ["/kaggle/input/aad-xai-artifacts/datasets/DTU",
                            "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU"]
                if os.path.isdir(p))
WINDOW_S = 5.0
FOLD_SEED = 42
OUT_DIR = Path("/kaggle/working/results_dtu_variance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pp = PreprocessConfig(bandpass_hz=(1.0, 31.0), envelope_bandpass_hz=(1.0, 8.0))
_t0 = time.time()
trials = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp, envelope_method="powerlaw_subbands").trials())
print(f"Trials: {len(trials)} in {time.time() - _t0:.0f}s")
folds = list(subject_group_story_out(trials, seed=FOLD_SEED))

results = []
for fold in folds:
    test_trials = [trials[i] for i in fold.test_idx]
    for subj in fold.meta["test_subjects"]:
        st = [t for t in test_trials if t.subject_id == subj]
        ds = WindowedEEGAudioDataset(st, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=False)
        x = torch.stack([ds[i][0] for i in range(len(ds))])                      # (N, C, T) raw windows
        results.append({"fold_id": fold.fold_id, "subject": subj, "n_windows": len(ds),
                        "variance": x.var(dim=-1, unbiased=False).mean(dim=0).tolist()})   # (C,)
with open(OUT_DIR / "dtu_variance.json", "w") as f:
    json.dump({"commit": GIT_COMMIT, "results": results}, f)
print(f"Done: {len(results)} subject-folds, {len({r['subject'] for r in results})} subjects")
