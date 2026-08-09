# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 diagnostic -- noise floor for "best-of-40" selection
#
# The 0.597 "best epoch by test accuracy" figure selects the single best of
# 40 epochs per fold, evaluated against small (down to a handful of trials),
# often severely class-imbalanced test folds (mean majority-class baseline
# 0.668 across all 144 folds). Best-of-40 selection on a small, imbalanced
# set can manufacture a good-looking number from pure noise even with ZERO
# real model capability -- this notebook measures exactly how much.
#
# Method: for each of the same 144 folds, build a FRESH RANDOMLY-INITIALIZED
# model (no training at all -- zero gradient steps) 40 times, evaluate each
# one's test accuracy, and take the best of the 40. Average that "best of 40
# untrained models" figure across all 144 folds. This is the noise floor:
# how good "best-of-40 on these exact folds" looks with no learning
# whatsoever. No backward pass anywhere in this notebook -- CPU is fine,
# avoids the GPU-compatibility lottery entirely.
#
# No changes to the model, adjacency, or data pipeline -- only which
# (untrained) weights get evaluated.
#
# **Kaggle setup requirements:** Internet enabled, CPU only,
# `dulanamanjitha/aad-xai-artifacts` dataset attached. No Kaggle Secret needed.

# %% [markdown]
# ## 1. Clone repository + install dependencies

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
    with open("requirements.txt") as _f:
        _reqs_no_torch = [ln for ln in _f if ln.strip() and not ln.strip().lower().startswith("torch")]
    with open("/tmp/requirements_no_torch.txt", "w") as _f:
        _f.writelines(_reqs_no_torch)
    subprocess.run(["pip", "install", "-q", "-r", "/tmp/requirements_no_torch.txt"], check=True)
except ImportError:
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

subprocess.run(["pip", "install", "-q", "-e", "."], check=True)

for extra in ("src", "external/AADNet", "stgcn"):
    p = os.path.join(REPO_DIR, extra)
    if p not in sys.path:
        sys.path.insert(0, p)

print("Setup done.")

# %% [markdown]
# ## 2. Device -- CPU by design (no training, no backward pass needed)

# %%
import torch

print(f"PyTorch version : {torch.__version__}")
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE} (forced CPU -- no gradients computed anywhere in this notebook)")

# %% [markdown]
# ## 3. Configuration -- IDENTICAL overrides to kaggle_train_stgcn_gcn_only.py

# %%
import yaml
import json
import time
from pathlib import Path

RANDOM_SEED = 42
N_TRIALS_PER_FOLD = 40  # matches N_EPOCHS in the real run -- same "best of 40" budget
BATCH_SIZE = 32
N_KERNELS = 5

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'."

BASE_CONFIG_PATH = os.path.join(REPO_DIR, "external", "AADNet", "config", "config_AADNet_SI_DTU_kaggle.yml")
with open(BASE_CONFIG_PATH, encoding="utf-8") as f:
    raw_config = yaml.safe_load(f)

raw_config["dataset"]["folder"] = os.path.join(DTU_ROOT, "eeg_new") + "/"
raw_config["dataset"]["stimuli_path"] = os.path.join(DTU_ROOT, "Audio")
raw_config["dataset"]["training_window"] = 1
raw_config["dataset"]["duplicate"] = False

from utils.config import Config
aadnet_config = Config.load_config(raw_config)

OUT_DIR = Path("/kaggle/working/stgcn_diag_noise_floor")
OUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import random
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# %% [markdown]
# ## 4. Fixed adjacency -- IDENTICAL to kaggle_train_stgcn_gcn_only.py

# %%
from adjacency import load_montage, build_adjacency_distance

montage = load_montage(os.path.join(REPO_DIR, "config", "aadnet_dtu_channel_montage.csv"))
ADJACENCY = build_adjacency_distance(montage, k=6)

# %% [markdown]
# ## 5. Model builder -- varies the seed per trial (genuinely independent
#    random re-initializations, not the same weights reused 40 times)

# %%
from model import STGCNGCNOnly


def build_random_model(seed):
    torch.manual_seed(seed)
    m = STGCNGCNOnly(ADJACENCY, n_kernels=N_KERNELS).to(DEVICE)
    m.eval()
    return m


# %% [markdown]
# ## 6. For each fold: best-of-40 UNTRAINED models' test accuracy

# %%
from torch.utils.data import DataLoader
from aadnet.dataset import DTUDataset


def eval_test_acc(model, test_ds):
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    correct, n = 0, 0
    with torch.no_grad():
        for eeg, _audio, y in loader:
            eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
            logits = model(eeg)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return correct / max(n, 1)


t_start = time.time()
subject_ids = list(range(len(raw_config["dataset"]["all_sbjs"])))
nFold = raw_config["learning"]["nFold"]
print(f"Subjects: {subject_ids}   Folds/subject: {nFold}   Total folds: {len(subject_ids) * nFold}")

fold_rows = []
for subject_id in subject_ids:
    t_subj_start = time.time()
    crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
    for fold in range(nFold):
        _tr_split, te_split = crossSIData[fold]
        te_eeg, te_aud, te_label = te_split
        test_ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)

        trial_accs = []
        for trial in range(N_TRIALS_PER_FOLD):
            seed = RANDOM_SEED + subject_id * 10000 + fold * 100 + trial
            model = build_random_model(seed)
            trial_accs.append(eval_test_acc(model, test_ds))

        fold_rows.append({
            "subject_id": subject_id, "fold": fold,
            "mean_untrained_acc": float(np.mean(trial_accs)),
            "best_of_40_untrained_acc": float(np.max(trial_accs)),
            "n_test_windows": len(test_ds),
        })
    print(f"[{time.time() - t_start:6.0f}s] subject {subject_id} done "
          f"({time.time() - t_subj_start:.0f}s)")

t_total = time.time() - t_start
print(f"\nTotal wall-clock: {t_total:.1f}s ({t_total/60:.1f} min) for {len(fold_rows)} folds")

# %% [markdown]
# ## 7. Write output

# %%
import pandas as pd

df = pd.DataFrame(fold_rows)
out_path = OUT_DIR / "noise_floor_all_folds.csv"
df.to_csv(out_path, index=False)

print(f"Mean of per-fold MEAN untrained acc (should be ~majority-class baseline on average): "
      f"{df['mean_untrained_acc'].mean():.4f}")
print(f"Mean of per-fold BEST-OF-{N_TRIALS_PER_FOLD} untrained acc (the noise floor): "
      f"{df['best_of_40_untrained_acc'].mean():.4f}")
print(f"Written {len(df)} rows to {out_path}")

with open(OUT_DIR / "diag_config.json", "w") as f:
    json.dump({
        "n_trials_per_fold": N_TRIALS_PER_FOLD,
        "total_wallclock_seconds": t_total,
        "purpose": "noise floor -- best-of-40 UNTRAINED models' test accuracy, same folds as the real run",
    }, f, indent=2)
