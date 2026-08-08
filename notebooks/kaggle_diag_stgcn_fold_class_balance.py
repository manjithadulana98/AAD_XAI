# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 diagnostic -- per-fold test-set class balance (all 144 folds)
#
# The 6-fold training-dynamics instrumentation found test-set class balance
# far from 50/50 in 5 of 6 sampled folds (0.625-0.875 majority-class
# baseline). This notebook gets the SAME count for all 18 subjects x 8 SI
# folds = 144 folds, WITHOUT training any model -- `createSICrossValidation`
# and window-level label counting are pure data-loading operations, no GPU
# needed. Combined with the already-saved v5 `gcn_only_fold_accuracy.csv`,
# this lets us recompute the population-level accuracy number relative to
# each fold's own chance level instead of a flat 0.5.
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
# ## 2. Configuration -- IDENTICAL overrides to kaggle_train_stgcn_gcn_only.py
#    (duplicate=False, training_window=1) -- same splits, same windows.

# %%
import yaml
import torch

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

# %% [markdown]
# ## 3. Count test-set window-level class balance for all 18 subjects x 8 folds
#    -- no model, no training, pure data loading.

# %%
from aadnet.dataset import DTUDataset
import time

t_start = time.time()
subject_ids = list(range(len(raw_config["dataset"]["all_sbjs"])))
nFold = raw_config["learning"]["nFold"]

rows = []
for subject_id in subject_ids:
    crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
    for fold in range(nFold):
        _tr_split, te_split = crossSIData[fold]
        te_eeg, te_aud, te_label = te_split
        test_ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)
        n_class0, n_class1 = 0, 0
        for i in range(len(test_ds)):
            _, _, y_i = test_ds[i]
            if int(y_i.item()) == 0:
                n_class0 += 1
            else:
                n_class1 += 1
        rows.append({
            "subject_id": subject_id, "fold": fold,
            "test_n_class0": n_class0, "test_n_class1": n_class1,
            "test_n_total": n_class0 + n_class1,
            "majority_frac": max(n_class0, n_class1) / (n_class0 + n_class1),
        })
    print(f"[{time.time()-t_start:6.0f}s] subject {subject_id} done "
          f"({nFold} folds)")

print(f"\nTotal wall-clock: {time.time()-t_start:.1f}s for {len(rows)} folds")

# %% [markdown]
# ## 4. Write output

# %%
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/kaggle/working/stgcn_diag_fold_class_balance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(rows)
out_path = OUT_DIR / "fold_class_balance.csv"
df.to_csv(out_path, index=False)
print(f"Written {len(df)} rows to {out_path}")
print(df.to_string(index=False))
print(f"\nFolds with majority_frac > 0.6: {(df['majority_frac'] > 0.6).sum()}/{len(df)}")
print(f"Mean majority_frac across all 144 folds: {df['majority_frac'].mean():.3f}")
