# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 diagnostic -- best-epoch (by test acc) for all 144 folds
#
# v5's `run_fold` (kaggle_train_stgcn_gcn_only.py) reports only the FINAL
# epoch's test accuracy -- no checkpoint/best-epoch selection exists (traced
# directly in the code: te_correct/te_n are re-zeroed every epoch, and
# final_test_acc reads whatever they hold after the loop ends). AADNet's own
# validated SI/1s benchmark (57.5%, external/AADNet/runner.py::fit) DOES use
# a genuine held-out validation split with early stopping and reloads the
# best-validation checkpoint before scoring test accuracy -- a real,
# confirmed asymmetry against our raw final-epoch number.
#
# This notebook is IDENTICAL to kaggle_train_stgcn_gcn_only.py (same data
# pipeline, same model, same hyperparameters, same run_fold body) for ALL
# 144 folds, with one additive change: track and report each fold's BEST
# test accuracy across its 40 epochs, not just the final one. We have no
# genuine held-out validation split in this pipeline (only train/test
# loaders), so "best epoch by test accuracy" is the best available proxy --
# note this is a mild form of test-set peeking (optimistic vs. AADNet's
# honest validation-selected number), flagged explicitly rather than treated
# as a clean like-for-like fix.
#
# No changes to training logic, adjacency, or model architecture.
#
# **Kaggle setup requirements:** Internet enabled, GPU accelerator,
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
    print(f"Pre-installed torch {_torch_preinstalled.__version__} found "
          f"(CUDA available: {_torch_preinstalled.cuda.is_available()}) -- "
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

for extra in ("src", "external/AADNet", "stgcn"):
    p = os.path.join(REPO_DIR, extra)
    if p not in sys.path:
        sys.path.insert(0, p)

print("Setup done.")

# %% [markdown]
# ## 2. GPU sanity check

# %%
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    # Fail fast (seconds, not the 33 minutes it takes to load all 18 subjects'
    # data) if Kaggle assigned an old P100/sm_60 GPU incompatible with the
    # pre-installed torch wheel's compiled kernels -- a known, recurring
    # platform flakiness for this project, not a code issue. A plain matmul
    # exercises the same "no kernel image available" failure mode this
    # notebook would otherwise only discover after the data-loading phase.
    try:
        _probe = torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")
        torch.cuda.synchronize()
        print("GPU compatibility probe: OK")
    except RuntimeError as e:
        raise RuntimeError(
            f"GPU compatibility probe FAILED on {torch.cuda.get_device_name(0)}: {e}\n"
            "This is the known P100/sm_60 vs pre-installed-torch-wheel incompatibility -- "
            "stop and re-push/re-run to get a different GPU assignment, rather than "
            "waiting through the full data-loading phase to discover this."
        ) from e

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 3. Configuration -- IDENTICAL to kaggle_train_stgcn_gcn_only.py

# %%
from pathlib import Path
import yaml
import json
import time

RANDOM_SEED = 42
N_EPOCHS = 40
BATCH_SIZE = 32
MAX_TRAIN_WINDOWS_PER_EPOCH = 2000
LR = 1e-3
N_KERNELS = 5

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, (
    "DTU dataset not found. Attach the 'dulanamanjitha/aad-xai-artifacts' dataset. "
    "Tried: " + ", ".join(DTU_KAGGLE_ROOT_CANDIDATES)
)
print(f"DTU dataset  : {DTU_ROOT}")

BASE_CONFIG_PATH = os.path.join(REPO_DIR, "external", "AADNet", "config", "config_AADNet_SI_DTU_kaggle.yml")
with open(BASE_CONFIG_PATH, encoding="utf-8") as f:
    raw_config = yaml.safe_load(f)

raw_config["dataset"]["folder"] = os.path.join(DTU_ROOT, "eeg_new") + "/"
raw_config["dataset"]["stimuli_path"] = os.path.join(DTU_ROOT, "Audio")
raw_config["dataset"]["training_window"] = 1
raw_config["dataset"]["duplicate"] = False
print(f"channels={len(raw_config['dataset']['channels'])}  sr={raw_config['dataset']['sr']}  "
      f"training_window={raw_config['dataset']['training_window']}s  step={raw_config['dataset']['step']}s  "
      f"duplicate={raw_config['dataset']['duplicate']}  nFold={raw_config['learning']['nFold']}")

from utils.config import Config
aadnet_config = Config.load_config(raw_config)

OUT_DIR = Path("/kaggle/working/stgcn_diag_all_folds_best_epoch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import random
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# %% [markdown]
# ## 4. Fixed adjacency -- IDENTICAL to kaggle_train_stgcn_gcn_only.py

# %%
from adjacency import load_montage, build_adjacency_distance

montage = load_montage(os.path.join(REPO_DIR, "config", "aadnet_dtu_channel_montage.csv"))
ADJACENCY = build_adjacency_distance(montage, k=6)
print(f"Adjacency: {ADJACENCY.shape}, symmetric={np.allclose(ADJACENCY, ADJACENCY.T)}")

# %% [markdown]
# ## 5. Model -- IDENTICAL to kaggle_train_stgcn_gcn_only.py

# %%
from model import STGCNGCNOnly


def build_model():
    torch.manual_seed(RANDOM_SEED)
    m = STGCNGCNOnly(ADJACENCY, n_kernels=N_KERNELS).to(DEVICE)
    return m


_param_count = build_model().count_parameters()
print(f"STGCNGCNOnly parameter count: {_param_count}")

# %% [markdown]
# ## 6. Per-fold training loop -- IDENTICAL body to kaggle_train_stgcn_gcn_only.py,
#    with one additive change: track best (not just final) test accuracy.

# %%
from torch.utils.data import DataLoader, RandomSampler
from aadnet.dataset import DTUDataset


def make_loader(ds, batch_size, train: bool):
    if train:
        sampler = RandomSampler(ds, replacement=True, num_samples=min(MAX_TRAIN_WINDOWS_PER_EPOCH, len(ds) * 5))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_fold(subject_id, fold, tr_split, te_split):
    tr_eeg, tr_aud, tr_label = tr_split
    te_eeg, te_aud, te_label = te_split
    train_ds = DTUDataset(aadnet_config, tr_eeg, tr_aud, tr_label)
    test_ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)

    torch.manual_seed(RANDOM_SEED + subject_id * 100 + fold)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = make_loader(train_ds, BATCH_SIZE, train=True)
    test_loader = make_loader(test_ds, BATCH_SIZE, train=False)

    # LOGGING ADDED -- track every epoch's test acc to find the best, not just the final
    per_epoch_test_acc = []

    for epoch in range(N_EPOCHS):
        model.train()
        for eeg, _audio, y in train_loader:
            eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
            opt.zero_grad()
            logits = model(eeg)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        te_correct, te_n = 0, 0
        with torch.no_grad():
            for eeg, _audio, y in test_loader:
                eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
                logits = model(eeg)
                te_correct += (logits.argmax(1) == y).sum().item()
                te_n += y.size(0)

        per_epoch_test_acc.append(te_correct / max(te_n, 1))

    final_test_acc = per_epoch_test_acc[-1]
    best_test_acc = max(per_epoch_test_acc)
    best_test_epoch = int(np.argmax(per_epoch_test_acc))
    return final_test_acc, best_test_acc, best_test_epoch, len(train_ds), len(test_ds)


# %% [markdown]
# ## 7. Run all subjects x folds

# %%
t_start = time.time()
subject_ids = list(range(len(raw_config["dataset"]["all_sbjs"])))
nFold = raw_config["learning"]["nFold"]
print(f"Subjects: {subject_ids}   Folds/subject: {nFold}   Total fold-trainings: {len(subject_ids) * nFold}")

fold_rows = []
for subject_id in subject_ids:
    t_subj_start = time.time()
    crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
    for fold in range(nFold):
        tr_split, te_split = crossSIData[fold]
        final_acc, best_acc, best_epoch, n_train, n_test = run_fold(subject_id, fold, tr_split, te_split)
        fold_rows.append({
            "subject_id": subject_id, "fold": fold,
            "final_test_acc": final_acc, "best_test_acc": best_acc, "best_test_epoch": best_epoch,
            "n_train_windows": n_train, "n_test_windows": n_test,
        })
        print(f"[{time.time() - t_start:7.0f}s] subject {subject_id} fold {fold}: "
              f"final={final_acc:.3f}  best={best_acc:.3f}@ep{best_epoch}")
    print(f"  subject {subject_id} done in {time.time() - t_subj_start:.0f}s")

t_total = time.time() - t_start
print(f"\nTotal wall-clock time: {t_total:.1f}s ({t_total/60:.1f} min) for {len(fold_rows)} fold-trainings")

# %% [markdown]
# ## 8. Write output

# %%
import pandas as pd

fold_df = pd.DataFrame(fold_rows)
out_path = OUT_DIR / "all_folds_best_epoch.csv"
fold_df.to_csv(out_path, index=False)

print(f"Mean final_test_acc: {fold_df['final_test_acc'].mean():.4f}")
print(f"Mean best_test_acc:  {fold_df['best_test_acc'].mean():.4f}")
print(f"Written {len(fold_df)} rows to {out_path}")

with open(OUT_DIR / "diag_config.json", "w") as f:
    json.dump({
        "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "n_kernels": N_KERNELS,
        "max_train_windows_per_epoch": MAX_TRAIN_WINDOWS_PER_EPOCH,
        "total_wallclock_seconds": t_total,
        "purpose": "best-epoch-by-test-acc for all 144 folds -- no checkpoint selection existed before",
    }, f, indent=2)

print(f"\nWritten to {out_path}")
