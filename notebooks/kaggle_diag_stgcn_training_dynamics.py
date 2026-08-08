# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 diagnostic -- training-dynamics instrumentation
#
# v5's representative fold (subject 0, fold 0) shows test accuracy peaking
# ~0.75 around epoch 2-3, then decaying to 0.25-0.40 by epoch 40, while train
# accuracy climbs only to ~50-54% and plateaus. `run_fold` reports the FINAL
# epoch's test accuracy with no checkpoint/best-epoch selection (confirmed by
# reading the code directly -- no `best`/`patience`/`checkpoint` logic exists
# anywhere in the training notebook), so the reported 0.404 population mean
# may be reading off an already-degraded tail state rather than reflecting
# the model's actual best capability.
#
# This notebook is PURE ADDITIVE LOGGING on top of the existing training
# loop -- same data pipeline, same model, same hyperparameters, same
# `run_fold` body -- for a handful of folds spanning different subjects, to
# distinguish candidate causes of the decay:
# - train LOSS trajectory (not just accuracy) -- flat accuracy with
#   improving loss vs. worsening loss are very different stories
# - L2 norm of `theta` (the learned per-kernel mixing coefficients) per epoch
# - BatchNorm running_mean/running_var norms per epoch
# - test-set class balance (sanity: is chance actually ~50%?)
#
# Nothing about `run_fold`'s training logic, the adjacency, or the model
# architecture is changed -- this only adds logging calls around the
# identical computation.
#
# **Kaggle setup requirements:** Internet enabled, GPU accelerator (optional),
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

# 6 folds spanning different subjects (fold 0 of each) -- not just subject 0
DIAG_FOLDS = [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0)]

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

OUT_DIR = Path("/kaggle/working/stgcn_diag_training_dynamics")
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
# ## 6. Instrumented fold-training loop
#
# Same body as `run_fold` in kaggle_train_stgcn_gcn_only.py (data loaders,
# optimizer, train/eval loop) -- no changes to the training logic itself.
# Additions are marked `# LOGGING ADDED` and never affect what the model
# sees or how gradients are computed.

# %%
from torch.utils.data import DataLoader, RandomSampler
from aadnet.dataset import DTUDataset


def make_loader(ds, batch_size, train: bool):
    if train:
        sampler = RandomSampler(ds, replacement=True, num_samples=min(MAX_TRAIN_WINDOWS_PER_EPOCH, len(ds) * 5))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_fold_instrumented(subject_id, fold, tr_split, te_split):
    tr_eeg, tr_aud, tr_label = tr_split
    te_eeg, te_aud, te_label = te_split
    train_ds = DTUDataset(aadnet_config, tr_eeg, tr_aud, tr_label)
    test_ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)

    # LOGGING ADDED -- test-set class balance at WINDOW level (te_label is
    # per-trial; DTUDataset expands each trial to many windows sharing that
    # trial's label, so counting te_label directly would give the wrong
    # denominator -- iterate the actual windowed dataset instead).
    n_class0, n_class1 = 0, 0
    for i in range(len(test_ds)):
        _, _, y_i = test_ds[i]
        if int(y_i.item()) == 0:
            n_class0 += 1
        else:
            n_class1 += 1

    torch.manual_seed(RANDOM_SEED + subject_id * 100 + fold)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = make_loader(train_ds, BATCH_SIZE, train=True)
    test_loader = make_loader(test_ds, BATCH_SIZE, train=False)

    rows = []
    for epoch in range(N_EPOCHS):
        model.train()
        tr_loss_sum, tr_correct, tr_n = 0.0, 0, 0
        for eeg, _audio, y in train_loader:
            eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
            opt.zero_grad()
            logits = model(eeg)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * y.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_n += y.size(0)

        model.eval()
        te_loss_sum, te_correct, te_n = 0.0, 0, 0
        with torch.no_grad():
            for eeg, _audio, y in test_loader:
                eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
                logits = model(eeg)
                loss = loss_fn(logits, y)
                te_loss_sum += loss.item() * y.size(0)
                te_correct += (logits.argmax(1) == y).sum().item()
                te_n += y.size(0)

        # LOGGING ADDED -- theta norm and BN running-stat norms, read-only, after the epoch's updates
        theta_norm = model.graph_conv.theta.detach().norm().item()
        bn_mean_norm = model.bn1.running_mean.detach().norm().item()
        bn_var_norm = model.bn1.running_var.detach().norm().item()

        rows.append({
            "subject_id": subject_id, "fold": fold, "epoch": epoch,
            "train_loss": tr_loss_sum / max(tr_n, 1),
            "train_acc": tr_correct / max(tr_n, 1),
            "test_loss": te_loss_sum / max(te_n, 1),
            "test_acc": te_correct / max(te_n, 1),
            "theta_norm": theta_norm,
            "bn1_running_mean_norm": bn_mean_norm,
            "bn1_running_var_norm": bn_var_norm,
            "test_n_class0": n_class0,
            "test_n_class1": n_class1,
        })

    return rows


# %% [markdown]
# ## 7. Run the instrumented folds

# %%
t_start = time.time()
all_rows = []
for subject_id, fold in DIAG_FOLDS:
    t_fold_start = time.time()
    crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
    tr_split, te_split = crossSIData[fold]
    fold_rows = run_fold_instrumented(subject_id, fold, tr_split, te_split)
    all_rows.extend(fold_rows)
    final = fold_rows[-1]
    peak_test = max(r["test_acc"] for r in fold_rows)
    peak_epoch = max(range(len(fold_rows)), key=lambda i: fold_rows[i]["test_acc"])
    print(f"[{time.time() - t_start:6.0f}s] subject {subject_id} fold {fold} "
          f"({time.time() - t_fold_start:.0f}s): final test_acc={final['test_acc']:.3f}  "
          f"peak test_acc={peak_test:.3f} @ epoch {peak_epoch}  "
          f"class balance={final['test_n_class0']}/{final['test_n_class1']}")

t_total = time.time() - t_start
print(f"\nTotal wall-clock time: {t_total:.1f}s ({t_total/60:.1f} min) for {len(DIAG_FOLDS)} folds")

# %% [markdown]
# ## 8. Write output

# %%
import pandas as pd

df = pd.DataFrame(all_rows)
out_path = OUT_DIR / "training_dynamics_per_epoch.csv"
df.to_csv(out_path, index=False)
print(f"Written {len(df)} rows ({len(DIAG_FOLDS)} folds x {N_EPOCHS} epochs) to {out_path}")
print(df.head(10).to_string(index=False))

with open(OUT_DIR / "diag_config.json", "w") as f:
    json.dump({
        "diag_folds": DIAG_FOLDS,
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "n_kernels": N_KERNELS,
        "max_train_windows_per_epoch": MAX_TRAIN_WINDOWS_PER_EPOCH,
        "total_wallclock_seconds": t_total,
        "purpose": "training-dynamics instrumentation -- peak-vs-final test acc, theta norm, BN stats",
    }, f, indent=2)

print(f"\nWritten to {out_path}")
