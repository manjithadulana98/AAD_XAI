# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 -- mean+variance pooling, full 144-fold population run
#
# First full-population run motivated by strong evidence: the 6-fold seed-
# variance study (3 seeds x 6 folds x 2 pooling configs, partition-hash-
# verified) found mean+variance pooling drops min train loss below the
# ~0.690 floor in 6/6 folds across all 3 seeds each (effect >> noise floor),
# while the accuracy effect was a positive point estimate (+0.045 mean
# paired difference) underpowered at n=6 folds due to small/imbalanced test
# partitions -- this run addresses that by scaling to all 144 folds.
#
# Single variable changed from the documented Phase 2 baseline
# (kaggle_train_stgcn_gcn_only.py): pooling is mean+variance instead of
# mean-only (fc1 input doubles, ~5,192 params instead of ~2,632). Everything
# else matches the original baseline procedure exactly (Adam lr=1e-3, no
# schedule, batch=32, RandomSampler 2000/epoch, 40 epochs) -- deliberately
# NOT NSR's substituted training procedure, which was ruled out as a fix on
# its own and is a separate, later follow-up.
#
# Selection: best VALIDATION-LOSS epoch, using a genuine held-out validation
# split carved from the training trials (test_size=0.2, random_state=
# subject_id) -- matching AADNet/NSR's real early_stop='loss' convention,
# not test-set peeking.
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
# ## 2. GPU sanity check (with fast compatibility probe)

# %%
import torch
import torch.nn as nn

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    try:
        _probe = torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")
        torch.cuda.synchronize()
        print("GPU compatibility probe: OK")
    except RuntimeError as e:
        raise RuntimeError(
            f"GPU compatibility probe FAILED on {torch.cuda.get_device_name(0)}: {e}\n"
            "Known P100/sm_60 incompatibility -- re-push/re-run for a different GPU."
        ) from e

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 3. Configuration -- IDENTICAL to the original Phase 2 baseline except pooling

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
FC_HIDDEN = 8
VAL_FRACTION = 0.2
POOLING = "meanvar"

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

OUT_DIR = Path("/kaggle/working/stgcn_meanvar_all_folds")
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

# %% [markdown]
# ## 5. Model -- mean+variance pooling (the tested variant), fc_hidden=8
#    (baseline, already ruled out as a separate variable)

# %%
from model import GraphConvKW


class STGCNMeanVar(nn.Module):
    def __init__(self, adjacency, n_kernels, fc_hidden, n_channels, dropout=0.3):
        super().__init__()
        self.graph_conv = GraphConvKW(adjacency, n_kernels=n_kernels)
        flat_dim = n_kernels * n_channels * 2  # mean + variance
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, 2)

    def forward(self, x):
        B = x.size(0)
        f = self.graph_conv(x)                      # (B, C, N, T)
        C, N, T = f.shape[1], f.shape[2], f.shape[3]
        f = f.reshape(B, C * N, T)
        mean = f.mean(dim=-1)
        var = f.var(dim=-1, unbiased=False)
        pooled = torch.cat([mean, var], dim=-1)      # (B, 2*C*N)
        h = self.act(self.bn1(self.fc1(pooled)))
        h = self.dropout(h)
        return self.fc2(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model():
    torch.manual_seed(RANDOM_SEED)
    m = STGCNMeanVar(ADJACENCY, n_kernels=N_KERNELS, fc_hidden=FC_HIDDEN, n_channels=64, dropout=0.3).to(DEVICE)
    return m


print(f"STGCNMeanVar parameter count: {build_model().count_parameters()}")

# %% [markdown]
# ## 6. Per-fold training loop -- val-split + honest best-VALIDATION-LOSS
#    epoch selection (matches runner.py::fit's early_stop='loss' convention)

# %%
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from aadnet.dataset import DTUDataset


def make_train_loader(ds, batch_size):
    sampler = RandomSampler(ds, replacement=True, num_samples=min(MAX_TRAIN_WINDOWS_PER_EPOCH, len(ds) * 5))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)


def run_fold(subject_id, fold, tr_split, te_split):
    tr_eeg, tr_aud, tr_label = tr_split
    te_eeg, te_aud, te_label = te_split

    tr_eeg2, va_eeg, tr_aud2, va_aud, tr_label2, va_label = train_test_split(
        tr_eeg, tr_aud, tr_label, test_size=VAL_FRACTION, random_state=subject_id
    )

    train_ds = DTUDataset(aadnet_config, tr_eeg2, tr_aud2, tr_label2)
    valid_ds = DTUDataset(aadnet_config, va_eeg, va_aud, va_label)
    test_ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)

    torch.manual_seed(RANDOM_SEED + subject_id * 100 + fold)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = make_train_loader(train_ds, BATCH_SIZE)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    epoch_rows = []
    for epoch in range(N_EPOCHS):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for eeg, _audio, y in train_loader:
            eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
            opt.zero_grad()
            logits = model(eeg)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * y.size(0)
            tr_n += y.size(0)

        model.eval()
        with torch.no_grad():
            va_loss_sum, va_n = 0.0, 0
            for eeg, _audio, y in valid_loader:
                eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
                logits = model(eeg)
                loss = loss_fn(logits, y)
                va_loss_sum += loss.item() * y.size(0)
                va_n += y.size(0)

            te_correct, te_n = 0, 0
            for eeg, _audio, y in test_loader:
                eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
                logits = model(eeg)
                te_correct += (logits.argmax(1) == y).sum().item()
                te_n += y.size(0)

        epoch_rows.append({
            "epoch": epoch,
            "train_loss": tr_loss_sum / max(tr_n, 1),
            "valid_loss": va_loss_sum / max(va_n, 1),
            "test_acc": te_correct / max(te_n, 1),
        })

    best_idx = min(range(len(epoch_rows)), key=lambda i: epoch_rows[i]["valid_loss"])
    return {
        "subject_id": subject_id, "fold": fold,
        "min_train_loss": min(r["train_loss"] for r in epoch_rows),
        "final_train_loss": epoch_rows[-1]["train_loss"],
        "best_val_epoch": epoch_rows[best_idx]["epoch"],
        "best_val_loss": epoch_rows[best_idx]["valid_loss"],
        "test_at_best_val": epoch_rows[best_idx]["test_acc"],
        "n_train_windows": len(train_ds), "n_valid_windows": len(valid_ds), "n_test_windows": len(test_ds),
    }


# %% [markdown]
# ## 7. Run all 18 subjects x 8 folds = 144 folds

# %%
t_start = time.time()
subject_ids = list(range(len(raw_config["dataset"]["all_sbjs"])))
nFold = raw_config["learning"]["nFold"]
print(f"Subjects: {subject_ids}   Folds/subject: {nFold}   Total fold-trainings: {len(subject_ids) * nFold}")

results = []
for subject_id in subject_ids:
    t_subj_start = time.time()
    crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
    for fold in range(nFold):
        tr_split, te_split = crossSIData[fold]
        row = run_fold(subject_id, fold, tr_split, te_split)
        results.append(row)
        print(f"[{time.time() - t_start:7.0f}s] subject {subject_id} fold {fold}: "
              f"min_train_loss={row['min_train_loss']:.4f}  best_val_epoch={row['best_val_epoch']:3d}  "
              f"test@best_val={row['test_at_best_val']:.3f}")
    print(f"  subject {subject_id} done in {time.time() - t_subj_start:.0f}s")

t_total = time.time() - t_start
print(f"\nTotal wall-clock time: {t_total:.1f}s ({t_total/60:.1f} min) for {len(results)} fold-trainings")

# %% [markdown]
# ## 8. Write output

# %%
import pandas as pd

df = pd.DataFrame(results)
out_path = OUT_DIR / "meanvar_all_folds.csv"
df.to_csv(out_path, index=False)

print(f"Mean test@best_val: {df['test_at_best_val'].mean():.4f}  SD: {df['test_at_best_val'].std(ddof=1):.4f}")
print(f"Mean min_train_loss: {df['min_train_loss'].mean():.4f}  SD: {df['min_train_loss'].std(ddof=1):.4f}")
print(f"Written {len(df)} rows to {out_path}")

with open(OUT_DIR / "diag_config.json", "w") as f:
    json.dump({
        "pooling": POOLING, "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "n_kernels": N_KERNELS, "fc_hidden": FC_HIDDEN, "val_fraction": VAL_FRACTION,
        "seed": RANDOM_SEED, "total_wallclock_seconds": t_total,
        "purpose": "full 144-fold population run of mean+variance pooling, honest val-loss selection",
    }, f, indent=2)
