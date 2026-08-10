# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 diagnostic -- does the pooling collapse destroy temporal signal?
#
# `GraphConvKW` never touches the time axis (its matmul only mixes nodes;
# `x` shape (B,N,T), `basis` shape (K,N,N), T passes through per-timestep
# untouched). `AdaptiveAvgPool1d(1)` then collapses the ENTIRE T=64-sample
# (1s @ 64Hz) window to one scalar per (kernel, node), with zero temporal
# processing beforehand. Band-pass-filtered EEG is close to zero-mean over
# a 1s window by construction -- averaging 64 oscillatory samples down to
# one number may throw away nearly all time-varying, attention-relevant
# structure.
#
# EEGNetAAD (the validated 0.546 EEG-only reference) never does this: its
# conv1/conv3 are TEMPORAL convolutions (learned filters along time),
# applied before any pooling, and its own pooling only shrinks time by 10x
# within a short 0.4s context -- never to a single global scalar.
#
# This tests three pooling variants on the SAME 2 sanity-check folds
# (subject 0, subject 12), same training procedure as the ORIGINAL baseline
# (Adam lr=1e-3, batch=32, RandomSampler 2000/epoch, 40 epochs, no
# scheduler -- deliberately NOT the NSR procedure already ruled out, to
# isolate this one variable), same val-split + honest best-epoch selection:
#
#   A. baseline  -- AdaptiveAvgPool1d(1), mean only (current architecture)
#   B. mean+var  -- concatenate mean AND variance per (kernel,node) before fc1
#   C. no-pool   -- skip pooling entirely, flatten the full (C*N,T) tensor into fc1
#
# fc_hidden stays at 8 throughout (already tested in isolation and ruled
# out). Parameter count rises substantially for variant C -- expected and
# fine for this diagnostic, which tests a mechanism, not a parameter budget.
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
# ## 3. Configuration -- ORIGINAL baseline training procedure (not NSR's,
#    already ruled out separately) -- isolates the pooling variable only

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

DIAG_FOLDS = [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0)]

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

OUT_DIR = Path("/kaggle/working/stgcn_diag_pooling_test")
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
# ## 5. Three model variants -- identical GraphConvKW, only the
#    pool/flatten step before fc1 changes

# %%
from model import GraphConvKW


class STGCNVariant(nn.Module):
    """Same as STGCNGCNOnly, but with a swappable pooling strategy.

    pooling='mean'    -- current architecture: AdaptiveAvgPool1d(1), mean only
    pooling='meanvar'  -- concatenate mean AND variance per (kernel, node)
    pooling='none'     -- skip pooling entirely, flatten the full (C*N, T) tensor
    """

    def __init__(self, adjacency, n_kernels, fc_hidden, n_channels, T, pooling, dropout=0.3):
        super().__init__()
        self.graph_conv = GraphConvKW(adjacency, n_kernels=n_kernels)
        self.pooling = pooling
        if pooling == "mean":
            flat_dim = n_kernels * n_channels
        elif pooling == "meanvar":
            flat_dim = n_kernels * n_channels * 2
        elif pooling == "none":
            flat_dim = n_kernels * n_channels * T
        else:
            raise ValueError(pooling)
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, 2)

    def forward(self, x):
        B = x.size(0)
        f = self.graph_conv(x)                        # (B, C, N, T)
        C, N, T = f.shape[1], f.shape[2], f.shape[3]
        f = f.reshape(B, C * N, T)                     # (B, C*N, T)
        if self.pooling == "mean":
            pooled = f.mean(dim=-1)                    # (B, C*N)
        elif self.pooling == "meanvar":
            mean = f.mean(dim=-1)
            var = f.var(dim=-1, unbiased=False)
            pooled = torch.cat([mean, var], dim=-1)     # (B, 2*C*N)
        else:  # "none"
            pooled = f.reshape(B, -1)                   # (B, C*N*T)
        h = self.act(self.bn1(self.fc1(pooled)))
        h = self.dropout(h)
        return self.fc2(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(pooling, T):
    torch.manual_seed(RANDOM_SEED)
    m = STGCNVariant(ADJACENCY, n_kernels=N_KERNELS, fc_hidden=FC_HIDDEN,
                      n_channels=64, T=T, pooling=pooling, dropout=0.3).to(DEVICE)
    return m


# %% [markdown]
# ## 6. Fold-training loop -- IDENTICAL to the original baseline procedure
#    (Adam, lr=1e-3, no schedule, RandomSampler(2000/epoch), batch=32),
#    only the model's pooling strategy varies. Val-split + honest
#    best-epoch selection (same as Part A), plus per-epoch train-loss.

# %%
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from aadnet.dataset import DTUDataset


def make_train_loader(ds, batch_size):
    sampler = RandomSampler(ds, replacement=True, num_samples=min(MAX_TRAIN_WINDOWS_PER_EPOCH, len(ds) * 5))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)


def run_fold(subject_id, fold, tr_split, te_split, pooling):
    tr_eeg, tr_aud, tr_label = tr_split
    te_eeg, te_aud, te_label = te_split

    tr_eeg2, va_eeg, tr_aud2, va_aud, tr_label2, va_label = train_test_split(
        tr_eeg, tr_aud, tr_label, test_size=VAL_FRACTION, random_state=subject_id
    )

    train_ds = DTUDataset(aadnet_config, tr_eeg2, tr_aud2, tr_label2)
    valid_ds = DTUDataset(aadnet_config, va_eeg, va_aud, va_label)
    test_ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)

    # Determine T from one real sample rather than assuming 64
    sample_eeg, _, _ = train_ds[0]
    T = sample_eeg.shape[-1]

    torch.manual_seed(RANDOM_SEED + subject_id * 100 + fold)
    model = build_model(pooling, T)
    param_count = model.count_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = make_train_loader(train_ds, BATCH_SIZE)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    rows = []
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
            va_correct, va_n = 0, 0
            for eeg, _audio, y in valid_loader:
                eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
                logits = model(eeg)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_n += y.size(0)

            te_correct, te_n = 0, 0
            for eeg, _audio, y in test_loader:
                eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
                logits = model(eeg)
                te_correct += (logits.argmax(1) == y).sum().item()
                te_n += y.size(0)

        rows.append({
            "subject_id": subject_id, "fold": fold, "pooling": pooling, "epoch": epoch,
            "train_loss": tr_loss_sum / max(tr_n, 1),
            "valid_acc": va_correct / max(va_n, 1),
            "test_acc": te_correct / max(te_n, 1),
        })

    return rows, param_count, T


# %% [markdown]
# ## 7. Run all 3 pooling variants on both folds

# %%
t_start = time.time()
all_rows = []
summary_rows = []
for pooling in ("meanvar",):  # baseline "mean" and "none" already on record from the 2-fold run; extending "meanvar" to all 6 folds now
    for subject_id, fold in DIAG_FOLDS:
        t_fold_start = time.time()
        crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
        tr_split, te_split = crossSIData[fold]
        fold_rows, param_count, T = run_fold(subject_id, fold, tr_split, te_split, pooling)
        all_rows.extend(fold_rows)

        best_va_idx = max(range(len(fold_rows)), key=lambda i: fold_rows[i]["valid_acc"])
        test_at_best_va = fold_rows[best_va_idx]["test_acc"]
        min_loss = min(r["train_loss"] for r in fold_rows)
        final_loss = fold_rows[-1]["train_loss"]
        summary_rows.append({
            "pooling": pooling, "subject_id": subject_id, "fold": fold,
            "param_count": param_count, "T": T,
            "min_train_loss": min_loss, "final_train_loss": final_loss,
            "best_val_epoch": int(fold_rows[best_va_idx]["epoch"]),
            "test_at_best_val_epoch": test_at_best_va,
        })
        print(f"[{time.time() - t_start:6.0f}s] pooling={pooling:8s} subject {subject_id} fold {fold} "
              f"({time.time() - t_fold_start:.0f}s, params={param_count}): "
              f"min_train_loss={min_loss:.4f}  test@best_val_epoch={test_at_best_va:.3f}")

t_total = time.time() - t_start
print(f"\nTotal wall-clock: {t_total:.1f}s ({t_total/60:.1f} min)")

# %% [markdown]
# ## 8. Write output

# %%
import pandas as pd

df = pd.DataFrame(all_rows)
df.to_csv(OUT_DIR / "pooling_test_per_epoch.csv", index=False)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "pooling_test_summary.csv", index=False)
print(summary.to_string(index=False))
