# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 -- final measurement (all 144 folds x 2 seeds, meanvar)
#
# Measurement job only. No architecture, hyperparameter, or validation-split
# logic changes versus the "meanvar" config already validated in the 6-fold
# seed-variance study (kaggle_diag_stgcn_seed_variance.py). That study found:
# meanvar's train-loss improvement reproduces in 6/6 folds across 3 seeds
# with no reproducible regression on honest (best-validation-LOSS-selected)
# test accuracy, and a positive point-estimate population effect (+0.045
# mean paired difference vs. the "mean"-only baseline, 95% CI crossing zero
# at n=6 folds). This run scales that config to the full population.
#
# Primary comparison target: **0.546** (NSR/EEGNetAAD, EEG-only, honest,
# identical folds/label -- established earlier in this investigation as the
# fair benchmark). 0.575 (AADNet-full) is a secondary reference only --
# audio-informed, ~2.9 points above the EEG-only floor, not the primary bar.
#
# Selection: best VALIDATION-LOSS epoch (matches runner.py::fit's
# early_stop='loss' convention, and the corrected convention used in the
# seed-variance study -- NOT best-validation-accuracy, which earlier
# diagnostics in this investigation mistakenly used).
#
# Partition-hash assertion carried over from the seed-variance study:
# train/val/test partition is recomputed fresh for every (fold, seed) run
# and hashed; fails loudly on any mismatch across seeds for the same fold.
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
# ## 3. Configuration -- UNCHANGED from the validated seed-variance study

# %%
from pathlib import Path
import yaml
import json
import time
import hashlib

N_EPOCHS = 40
BATCH_SIZE = 32
MAX_TRAIN_WINDOWS_PER_EPOCH = 2000
LR = 1e-3
N_KERNELS = 5
FC_HIDDEN = 8
VAL_FRACTION = 0.2

SEEDS = [42, 43]
POOLING = "meanvar"  # winner config -- see decision rationale in the accompanying report

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

OUT_DIR = Path("/kaggle/working/stgcn_phase2_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np

# %% [markdown]
# ## 4. Fixed adjacency -- IDENTICAL to every prior ST-GCN run

# %%
from adjacency import load_montage, build_adjacency_distance

montage = load_montage(os.path.join(REPO_DIR, "config", "aadnet_dtu_channel_montage.csv"))
ADJACENCY = build_adjacency_distance(montage, k=6)

# %% [markdown]
# ## 5. Model -- IDENTICAL STGCNVariant("meanvar") from the seed-variance study

# %%
from model import GraphConvKW


class STGCNVariant(nn.Module):
    def __init__(self, adjacency, n_kernels, fc_hidden, n_channels, T, pooling, dropout=0.3):
        super().__init__()
        self.graph_conv = GraphConvKW(adjacency, n_kernels=n_kernels)
        self.pooling = pooling
        if pooling == "mean":
            flat_dim = n_kernels * n_channels
        elif pooling == "meanvar":
            flat_dim = n_kernels * n_channels * 2
        else:
            raise ValueError(pooling)
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, 2)

    def forward(self, x):
        B = x.size(0)
        f = self.graph_conv(x)
        C, N, T = f.shape[1], f.shape[2], f.shape[3]
        f = f.reshape(B, C * N, T)
        if self.pooling == "mean":
            pooled = f.mean(dim=-1)
        else:
            mean = f.mean(dim=-1)
            var = f.var(dim=-1, unbiased=False)
            pooled = torch.cat([mean, var], dim=-1)
        h = self.act(self.bn1(self.fc1(pooled)))
        h = self.dropout(h)
        return self.fc2(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(pooling, T, seed):
    torch.manual_seed(seed)
    m = STGCNVariant(ADJACENCY, n_kernels=N_KERNELS, fc_hidden=FC_HIDDEN,
                      n_channels=64, T=T, pooling=pooling, dropout=0.3).to(DEVICE)
    return m


# %% [markdown]
# ## 6. Partition construction + hash assertion (carried over from Part A)

# %%
from sklearn.model_selection import train_test_split
from aadnet.dataset import DTUDataset


def hash_partition(tr_label, va_label, te_label):
    payload = repr((
        [int(x) for x in tr_label], [int(x) for x in va_label], [int(x) for x in te_label]
    )).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_partition(subject_id, fold):
    crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
    tr_split, te_split = crossSIData[fold]
    tr_eeg, tr_aud, tr_label = tr_split
    te_eeg, te_aud, te_label = te_split
    tr_eeg2, va_eeg, tr_aud2, va_aud, tr_label2, va_label = train_test_split(
        tr_eeg, tr_aud, tr_label, test_size=VAL_FRACTION, random_state=subject_id
    )
    h = hash_partition(tr_label2, va_label, te_label)
    return (tr_eeg2, tr_aud2, tr_label2), (va_eeg, va_aud, va_label), (te_eeg, te_aud, te_label), h


# %% [markdown]
# ## 7. Fold-training loop -- IDENTICAL to the seed-variance study's run_one

# %%
def make_train_loader(ds, batch_size):
    sampler = torch.utils.data.RandomSampler(ds, replacement=True,
                                              num_samples=min(MAX_TRAIN_WINDOWS_PER_EPOCH, len(ds) * 5))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)


def run_one(subject_id, fold, seed, tr_parts, va_parts, te_parts, ref_hash):
    tr_eeg2, tr_aud2, tr_label2 = tr_parts
    va_eeg, va_aud, va_label = va_parts
    te_eeg, te_aud, te_label = te_parts

    this_hash = hash_partition(tr_label2, va_label, te_label)
    assert this_hash == ref_hash, (
        f"PARTITION DRIFT DETECTED for subject {subject_id} fold {fold} seed={seed}: "
        f"hash {this_hash} != reference {ref_hash}"
    )

    train_ds = DTUDataset(aadnet_config, tr_eeg2, tr_aud2, tr_label2)
    valid_ds = DTUDataset(aadnet_config, va_eeg, va_aud, va_label)
    test_ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)
    T = train_ds[0][0].shape[-1]

    model = build_model(POOLING, T, seed)
    param_count = model.count_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    torch.manual_seed(seed)
    train_loader = make_train_loader(train_ds, BATCH_SIZE)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
        "subject_id": subject_id, "fold": fold, "seed": seed,
        "param_count": param_count,
        "min_train_loss": min(r["train_loss"] for r in epoch_rows),
        "final_train_loss": epoch_rows[-1]["train_loss"],
        "best_val_epoch": epoch_rows[best_idx]["epoch"],
        "best_val_loss": epoch_rows[best_idx]["valid_loss"],
        "test_at_best_val": epoch_rows[best_idx]["test_acc"],
        "partition_hash": this_hash,
    }


# %% [markdown]
# ## 8. Run all 144 folds x 2 seeds

# %%
t_start = time.time()
subject_ids = list(range(len(raw_config["dataset"]["all_sbjs"])))
nFold = raw_config["learning"]["nFold"]
print(f"Subjects: {subject_ids}   Folds/subject: {nFold}   Seeds: {SEEDS}   Pooling: {POOLING}")
print(f"Total runs: {len(subject_ids) * nFold * len(SEEDS)}")

results = []
for subject_id in subject_ids:
    t_subj_start = time.time()
    for fold in range(nFold):
        tr_parts, va_parts, te_parts, ref_hash = build_partition(subject_id, fold)
        for seed in SEEDS:
            row = run_one(subject_id, fold, seed, tr_parts, va_parts, te_parts, ref_hash)
            results.append(row)
        print(f"[{time.time() - t_start:7.0f}s] subject {subject_id} fold {fold}: "
              f"seed42_test={results[-2]['test_at_best_val']:.3f}  "
              f"seed43_test={results[-1]['test_at_best_val']:.3f}  "
              f"min_train_loss=[{results[-2]['min_train_loss']:.4f}, {results[-1]['min_train_loss']:.4f}]")
    print(f"  subject {subject_id} done in {time.time() - t_subj_start:.0f}s")

t_total = time.time() - t_start
print(f"\nTotal wall-clock: {t_total:.1f}s ({t_total/60:.1f} min) for {len(results)} runs")

# %% [markdown]
# ## 9. Write output + partition-hash consistency check

# %%
import pandas as pd

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "phase2_final_results.csv", index=False)

print("\n=== Partition hash consistency check ===")
n_mismatch = 0
for (subject_id, fold), g in results_df.groupby(["subject_id", "fold"]):
    n_unique = g["partition_hash"].nunique()
    if n_unique != 1:
        n_mismatch += 1
        print(f"subject {subject_id} fold {fold}: {n_unique} unique hashes -- MISMATCH!!!")
print(f"Folds with hash mismatch: {n_mismatch}/{len(subject_ids) * nFold}")

# %% [markdown]
# ## 10. Summary statistics

# %%
print("\n=== Population summary ===")
print(f"Mean test_at_best_val across all {len(results_df)} runs: {results_df['test_at_best_val'].mean():.4f}")
print(f"SD across all runs: {results_df['test_at_best_val'].std(ddof=1):.4f}")

per_fold = results_df.groupby(["subject_id", "fold"])["test_at_best_val"].mean()
print(f"\nMean of per-fold (seed-averaged) accuracy: {per_fold.mean():.4f}")
print(f"SD across the 144 per-fold means: {per_fold.std(ddof=1):.4f}")

per_subject = results_df.groupby("subject_id")["test_at_best_val"].mean()
print("\nPer-subject mean accuracy (averaged over folds and seeds):")
print(per_subject.round(4).to_string())

seed_sd_per_fold = results_df.groupby(["subject_id", "fold"])["test_at_best_val"].std(ddof=1)
print(f"\nPooled seed-to-seed SD (2 seeds, full population): {np.sqrt((seed_sd_per_fold**2).mean()):.4f}")

print(f"\nMean min_train_loss: {results_df['min_train_loss'].mean():.4f}")
print(f"Folds where BOTH seeds have min_train_loss < 0.690: "
      f"{(results_df.groupby(['subject_id','fold'])['min_train_loss'].max() < 0.690).sum()}/{len(subject_ids)*nFold}")

print("\nBest-val-epoch distribution:")
print(results_df["best_val_epoch"].describe())
print(f"Median best_val_epoch: {results_df['best_val_epoch'].median()}")

below_035 = per_fold[per_fold < 0.35]
print(f"\nFolds (seed-averaged) below 0.35: {len(below_035)}/144")
print(below_035.round(4).to_string())
