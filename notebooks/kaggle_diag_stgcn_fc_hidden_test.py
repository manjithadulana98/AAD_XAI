# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 diagnostic -- does widening fc_hidden unstick the loss floor?
#
# Train loss pins at exactly ln(2) (0.693 +/- 0.003) for the back half of
# training, uniformly across every subject -- the signature of a model
# outputting near-constant ~0.5/0.5 regardless of input. The forward pass
# has exactly one nonlinearity (`nn.ELU()`), sitting downstream of TWO
# purely linear reductions: the graph convolution, then global-average-
# pooling over the entire 64-sample time axis, compressing 320 features
# (n_kernels=5 x n_channels=64) down to fc_hidden=8 BEFORE the nonlinearity
# ever sees the data -- a 40:1 linear squeeze ahead of the only nonlinear
# stage.
#
# `fc_hidden=8` has no documented rationale (no comment/docstring anywhere)
# and reverse-engineers exactly to the paper's ~2,930 reference parameter
# count (32 + 325*fc_hidden = 2632 at fc_hidden=8) -- consistent with being
# a parameter-budget artifact, not a deliberate capacity choice.
#
# This notebook tests the hypothesis directly: same everything (data,
# adjacency, val-split + honest best-epoch selection from the Part A
# sanity check), except `fc_hidden` is widened from 8 to 64, on 2 folds
# only (subject 0 fold 0; subject 12 fold 0 -- the one perfectly
# class-balanced fold, cleanest to read). Parameter count will rise well
# above 2,930 -- that's fine here, this tests a mechanism, not a
# parameter-matched architecture.
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
# ## 3. Configuration -- IDENTICAL to kaggle_train_stgcn_gcn_only.py except
#    FC_HIDDEN, and only 2 folds instead of 144

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
VAL_FRACTION = 0.2

FC_HIDDEN_BASELINE = 8    # current value, for reference / re-verification in this same run
FC_HIDDEN_WIDE = 64       # the hypothesis under test

DIAG_FOLDS = [(0, 0), (12, 0)]  # subject 0 (heavily imbalanced test fold) + subject 12 (perfectly balanced)

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

OUT_DIR = Path("/kaggle/working/stgcn_diag_fc_hidden_test")
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
# ## 5. Model builder -- fc_hidden is now a parameter

# %%
from model import STGCNGCNOnly


def build_model(fc_hidden):
    torch.manual_seed(RANDOM_SEED)
    m = STGCNGCNOnly(ADJACENCY, n_kernels=N_KERNELS, fc_hidden=fc_hidden).to(DEVICE)
    return m


for fh in (FC_HIDDEN_BASELINE, FC_HIDDEN_WIDE):
    print(f"fc_hidden={fh}: parameter count = {build_model(fh).count_parameters()}")

# %% [markdown]
# ## 6. Fold-training loop -- IDENTICAL val-split + honest best-epoch
#    selection as the Part A sanity check, plus per-epoch train LOSS logging
#    (not tracked before) to directly check whether it drops below ln(2).

# %%
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from aadnet.dataset import DTUDataset


def make_loader(ds, batch_size, train: bool):
    if train:
        sampler = RandomSampler(ds, replacement=True, num_samples=min(MAX_TRAIN_WINDOWS_PER_EPOCH, len(ds) * 5))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_fold(subject_id, fold, tr_split, te_split, fc_hidden):
    tr_eeg, tr_aud, tr_label = tr_split
    te_eeg, te_aud, te_label = te_split

    tr_eeg2, va_eeg, tr_aud2, va_aud, tr_label2, va_label = train_test_split(
        tr_eeg, tr_aud, tr_label, test_size=VAL_FRACTION, random_state=subject_id
    )

    train_ds = DTUDataset(aadnet_config, tr_eeg2, tr_aud2, tr_label2)
    valid_ds = DTUDataset(aadnet_config, va_eeg, va_aud, va_label)
    test_ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)

    torch.manual_seed(RANDOM_SEED + subject_id * 100 + fold)
    model = build_model(fc_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = make_loader(train_ds, BATCH_SIZE, train=True)
    valid_loader = make_loader(valid_ds, BATCH_SIZE, train=False)
    test_loader = make_loader(test_ds, BATCH_SIZE, train=False)

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
            "subject_id": subject_id, "fold": fold, "fc_hidden": fc_hidden, "epoch": epoch,
            "train_loss": tr_loss_sum / max(tr_n, 1),
            "valid_acc": va_correct / max(va_n, 1),
            "test_acc": te_correct / max(te_n, 1),
        })

    return rows, len(train_ds), len(valid_ds), len(test_ds)


# %% [markdown]
# ## 7. Run both fc_hidden settings on both folds

# %%
t_start = time.time()
all_rows = []
for fc_hidden in (FC_HIDDEN_BASELINE, FC_HIDDEN_WIDE):
    for subject_id, fold in DIAG_FOLDS:
        t_fold_start = time.time()
        crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
        tr_split, te_split = crossSIData[fold]
        fold_rows, n_tr, n_va, n_te = run_fold(subject_id, fold, tr_split, te_split, fc_hidden)
        all_rows.extend(fold_rows)

        best_va_idx = max(range(len(fold_rows)), key=lambda i: fold_rows[i]["valid_acc"])
        test_at_best_va = fold_rows[best_va_idx]["test_acc"]
        final_loss = fold_rows[-1]["train_loss"]
        min_loss = min(r["train_loss"] for r in fold_rows)
        print(f"[{time.time() - t_start:6.0f}s] fc_hidden={fc_hidden:3d} subject {subject_id} fold {fold} "
              f"({time.time() - t_fold_start:.0f}s): "
              f"final_train_loss={final_loss:.4f}  min_train_loss={min_loss:.4f}  "
              f"test@best_val_epoch({best_va_idx})={test_at_best_va:.3f}")

t_total = time.time() - t_start
print(f"\nTotal wall-clock: {t_total:.1f}s ({t_total/60:.1f} min)")

# %% [markdown]
# ## 8. Write output

# %%
import pandas as pd

df = pd.DataFrame(all_rows)
out_path = OUT_DIR / "fc_hidden_test_per_epoch.csv"
df.to_csv(out_path, index=False)
print(f"Written {len(df)} rows to {out_path}")

summary_rows = []
for (fh, sid, fold), g in df.groupby(["fc_hidden", "subject_id", "fold"]):
    g = g.sort_values("epoch").reset_index(drop=True)
    best_va_idx = g["valid_acc"].idxmax()
    summary_rows.append({
        "fc_hidden": fh, "subject_id": sid, "fold": fold,
        "min_train_loss": g["train_loss"].min(),
        "final_train_loss": g.iloc[-1]["train_loss"],
        "best_val_epoch": int(g.loc[best_va_idx, "epoch"]),
        "test_at_best_val_epoch": g.loc[best_va_idx, "test_acc"],
        "final_test_acc": g.iloc[-1]["test_acc"],
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "fc_hidden_test_summary.csv", index=False)
print(summary.to_string(index=False))
