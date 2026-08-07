# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 — GCN-only training (no temporal attention)
#
# Stages (a)+(b) of Wang, Cai & Li (Interspeech 2023) as a standalone
# `nn.Module` (`stgcn/model.py::STGCNGCNOnly`), trained on the
# `external/AADNet` pipeline's real subject-independent (LOSO) splits at the
# 1-second decision window, using the Phase-1 distance-based k=6 adjacency
# (`config/aadnet_dtu_channel_montage.csv` order, verified 64/64 against
# AADNet's real channel order).
#
# **Reused as-is, unmodified:** `DTUDataset.createSICrossValidation` (leakage
# rules: disjoint train/test subjects, and any other-subject trial sharing a
# test-fold's attended stimulus is excluded from training -- both already
# implemented there, not reimplemented here).
#
# **One pipeline-compatibility fix, flagged rather than silently applied:**
# `AADDataset.__len__`/`__getitem__` implement a `duplicate` augmentation --
# every window is optionally doubled by swapping the two audio-envelope
# channels and flipping the label, which is valid for AADNet's own dual-
# stream (EEG+audio) architecture but would silently corrupt an EEG-only
# model: the identical EEG window would appear twice with opposite labels.
# This run sets `dataset.duplicate = False` in a mutated copy of
# `config_AADNet_SI_DTU_kaggle.yml` (`external/AADNet/utils/config.py`'s
# `Config.load_config` accepts a plain dict, so no new YAML file is added to
# the vendored `external/AADNet/` tree). Also overrides `dataset.training_window
# = 1` (this phase's 1-second window; the base config's default is 10s).
# Everything else (paths, preprocessing, channel list, scaler) is reused
# unchanged from the validated AADNet SI config.
#
# No GCS/checkpoint access needed this phase -- training a brand-new model
# from scratch per fold, not touching AADNet's own trained checkpoints.
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 3. Configuration
#
# Mutates a copy of the validated AADNet SI config -- window to 1s,
# `duplicate` off (see markdown note above). Everything else unchanged.

# %%
from pathlib import Path
import yaml
import json
import time

RANDOM_SEED = 42
N_EPOCHS = 40
BATCH_SIZE = 32
MAX_TRAIN_WINDOWS_PER_EPOCH = 2000   # RandomSampler draw (with replacement) -- bounds per-epoch cost
LR = 1e-3
N_KERNELS = 5
REPRESENTATIVE_SUBJECT_FOLD = (0, 0)   # curves saved for this (subject, fold) only

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

# Fix paths for this Kaggle session's actual mount point (base config hardcodes
# a specific dataset slug's path layout; DTU_ROOT above already resolved it).
raw_config["dataset"]["folder"] = os.path.join(DTU_ROOT, "eeg_new") + "/"
raw_config["dataset"]["stimuli_path"] = os.path.join(DTU_ROOT, "Audio")

raw_config["dataset"]["training_window"] = 1     # Phase 2: 1-second decision window
raw_config["dataset"]["duplicate"] = False       # EEG-only model -- see markdown note above
print(f"channels={len(raw_config['dataset']['channels'])}  sr={raw_config['dataset']['sr']}  "
      f"training_window={raw_config['dataset']['training_window']}s  step={raw_config['dataset']['step']}s  "
      f"duplicate={raw_config['dataset']['duplicate']}  nFold={raw_config['learning']['nFold']}")

from utils.config import Config
aadnet_config = Config.load_config(raw_config)

OUT_DIR = Path("/kaggle/working/stgcn_gcn_only")
OUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import random
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# %% [markdown]
# ## 4. Fixed adjacency (Phase 1, distance-based k=6, AADNet channel order)

# %%
from adjacency import load_montage, build_adjacency_distance

montage = load_montage(os.path.join(REPO_DIR, "config", "aadnet_dtu_channel_montage.csv"))
ADJACENCY = build_adjacency_distance(montage, k=6)
print(f"Adjacency: {ADJACENCY.shape}, symmetric={np.allclose(ADJACENCY, ADJACENCY.T)}, "
      f"no zero rows={not np.any(ADJACENCY.sum(axis=1) == 0)}")

# %% [markdown]
# ## 5. Model

# %%
from model import STGCNGCNOnly


def build_model():
    torch.manual_seed(RANDOM_SEED)  # model init reproducible; fold-level seeding done separately below
    m = STGCNGCNOnly(ADJACENCY, n_kernels=N_KERNELS).to(DEVICE)
    return m


_param_count = build_model().count_parameters()
print(f"STGCNGCNOnly parameter count: {_param_count}  (paper reference: ~2930)")

# %% [markdown]
# ## 6. Per-fold training loop

# %%
from torch.utils.data import DataLoader, RandomSampler
from aadnet.dataset import DTUDataset


def make_loader(ds, batch_size, train: bool):
    if train:
        sampler = RandomSampler(ds, replacement=True, num_samples=min(MAX_TRAIN_WINDOWS_PER_EPOCH, len(ds) * 5))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_fold(subject_id, fold, tr_split, te_split, keep_curve: bool):
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

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []} if keep_curve else None

    for epoch in range(N_EPOCHS):
        model.train()
        tr_loss_sum, tr_correct, tr_n = 0.0, 0, 0
        for eeg, _audio, y in train_loader:
            # DTUDataset yields float64 (the raw .mat EEG arrays are double-precision);
            # the graph-conv basis buffer is float32, so this cast is required.
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

        if keep_curve:
            history["train_loss"].append(tr_loss_sum / max(tr_n, 1))
            history["train_acc"].append(tr_correct / max(tr_n, 1))
            history["test_loss"].append(te_loss_sum / max(te_n, 1))
            history["test_acc"].append(te_correct / max(te_n, 1))

    final_test_acc = te_correct / max(te_n, 1)
    return final_test_acc, len(train_ds), len(test_ds), history


# %% [markdown]
# ## 7. Run all subjects x folds

# %%
t_start = time.time()
subject_ids = list(range(len(raw_config["dataset"]["all_sbjs"])))
nFold = raw_config["learning"]["nFold"]
print(f"Subjects: {subject_ids}   Folds/subject: {nFold}   Total fold-trainings: {len(subject_ids) * nFold}")

fold_rows = []
per_subject_seconds = {}
representative_history = None

for subject_id in subject_ids:
    t_subj_start = time.time()
    crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
    for fold in range(nFold):
        tr_split, te_split = crossSIData[fold]
        keep_curve = (subject_id, fold) == REPRESENTATIVE_SUBJECT_FOLD
        acc, n_train, n_test, history = run_fold(subject_id, fold, tr_split, te_split, keep_curve)
        fold_rows.append({"subject_id": subject_id, "fold": fold, "accuracy": acc,
                          "n_train_windows": n_train, "n_test_windows": n_test})
        if keep_curve:
            representative_history = history
        print(f"[{time.time() - t_start:7.0f}s] subject {subject_id} fold {fold}: "
              f"acc={acc:.3f}  (train_windows={n_train}, test_windows={n_test})")
    per_subject_seconds[subject_id] = time.time() - t_subj_start

t_total = time.time() - t_start
print(f"\nTotal wall-clock time: {t_total:.1f}s ({t_total/60:.1f} min) for "
      f"{len(fold_rows)} fold-trainings across {len(subject_ids)} subjects")

# %% [markdown]
# ## 8. Write outputs

# %%
import pandas as pd

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT_DIR / "gcn_only_fold_accuracy.csv", index=False)

accs = fold_df["accuracy"].to_numpy()
summary = {
    "mean_accuracy": float(accs.mean()),
    "sd_accuracy": float(accs.std(ddof=1)),
    "min_accuracy": float(accs.min()),
    "max_accuracy": float(accs.max()),
    "n_folds_total": len(accs),
    "n_subjects": len(subject_ids),
    "n_folds_per_subject": nFold,
    "parameter_count": _param_count,
    "paper_reference_accuracy": {"mean": 0.731, "sd": 0.0761},
    "paper_reference_param_count": 2930,
    "window_seconds": 1,
    "n_epochs": N_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "n_kernels": N_KERNELS,
    "max_train_windows_per_epoch": MAX_TRAIN_WINDOWS_PER_EPOCH,
    "total_wallclock_seconds": t_total,
    "total_wallclock_minutes": t_total / 60,
    "per_subject_seconds": per_subject_seconds,
    "adjacency_variant": "distance_k6",
    "montage_file": "config/aadnet_dtu_channel_montage.csv",
    "duplicate_augmentation_disabled": True,
}
with open(OUT_DIR / "gcn_only_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Mean accuracy: {summary['mean_accuracy']:.4f}  SD: {summary['sd_accuracy']:.4f}")
print(f"Parameter count: {_param_count}  (paper reference: ~2930)")

# %% [markdown]
# ## 9. Training curves (representative fold)

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = representative_history
assert h is not None, f"No history captured for representative fold {REPRESENTATIVE_SUBJECT_FOLD}"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
epochs = range(1, N_EPOCHS + 1)
axes[0].plot(epochs, h["train_loss"], label="train")
axes[0].plot(epochs, h["test_loss"], label="test")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss")
axes[0].legend()

axes[1].plot(epochs, h["train_acc"], label="train")
axes[1].plot(epochs, h["test_acc"], label="test")
axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].set_title("Accuracy")
axes[1].legend()

fig.suptitle(f"ST-GCN GCN-only -- subject {REPRESENTATIVE_SUBJECT_FOLD[0]} fold {REPRESENTATIVE_SUBJECT_FOLD[1]} "
            f"(representative fold)")
fig.tight_layout()
fig.savefig(OUT_DIR / "gcn_only_training_curves.png", dpi=150)
print(f"Saved {OUT_DIR / 'gcn_only_training_curves.png'}")

# %% [markdown]
# ## 10. Stop-condition self-check (reports, does not auto-tune)

# %%
mean_acc = summary["mean_accuracy"]
collapsed = (np.array(h["test_acc"][-5:]).std() < 1e-6) and abs(h["test_acc"][-1] - 0.5) < 1e-6
print(f"Mean accuracy: {mean_acc:.4f} (target range 0.65-0.80, paper: 0.731 +/- 0.0761)")
print(f"Representative-fold test acc collapsed to a constant ~0.5 in the last 5 epochs: {collapsed}")
if mean_acc < 0.60 or mean_acc > 0.85:
    print("STOP CONDITION FLAG: mean accuracy outside the [0.60, 0.85] plausibility band -- "
          "report and debug before proceeding to Phase 3, do not retune blindly.")
else:
    print("Mean accuracy within the plausibility band.")
