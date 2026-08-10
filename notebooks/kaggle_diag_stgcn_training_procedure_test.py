# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 diagnostic -- substitute NSR/EEGNetAAD's training procedure
#
# NSR (EEGNetAAD), the validated EEG-only reference (0.546 SI/1s), is
# trained via `runner.py::fit` + `config_NSR_SI_DTU.yml`. Side by side with
# our ST-GCN training notebook:
#
# |                  | Ours (kaggle_train_stgcn_gcn_only.py) | NSR (config_NSR_SI_DTU.yml) |
# |------------------|----------------------------------------|------------------------------|
# | optimizer        | Adam                                    | NAdam                        |
# | lr               | 1e-3                                    | 1e-4 (10x smaller)           |
# | LR schedule      | none                                     | StepLR(step=20, gamma=0.1)   |
# | weight_decay     | 0                                        | 1e-6                          |
# | batch_size       | 32                                      | 128                           |
# | train sampling   | RandomSampler(replacement=True, 2000/ep)| full-epoch shuffle, no cap    |
# | epochs           | 40 fixed, no early stop                 | up to 50, early_stop='loss',  |
# |                  |                                          | patience=5 (runner.py L117)  |
#
# (AADNet-full, for reference, is even further out: lr=5e-5, StepLR(step=10,
# gamma=0.2), weight_decay=0.01, batch_size=256.)
#
# This substitutes ALL of NSR's training-procedure hyperparameters (not the
# model, adjacency, or data) for ours, on the same 2 sanity-check folds
# (subject 0, subject 12), fc_hidden left at the baseline 8 to isolate the
# training-procedure variable from the bottleneck-width one already tested.
# Reports whether train loss escapes ln(2), whether early stopping actually
# fires before 40/50 epochs, and where honest test accuracy lands.
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
# ## 3. Configuration -- data/adjacency/model identical to baseline; training
#    procedure substituted with NSR's (config_NSR_SI_DTU.yml)

# %%
from pathlib import Path
import yaml
import json
import time

RANDOM_SEED = 42
N_KERNELS = 5
FC_HIDDEN = 8          # baseline architecture -- isolate training-procedure variable only
VAL_FRACTION = 0.2

# --- NSR's training procedure, substituted in verbatim ---
MAX_EPOCHS = 50        # NSR's epochs budget (early stopping will likely cut this short)
PATIENCE = 5           # runner.py::fit line ~18/117, hardcoded
BATCH_SIZE = 128        # NSR's batch_size (vs. our 32)
LR = 1e-4              # NSR's lr (vs. our 1e-3)
WEIGHT_DECAY = 1e-6    # NSR's weight_decay (vs. our 0)
LR_DECAY_STEP = 20     # NSR's lr_decay_step
LR_DECAY_GAMMA = 0.1   # NSR's lr_decay_gamma

DIAG_FOLDS = [(0, 0), (12, 0)]

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

OUT_DIR = Path("/kaggle/working/stgcn_diag_training_procedure_test")
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
# ## 5. Model -- IDENTICAL architecture to kaggle_train_stgcn_gcn_only.py
#    (fc_hidden=8, the baseline -- this test isolates training procedure only)

# %%
from model import STGCNGCNOnly


def build_model():
    torch.manual_seed(RANDOM_SEED)
    m = STGCNGCNOnly(ADJACENCY, n_kernels=N_KERNELS, fc_hidden=FC_HIDDEN).to(DEVICE)
    return m


print(f"STGCNGCNOnly parameter count: {build_model().count_parameters()}")

# %% [markdown]
# ## 6. Fold-training loop -- NSR's training procedure substituted in:
#    NAdam, lr=1e-4, StepLR(20, 0.1), weight_decay=1e-6, batch_size=128,
#    full-epoch shuffle (no RandomSampler subsampling), real early stopping
#    on validation loss with patience=5 (runner.py::fit's exact logic,
#    including in-memory best-checkpoint tracking).

# %%
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from aadnet.dataset import DTUDataset
import copy


def run_fold(subject_id, fold, tr_split, te_split):
    tr_eeg, tr_aud, tr_label = tr_split
    te_eeg, te_aud, te_label = te_split

    tr_eeg2, va_eeg, tr_aud2, va_aud, tr_label2, va_label = train_test_split(
        tr_eeg, tr_aud, tr_label, test_size=VAL_FRACTION, random_state=subject_id
    )

    train_ds = DTUDataset(aadnet_config, tr_eeg2, tr_aud2, tr_label2)
    valid_ds = DTUDataset(aadnet_config, va_eeg, va_aud, va_label)
    test_ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)

    # NSR-style: full-epoch shuffle, no fixed per-epoch sample cap
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  train_ds={len(train_ds)} windows, {len(train_loader)} batches/epoch "
          f"(vs. our usual ~{2000 // 32} batches/epoch of 2000 sampled windows)")

    torch.manual_seed(RANDOM_SEED + subject_id * 100 + fold)
    model = build_model()
    opt = torch.optim.NAdam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                             eps=1e-8, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
    loss_fn = torch.nn.CrossEntropyLoss()

    rows = []
    best_val_loss = None
    best_state = None
    best_epoch = -1
    waiting = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for eeg, _audio, y in train_loader:
            eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
            if y.size(0) == 1:
                continue
            opt.zero_grad()
            logits = model(eeg)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * y.size(0)
            tr_n += y.size(0)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_loss_sum, va_correct, va_n = 0.0, 0, 0
            for eeg, _audio, y in valid_loader:
                eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
                logits = model(eeg)
                loss = loss_fn(logits, y)
                va_loss_sum += loss.item() * y.size(0)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_n += y.size(0)

            te_correct, te_n = 0, 0
            for eeg, _audio, y in test_loader:
                eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
                logits = model(eeg)
                te_correct += (logits.argmax(1) == y).sum().item()
                te_n += y.size(0)

        val_loss = va_loss_sum / max(va_n, 1)
        rows.append({
            "subject_id": subject_id, "fold": fold, "epoch": epoch,
            "train_loss": tr_loss_sum / max(tr_n, 1),
            "valid_loss": val_loss,
            "valid_acc": va_correct / max(va_n, 1),
            "test_acc": te_correct / max(te_n, 1),
            "lr": opt.param_groups[0]["lr"],
        })

        # runner.py::fit's exact early-stopping logic (early_stop='loss')
        if best_val_loss is None or val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            waiting = 0
        else:
            waiting += 1
        if waiting > PATIENCE:
            print(f"  early stopping at epoch {epoch} (best epoch was {best_epoch})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_correct, te_n = 0, 0
        for eeg, _audio, y in test_loader:
            eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
            logits = model(eeg)
            te_correct += (logits.argmax(1) == y).sum().item()
            te_n += y.size(0)
    test_at_best_checkpoint = te_correct / max(te_n, 1)

    return rows, best_epoch, test_at_best_checkpoint, len(train_ds), len(valid_ds), len(test_ds)


# %% [markdown]
# ## 7. Run both sanity-check folds

# %%
t_start = time.time()
all_rows = []
summary_rows = []
for subject_id, fold in DIAG_FOLDS:
    t_fold_start = time.time()
    crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
    tr_split, te_split = crossSIData[fold]
    fold_rows, best_epoch, test_at_best, n_tr, n_va, n_te = run_fold(subject_id, fold, tr_split, te_split)
    all_rows.extend(fold_rows)

    n_epochs_run = len(fold_rows)
    min_loss = min(r["train_loss"] for r in fold_rows)
    final_loss = fold_rows[-1]["train_loss"]
    summary_rows.append({
        "subject_id": subject_id, "fold": fold,
        "n_epochs_run": n_epochs_run, "stopped_early": n_epochs_run < MAX_EPOCHS,
        "best_epoch": best_epoch,
        "min_train_loss": min_loss, "final_train_loss": final_loss,
        "test_at_best_checkpoint": test_at_best,
    })
    print(f"[{time.time() - t_start:6.0f}s] subject {subject_id} fold {fold} "
          f"({time.time() - t_fold_start:.0f}s, n_tr={n_tr} n_va={n_va} n_te={n_te}): "
          f"ran {n_epochs_run} epochs (stopped_early={n_epochs_run < MAX_EPOCHS}), "
          f"best_epoch={best_epoch}  min_train_loss={min_loss:.4f}  final_train_loss={final_loss:.4f}  "
          f"test@best_checkpoint={test_at_best:.3f}")

t_total = time.time() - t_start
print(f"\nTotal wall-clock: {t_total:.1f}s ({t_total/60:.1f} min)")

# %% [markdown]
# ## 8. Write output

# %%
import pandas as pd

df = pd.DataFrame(all_rows)
df.to_csv(OUT_DIR / "training_procedure_test_per_epoch.csv", index=False)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "training_procedure_test_summary.csv", index=False)
print(summary.to_string(index=False))
