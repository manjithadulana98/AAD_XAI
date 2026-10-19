# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT -- Train-vs-Test Accuracy Gap on Real DTU Data (Kaggle)
#
# Every StimulusGAT LOSO run (v1 default schedule, v2 aggressive schedule,
# v3 FiLM-init fix) landed at chance on held-out TEST subjects. A separate
# diagnostic just proved `WindowedEEGAudioDataset` (the data path
# StimulusGAT trains on) is NOT the problem -- a dead-simple TRF decoder
# sourced through the exact same tensors gets 0.6006 +/- 0.0420 across the
# full 18-fold LOSO, matching this project's own established TRF baseline.
#
# So the problem is specific to StimulusGAT. But we've never actually
# checked whether StimulusGAT fits its own TRAINING population on real
# data at scale -- only a 16-window toy memorization test (too easy to be
# informative: any overparameterized net can memorize 16 arbitrary labels)
# and final held-out TEST accuracy (stuck at chance). This notebook runs
# one real training session via `aad_xai.train` (which prints live
# per-epoch train accuracy, unlike the LOSO harness) with v3's settings
# (FiLM-init fix, lr=1e-2, enough epochs to see a trend) on a reasonably
# large multi-subject training pool, and tracks train vs. val vs. test
# accuracy across epochs.
#
# - **If train accuracy ALSO stays near chance**: something deeper than
#   FiLM is broken -- vanishing gradients further up the stack, batch-norm
#   instability across heterogeneous real subjects (unlike the toy 16-window
#   batch, which may have come from a single subject/trial), or a
#   genuinely bad interaction at full-dataset scale. Points toward
#   architecture investigation, not just schedule/init.
# - **If train accuracy climbs well above chance but test stays flat**:
#   classic overfitting / failure to generalize across subjects. Points
#   toward regularization (dropout, weight decay), reduced capacity, or a
#   subject-invariance mechanism -- not a "broken" architecture, just one
#   that's currently memorizing subject-specific EEG idiosyncrasies instead
#   of the true attention-decoding relationship.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - GPU recommended (one real training run, more subjects/epochs than the
#   smoke test, though still a single fold not a sweep)

# %% [markdown]
# ## 1. Clone repository and install dependencies

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
    print(f"Repository already present at {REPO_DIR} -- pulling latest.")
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)

os.chdir(REPO_DIR)

try:
    import torch as _torch_preinstalled
    print(f"Pre-installed torch {_torch_preinstalled.__version__} found -- "
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

_src_dir = os.path.join(REPO_DIR, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import aad_xai  # noqa: F401
print(f"aad_xai importable from: {aad_xai.__file__}")

# %% [markdown]
# ## 2. Confirm the FiLM-init fix is live

# %%
import torch
from aad_xai.models.sgat_gat_layer import CandidateConditionedGAT

_probe = CandidateConditionedGAT(
    n_channels=4, in_features=6, speech_dim=4, out_features=8,
    adjacency=torch.eye(4), n_heads=2,
)
assert torch.all(_probe.film.weight == 0.0), "FiLM-init fix NOT applied -- git pull/re-clone needed."
print("FiLM-init fix confirmed live.")

# %% [markdown]
# ## 3. Resolve the DTU dataset path

# %%
from pathlib import Path

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, (
    "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'. "
    "Tried: " + ", ".join(DTU_KAGGLE_ROOT_CANDIDATES)
)
print(f"DTU dataset: {DTU_ROOT}")

# %% [markdown]
# ## 4. GPU sanity check

# %%
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 5. Load 14 subjects (reasonably large pool), build splits
#
# 14 subjects (not all 18) -- fast enough to get an answer quickly,
# large enough that the training population is genuinely heterogeneous
# across subjects (the point of this check), unlike the toy 16-window
# memorization test which likely came from very few trials/subjects.

# %%
from dataclasses import asdict

from aad_xai.config import RunConfig, PreprocessConfig, TrainConfig, SplitConfig, WindowConfig
from aad_xai.data.base import BaseDataset
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.train import _prepare_data
from aad_xai.utils.logging import get_run_dir, log_run_metadata

N_SUBJECTS = 14
WINDOW_S = 5.0
SEED = 42
EPOCHS = 40
LR = 1e-2
PATIENCE = 40  # effectively disabled -- we want the FULL train/val/test curve, not early stop
OUTPUT_DIR = "/kaggle/working/results_sgat_train_test_gap"

_all_subject_ids = sorted(p.stem for p in (Path(DTU_ROOT) / "eeg_new").glob("S*.mat"))
diag_subjects = _all_subject_ids[:N_SUBJECTS]
print(f"Using {len(diag_subjects)} of {len(_all_subject_ids)} subjects: {diag_subjects}")


class _SubjectSubset(BaseDataset):
    def __init__(self, inner, subjects):
        self.inner = inner
        self.subjects = set(subjects)
    def trials(self):
        return (t for t in self.inner.trials() if t.subject_id in self.subjects)


cfg = RunConfig(
    preprocess=PreprocessConfig(),
    window=WindowConfig(),
    split=SplitConfig(seed=SEED),
    train=TrainConfig(model="sgat", epochs=EPOCHS, num_seeds=1, device=str(DEVICE), lr=LR, patience=PATIENCE),
    dataset="dtu",
    dataset_root=DTU_ROOT,
    output_dir=OUTPUT_DIR,
)

_full_dtu = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=cfg.preprocess)
dataset = _SubjectSubset(_full_dtu, diag_subjects)
trials, split, ds_train, ds_val, ds_test = _prepare_data(dataset, cfg, WINDOW_S)
print(f"Windows: train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")
print(f"Split -- train subjects: {split.train}")
print(f"Split -- val subject:    {split.val}")
print(f"Split -- test subject:   {split.test}")

# %% [markdown]
# ## 6. Train, tracking train/val/test accuracy every epoch
#
# Hand-rolled training loop (not `aad_xai.train.train_deep`, which only
# returns final test accuracy) so we can record the FULL per-epoch
# trajectory for all three splits -- that trajectory is the actual
# diagnostic signal here, not just the endpoint.

# %%
import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.models.sgat import StimulusGAT
from aad_xai.utils.seed import seed_everything

seed_everything(SEED)
n_channels = ds_train[0][0].shape[0]
model = StimulusGAT(n_channels=n_channels, sfreq=cfg.preprocess.sfreq_out).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg.train.weight_decay)
loss_fn = nn.CrossEntropyLoss()

BATCH_SIZE = 64
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def _eval(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, envb, yb in loader:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            logits = model(xb, envb)
            loss_sum += loss_fn(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_correct, train_total, train_loss_sum = 0, 0, 0.0
    for xb, envb, yb in dl_train:
        xb, envb = xb.to(DEVICE), envb.to(DEVICE)
        yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(xb, envb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        train_loss_sum += loss.item() * xb.size(0)
        train_correct += (logits.argmax(1) == yb).sum().item()
        train_total += xb.size(0)

    train_loss = train_loss_sum / max(train_total, 1)
    train_acc = train_correct / max(train_total, 1)
    val_loss, val_acc = _eval(dl_val)
    test_loss, test_acc = _eval(dl_test)

    history.append({
        "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
        "val_loss": val_loss, "val_acc": val_acc, "test_loss": test_loss, "test_acc": test_acc,
    })
    print(f"  epoch {epoch:>3}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
          f"val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

# %% [markdown]
# ## 7. Summary

# %%
import json

final = history[-1]
best_train_acc = max(h["train_acc"] for h in history)
best_test_acc = max(h["test_acc"] for h in history)

print(f"\nFinal epoch ({EPOCHS}): train_acc={final['train_acc']:.4f}  "
      f"val_acc={final['val_acc']:.4f}  test_acc={final['test_acc']:.4f}")
print(f"Best train_acc across all epochs: {best_train_acc:.4f}")
print(f"Best test_acc across all epochs:  {best_test_acc:.4f}")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
with open(Path(OUTPUT_DIR) / "train_vs_test_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"\nFull per-epoch history written to {OUTPUT_DIR}/train_vs_test_history.json")

# %% [markdown]
# ## 8. Decision

# %%
print("Decision:")
if best_train_acc < 0.55:
    print(f"  best_train_acc={best_train_acc:.4f} -- TRAIN accuracy ALSO stuck near chance.")
    print("  StimulusGAT cannot even fit its own training population at real-dataset scale,")
    print("  despite fitting 16 toy windows perfectly. Something deeper than FiLM init is at play:")
    print("  likely candidates -- vanishing gradients further up the stack (temporal/frequency")
    print("  encoders, node fusion), batch-norm instability across heterogeneous real subjects")
    print("  within a batch, or the GAT's attention collapsing to a degenerate pattern at scale.")
    print("  -> Next: inspect per-layer gradient norms during training; check attention weight")
    print("     entropy (collapsed to uniform or one-hot?); verify adjacency correctness.")
elif best_train_acc > 0.65 and best_test_acc < 0.55:
    print(f"  best_train_acc={best_train_acc:.4f}, best_test_acc={best_test_acc:.4f} -- classic")
    print("  overfitting / failure to generalize across subjects. The architecture CAN learn the")
    print("  training population but is memorizing subject-specific idiosyncrasies rather than")
    print("  the general attention-decoding relationship.")
    print("  -> Next: increase dropout/weight_decay, reduce model capacity, or add an explicit")
    print("     subject-invariance mechanism (e.g. subject-adversarial training, batch norm")
    print("     statistics computed per-subject not per-batch).")
else:
    print(f"  best_train_acc={best_train_acc:.4f}, best_test_acc={best_test_acc:.4f} -- mixed/partial")
    print("  signal. Re-examine the full per-epoch trajectory above for the actual shape (e.g. does")
    print("  train_acc rise slowly and monotonically, or spike-and-collapse?) before deciding.")
