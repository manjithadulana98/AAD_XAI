# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT with Time-Resolved FiLM -- First Real-Data Check (Kaggle)
#
# Commit d00c768 moves speech conditioning before time-pooling, so node
# features can carry EEG-envelope temporal covariance. On the synthetic
# tracking task this took StimulusGAT from 0.520 to 1.000 (TRF: 1.000).
#
# Same 14-subject pool, split (test S10/S17), window (5 s) and epochs (30) as
# every earlier score-gap diagnostic, so train/test accuracy is directly
# comparable: every prior variant stayed at train ~0.50, test 0.50-0.52.
# lr = 1e-3 (config default; also what passed the synthetic gate).
#
# Reference on real DTU: TRF via the same pipeline gets 0.6006 +/- 0.0420
# LOSO at 5 s.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - GPU recommended

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
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

# %% [markdown]
# ## 2. Confirm the time-resolved FiLM model is live

# %%
import torch
from aad_xai.models.sgat import StimulusGAT
from aad_xai.models.sgat_film import TimeResolvedFiLM

_probe = StimulusGAT(n_channels=4, sfreq=64.0, adjacency=torch.eye(4))
assert isinstance(getattr(_probe, "film", None), TimeResolvedFiLM), "TimeResolvedFiLM NOT live."
assert _probe.readout.bias is None, "Bias-free readout NOT live."
assert _probe.temporal_encoder(torch.randn(1, 4, 64)).dim() == 4, "Temporal encoder still pools over time."
print("Time-resolved FiLM model confirmed live.")

# %% [markdown]
# ## 3. Resolve the DTU dataset path, GPU check

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
print(f"CUDA available: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 4. Load data (same 14-subject pool and split as prior diagnostics)

# %%
from aad_xai.config import RunConfig, PreprocessConfig, TrainConfig, SplitConfig, WindowConfig
from aad_xai.data.base import BaseDataset
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.train import _prepare_data

N_SUBJECTS = 14
WINDOW_S = 5.0
SEED = 42
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 64

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
    train=TrainConfig(model="sgat", epochs=EPOCHS, num_seeds=1, device=str(DEVICE), lr=LR),
    dataset="dtu",
    dataset_root=DTU_ROOT,
    output_dir="/kaggle/working/results_sgat_time_resolved_film",
)

_full_dtu = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=cfg.preprocess)
dataset = _SubjectSubset(_full_dtu, diag_subjects)
trials, split, ds_train, ds_val, ds_test = _prepare_data(dataset, cfg, WINDOW_S)
print(f"Windows: train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")
print(f"Split -- val: {split.val}  test: {split.test}")

# %% [markdown]
# ## 5. Train, tracking train/val/test accuracy and the candidate score gap

# %%
import time
import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.utils.seed import seed_everything

seed_everything(SEED)
n_channels = ds_train[0][0].shape[0]
model = StimulusGAT(n_channels=n_channels, sfreq=cfg.preprocess.sfreq_out).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg.train.weight_decay)
loss_fn = nn.CrossEntropyLoss()

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def _evaluate(loader):
    model.eval()
    correct, total, gaps, scales = 0, 0, [], []
    with torch.no_grad():
        for xb, envb, yb in loader:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            logits = model(xb, envb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
            gaps.append((logits[:, 0] - logits[:, 1]).abs().cpu())
            scales.append(logits.abs().mean(1).cpu())
    return correct / max(total, 1), torch.cat(gaps).mean().item(), torch.cat(scales).mean().item()


history = []
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
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

    val_acc, _, _ = _evaluate(dl_val)
    test_acc, gap, scale = _evaluate(dl_test)
    film_w = model.film.proj.weight.norm().item()
    row = {
        "epoch": epoch, "train_loss": train_loss_sum / train_total, "train_acc": train_correct / train_total,
        "val_acc": val_acc, "test_acc": test_acc, "test_score_gap": gap, "test_score_scale": scale,
        "film_weight_norm": film_w,
    }
    history.append(row)
    print(f"epoch {epoch:>3}: loss={row['train_loss']:.4f} train={row['train_acc']:.4f} "
          f"val={val_acc:.4f} test={test_acc:.4f}  gap={gap:.4f} |score|={scale:.4f}  "
          f"||film||={film_w:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# %% [markdown]
# ## 6. Summary

# %%
import json

Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
with open(Path(cfg.output_dir) / "history.json", "w") as f:
    json.dump({"lr": LR, "epochs": EPOCHS, "split": {"val": list(split.val), "test": list(split.test)},
               "history": history}, f, indent=2)

best = max(history, key=lambda h: h["val_acc"])
print(f"train_acc: max={max(h['train_acc'] for h in history):.4f}  final={history[-1]['train_acc']:.4f}")
print(f"test_acc:  max={max(h['test_acc'] for h in history):.4f}  final={history[-1]['test_acc']:.4f}")
print(f"best-val epoch {best['epoch']}: val={best['val_acc']:.4f}  test={best['test_acc']:.4f}")
print("Prior variants on this split: train ~0.50, test 0.50-0.52. TRF LOSO reference: 0.60.")
