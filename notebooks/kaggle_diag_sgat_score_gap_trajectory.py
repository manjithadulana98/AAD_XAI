# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT -- Candidate Score-Gap Trajectory, All Fixes Applied (Kaggle)
#
# Five hypotheses tested with direct evidence, in order:
# 1. FiLM never breaks candidate symmetry at init -> fixed (commit 5f49991),
#    still chance on full LOSO (0.4952 +/- 0.0261).
# 2. Training schedule too weak -> ruled out (v2: 10x lr, 4x epochs, still chance).
# 3. Envelope/data pipeline uninformative -> ruled out (TRF via the same
#    WindowedEEGAudioDataset path gets 0.6006 +/- 0.0420 LOSO, matching its
#    own established baseline).
# 4. EEG amplitude scale -> standardization added, train accuracy STILL
#    pinned at chance (best 0.5120).
# 5. node_fusion activation explosion + attention collapse -> both directly
#    confirmed and fixed (LayerNorm, commit 387fec8): node_fusion output
#    now stable (std ~0.65-1.0, was exploding to 20.5), attention entropy
#    stays near its uniform-reference max throughout training (was
#    collapsing to ~2% of it) -- but train/test accuracy are STILL
#    unchanged, pinned almost exactly at 0.50.
#
# Every internal pathology found so far has been fixed, yet accuracy hasn't
# moved even slightly. This notebook checks the one thing that was never
# directly measured on a real, fully-fixed model: does conditioning on
# candidate A vs. candidate B produce ANY separation in the output scores
# at all? `score_a - score_b` is the literal quantity that has to differ,
# per-sample, for the model to ever predict anything but a coin flip.
#
# Tracked every epoch on a fixed real test batch:
# - mean |score_a - score_b| (the candidate-conditioning "effect size")
# - accuracy
# - for reference: mean |score_a| (the output scale, to judge whether the
#   gap is small in absolute terms or small only relative to a large scale)
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

# %% [markdown]
# ## 2. Confirm all three fixes are live

# %%
import torch
from aad_xai.models.sgat_gat_layer import CandidateConditionedGAT
from aad_xai.models.sgat import NodeFeatureFusion
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
import inspect

_probe = CandidateConditionedGAT(
    n_channels=4, in_features=6, speech_dim=4, out_features=8,
    adjacency=torch.eye(4), n_heads=2,
)
assert torch.all(_probe.film.weight == 0.0), "FiLM-init fix NOT live."

_fusion_probe = NodeFeatureFusion(n_channels=4, in_features=6, out_features=8)
assert hasattr(_fusion_probe, "norm") and isinstance(_fusion_probe.norm, torch.nn.LayerNorm), \
    "NodeFeatureFusion LayerNorm fix NOT live."

assert "standardize_eeg" in inspect.signature(WindowedEEGAudioDataset.__init__).parameters, \
    "EEG standardization fix NOT live."

print("All three fixes confirmed live: FiLM-init, NodeFeatureFusion LayerNorm, EEG standardization.")

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
# ## 4. Load data (same 14-subject pool as prior diagnostics)

# %%
from dataclasses import asdict

from aad_xai.config import RunConfig, PreprocessConfig, TrainConfig, SplitConfig, WindowConfig
from aad_xai.data.base import BaseDataset
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.train import _prepare_data

N_SUBJECTS = 14
WINDOW_S = 5.0
SEED = 42
EPOCHS = 30
LR = 1e-2
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
    output_dir="/kaggle/working/results_sgat_score_gap",
)

_full_dtu = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=cfg.preprocess)
dataset = _SubjectSubset(_full_dtu, diag_subjects)
trials, split, ds_train, ds_val, ds_test = _prepare_data(dataset, cfg, WINDOW_S)
print(f"Windows: train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")
print(f"Split -- test subject: {split.test}")

# %% [markdown]
# ## 5. Train, tracking the candidate score-gap every epoch on a fixed test batch

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

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Fixed test batch -- same batch probed every epoch, for a clean trend.
_fixed_eeg, _fixed_env, _fixed_labels = next(iter(dl_test))
_fixed_eeg, _fixed_env = _fixed_eeg.to(DEVICE), _fixed_env.to(DEVICE)
_fixed_labels = torch.as_tensor(_fixed_labels, dtype=torch.long, device=DEVICE)


def _score_gap_stats():
    model.eval()
    with torch.no_grad():
        logits = model(_fixed_eeg, _fixed_env)  # (B, 2) = [score_a, score_b]
        gap = (logits[:, 0] - logits[:, 1]).abs()
        acc = (logits.argmax(dim=1) == _fixed_labels).float().mean().item()
        return gap.mean().item(), gap.std().item(), logits.abs().mean().item(), acc


def _eval_acc(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, envb, yb in loader:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            logits = model(xb, envb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / max(total, 1)


gap0, gapstd0, scale0, acc0 = _score_gap_stats()
print(f"epoch   0 (init): score_gap={gap0:.6f} +/- {gapstd0:.6f}  |score|={scale0:.6f}  fixed_batch_acc={acc0:.4f}")

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

    train_acc = train_correct / max(train_total, 1)
    train_loss = train_loss_sum / max(train_total, 1)
    test_acc = _eval_acc(dl_test)
    gap_mean, gap_std, scale, fixed_acc = _score_gap_stats()

    history.append({
        "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc,
        "score_gap_mean": gap_mean, "score_gap_std": gap_std, "score_scale": scale,
    })
    print(f"epoch {epoch:>3}: loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}  "
          f"score_gap={gap_mean:.6f}+/-{gap_std:.6f}  |score|={scale:.6f}  "
          f"gap/scale={100*gap_mean/max(scale,1e-12):.1f}%")

# %% [markdown]
# ## 6. Summary

# %%
import json

gaps = [h["score_gap_mean"] for h in history]
print(f"\nscore_gap: init={gap0:.6f}  min={min(gaps):.6f}  max={max(gaps):.6f}  final={gaps[-1]:.6f}")
print(f"test_acc:  min={min(h['test_acc'] for h in history):.4f}  "
      f"max={max(h['test_acc'] for h in history):.4f}  final={history[-1]['test_acc']:.4f}")

Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
with open(Path(cfg.output_dir) / "score_gap_history.json", "w") as f:
    json.dump({"init": {"gap": gap0, "gap_std": gapstd0, "scale": scale0, "acc": acc0}, "history": history}, f, indent=2)
print(f"Full history written to {cfg.output_dir}/score_gap_history.json")

# %% [markdown]
# ## 7. Decision

# %%
import numpy as np

final_gap = history[-1]["score_gap_mean"]
final_scale = history[-1]["score_scale"]
gap_growth = final_gap / max(gap0, 1e-12)
max_gap_relative = max(h["score_gap_mean"] / max(h["score_scale"], 1e-12) for h in history)

print("Decision:")
print(f"  score_gap grew {gap_growth:.1f}x from init ({gap0:.6f}) to final epoch ({final_gap:.6f})")
print(f"  score_gap as %% of |score| scale: max over training = {100*max_gap_relative:.1f}%%")

if gap_growth < 3 and max_gap_relative < 0.05:
    print("  -> score_gap stayed essentially flat and tiny relative to output scale throughout")
    print("     training. Candidate conditioning via FiLM is contributing negligibly to the")
    print("     output -- NOT because of a numerical bug (all previously-found pathologies are")
    print("     fixed and confirmed healthy), but because the conditioning pathway's effect size")
    print("     is too small to matter. This is now a capacity/design question, not a bug:")
    print("     the single Linear(speech_dim, 2*node_features) FiLM layer may be too thin a")
    print("     pathway to meaningfully steer a 64-node graph-attention + mean-pool pipeline.")
    print("  -> Flag for discussion: strengthening the conditioning mechanism (e.g. deeper FiLM")
    print("     MLP, conditioning applied at multiple points, or an architecturally different")
    print("     conditioning approach) would be a genuine architecture change needing sign-off.")
elif gap_growth > 3 and max(h["test_acc"] for h in history) < 0.53:
    print("  -> score_gap IS growing, but accuracy still isn't moving -- the model is learning")
    print("     to separate the two candidates' scores, but not in a way that correlates with")
    print("     the true attended/unattended label. Worth checking: is the gap's SIGN correlated")
    print("     with the correct label at all, or just growing in an arbitrary/unhelpful direction?")
else:
    print("  -> Mixed signal -- inspect the per-epoch table above directly.")
