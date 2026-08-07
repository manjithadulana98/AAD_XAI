# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 diagnostic -- can the model overfit a tiny fixed batch?
#
# `kaggle_train_stgcn_gcn_only.py` (v5, dtype fix applied) trains cleanly but
# lands at mean test accuracy 0.404 (target band 0.60-0.85) -- the notebook's
# own stop condition. The representative fold's curve shows train accuracy
# stuck at ~50-54% for all 40 epochs (never rises), while test accuracy
# starts at ~0.75 and decays toward/below chance -- consistent with "the
# GCN-only submodel (stages a+b, no temporal attention) has very little
# discriminative signal at a 1s window," but that can't be distinguished from
# a real bug without a sanity check.
#
# Standard sanity check: take a small, FIXED batch of real windows (one
# subject's own SI-fold training split, first N windows, same batch every
# epoch, no held-out eval) and train for many epochs. If train accuracy
# climbs to ~100%, the model/gradients/adjacency are mechanically sound and
# the population-level result is a genuine capacity/signal finding. If it
# can't even memorize this tiny fixed batch, something upstream is still
# broken (dead gradients, degenerate graph conv, etc).
#
# **Kaggle setup requirements:** Internet enabled, GPU accelerator (optional --
# tiny job, CPU is fine too), `dulanamanjitha/aad-xai-artifacts` dataset
# attached. No Kaggle Secret needed.

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
# ## 2. Device + config (same mutation as the main training notebook)

# %%
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import yaml

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

RANDOM_SEED = 42
import numpy as np
import random
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# %% [markdown]
# ## 3. Fixed adjacency (identical to the main training notebook)

# %%
from adjacency import load_montage, build_adjacency_distance

montage = load_montage(os.path.join(REPO_DIR, "config", "aadnet_dtu_channel_montage.csv"))
ADJACENCY = build_adjacency_distance(montage, k=6)

# %% [markdown]
# ## 4. One subject's SI-fold-0 training split -- take a small, CLASS-BALANCED
#    fixed batch
#
# DTU windows are drawn sequentially within a trial, and each trial has one
# fixed attended speaker throughout -- so "the first N windows" of a training
# split can land entirely inside a single trial (all one label), making
# "memorize this batch" trivial (just bias toward the one class). Scan until
# N_PER_CLASS examples of EACH label are found instead.

# %%
from aadnet.dataset import DTUDataset

SUBJECT_ID = 0
N_PER_CLASS = 16   # -> 32 total, but guaranteed a real mix of both labels

crossSIData = DTUDataset.createSICrossValidation(SUBJECT_ID, aadnet_config)
tr_split, te_split = crossSIData[0]
tr_eeg, tr_aud, tr_label = tr_split
train_ds = DTUDataset(aadnet_config, tr_eeg, tr_aud, tr_label)
print(f"Full training split size: {len(train_ds)} windows -- scanning for {N_PER_CLASS} examples of each label")

by_class = {0: [], 1: []}
for i in range(len(train_ds)):
    eeg, _audio, y = train_ds[i]
    cls = int(y.item())
    if len(by_class[cls]) < N_PER_CLASS:
        by_class[cls].append(eeg)
    if len(by_class[0]) >= N_PER_CLASS and len(by_class[1]) >= N_PER_CLASS:
        break

fixed_eeg = torch.stack(by_class[0] + by_class[1]).to(DEVICE).float()
fixed_y = torch.tensor([0] * len(by_class[0]) + [1] * len(by_class[1])).to(DEVICE).long()
print(f"Fixed batch: eeg {tuple(fixed_eeg.shape)}, labels {fixed_y.tolist()}")
print(f"Class balance: {int((fixed_y == 0).sum())} zeros, {int((fixed_y == 1).sum())} ones")
assert by_class[0] and by_class[1], "Could not find both classes in this training split -- investigate labels."

# %% [markdown]
# ## 5. Train on ONLY this fixed batch for many epochs -- can it memorize?

# %%
from model import STGCNGCNOnly

N_EPOCHS_OVERFIT = 300
LR = 1e-3
N_KERNELS = 5

torch.manual_seed(RANDOM_SEED)
model = STGCNGCNOnly(ADJACENCY, n_kernels=N_KERNELS).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

print(f"Parameter count: {model.count_parameters()}")

acc_curve = []
loss_curve = []
for epoch in range(N_EPOCHS_OVERFIT):
    model.train()
    opt.zero_grad()
    logits = model(fixed_eeg)
    loss = loss_fn(logits, fixed_y)
    loss.backward()
    opt.step()
    acc = (logits.argmax(1) == fixed_y).float().mean().item()
    acc_curve.append(acc)
    loss_curve.append(loss.item())
    if epoch % 20 == 0 or epoch == N_EPOCHS_OVERFIT - 1:
        print(f"epoch {epoch:3d}: loss={loss.item():.4f}  train_acc={acc:.3f}")

final_acc = acc_curve[-1]
max_acc = max(acc_curve)
print(f"\nFinal train accuracy on the fixed {fixed_y.shape[0]}-window batch: {final_acc:.3f}")
print(f"Max train accuracy reached at any epoch: {max_acc:.3f}")

# %% [markdown]
# ## 6. Verdict (reports, does not auto-tune)

# %%
if max_acc >= 0.95:
    print("VERDICT: Model CAN memorize a tiny fixed batch (>=95% train acc reached) -- "
          "gradients/adjacency/architecture are mechanically sound. The population-level "
          "0.40 mean accuracy is a genuine capacity/signal-at-1s finding for the GCN-only "
          "submodel, not a bug. Supports proceeding to Phase 3 (temporal attention) as the "
          "next real lever, rather than further debugging Phase 2.")
elif max_acc >= 0.80:
    print("VERDICT: Partial memorization (80-95% train acc) -- some learning signal reaches "
          "the parameters, but weakly. Worth checking learning rate / theta init scale before "
          "concluding either way.")
else:
    print("VERDICT: Model CANNOT memorize even a tiny fixed batch (<80% train acc) -- "
          "this points to a real upstream bug (dead gradients, degenerate graph conv, or a "
          "label/data issue), not an architecture-capacity limit. Debug before Phase 3.")
