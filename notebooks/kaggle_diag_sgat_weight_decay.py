# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT -- Score-Gap Trajectory under Strong Weight Decay (Kaggle)
#
# Context: after the readout LayerNorm (commit 7f1a93f), the score-gap
# diagnostic still showed |score| growing ~270x (0.035 -> 9.56) over 30
# epochs while train/test accuracy stayed at chance. Hypothesis: with
# predictions near chance, readout.weight's gradient is near-zero-mean noise;
# AdamW normalises step size, so the weight norm random-walks upward. The
# existing decoupled decay (lr * wd = 1e-2 * 1e-4 = 1e-6 per step) is far too
# weak to oppose it.
#
# Identical to kaggle_diag_sgat_score_gap_trajectory except:
# - weight_decay = 1e-2 (per-step shrink 1e-4, 100x stronger)
# - logs ||readout.weight|| per epoch, to measure where the growth lives
#
# Competing hypothesis (structural bottleneck): EEG and speech are each
# time-pooled before they interact, so the EEG-envelope temporal covariance
# that carries attention information cannot be represented. Under that
# hypothesis, weight decay should tame |score| but leave accuracy at chance.
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
# ## 2. Confirm all four fixes are live

# %%
import torch
from aad_xai.models.sgat_gat_layer import CandidateConditionedGAT
from aad_xai.models.sgat import NodeFeatureFusion, StimulusGAT
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
import inspect

_probe = CandidateConditionedGAT(
    n_channels=4, in_features=6, speech_dim=4, out_features=8,
    adjacency=torch.eye(4), n_heads=2,
)
assert torch.all(_probe.film.weight == 0.0), "FiLM-init fix NOT live."

_fusion_probe = NodeFeatureFusion(n_channels=4, in_features=6, out_features=8)
assert isinstance(getattr(_fusion_probe, "norm", None), torch.nn.LayerNorm), \
    "NodeFeatureFusion LayerNorm fix NOT live."

assert "standardize_eeg" in inspect.signature(WindowedEEGAudioDataset.__init__).parameters, \
    "EEG standardization fix NOT live."

_sgat_probe = StimulusGAT(n_channels=4, sfreq=64.0, adjacency=torch.eye(4))
assert isinstance(getattr(_sgat_probe, "readout_norm", None), torch.nn.LayerNorm), \
    "Readout LayerNorm fix NOT live."

print("All four fixes confirmed live: FiLM-init, NodeFeatureFusion LayerNorm, "
      "EEG standardization, readout LayerNorm.")

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
LR = 1e-2
WEIGHT_DECAY = 1e-2
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
    train=TrainConfig(model="sgat", epochs=EPOCHS, num_seeds=1, device=str(DEVICE),
                      lr=LR, weight_decay=WEIGHT_DECAY),
    dataset="dtu",
    dataset_root=DTU_ROOT,
    output_dir="/kaggle/working/results_sgat_weight_decay",
)

_full_dtu = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=cfg.preprocess)
dataset = _SubjectSubset(_full_dtu, diag_subjects)
trials, split, ds_train, ds_val, ds_test = _prepare_data(dataset, cfg, WINDOW_S)
print(f"Windows: train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")
print(f"Split -- test subject: {split.test}")

# %% [markdown]
# ## 5. Train, tracking score-gap, output scale, and readout weight norm

# %%
import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.utils.seed import seed_everything

seed_everything(SEED)
n_channels = ds_train[0][0].shape[0]
model = StimulusGAT(n_channels=n_channels, sfreq=cfg.preprocess.sfreq_out).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss()

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

_fixed_eeg, _fixed_env, _fixed_labels = next(iter(dl_test))
_fixed_eeg, _fixed_env = _fixed_eeg.to(DEVICE), _fixed_env.to(DEVICE)
_fixed_labels = torch.as_tensor(_fixed_labels, dtype=torch.long, device=DEVICE)


def _score_gap_stats():
    model.eval()
    with torch.no_grad():
        logits = model(_fixed_eeg, _fixed_env)  # (B, 2)
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
            correct += (model(xb, envb).argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / max(total, 1)


def _readout_norms():
    return model.readout.weight.norm().item(), model.readout.bias.abs().item()


gap0, gapstd0, scale0, acc0 = _score_gap_stats()
w0, b0 = _readout_norms()
print(f"epoch   0 (init): score_gap={gap0:.6f}  |score|={scale0:.6f}  "
      f"||w_readout||={w0:.4f} |b_readout|={b0:.4f}")

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
    gap_mean, gap_std, scale, _ = _score_gap_stats()
    w_norm, b_abs = _readout_norms()

    history.append({
        "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc,
        "score_gap_mean": gap_mean, "score_gap_std": gap_std, "score_scale": scale,
        "readout_w_norm": w_norm, "readout_b_abs": b_abs,
    })
    print(f"epoch {epoch:>3}: loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}  "
          f"score_gap={gap_mean:.6f}  |score|={scale:.4f}  "
          f"gap/scale={100*gap_mean/max(scale,1e-12):.1f}%  "
          f"||w||={w_norm:.3f} |b|={b_abs:.3f}")

# %% [markdown]
# ## 6. Summary and decision

# %%
import json

Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
with open(Path(cfg.output_dir) / "weight_decay_history.json", "w") as f:
    json.dump({"init": {"gap": gap0, "scale": scale0, "w_norm": w0, "b_abs": b0},
               "weight_decay": WEIGHT_DECAY, "lr": LR, "history": history}, f, indent=2)

scales = [h["score_scale"] for h in history]
test_accs = [h["test_acc"] for h in history]
train_accs = [h["train_acc"] for h in history]
print(f"|score|:   first={scales[0]:.4f}  final={scales[-1]:.4f}  growth={scales[-1]/max(scales[0],1e-12):.1f}x "
      f"(previous run with wd=1e-4: 0.035 -> 9.56, ~270x)")
print(f"train_acc: max={max(train_accs):.4f}  final={train_accs[-1]:.4f}")
print(f"test_acc:  max={max(test_accs):.4f}  final={test_accs[-1]:.4f}")

scale_tamed = scales[-1] / max(scales[0], 1e-12) < 20
learned = max(train_accs) > 0.55
print("Decision:")
if scale_tamed and not learned:
    print("  -> Weight decay bounds the output scale, but train accuracy stays at chance.")
    print("     Optimizer drift was a symptom, not the cause. Supports the structural")
    print("     bottleneck hypothesis (time-pooling before speech/EEG interaction).")
elif learned:
    print("  -> Train accuracy rose above chance: optimizer drift was masking learning.")
else:
    print("  -> Output scale still grows under strong decay; inspect ||w|| and |b| columns.")
