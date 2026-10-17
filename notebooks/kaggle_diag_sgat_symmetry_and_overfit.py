# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT -- Stuck-at-Chance Diagnostic (Kaggle)
#
# The 18-fold LOSO sweep returned mean accuracy 0.4887 +/- 0.0247 -- not
# distinguishable from the 0.5 chance baseline. Earlier smoke-training logs
# showed the exact same signature: val_loss pinned near 0.693 (= ln 2) across
# every epoch, which is what you see when the model assigns ~50/50 probability
# to both candidates and never breaks that symmetry.
#
# This notebook runs two bounded, cheap diagnostics -- no architecture changes
# -- to localize the cause before anything in `src/aad_xai/models/sgat.py` is
# touched (per CLAUDE.md, architecture changes need discussion first):
#
# 1. **Symmetry probe**: measure `|score_a - score_b|` across a batch, both
#    on a FRESHLY-INITIALIZED model (expected near zero -- FiLM output is
#    small at init, so both candidate passes return nearly identical scores)
#    and on the TRAINED LOSO/smoke checkpoint (if still near zero, the model
#    never learned to differentiate the two candidates -- consistent with
#    stuck-at-chance). If the trained checkpoint shows meaningful separation
#    but accuracy is still ~0.5, that would mean the model distinguishes the
#    streams but doesn't correctly identify the attended one -- a different
#    failure mode.
#
# 2. **Overfit sanity check**: can the model memorize a tiny fixed batch at
#    all, given a higher LR and many more epochs? Same discipline this repo
#    already used on ST-GCN Phase 2 (see `stgcn/README.md`'s "overfit sanity
#    check" section, which caught a related failure mode). If the model
#    cannot overfit even 16 windows with a 10x LR, something structural
#    (not training-schedule) is wrong.
#
# What this does NOT do:
# - Change the architecture
# - Change the loss function
# - Add any new attention/transformer blocks
# - Modify any file in `src/`
#
# All it needs is a trained checkpoint path. Default is the smoke-notebook's
# checkpoint; change `CHECKPOINT_PATH` to point at one of the LOSO run's
# per-fold models if those get persisted later.

# %% [markdown]
# ## 1. Clone repo, install deps, resolve dataset

# %%
import os
import subprocess
import sys
from pathlib import Path

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

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, (
    "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'."
)
print(f"DTU dataset: {DTU_ROOT}")

# %% [markdown]
# ## 2. Configuration
#
# Default `CHECKPOINT_PATH` is the smoke notebook's output. If you've since
# trained a fuller checkpoint (e.g. via `aad_xai.train --dataset dtu --model
# sgat` on more subjects/epochs), point this at its `best_model.pt`.

# %%
import torch
from aad_xai.config import PreprocessConfig

CHECKPOINT_PATH = "/kaggle/working/results_sgat_dtu_smoke/sgat/seed42/win5.0/best_model.pt"
N_SUBJECTS_FOR_PROBE = 2       # just need a batch of real windows; keep this small/fast
WINDOW_S = 5.0
SEED = 42
PROBE_BATCH_SIZE = 64

OVERFIT_BATCH_SIZE = 16         # 8 attended + 8 unattended, scanned not sliced -- see cell 4
OVERFIT_LR = 1e-2               # 10x the main training's 1e-3
OVERFIT_EPOCHS = 300
OVERFIT_SFREQ = PreprocessConfig().sfreq_out  # same as training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %% [markdown]
# ## 3. Symmetry probe

# %%
from dataclasses import asdict
from torch.utils.data import DataLoader

from aad_xai.config import RunConfig, TrainConfig, SplitConfig, WindowConfig
from aad_xai.data.base import BaseDataset
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.train import _prepare_data
from aad_xai.models.sgat import StimulusGAT


class _SubjectSubset(BaseDataset):
    def __init__(self, inner, subjects):
        self.inner = inner
        self.subjects = set(subjects)
    def trials(self):
        return (t for t in self.inner.trials() if t.subject_id in self.subjects)


_all_subject_ids = sorted(p.stem for p in (Path(DTU_ROOT) / "eeg_new").glob("S*.mat"))
probe_subjects = _all_subject_ids[:N_SUBJECTS_FOR_PROBE]
print(f"Probe subjects: {probe_subjects}")

cfg = RunConfig(
    preprocess=PreprocessConfig(),
    window=WindowConfig(),
    split=SplitConfig(seed=SEED),
    train=TrainConfig(model="sgat", epochs=1, num_seeds=1, device=str(DEVICE)),
    dataset="dtu",
    dataset_root=DTU_ROOT,
    output_dir="/kaggle/working/results_sgat_diag",
)
_full_dtu = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=cfg.preprocess)
probe_dataset = _SubjectSubset(_full_dtu, probe_subjects)
_, _, ds_tr, ds_val, ds_te = _prepare_data(probe_dataset, cfg, WINDOW_S)
probe_loader = DataLoader(ds_te, batch_size=min(PROBE_BATCH_SIZE, len(ds_te)), shuffle=True)
eeg_probe, env_probe, labels_probe = next(iter(probe_loader))
eeg_probe, env_probe, labels_probe = eeg_probe.to(DEVICE), env_probe.to(DEVICE), labels_probe.long().to(DEVICE)
print(f"Probe batch: eeg={tuple(eeg_probe.shape)} env={tuple(env_probe.shape)}")

n_channels = eeg_probe.shape[1]


def _score_separation(model):
    """Return mean |score_a - score_b| and mean absolute score across the probe batch."""
    model.eval()
    with torch.no_grad():
        logits = model(eeg_probe, env_probe)  # (B, 2) = [score_a, score_b]
    gap = (logits[:, 0] - logits[:, 1]).abs().mean().item()
    scale = logits.abs().mean().item()
    acc = (logits.argmax(dim=1) == labels_probe).float().mean().item()
    return {"mean_abs_score_gap": gap, "mean_abs_score": scale, "accuracy": acc}


# Fresh model
fresh = StimulusGAT(n_channels=n_channels, sfreq=cfg.preprocess.sfreq_out).to(DEVICE)
fresh_stats = _score_separation(fresh)
print("\n[FRESH model, random init]")
for k, v in fresh_stats.items():
    print(f"  {k}: {v:.6f}")

# Trained checkpoint
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    trained = StimulusGAT(n_channels=n_channels, sfreq=cfg.preprocess.sfreq_out).to(DEVICE)
    trained.load_state_dict(ckpt["state_dict"])
    trained_stats = _score_separation(trained)
    print(f"\n[TRAINED model, {CHECKPOINT_PATH}]")
    for k, v in trained_stats.items():
        print(f"  {k}: {v:.6f}")

    ratio = trained_stats["mean_abs_score_gap"] / max(fresh_stats["mean_abs_score_gap"], 1e-12)
    print(f"\n  score_gap(trained) / score_gap(fresh) = {ratio:.2f}x")
    print("  Interpretation:")
    print("    ~1x  = training did NOT break candidate symmetry (stuck-at-chance root cause).")
    print("    >>1x = model distinguishes the two candidates but accuracy still at chance")
    print("           -> different failure (distinguishes streams but mislabels attended).")
else:
    print(f"\n[TRAINED] Checkpoint not found: {CHECKPOINT_PATH} -- skipping trained-model comparison.")
    print("         Run `kaggle_train_sgat_dtu_smoke.ipynb` first to produce one.")

# %% [markdown]
# ## 4. Overfit sanity check
#
# Scan `ds_tr` for 8 attended + 8 unattended windows (not "first 16" -- see
# the ST-GCN overfit-check note: taking the first N windows from one trial
# landed entirely inside one attended-speaker trial, making the batch all
# one class, and the model "overfit" to a constant bias, not real
# discrimination). Then try to memorize that fixed, balanced batch.

# %%
import numpy as np

attended_idx = []
unattended_idx = []
TARGET_PER_CLASS = OVERFIT_BATCH_SIZE // 2
for i in range(len(ds_tr)):
    sample = ds_tr[i]
    label = int(sample[2])
    if label == 0 and len(attended_idx) < TARGET_PER_CLASS:
        attended_idx.append(i)
    elif label == 1 and len(unattended_idx) < TARGET_PER_CLASS:
        unattended_idx.append(i)
    if len(attended_idx) == TARGET_PER_CLASS and len(unattended_idx) == TARGET_PER_CLASS:
        break

assert len(attended_idx) == TARGET_PER_CLASS and len(unattended_idx) == TARGET_PER_CLASS, (
    f"Could not find {TARGET_PER_CLASS} windows per class in ds_tr "
    f"(found {len(attended_idx)}/{len(unattended_idx)})."
)
selected = attended_idx + unattended_idx
eeg_fix, env_fix, labels_fix = zip(*(ds_tr[i] for i in selected))
eeg_fix = torch.stack(eeg_fix).to(DEVICE)
env_fix = torch.stack(env_fix).to(DEVICE)
labels_fix = torch.as_tensor(labels_fix, dtype=torch.long, device=DEVICE)
print(f"Overfit batch: eeg={tuple(eeg_fix.shape)} env={tuple(env_fix.shape)} "
      f"attended={int((labels_fix == 0).sum())} unattended={int((labels_fix == 1).sum())}")

overfit_model = StimulusGAT(n_channels=n_channels, sfreq=OVERFIT_SFREQ).to(DEVICE)
opt = torch.optim.AdamW(overfit_model.parameters(), lr=OVERFIT_LR)
loss_fn = torch.nn.CrossEntropyLoss()

history = []
for epoch in range(OVERFIT_EPOCHS + 1):
    overfit_model.train()
    opt.zero_grad(set_to_none=True)
    logits = overfit_model(eeg_fix, env_fix)
    loss = loss_fn(logits, labels_fix)
    loss.backward()
    opt.step()

    overfit_model.eval()
    with torch.no_grad():
        acc = (overfit_model(eeg_fix, env_fix).argmax(dim=1) == labels_fix).float().mean().item()
    history.append({"epoch": epoch, "loss": float(loss.item()), "acc": acc})
    if epoch % 50 == 0 or epoch == OVERFIT_EPOCHS:
        print(f"  epoch {epoch:>4}: loss={loss.item():.4f}  acc={acc:.3f}")

final_acc = history[-1]["acc"]
print(f"\nFinal overfit accuracy: {final_acc:.3f}")
print("Interpretation:")
print("  ~1.00 -> the architecture CAN fit the data; the main training is undertrained /")
print("          needs schedule changes (LR, epochs, warmup), not architectural changes.")
print("  ~0.50 -> architecture cannot even memorize 16 windows at 10x LR -- structural issue")
print("          (likely the FiLM-symmetry mechanism never gets gradient signal strong enough")
print("          to break the near-identical two-pass output). Flag for CLAUDE.md-compliant")
print("          architecture discussion before any patch.")

# %% [markdown]
# ## 5. Summary / what to do next
#
# Read the two outputs above together:
#
# | Symmetry probe (trained) | Overfit accuracy | Likely cause |
# | --- | --- | --- |
# | gap ~0 (no change from fresh) | ~0.5 | Structural: FiLM path never learns to break candidate symmetry |
# | gap ~0 (no change from fresh) | ~1.0 | Training schedule insufficient to break symmetry at normal batch sizes/LR |
# | gap meaningful | ~0.5 | Model distinguishes streams but mislabels which is attended -- different bug |
# | gap meaningful | ~1.0 | Needs more capacity / data / epochs; architecture is viable |
#
# Only the "structural" row would justify a CLAUDE.md architecture discussion.
# The other three are schedule/data questions, resolvable without touching
# `src/aad_xai/models/sgat.py`.
