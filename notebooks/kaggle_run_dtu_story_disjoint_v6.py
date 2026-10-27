# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Story-Disjoint Protocol, Round 2: wide EEG band (Kaggle)
#
# Single factor changed vs. round 1 (v5, commit 949f485): the EEG band-pass
# goes from 1-8 Hz to 1-31 Hz. The speech envelope stays at 1-8 Hz
# (`PreprocessConfig.envelope_bandpass_hz`), sampling rate stays 64 Hz.
#
# Why: StimulusGAT's frequency encoder computes 8 log band powers from 1 to
# 32 Hz, but with 1-8 Hz EEG the six bands above 8 Hz only see filter
# roll-off. Hypothesis: giving it (and the temporal convs) alpha/beta
# activity raises accuracy.
#
# Everything else equals the round-1 "overlap" arm: same story-disjoint folds
# (subject_group_story_out, seed 42), same StimulusGAT config and seed, 2.5 s
# training-window hop. The TRF baseline (1-8 Hz, closed-form, deterministic)
# and the round-1 reference arm are reused from round 1 and compared offline,
# per subject.
#
# Pre-registered decision: keep 1-31 Hz iff mean pooled validation accuracy
# over folds beats round-1 overlap (0.6476) AND the shifted-envelope test
# control is not above chance at subject level (Wilcoxon vs 0.5, p >= 0.05).
# A is always the male talker, so any EEG-only cue to the attended gender
# would survive the envelope shift and show up in that control.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - GPU required (~2-2.5 h)

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
GIT_COMMIT = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
print(f"Commit: {GIT_COMMIT}")

# %% [markdown]
# ## 2. Confirm the model, the split and the decoupled envelope band are live

# %%
import torch
from aad_xai.config import PreprocessConfig
from aad_xai.models.sgat import StimulusGAT
from aad_xai.models.sgat_film import TimeResolvedFiLM
from aad_xai.data.cv_splits import (
    assert_story_disjoint_fold_integrity,
    dtu_content_group,
    subject_group_story_out,
)

_probe = StimulusGAT(n_channels=4, sfreq=64.0, adjacency=torch.eye(4))
assert isinstance(getattr(_probe, "film", None), TimeResolvedFiLM), "TimeResolvedFiLM NOT live."
assert _probe.readout.bias is None, "Bias-free readout NOT live."
assert PreprocessConfig(bandpass_hz=(1.0, 31.0), envelope_bandpass_hz=(1.0, 8.0)).envelope_band == (1.0, 8.0), \
    "envelope_bandpass_hz NOT live."
print("Model, story-disjoint split and envelope_bandpass_hz confirmed live.")

# %% [markdown]
# ## 3. Load all subjects (EEG 1-31 Hz, envelope 1-8 Hz)

# %%
import time
from collections import Counter
from pathlib import Path

import numpy as np

from aad_xai.data.dtu_dataset import DTUDataset

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DTU dataset: {DTU_ROOT}   device: {DEVICE}")

WINDOW_S = 5.0
TRAIN_OVERLAP_S = 2.5
FOLD_SEED = 42
SGAT_SEED = 42
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
SHIFT_S = 2.5
EEG_BAND_HZ = (1.0, 31.0)
ENVELOPE_BAND_HZ = (1.0, 8.0)
SGAT_ARMS = {"wide_overlap": TRAIN_OVERLAP_S}
OUT_DIR = Path("/kaggle/working/results_story_disjoint_v6")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

preprocess = PreprocessConfig(bandpass_hz=EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
SFREQ = preprocess.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)

_t0 = time.time()
all_trials = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=preprocess).trials())
print(f"Loaded {len(all_trials)} trials, {len({t.subject_id for t in all_trials})} subjects "
      f"in {time.time() - _t0:.0f}s  (EEG {EEG_BAND_HZ} Hz, envelope {preprocess.envelope_band} Hz, {SFREQ} Hz)")
print("Content groups:", sorted(Counter(dtu_content_group(t) for t in all_trials).items(), key=str))

# %% [markdown]
# ## 4. StimulusGAT helpers, with per-subject counts (unchanged from v5)

# %%
import copy

import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.utils.seed import seed_everything


def _sgat_counts(model, ds, shift_samples=0):
    """Per-subject [correct, total] on a non-shuffled dataset."""
    model.eval()
    hits = []
    with torch.no_grad():
        for xb, envb, yb in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            if shift_samples:
                envb = torch.roll(envb, shifts=shift_samples, dims=-1)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            hits.append((model(xb, envb).argmax(1) == yb).cpu())
    hits = torch.cat(hits).numpy()
    out = {}
    for h, w in zip(hits, ds.window_indices):
        c = out.setdefault(w.subject_id, [0, 0])
        c[0] += int(h); c[1] += 1
    return out


def _pooled(counts):
    return sum(c for c, _ in counts.values()) / max(sum(n for _, n in counts.values()), 1)


def train_sgat_fold(ds_tr, ds_va, ds_te, n_channels, seed):
    seed_everything(seed)
    model = StimulusGAT(n_channels=n_channels, sfreq=SFREQ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    best = {"val_acc": -1.0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, envb, yb in dl_tr:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb, envb), yb).backward()
            opt.step()
        val_acc = _pooled(_sgat_counts(model, ds_va))
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}

    model.load_state_dict(best["state"])
    return model, {
        "best_epoch": best["epoch"],
        "val": _sgat_counts(model, ds_va),
        "test": _sgat_counts(model, ds_te),
        "test_shifted": _sgat_counts(model, ds_te, shift_samples=SHIFT_SAMPLES),
        "n_train_windows": len(ds_tr),
    }

# %% [markdown]
# ## 5. Story-disjoint CV (same folds as round 1)

# %%
import json

folds = list(subject_group_story_out(all_trials, seed=FOLD_SEED))
for fold in folds:
    assert_story_disjoint_fold_integrity(all_trials, fold)
print(f"{len(folds)} story-disjoint folds (integrity checked); SGAT arms {SGAT_ARMS}")

fold_results = []
for fi, fold in enumerate(folds):
    t0 = time.time()
    tr = [all_trials[i] for i in fold.train_idx]
    va = [all_trials[i] for i in fold.val_idx]
    te = [all_trials[i] for i in fold.test_idx]

    r = {"fold_id": fold.fold_id, "meta": fold.meta, "sgat": {}}
    ds_va, ds_te = (WindowedEEGAudioDataset(t, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=True)
                    for t in (va, te))
    for arm, overlap_s in SGAT_ARMS.items():
        ds_tr = WindowedEEGAudioDataset(tr, window_s=WINDOW_S, overlap_s=overlap_s, standardize_eeg=True)
        model, rs = train_sgat_fold(ds_tr, ds_va, ds_te, n_channels=ds_tr[0][0].shape[0], seed=SGAT_SEED)
        torch.save(model.state_dict(), CKPT_DIR / f"sgat_v6_{arm}_{fold.fold_id}.pt")
        r["sgat"][arm] = rs
    fold_results.append(r)

    sg = "  ".join(f"{arm}: val={_pooled(r['sgat'][arm]['val']):.3f} test={_pooled(r['sgat'][arm]['test']):.3f} "
                   f"(shift {_pooled(r['sgat'][arm]['test_shifted']):.3f}, ep {r['sgat'][arm]['best_epoch']})"
                   for arm in SGAT_ARMS)
    print(f"[{fi+1:>2}/{len(folds)}] {fold.fold_id} test={fold.meta['test_subjects']} "
          f"stories={fold.meta['held_out_content_groups']}: {sg}  ({time.time()-t0:.0f}s)", flush=True)

    with open(OUT_DIR / "story_disjoint_v6.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {
                       "protocol": "subject_group_story_out (6 subject groups x 3 story folds, 2 val subjects)",
                       "eeg_band_hz": list(EEG_BAND_HZ), "envelope_band_hz": list(preprocess.envelope_band),
                       "sfreq": SFREQ, "window_s": WINDOW_S, "fold_seed": FOLD_SEED, "sgat_seed": SGAT_SEED,
                       "sgat_arms": SGAT_ARMS, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
                       "shift_s": SHIFT_S},
                   "folds": fold_results}, f, indent=2)

# %% [markdown]
# ## 6. Summary (comparison with round 1 and the TRF is done offline, per subject)

# %%
from scipy.stats import wilcoxon


def per_subject(get):
    acc = {}
    for r in fold_results:
        for s, (c, n) in get(r).items():
            a = acc.setdefault(s, [0, 0]); a[0] += c; a[1] += n
    subjects = sorted(acc, key=lambda s: int(s[1:]))
    return subjects, np.array([acc[s][0] / acc[s][1] for s in subjects])


for arm in SGAT_ARMS:
    _, test = per_subject(lambda r: r["sgat"][arm]["test"])
    _, shifted = per_subject(lambda r: r["sgat"][arm]["test_shifted"])
    val = np.mean([_pooled(r["sgat"][arm]["val"]) for r in fold_results])
    print(f"SGAT {arm}: test {test.mean():.4f} +/- {test.std():.4f}  "
          f"shifted {shifted.mean():.4f} (above 0.5 on {int((shifted > 0.5).sum())}/{len(shifted)}, "
          f"Wilcoxon vs 0.5 p={wilcoxon(shifted - 0.5).pvalue:.4g})  mean pooled val acc {val:.4f}")
