# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT v4 (Time-Resolved FiLM) -- Full 18-Subject LOSO vs. Paired TRF (Kaggle)
#
# v1-v3 LOSO were all at chance (0.4887 / 0.4814 / 0.4952). Commit d00c768
# conditions EEG on speech before time-pooling; on a single 14-subject split
# it reached test 0.613 at the best-validation epoch (train 0.50 -> 0.82).
#
# Per fold (same folds as the TRF informativeness diagnostic:
# `leave_one_subject_out(all_trials, seed=42)`):
# 1. Train StimulusGAT, keep the best-validation-accuracy checkpoint.
# 2. Test accuracy.
# 3. **Alignment control**: test again with both envelopes circularly shifted
#    by 2.5 s. Every time-averaged statistic of the speech is unchanged but
#    its alignment with the EEG is broken. If the model decodes attention by
#    tracking, accuracy should fall to chance; if it stays up, the model is
#    using a shortcut (e.g. candidate A is always the male speaker).
# 4. TRF (tmin 0, tmax 0.4 s, alpha 100) on the same fold -> paired comparison.
# 5. Save the checkpoint for the interpretability pipeline.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - GPU required (~2-3 h)

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
print(f"aad_xai importable from: {aad_xai.__file__}")
print(f"Commit: {GIT_COMMIT}")

# %% [markdown]
# ## 2. Confirm the time-resolved FiLM model is live

# %%
import torch
from aad_xai.models.sgat import StimulusGAT
from aad_xai.models.sgat_film import TimeResolvedFiLM

_probe = StimulusGAT(n_channels=4, sfreq=64.0, adjacency=torch.eye(4))
assert isinstance(getattr(_probe, "film", None), TimeResolvedFiLM), "TimeResolvedFiLM NOT live."
assert _probe.readout.bias is None, "Bias-free readout NOT live."
print("Time-resolved FiLM model confirmed live.")

# %% [markdown]
# ## 3. Load all subjects

# %%
import time
from pathlib import Path

from aad_xai.config import PreprocessConfig
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
SEED = 42
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
SHIFT_S = 2.5
OUT_DIR = Path("/kaggle/working/results_sgat_v4_loso")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

preprocess = PreprocessConfig()
SFREQ = preprocess.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)

_t0 = time.time()
all_trials = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=preprocess).trials())
print(f"Loaded {len(all_trials)} trials, {len({t.subject_id for t in all_trials})} subjects "
      f"in {time.time() - _t0:.0f}s")

# %% [markdown]
# ## 4. Per-fold helpers

# %%
import copy

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.data.cv_splits import leave_one_subject_out
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.models.trf_baseline import TRFDecoder, _safe_corrcoef
from aad_xai.utils.seed import seed_everything


def _sgat_accuracy(model, loader, shift_samples=0):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, envb, yb in loader:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            if shift_samples:
                envb = torch.roll(envb, shifts=shift_samples, dims=-1)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            correct += (model(xb, envb).argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / max(total, 1)


def train_sgat_fold(ds_tr, ds_va, ds_te, n_channels):
    seed_everything(SEED)
    model = StimulusGAT(n_channels=n_channels, sfreq=SFREQ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best = {"val_acc": -1.0}
    curve = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct, total = 0, 0
        for xb, envb, yb in dl_tr:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, envb)
            loss_fn(logits, yb).backward()
            opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        val_acc = _sgat_accuracy(model, dl_va)
        curve.append({"epoch": epoch, "train_acc": correct / total, "val_acc": val_acc})
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}

    model.load_state_dict(best["state"])
    return model, {
        "best_epoch": best["epoch"], "val_acc": best["val_acc"],
        "test_acc": _sgat_accuracy(model, dl_te),
        "test_acc_shifted": _sgat_accuracy(model, dl_te, shift_samples=SHIFT_SAMPLES),
        "final_train_acc": curve[-1]["train_acc"], "curve": curve,
    }


def trf_fold(train_trials, test_trials):
    ds_tr = WindowedEEGAudioDataset(train_trials, window_s=WINDOW_S, overlap_s=0.0)
    ds_te = WindowedEEGAudioDataset(test_trials, window_s=WINDOW_S, overlap_s=0.0)
    eeg_segs, env_segs = [], []
    for i in range(len(ds_tr)):
        eeg_i, env_i, label_i = ds_tr[i]
        eeg_segs.append(eeg_i.numpy())
        env_segs.append(env_i.numpy()[int(label_i)].astype(np.float32))
    decoder = TRFDecoder(tmin_s=0.0, tmax_s=0.4, alpha=100.0)
    decoder.fit(np.concatenate(eeg_segs, axis=1), np.concatenate(env_segs, axis=0), SFREQ)
    correct = 0
    for i in range(len(ds_te)):
        eeg_i, env_i, label_i = ds_te[i]
        pred = decoder.predict(eeg_i.numpy())
        env_np = env_i.numpy()
        correct += int((0 if _safe_corrcoef(pred, env_np[0]) > _safe_corrcoef(pred, env_np[1]) else 1) == int(label_i))
    return correct / max(len(ds_te), 1)

# %% [markdown]
# ## 5. Full LOSO

# %%
import json

folds = list(leave_one_subject_out(all_trials, seed=SEED))
print(f"{len(folds)} LOSO folds")

fold_results = []
for fi, fold in enumerate(folds):
    t0 = time.time()
    test_subject = fold.meta["test_subject"]
    tr = [all_trials[i] for i in fold.train_idx]
    va = [all_trials[i] for i in fold.val_idx]
    te = [all_trials[i] for i in fold.test_idx]

    ds_tr, ds_va, ds_te = (WindowedEEGAudioDataset(t, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=True)
                           for t in (tr, va, te))
    model, r = train_sgat_fold(ds_tr, ds_va, ds_te, n_channels=ds_tr[0][0].shape[0])
    torch.save(model.state_dict(), CKPT_DIR / f"sgat_v4_{test_subject}.pt")
    r["trf_test_acc"] = trf_fold(tr, te)
    r.update({"test_subject": test_subject, "val_subject": va[0].subject_id, "n_test_windows": len(ds_te)})
    fold_results.append(r)

    print(f"[{fi+1:>2}/{len(folds)}] {test_subject:>4} (val {r['val_subject']:>4}): "
          f"SGAT={r['test_acc']:.4f}  shifted={r['test_acc_shifted']:.4f}  TRF={r['trf_test_acc']:.4f}  "
          f"best_ep={r['best_epoch']:>2} val={r['val_acc']:.4f} train={r['final_train_acc']:.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    with open(OUT_DIR / "loso_v4_results.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {"window_s": WINDOW_S, "epochs": EPOCHS, "lr": LR,
                   "weight_decay": WEIGHT_DECAY, "shift_s": SHIFT_S, "seed": SEED},
                   "folds": fold_results}, f, indent=2)

# %% [markdown]
# ## 6. Summary and paired statistics

# %%
from scipy.stats import wilcoxon

sgat = np.array([r["test_acc"] for r in fold_results])
shifted = np.array([r["test_acc_shifted"] for r in fold_results])
trf = np.array([r["trf_test_acc"] for r in fold_results])

print(f"\n{len(fold_results)}-fold LOSO, {WINDOW_S:.0f} s windows (mean +/- std across subjects):")
print(f"  StimulusGAT v4              {sgat.mean():.4f} +/- {sgat.std():.4f}")
print(f"  StimulusGAT v4, env shifted {shifted.mean():.4f} +/- {shifted.std():.4f}")
print(f"  TRF (same folds)            {trf.mean():.4f} +/- {trf.std():.4f}")

d = sgat - trf
print(f"\nSGAT - TRF: mean {d.mean():+.4f}, SGAT better on {int((d > 0).sum())}/{len(d)} subjects, "
      f"Wilcoxon p={wilcoxon(sgat, trf).pvalue:.4g}")
print(f"SGAT - shifted: mean {(sgat - shifted).mean():+.4f}, "
      f"Wilcoxon p={wilcoxon(sgat, shifted).pvalue:.4g}")
print(f"SGAT > 0.5: Wilcoxon (one-sided) p={wilcoxon(sgat - 0.5, alternative='greater').pvalue:.4g}")

print("\nPer subject:")
for r in sorted(fold_results, key=lambda r: int(r["test_subject"][1:])):
    print(f"  {r['test_subject']:>4}: SGAT={r['test_acc']:.4f}  shifted={r['test_acc_shifted']:.4f}  "
          f"TRF={r['trf_test_acc']:.4f}")

print("\nPrior StimulusGAT LOSO: v1 0.4887 +/- 0.0247, v2 0.4814 +/- 0.0192, v3 0.4952 +/- 0.0261")
print("Prior TRF via WindowedEEGAudioDataset LOSO: 0.6006 +/- 0.0420")
