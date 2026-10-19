# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT -- Envelope Informativeness Check, Full 18-Subject LOSO (Kaggle)
#
# Follow-up to `kaggle_diag_envelope_informativeness.py`, which ran a single
# 12-subject split (2 held-out test subjects: S10, S17) and got 0.5750
# (690/1200 windows) -- directionally above chance, but statistically
# ambiguous with only 2 independent test subjects, and S10 is known to be
# one of the weakest subjects across all three StimulusGAT LOSO runs
# (0.483, 0.465, 0.463), so a 2-subject estimate could easily be unlucky.
#
# This version runs the same method -- TRF fit via `WindowedEEGAudioDataset`-
# sourced windows (not TRF's own raw-trial-slicing path) -- across all 18
# subjects, true leave-one-subject-out, matching the methodology this
# project already trusts for its real TRF baseline (0.6328 +/- 0.0609 at 5s,
# `RESULTS_LOG.txt`).
#
# - **If mean accuracy lands near TRF's known ~0.63**: WindowedEEGAudioDataset
#   is NOT the problem. The issue is specific to StimulusGAT's architecture/
#   training.
# - **If mean accuracy lands near 0.50**: real evidence of a
#   WindowedEEGAudioDataset-specific bug (would also affect ExternalAADNet).
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - No GPU needed (ridge regression, CPU-only)
# - Expect ~15-25 min (one-time ~13min load of all 18 subjects + ~18 fast folds)

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
# ## 2. Resolve the DTU dataset path, load ALL 18 subjects once

# %%
from pathlib import Path
from aad_xai.config import PreprocessConfig
from aad_xai.data.dtu_dataset import DTUDataset

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

WINDOW_S = 5.0
SEED = 42
preprocess = PreprocessConfig()
SFREQ = preprocess.sfreq_out

import time as _time
_t0 = _time.time()
_full_dtu = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=preprocess)
all_trials = list(_full_dtu.trials())
print(f"Loaded {len(all_trials)} trials across "
      f"{len({t.subject_id for t in all_trials})} subjects in {_time.time() - _t0:.0f}s")

# %% [markdown]
# ## 3. Full 18-fold LOSO: TRF fit via WindowedEEGAudioDataset-sourced windows
#
# Same ridge-TRF method as `train.py::train_trf` (tmin=0, tmax=0.4s,
# alpha=100) -- the only thing varied across the whole investigation is
# which code path extracts (eeg, env) windows from the Trial objects.

# %%
import numpy as np

from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.data.cv_splits import leave_one_subject_out
from aad_xai.models.trf_baseline import TRFDecoder, _safe_corrcoef

folds = list(leave_one_subject_out(all_trials, seed=SEED))
print(f"{len(folds)} LOSO folds (1 per subject)")

fold_results = []
for fi, fold in enumerate(folds):
    test_subject = fold.meta["test_subject"]
    train_trials = [all_trials[i] for i in fold.train_idx]
    test_trials = [all_trials[i] for i in fold.test_idx]

    ds_tr = WindowedEEGAudioDataset(train_trials, window_s=WINDOW_S, overlap_s=0.0)
    ds_te = WindowedEEGAudioDataset(test_trials, window_s=WINDOW_S, overlap_s=0.0)

    if len(ds_tr) == 0 or len(ds_te) == 0:
        print(f"  [{fi+1}/{len(folds)}] {test_subject}: skip (empty dataset: "
              f"train={len(ds_tr)} test={len(ds_te)})")
        continue

    eeg_segs, env_segs = [], []
    for i in range(len(ds_tr)):
        eeg_i, env_i, label_i = ds_tr[i]
        env_np = env_i.numpy()
        attended_env = env_np[0] if int(label_i) == 0 else env_np[1]
        eeg_segs.append(eeg_i.numpy())
        env_segs.append(attended_env.astype(np.float32))
    eeg_cat = np.concatenate(eeg_segs, axis=1)
    env_cat = np.concatenate(env_segs, axis=0)

    decoder = TRFDecoder(tmin_s=0.0, tmax_s=0.4, alpha=100.0)
    decoder.fit(eeg_cat, env_cat, SFREQ)

    correct, total = 0, 0
    for i in range(len(ds_te)):
        eeg_i, env_i, label_i = ds_te[i]
        eeg_np = eeg_i.numpy()
        env_np = env_i.numpy()
        pred_env = decoder.predict(eeg_np)
        corr_a = _safe_corrcoef(pred_env, env_np[0])
        corr_b = _safe_corrcoef(pred_env, env_np[1])
        pred_label = 0 if corr_a > corr_b else 1
        correct += int(pred_label == int(label_i))
        total += 1

    acc = correct / total if total else 0.0
    fold_results.append({"test_subject": test_subject, "accuracy": acc, "n_windows": total})
    print(f"  [{fi+1}/{len(folds)}] {test_subject}: acc={acc:.4f} ({correct}/{total} windows)")

# %% [markdown]
# ## 4. Summary

# %%
accs = [r["accuracy"] for r in fold_results]
mean_acc = float(np.mean(accs))
std_acc = float(np.std(accs))

print(f"\nTRF-on-WindowedEEGAudioDataset-tensors, full {len(fold_results)}-fold LOSO:")
print(f"  mean +/- std: {mean_acc:.4f} +/- {std_acc:.4f}")
print(f"\nPer-fold:")
for r in sorted(fold_results, key=lambda r: r["test_subject"]):
    print(f"  {r['test_subject']}: acc={r['accuracy']:.4f} ({r['n_windows']} windows)")

print("\nFor reference:")
print("  TRF LOSO (own raw-trial-slicing path, RESULTS_LOG.txt): 0.6328 +/- 0.0609 (5s)")
print("  StimulusGAT LOSO v1 (default schedule):                 0.4887 +/- 0.0247")
print("  StimulusGAT LOSO v2 (aggressive schedule):               0.4814 +/- 0.0192")
print("  StimulusGAT LOSO v3 (FiLM-init fix):                     0.4952 +/- 0.0261")
print(f"  TRF via WindowedEEGAudioDataset (THIS RUN):              {mean_acc:.4f} +/- {std_acc:.4f}")

# %% [markdown]
# ## 5. Decision

# %%
print("Decision:")
if mean_acc > 0.58:
    print(f"  mean={mean_acc:.4f} -- reproduces TRF's known DTU performance via the SAME data path")
    print("  StimulusGAT trains on. WindowedEEGAudioDataset is NOT the problem.")
    print("  -> The issue is specific to StimulusGAT's architecture/training, not the data pipeline.")
    print("  -> Next: adjacency verification, or deeper architecture/training investigation")
    print("     (e.g. speech-encoder capacity, regularization, learning-rate schedule per-component).")
elif mean_acc < 0.53:
    print(f"  mean={mean_acc:.4f} -- indistinguishable from chance across all 18 subjects, despite TRF")
    print("  succeeding on the same underlying trials via its own raw-trial-slicing path.")
    print("  -> Strong evidence of a bug in WindowedEEGAudioDataset (windowing/alignment) that doesn't")
    print("     affect TRF's own path. Would also affect ExternalAADNet.")
    print("  -> Next: audit WindowedEEGAudioDataset's window index <-> eeg/audio_a/audio_b alignment")
    print("     directly against DTUDataset's trial construction and make_windows().")
else:
    print(f"  mean={mean_acc:.4f} -- partial signal, below TRF's own-path performance but clearly above")
    print("  chance. WindowedEEGAudioDataset likely preserves SOME signal but perhaps degrades it")
    print("  (e.g. different window boundary handling, min() length truncation across eeg/audio_a/audio_b).")
    print("  -> Worth auditing WindowedEEGAudioDataset for a partial degradation, in parallel with")
    print("     investigating StimulusGAT-specific causes.")
