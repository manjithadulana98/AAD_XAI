# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT -- Envelope Informativeness Check (Kaggle)
#
# Three LOSO runs (v1 default schedule, v2 aggressive schedule, v3 FiLM-init
# fix) all landed at chance (0.48-0.50). The overfit sanity check reaching
# 100% train accuracy on 16 windows doesn't actually rule out "the envelope
# carries no usable signal through this pipeline" -- an overparameterized
# model can memorize 16 arbitrarily-labeled examples regardless of whether
# the features are meaningful (this is a well-known failure mode, not
# evidence the model is using eeg/env correctly).
#
# This repo's TRF baseline DOES decode DTU above chance (60-85% at 5s,
# see `results_trf_vandecappelle2021_linear/`) -- but via a DIFFERENT data
# path: `run_experiments.py`'s `_train_trf_fold`/`_aad_decoding_accuracy`
# slice `Trial.eeg`/`audio_a`/`audio_b` directly. `StimulusGAT` trains on
# `WindowedEEGAudioDataset` instead -- a different windowing/tensor-assembly
# code path over the same underlying `Trial` objects. TRF's success proves
# the RAW per-trial data carries signal; it does NOT prove
# `WindowedEEGAudioDataset`'s windowing/alignment preserves that signal for
# deep models.
#
# This notebook isolates exactly that variable: fit the same TRF method,
# but source its (eeg, attended_env) pairs through `WindowedEEGAudioDataset`
# instead of TRF's own path, then evaluate the standard way (correlate
# predicted envelope against both candidates, pick higher).
#
# - **If this also lands near chance**: real evidence of a bug specific to
#   `WindowedEEGAudioDataset` (windowing/alignment) -- would affect
#   `StimulusGAT` AND `ExternalAADNet` (the other `WindowedEEGAudioDataset`
#   consumer), not a StimulusGAT-architecture problem.
# - **If this reproduces TRF's known ~60-85% range**: the pipeline is fine;
#   the problem is specific to `StimulusGAT`'s architecture/training, not
#   the data it's fed.
#
# This is a quick diagnostic (ridge regression, CPU-only, no deep training,
# a handful of subjects) -- not a full LOSO sweep. A clear separation from
# chance either way is all that's needed to decide where to look next.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - No GPU needed (ridge regression is CPU-only, like the TRF baseline)

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
# ## 2. Resolve the DTU dataset path

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

_all_subject_ids = sorted(p.stem for p in (Path(DTU_ROOT) / "eeg_new").glob("S*.mat"))
print(f"Subjects found ({len(_all_subject_ids)}):", _all_subject_ids)

# %% [markdown]
# ## 3. Load via WindowedEEGAudioDataset (the StimulusGAT data path)
#
# Same `_prepare_data` call StimulusGAT training uses -- `cfg.train.model =
# "sgat"` forces `WindowedEEGAudioDataset` (not TRF's own raw-trial-slicing
# path). N_SUBJECTS=12 gives a reasonable train/val/test split quickly
# without a full 18-subject sweep.

# %%
from dataclasses import asdict

from aad_xai.config import RunConfig, PreprocessConfig, TrainConfig, SplitConfig, WindowConfig
from aad_xai.data.base import BaseDataset
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.train import _prepare_data

N_SUBJECTS = 12
WINDOW_S = 5.0
SEED = 42

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
    train=TrainConfig(model="sgat", epochs=1, num_seeds=1, device="cpu"),  # model="sgat" only to select WindowedEEGAudioDataset
    dataset="dtu",
    dataset_root=DTU_ROOT,
    output_dir="/kaggle/working/results_envelope_diag",
)

_full_dtu = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=cfg.preprocess)
dataset = _SubjectSubset(_full_dtu, diag_subjects)
trials, split, ds_train, ds_val, ds_test = _prepare_data(dataset, cfg, WINDOW_S)
print(f"Windows: train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")
print(f"Test subjects: {split.test}")

# %% [markdown]
# ## 4. Fit a TRF decoder on WindowedEEGAudioDataset-sourced windows
#
# Same method as `train.py::train_trf` (ridge, tmin=0, tmax=0.4s, alpha=100)
# -- but (eeg, attended_env) pairs come from `ds_train[i]` (the
# `WindowedEEGAudioDataset` tensors `StimulusGAT` actually trains on), not
# from re-slicing `Trial.eeg`/`audio_a`/`audio_b` directly.

# %%
import numpy as np

from aad_xai.models.trf_baseline import TRFDecoder, _safe_corrcoef

SFREQ = cfg.preprocess.sfreq_out

eeg_segs, env_segs = [], []
for i in range(len(ds_train)):
    eeg_i, env_i, label_i = ds_train[i]
    eeg_np = eeg_i.numpy()                                  # (C, T)
    env_np = env_i.numpy()                                  # (2, T) -- [0]=audio_a, [1]=audio_b
    attended_env = env_np[0] if int(label_i) == 0 else env_np[1]
    eeg_segs.append(eeg_np)
    env_segs.append(attended_env.astype(np.float32))

eeg_cat = np.concatenate(eeg_segs, axis=1)     # (C, total_T)
env_cat = np.concatenate(env_segs, axis=0)     # (total_T,)
print(f"Training on {len(eeg_segs)} windows, {eeg_cat.shape[1]} total samples "
      f"({eeg_cat.shape[1] / SFREQ:.0f}s of concatenated EEG)")

decoder = TRFDecoder(tmin_s=0.0, tmax_s=0.4, alpha=100.0)
decoder.fit(eeg_cat, env_cat, SFREQ)
print("TRF decoder fit complete.")

# %% [markdown]
# ## 5. Evaluate: per-window accuracy on held-out test windows
#
# For each test window, predict an envelope from EEG, correlate it against
# BOTH candidate envelopes (also sourced from the same `ds_test[i]` tensor,
# not re-sliced from the raw trial), and pick the higher-correlation one.

# %%
correct, total = 0, 0
for i in range(len(ds_test)):
    eeg_i, env_i, label_i = ds_test[i]
    eeg_np = eeg_i.numpy()
    env_np = env_i.numpy()
    pred_env = decoder.predict(eeg_np)
    corr_a = _safe_corrcoef(pred_env, env_np[0])
    corr_b = _safe_corrcoef(pred_env, env_np[1])
    pred_label = 0 if corr_a > corr_b else 1
    correct += int(pred_label == int(label_i))
    total += 1

acc = correct / total if total else 0.0
print(f"\nTRF-on-WindowedEEGAudioDataset-tensors test accuracy: {acc:.4f} ({correct}/{total} windows)")
print(f"Test subjects: {split.test}")

# %% [markdown]
# ## 6. Interpretation
#
# For reference: this repo's own TRF LOSO baseline (via the OTHER data path)
# gets 0.6809 +/- 0.0849 at 10s windows, 0.6328 +/- 0.0609 at 5s windows
# (`RESULTS_LOG.txt`, `results_trf_vandecappelle2021_linear/`). Chance is 0.5.
# This run uses fewer subjects (12 vs. 18) and a single split (not LOSO), so
# expect more noise than those numbers -- but a clear separation from chance
# either way is the signal we need.

# %%
print("Decision:")
if acc > 0.58:
    print(f"  acc={acc:.4f} -- clearly above chance, consistent with TRF's known DTU performance.")
    print("  WindowedEEGAudioDataset's windowing/alignment is NOT the problem.")
    print("  -> The issue is specific to StimulusGAT's architecture/training, not the data pipeline.")
    print("  -> Next: adjacency verification, or deeper architecture/training investigation.")
elif acc < 0.53:
    print(f"  acc={acc:.4f} -- indistinguishable from chance, despite TRF succeeding on the same")
    print("  underlying trials via a DIFFERENT data path.")
    print("  -> Real evidence of a bug in WindowedEEGAudioDataset (windowing/alignment) that doesn't")
    print("     affect TRF's own raw-trial-slicing path. Would also affect ExternalAADNet.")
    print("  -> Next: audit WindowedEEGAudioDataset's window index <-> eeg/audio_a/audio_b alignment")
    print("     directly against DTUDataset's trial construction.")
else:
    print(f"  acc={acc:.4f} -- ambiguous given the small, non-LOSO sample here.")
    print("  -> Re-run with more subjects or as a proper LOSO sweep before concluding either way.")
