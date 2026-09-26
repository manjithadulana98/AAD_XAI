# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT — DTU Real-Data Smoke Test (Kaggle)
#
# Validates the new stimulus-conditioned graph-attention model
# (`aad_xai.models.sgat.StimulusGAT`) against *real* DTU EEG+audio data
# end-to-end: raw `.mat` parsing, EEG/audio envelope alignment
# (`PreprocessConfig`), leakage-safe subject-independent splitting, and the
# model's forward/backward pass on real 64-channel DTU windows.
#
# This is a **smoke test, not a benchmark result** -- it trains on a handful
# of subjects for a few epochs to validate the pipeline quickly. For a real
# accuracy number, scale `N_SUBJECTS`/`EPOCHS` up or run the full
# `aad_xai.run_experiments` CV harness with `--model sgat`.
#
# **Kaggle setup requirements**
# - Enable Internet in notebook settings (for git clone + pip install)
# - Attach dataset `dulanamanjitha/aad-xai-artifacts` (holds DTU EEG + Audio)
# - GPU optional -- helps, not required at this smoke scale
#
# **Known open item** (see the project plan): the default electrode
# adjacency `StimulusGAT` builds (`config/dtu_channel_montage.csv`,
# distance-based k=6) has not yet been verified against this dataset's true
# native channel order. This notebook still runs and trains regardless --
# an unverified spatial *prior* affects which channel pairs the GAT can
# attend between, not whether the model runs -- but treat any channel-level
# interpretability from a run built on this notebook as provisional until
# that verification (`stgcn/channel_order_check.py`-style, against a real
# `S1.mat`'s `data.dim.chan.eeg`) is done.

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
    # A kernel restart (as opposed to a fresh session) can leave REPO_DIR
    # sitting on disk from a previous run -- pull instead of silently
    # training against a stale checkout.
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

# Belt-and-suspenders: `pip install -e .` should put `aad_xai` on sys.path,
# but a running kernel doesn't always pick up a package installed mid-session
# (this bit a real user on the first run of this notebook). Insert `src/`
# directly, matching kaggle_train_stgcn_gcn_only.py's own precedent.
_src_dir = os.path.join(REPO_DIR, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import aad_xai  # noqa: F401 -- fail fast here, not three cells later
print(f"aad_xai importable from: {aad_xai.__file__}")
print("Setup done.")

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
    "DTU dataset not found. Attach the 'dulanamanjitha/aad-xai-artifacts' dataset "
    "to this notebook. Tried: " + ", ".join(DTU_KAGGLE_ROOT_CANDIDATES)
)
print(f"DTU dataset: {DTU_ROOT}")

_all_subject_ids = sorted(p.stem for p in (Path(DTU_ROOT) / "eeg_new").glob("S*.mat"))
print(f"Subjects found ({len(_all_subject_ids)}):", _all_subject_ids)

# %% [markdown]
# ## 3. GPU sanity check

# %%
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 4. Configuration
#
# A handful of subjects, few epochs, one window length -- fast enough to
# validate the pipeline, not to produce a headline accuracy number.
#
# Note: `DTUDataset.trials()` parses and preprocesses every subject file
# present in the attached dataset before this notebook filters down to
# `N_SUBJECTS` (the same behaviour as `evaluation/loso_runner.py`'s own
# `--subjects` flag) -- wall-clock for this step scales with how many
# subjects are attached, not with `N_SUBJECTS` alone.

# %%
N_SUBJECTS = 6          # first N subjects found under eeg_new/ (string-sorted)
WINDOW_S = 5.0          # matches the project's primary decision window (WindowConfig.primary_s)
EPOCHS = 5
SEED = 42
OUTPUT_DIR = "/kaggle/working/results_sgat_dtu_smoke"

smoke_subjects = _all_subject_ids[:N_SUBJECTS]
print(f"Using {len(smoke_subjects)} of {len(_all_subject_ids)} subjects: {smoke_subjects}")

# %% [markdown]
# ## 5. Load DTU (subject subset), leakage-safe split, window
#
# Reuses the library's own pipeline unmodified: `DTUDataset` (raw `.mat` +
# `.wav` parsing, EEG/audio alignment via `PreprocessConfig`),
# `aad_xai.train._prepare_data` (subject-independent split + leakage
# assertions + windowing + cross-split-overlap assertion). `_SubjectSubset`
# below is a thin, notebook-local filter -- not a change to the dataset
# pipeline itself.

# %%
from dataclasses import asdict

from aad_xai.config import RunConfig, PreprocessConfig, TrainConfig, SplitConfig, WindowConfig
from aad_xai.data.base import BaseDataset
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.train import _prepare_data, train_deep
from aad_xai.utils.logging import get_run_dir, log_run_metadata


class _SubjectSubset(BaseDataset):
    """Restrict an existing BaseDataset to a fixed subject list, for a fast
    smoke run. Does not change how `inner` parses/preprocesses trials."""

    def __init__(self, inner: BaseDataset, subjects: list[str]):
        self.inner = inner
        self.subjects = set(subjects)

    def trials(self):
        return (t for t in self.inner.trials() if t.subject_id in self.subjects)


cfg = RunConfig(
    preprocess=PreprocessConfig(),
    window=WindowConfig(),
    split=SplitConfig(seed=SEED),
    train=TrainConfig(model="sgat", epochs=EPOCHS, num_seeds=1, device="cuda"),
    dataset="dtu",
    dataset_root=DTU_ROOT,
    output_dir=OUTPUT_DIR,
)

_full_dtu = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=cfg.preprocess)
dataset = _SubjectSubset(_full_dtu, smoke_subjects)

trials, split, ds_train, ds_val, ds_test = _prepare_data(dataset, cfg, WINDOW_S)

# %% [markdown]
# ## 6. Train StimulusGAT for a few epochs

# %%
run_dir = get_run_dir(cfg.output_dir, cfg.train.model, SEED, WINDOW_S)
log_run_metadata(
    run_dir,
    split={"train": split.train, "val": split.val, "test": split.test},
    window_s=WINDOW_S, seed=SEED, config=asdict(cfg),
)

result = train_deep(cfg, ds_train, ds_val, ds_test, SEED, run_dir)
print("\nSmoke-test result:", result)
print(f"Checkpoint: {run_dir / 'best_model.pt'}")

# %% [markdown]
# ## 7. Next steps
#
# - This validates the real-DTU path end-to-end -- not a benchmark result
#   (too few subjects/epochs for that).
# - To scale up: raise `N_SUBJECTS` (or drop `_SubjectSubset` to use all 18
#   subjects present), raise `EPOCHS`, and/or run the full
#   `aad_xai.run_experiments --model sgat` CV harness for a proper
#   LOSO/subject-independent evaluation across all subjects.
# - Resolve the channel-order-vs-montage open item (see cell 1's note)
#   before treating any channel-level attention/interpretability output from
#   a scaled-up run as trustworthy.
