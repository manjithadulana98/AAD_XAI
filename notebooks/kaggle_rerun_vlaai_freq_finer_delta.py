# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VLAAI — finer-resolution delta-band rerun
#
# **Why this notebook exists.** Same motivation as
# `kaggle_rerun_aadnet_freq_finer_delta.py` for AADNet: every delta-band
# (0.5-4 Hz) row in `subject_level_roi_frequency_stats.csv` carries a
# caution about limited frequency resolution, applied uniformly regardless
# of window length. This tests it directly by splitting delta into three
# narrower sub-bands --
#
#   delta_1: 0.5-1.5 Hz   delta_2: 1.5-2.5 Hz   delta_3: 2.5-4.0 Hz
#
# -- while theta/alpha/beta stay exactly as they were.
#
# **Targeted rerun, not a full pipeline run.** VLAAI's real Kaggle notebook
# (`kaggle_run_xai.py`) invokes the ENTIRE `scripts/run_focused_xai.py` as a
# subprocess -- occlusion, permutation, IG, architecture ablation, ROI-group
# ablation, top-k ablation, sanity checks, faithfulness curves, *and*
# Section H.6 (the ROI x frequency stats this notebook cares about) -- all
# in one run. None of the other sections read `BANDS`, so redoing all of
# them just to get a different frequency-band split would be wasted compute.
# Instead, this notebook imports `scripts/run_focused_xai.py` as a module,
# monkey-patches its module-level `BANDS` OrderedDict to the 6-band scheme
# above, and calls `run_subject_level_roi_frequency_stats(...)` directly with
# the same model, same data, and the same arguments the real run used.
#
# **Simpler than the AADNet version of this notebook**: VLAAI is a single
# pretrained model (`models/vlaai.h5`) evaluated once over all windows, not
# 144 per-subject-per-fold checkpoints -- no GCS auth, no per-fold model
# reloading. Its data (`data/vlaai_dtu_npz/`) and model are both already in
# the git repo, so this doesn't even need the Kaggle dataset attached
# (matching `kaggle_run_xai.py`'s own comment on that point).
#
# **Matches the original run's settings** (`kaggle_run_xai.py`): all
# available windows (`--max-samples -1`, ~8,100 windows/18 subjects),
# `n_boot=2000`, `seed=42`, `fdr_alpha=0.05` -- so the only difference from
# the existing `subject_level_roi_frequency_stats.csv` is the band split.
#
# **Kaggle setup requirements:** Internet enabled (git clone + pip install
# only). GPU accelerator recommended but not required-required -- VLAAI is a
# single small forward pass per window, not per-fold retraining. No Kaggle
# Secret needed, no dataset attachment needed.
#
# Output: `/kaggle/working/xai_results_vlaai_freq_finer_delta/subject_level_roi_frequency_stats_finer_delta.csv`.

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

sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))

print("Setup done.")

# %% [markdown]
# ## 2. GPU sanity check

# %%
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 3. Configuration + verify local data/model files

# %%
from pathlib import Path
import time

RANDOM_SEED = 42
N_BOOT = 2000
FDR_ALPHA = 0.05

DATA_DIR = os.path.join(REPO_DIR, "data", "vlaai_dtu_npz")
H5_PATH = os.path.join(REPO_DIR, "models", "vlaai.h5")
MONTAGE_PATH = os.path.join(REPO_DIR, "config", "dtu_channel_montage.csv")

assert os.path.isdir(DATA_DIR), f"Missing data dir: {DATA_DIR}"
assert os.path.isfile(H5_PATH), f"Missing model: {H5_PATH}"
assert os.path.isfile(MONTAGE_PATH), f"Missing montage: {MONTAGE_PATH}"

OUT_DIR = Path("/kaggle/working/xai_results_vlaai_freq_finer_delta")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}  (scoped rerun -- NOT xai_results/)")

import numpy as np
import random
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# %% [markdown]
# ## 4. Import `run_focused_xai` as a module and patch `BANDS`
#
# `run_subject_level_roi_frequency_stats` reads the module-level `BANDS`
# OrderedDict directly (not a function argument), so patching it before
# calling the function is how the 6-band scheme takes effect.

# %%
from collections import OrderedDict
import run_focused_xai as rfx

ORIGINAL_BANDS = dict(rfx.BANDS)
print(f"Original BANDS: {ORIGINAL_BANDS}")

rfx.BANDS = OrderedDict([
    ("delta_1", (0.5, 1.5)),
    ("delta_2", (1.5, 2.5)),
    ("delta_3", (2.5, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 13.0)),
    ("beta", (13.0, 30.0)),
])
print(f"Patched BANDS: {dict(rfx.BANDS)}")
assert len(rfx.BANDS) == 6

# %% [markdown]
# ## 5. Load montage, data, and model — identical to `run_focused_xai.py::main()`
#
# Replicates exactly the loading section of `main()` (montage, all windows
# via `VLAAIDTUDataset`, the pretrained model + `AADDecisionEEGOnly`
# wrapper) so `decision`/`eeg_all`/`att_all`/`unatt_all`/`selected_subject_ids`
# match what the real run used -- everything after that point in `main()`
# (occlusion, permutation, IG, ROI-group/top-k ablation, sanity checks,
# faithfulness curves) is NOT reused code here because it's simply not
# needed to get to Section H.6, not because of any equivalence assumption.

# %%
montage = rfx.load_montage(MONTAGE_PATH)
print(f"Montage: {len(montage['rois'])} ROIs: {list(montage['rois'])}")

print("\nLoading dataset (all windows, matching the original --max-samples -1)...")
from aad_xai.data.vlaai_dataset import VLAAIDTUDataset

ds = VLAAIDTUDataset(data_dir=DATA_DIR, window_length=320, hop=64, subjects=None)
N = len(ds)
selected_indices = list(range(N))
selected_subject_ids = np.asarray([ds.subject_ids[i] for i in selected_indices])
n_subjects = len(set(selected_subject_ids.tolist()))
print(f"Total windows: {N}  across {n_subjects} subjects")

t_load_start = time.time()
eeg_all = torch.stack([ds[i][0] for i in selected_indices]).to(DEVICE)
att_all = torch.stack([ds[i][1] for i in selected_indices]).to(DEVICE)
unatt_all = torch.stack([ds[i][2] for i in selected_indices]).to(DEVICE)
print(f"Windows loaded in {time.time() - t_load_start:.1f}s. eeg_all shape: {tuple(eeg_all.shape)}")

print("\nLoading VLAAI model...")
from aad_xai.models import VLAAIPyTorch, AADDecisionEEGOnly

model = VLAAIPyTorch.from_h5(H5_PATH)
model.eval().to(DEVICE)

decision = AADDecisionEEGOnly(model)
decision.eval().to(DEVICE)

decision.set_envelopes(att_all[:3], unatt_all[:3])
with torch.no_grad():
    test_logits = decision(eeg_all[:3])
print(f"Smoke-test decision logits: {test_logits[0].detach().cpu().numpy()}")

decision.set_envelopes(att_all, unatt_all)
print("Model loaded and verified.")

# %% [markdown]
# ## 6. Run Section H.6 only (the 6-band frequency stats)

# %%
t_start = time.time()
roi_freq_subj_stats = rfx.run_subject_level_roi_frequency_stats(
    decision, eeg_all, att_all, unatt_all,
    selected_subject_ids, montage,
    OUT_DIR, FDR_ALPHA, N_BOOT, RANDOM_SEED,
)
t_total = time.time() - t_start
print(f"\nTotal wall-clock time: {t_total:.1f}s ({t_total / 60:.1f} min)")

# %% [markdown]
# ## 7. Verify + rename output, write timing record
#
# `run_subject_level_roi_frequency_stats` writes `subject_level_roi_frequency_stats.csv`
# directly into `out_dir`; renamed here to make the 6-band scheme unambiguous
# in the filename itself.

# %%
import pandas as pd
import json

written_path = OUT_DIR / "subject_level_roi_frequency_stats.csv"
final_path = OUT_DIR / "subject_level_roi_frequency_stats_finer_delta.csv"
assert written_path.exists(), f"Expected output not found: {written_path}"
written_path.rename(final_path)

df = pd.read_csv(final_path)
roi_values = sorted(df["roi"].unique().tolist())
band_values = sorted(df["band"].unique().tolist())
print(f"ROIs present ({len(roi_values)}): {roi_values}")
print(f"Bands present ({len(band_values)}): {band_values}")
assert len(band_values) == 6, f"Expected 6 bands, got {len(band_values)}: {band_values}"
n_sig = int(df["fdr_sig"].sum())
print(f"FDR-significant ROI x band combinations: {n_sig}/{len(df)}")

with open(OUT_DIR / "rerun_timing.json", "w") as f:
    json.dump({
        "total_seconds": t_total,
        "total_minutes": t_total / 60,
        "n_windows": N,
        "n_subjects": n_subjects,
        "n_rois": len(montage["rois"]),
        "n_bands": len(rfx.BANDS),
        "bands_config": {k: v for k, v in rfx.BANDS.items()},
        "n_boot": N_BOOT,
        "fdr_alpha": FDR_ALPHA,
        "seed": RANDOM_SEED,
        "purpose": "finer delta-band resolution test (0.5-1.5 / 1.5-2.5 / 2.5-4 Hz), theta/alpha/beta unchanged",
        "montage_file": "config/dtu_channel_montage.csv",
    }, f, indent=2)

print(f"\nWritten to {final_path}")
print(f"Timing/config record: {OUT_DIR / 'rerun_timing.json'}")
