# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VLAAI — global (all-channel) frequency-band ablation
#
# **Why this notebook is different from `kaggle_rerun_vlaai_freq_finer_delta.py`.**
# That notebook ablates a band only within one ROI's channels at a time (9
# ROIs x 6 bands = 54 numbers). This notebook answers a simpler, different
# question: **remove one frequency band from all 64 channels at once, and
# see whether that band matters to the decision at all** -- one number per
# band, 6 numbers total, no ROI dimension.
#
# Same 6-band split as the finer-delta notebook:
#   delta_1: 0.5-1.5 Hz   delta_2: 1.5-2.5 Hz   delta_3: 2.5-4.0 Hz
#   theta: 4-8 Hz   alpha: 8-13 Hz   beta: 13-30 Hz
#
# **Implementation note:** `run_subject_level_roi_frequency_stats` only ever
# reads `montage["rois"]` (a dict of group-name -> channel-index list) --
# nothing else in the function depends on it being a real anatomical ROI
# grouping. So instead of writing new statistical code, this notebook loads
# the real montage (for the channel count) and then overrides just the
# `"rois"` key with a single synthetic group spanning all 64 channels before
# calling the exact same function used by the ROI-wise notebooks. Same
# reuse-everything-else approach otherwise (single pretrained model, all
# windows, no GCS/dataset-attach needed).
#
# **Kaggle setup requirements:** Internet enabled (git clone + pip install
# only). No GPU strictly required, no Kaggle Secret, no dataset attachment.
#
# Output: `/kaggle/working/xai_results_vlaai_freq_global/frequency_global_subject.csv`
# (schema: `subject,band,mean_dp,n_windows` -- `roi` column will read
# "AllChannels" for every row, dropped for clarity in the final CSV).

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

OUT_DIR = Path("/kaggle/working/xai_results_vlaai_freq_global")
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
# ## 4. Import `run_focused_xai` as a module, patch `BANDS`, and build a
#    single all-channel "ROI" group

# %%
from collections import OrderedDict
import run_focused_xai as rfx

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
# ## 5. Load montage (for channel count only), data, and model
#
# Loads the real montage first (so N_CHANNELS is read from real data, not
# hardcoded), then overrides `montage["rois"]` with a single group spanning
# every channel -- see markdown note above on why this is safe.

# %%
montage = rfx.load_montage(MONTAGE_PATH)
n_channels_real = sum(len(chs) for chs in montage["rois"].values())
print(f"Real montage: {len(montage['rois'])} ROIs, {n_channels_real} channels total")

montage["rois"] = OrderedDict([("AllChannels", list(range(n_channels_real)))])
print(f"Overridden montage['rois']: {list(montage['rois'].keys())} "
      f"({len(montage['rois']['AllChannels'])} channels in the one group)")

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
# ## 6. Run the global (all-channel) per-band ablation
#
# Same function as the ROI-wise notebooks -- just 1 group x 6 bands = 6
# combos instead of 9 x 6 = 54, so this should run considerably faster.

# %%
t_start = time.time()
freq_stats = rfx.run_subject_level_roi_frequency_stats(
    decision, eeg_all, att_all, unatt_all,
    selected_subject_ids, montage,
    OUT_DIR, FDR_ALPHA, N_BOOT, RANDOM_SEED,
)
t_total = time.time() - t_start
print(f"\nTotal wall-clock time: {t_total:.1f}s ({t_total / 60:.1f} min)")

# %% [markdown]
# ## 7. Verify + clean up output, write timing record

# %%
import pandas as pd
import json

written_path = OUT_DIR / "subject_level_roi_frequency_stats.csv"
final_path = OUT_DIR / "frequency_global_subject.csv"
assert written_path.exists(), f"Expected output not found: {written_path}"

df = pd.read_csv(written_path)
assert set(df["roi"].unique()) == {"AllChannels"}, f"Expected only 'AllChannels', got {df['roi'].unique()}"
df = df.drop(columns=["roi"])  # only one value by construction, not informative as a column
df.to_csv(final_path, index=False)
written_path.unlink()

band_values = sorted(df["band"].unique().tolist())
print(f"Bands present ({len(band_values)}): {band_values}")
assert len(band_values) == 6, f"Expected 6 bands, got {len(band_values)}: {band_values}"
n_sig = int(df["fdr_sig"].sum())
print(f"FDR-significant bands: {n_sig}/{len(df)}")
print(df.to_string(index=False))

with open(OUT_DIR / "rerun_timing.json", "w") as f:
    json.dump({
        "total_seconds": t_total,
        "total_minutes": t_total / 60,
        "n_windows": N,
        "n_subjects": n_subjects,
        "n_channels_ablated_together": n_channels_real,
        "n_bands": len(rfx.BANDS),
        "bands_config": {k: v for k, v in rfx.BANDS.items()},
        "n_boot": N_BOOT,
        "fdr_alpha": FDR_ALPHA,
        "seed": RANDOM_SEED,
        "purpose": "global (all-channel) per-band ablation -- no ROI dimension",
        "montage_file": "config/dtu_channel_montage.csv",
    }, f, indent=2)

print(f"\nWritten to {final_path}")
print(f"Timing/config record: {OUT_DIR / 'rerun_timing.json'}")
