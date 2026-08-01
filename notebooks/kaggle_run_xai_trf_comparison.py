# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TRF vs VLAAI/AADNet XAI Comparison — Kaggle Notebook
#
# Runs `scripts/run_xai_trf_comparison.py`: trains a linear backward TRF
# decoder on the same DTU windows VLAAI uses, runs the model-agnostic XAI
# sections (C, D, H, I, J) on it, applies the Haufe transform to compare
# against VLAAI's `combined_score` channel ranking (Section K, Phase 3) and,
# if AADNet's kernel output is attached, against AADNet's own
# `combined_score` ranking via name-based channel alignment (Section K2,
# Phase 3 extension), and generates a side-by-side comparison report against
# VLAAI's existing XAI results.
#
# **Kaggle setup requirements**
# - Attach the dataset `dulanamanjitha/aad-xai-artifacts` (holds the DTU
#   preprocessed `.npz` files VLAAI itself uses)
# - Add Data → Kernel Output → attach `dulanamanjitha/vlaai-xai` (needed for
#   `channel_importance.csv`, i.e. VLAAI's `combined_score` ranking)
# - Add Data → Kernel Output → attach `dulanamanjitha/aadnet-xai` (needed for
#   Section K2 — AADNet's own `channel_importance.csv` with `combined_score`,
#   from a run that post-dates the Part 0 montage-label fix. Optional: if not
#   attached, Section K2 is skipped with an explicit message, everything else
#   still runs.)
# - Enable Internet + GPU (same as the other XAI notebooks — TRF fitting
#   itself is CPU-friendly, but the shared occlusion/permutation XAI
#   machinery this script reuses is written against torch tensors, so
#   keeping GPU enabled avoids a second image-compatibility hunt)

# %% [markdown]
# ## 1. Clone repository and install dependencies

# %%
import os
REPO_DIR = "/kaggle/working/AAD_XAI"

if not os.path.exists(REPO_DIR):
    os.system("git clone https://github.com/manjithadulana98/AAD_XAI.git " + REPO_DIR)
else:
    print(f"Repository already cloned at {REPO_DIR}")

os.chdir(REPO_DIR)

# Avoid clobbering Kaggle's own GPU-matched torch build with a generic PyPI
# wheel (see notebooks/kaggle_run_xai.py for the full explanation of this
# fix). If a working torch is already importable, keep it and only install
# the rest of requirements.txt.
try:
    import torch as _torch_preinstalled
    print(f"Pre-installed torch {_torch_preinstalled.__version__} found "
          f"(CUDA available: {_torch_preinstalled.cuda.is_available()}) -- "
          "keeping it; installing the rest of requirements.txt without touching torch.")
    with open("requirements.txt") as _f:
        _reqs_no_torch = [ln for ln in _f if ln.strip() and not ln.strip().lower().startswith("torch")]
    with open("/tmp/requirements_no_torch.txt", "w") as _f:
        _f.writelines(_reqs_no_torch)
    os.system("pip install -q -r /tmp/requirements_no_torch.txt")
except ImportError:
    print("No pre-installed torch found -- installing requirements.txt as-is.")
    os.system("pip install -q -r requirements.txt")

os.system("pip install -q -e .")

# %% [markdown]
# ## 2. Locate the attached VLAAI and (optional) AADNet kernel-output data
#
# Kernel-output data sources have been observed to mount at
# `/kaggle/input/notebooks/<owner>/<slug>/`, not the flat `/kaggle/input/<slug>/`
# one might expect -- so this searches recursively for each model's
# `channel_importance.csv` rather than assuming a fixed path. Both models'
# output directories can contain a same-named `channel_importance.csv`, so
# matches are disambiguated by checking for `aadnet-xai` / `vlaai-xai` in the
# matched path (the kernel-output mount always embeds the source kernel's
# slug).
#
# Both notebooks also clone the *full* AAD_XAI git repo into their working
# directory (`REPO_DIR = "/kaggle/working/AAD_XAI"`), and the repo happens to
# have a stale, pre-existing `channel_importance.csv` committed under
# `Kaggle_results/Full run results/` (an old AADNet run that predates the
# Part 0 montage-label fix -- confirmed by its channel 13 = "Cz", channel 42
# = "FC4" labeling, the exact mislabeling that fix corrects). That stale file
# therefore shows up as a match on BOTH kernels' outputs, nested inside their
# cloned-repo copy, alongside the genuine fresh output written outside the
# repo tree (`/kaggle/working/xai_results/...` or
# `/kaggle/working/xai_results_aadnet/...`). Any match path containing
# "/AAD_XAI/" is therefore excluded -- it can only be a file committed to the
# repo itself, never a notebook's actual output directory.

# %%
import glob

_all_ci_matches = glob.glob("/kaggle/input/**/channel_importance.csv", recursive=True)
_stale_repo_matches = [p for p in _all_ci_matches if "/AAD_XAI/" in p.replace("\\", "/")]
if _stale_repo_matches:
    print(f"Ignoring {len(_stale_repo_matches)} channel_importance.csv match(es) committed "
          f"inside the cloned repo tree (not a notebook's real output): {_stale_repo_matches}")
_all_ci_matches = [p for p in _all_ci_matches if "/AAD_XAI/" not in p.replace("\\", "/")]

_vlaai_matches = [p for p in _all_ci_matches if "vlaai-xai" in p.replace("\\", "/")]
if not _vlaai_matches:
    _all_input = glob.glob("/kaggle/input/**/*", recursive=True)
    raise FileNotFoundError(
        "Could not find VLAAI's channel_importance.csv anywhere under /kaggle/input/.\n"
        f"Full contents of /kaggle/input ({len(_all_input)} entries): {_all_input[:200]}\n"
        "Check that dulanamanjitha/vlaai-xai is attached as a Kernel Output data source."
    )
VLAAI_RESULTS_DIR = os.path.dirname(_vlaai_matches[0])
print("VLAAI results dir:", VLAAI_RESULTS_DIR)

_aadnet_matches = [p for p in _all_ci_matches if "aadnet-xai" in p.replace("\\", "/")]
if _aadnet_matches:
    AADNET_CHANNEL_IMPORTANCE = _aadnet_matches[0]
    print("AADNet channel_importance.csv:", AADNET_CHANNEL_IMPORTANCE)
else:
    AADNET_CHANNEL_IMPORTANCE = None
    print("AADNet kernel output not attached (or channel_importance.csv not found) "
          "-- Section K2 (Haufe vs AADNet) will be skipped. "
          "Attach dulanamanjitha/aadnet-xai as a Kernel Output data source to enable it.")

# %% [markdown]
# ## 3. Verify GPU and data paths

# %%
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Reuse the exact same preprocessed DTU windows VLAAI's own notebook uses --
# NOT run_xai_trf_comparison.py's own default (external/vlaai/evaluation_datasets/DTU),
# since external/vlaai is an uninitialized git submodule in this repo and that
# path won't exist on a fresh clone.
DATA_DIR = os.path.join(REPO_DIR, "data", "vlaai_dtu_npz")
MONTAGE_FILE = os.path.join(REPO_DIR, "config", "dtu_channel_montage.csv")
AADNET_MONTAGE_FILE = os.path.join(REPO_DIR, "config", "aadnet_dtu_channel_montage.csv")
OUTPUT = "/kaggle/working/xai_results_trf_comparison"

assert os.path.isdir(DATA_DIR), f"Missing data dir: {DATA_DIR}"
npz_count = len([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])
print(f"Data dir: {DATA_DIR} ({npz_count} .npz files)")
print(f"VLAAI results: {VLAAI_RESULTS_DIR}")
print(f"AADNet channel_importance: {AADNET_CHANNEL_IMPORTANCE}")
print(f"Output: {OUTPUT}")

# %% [markdown]
# ## 4. Run TRF vs VLAAI/AADNet XAI comparison (Sections C, D, H, I, J, K, K2)
#
# Section K (Phase 3) is the Haufe-transform comparison against VLAAI's
# `combined_score`; Section K2 (Phase 3 extension) is the same comparison
# against AADNet's `combined_score`, run only if AADNet's kernel output was
# found in step 2; the rest (C/D/H/I/J) are the pre-existing occlusion-based
# TRF-vs-VLAAI comparison this script already did before Phase 3 was added.

# %%
import subprocess, sys

cmd = [
    sys.executable, "scripts/run_xai_trf_comparison.py",
    "--data-dir", DATA_DIR,
    "--vlaai-results", VLAAI_RESULTS_DIR,
    "--output-dir", OUTPUT,
    "--montage-file", MONTAGE_FILE,
    "--max-samples", "200",
    "--n-boot", "1000",
]
if AADNET_CHANNEL_IMPORTANCE:
    cmd += [
        "--aadnet-montage-file", AADNET_MONTAGE_FILE,
        "--aadnet-channel-importance", AADNET_CHANNEL_IMPORTANCE,
    ]

print("Command:", " ".join(cmd))
print("=" * 70)

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in process.stdout:
    print(line, end="")
process.wait()
print("=" * 70)
print(f"Exit code: {process.returncode}")

# %% [markdown]
# ## 5. Display results

# %%
report_path = os.path.join(OUTPUT, "comparison", "COMPARISON_REPORT.txt")
if os.path.isfile(report_path):
    with open(report_path, "r") as f:
        print(f.read())
else:
    print("Comparison report not found — check for errors above.")

# %%
print("Output files:")
for root, _dirs, files in os.walk(OUTPUT):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        print(f"  {os.path.relpath(fpath, OUTPUT):60s}  {os.path.getsize(fpath):>10,} bytes")

# %% [markdown]
# ## 6. Display the Haufe-transform comparison (Phase 3)

# %%
haufe_dir = os.path.join(OUTPUT, "trf_xai", "K_haufe_transform")
haufe_summary_path = os.path.join(haufe_dir, "haufe_summary.json")
if os.path.isfile(haufe_summary_path):
    import json
    with open(haufe_summary_path) as f:
        print(json.dumps(json.load(f), indent=2))
else:
    print("haufe_summary.json not found — check Section K output above for errors.")

# %%
from IPython.display import Image, display

haufe_plot = os.path.join(haufe_dir, "haufe_vs_combined_score.png")
if os.path.isfile(haufe_plot):
    display(Image(filename=haufe_plot))
else:
    print("haufe_vs_combined_score.png not found.")

# %% [markdown]
# ## 7. Display the Haufe-vs-AADNet comparison (Phase 3 extension, Section K2)
#
# Only produced if AADNet's kernel output was attached (step 2). Uses
# name-based channel alignment across the 58 electrode names shared between
# VLAAI's and AADNet's different montages.

# %%
k2_summary_path = os.path.join(haufe_dir, "haufe_vs_aadnet_summary.json")
if os.path.isfile(k2_summary_path):
    import json
    with open(k2_summary_path) as f:
        print(json.dumps(json.load(f), indent=2))
else:
    print("haufe_vs_aadnet_summary.json not found "
          "(AADNet kernel output not attached, or Section K2 skipped/failed above).")

# %%
k2_plot = os.path.join(haufe_dir, "haufe_vs_aadnet_scatter.png")
if os.path.isfile(k2_plot):
    display(Image(filename=k2_plot))
else:
    print("haufe_vs_aadnet_scatter.png not found.")

# %% [markdown]
# ## 8. Download results
#
# All results are saved under `/kaggle/working/xai_results_trf_comparison/`.
# Use the Kaggle **"Save Version"** button to download them as an output
# artifact (and to make them available as a Kernel Output data source for
# further downstream analysis).
