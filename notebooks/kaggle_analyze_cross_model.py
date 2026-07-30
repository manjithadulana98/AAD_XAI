# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-Model Composite Analysis — Kaggle Notebook
#
# Runs the region-wise / top-channel / subject-wise-cross-validation composite
# analysis (same statistics as `scripts/analyze_region_channel_stability.py`,
# shared via `aad_xai.xai.composite_stability`) directly against the
# **already-completed outputs** of the two GPU pipelines:
# - `dulanamanjitha/aadnet-xai`
# - `dulanamanjitha/vlaai-xai`
#
# This does **not** re-run either GPU pipeline — it only reads their result
# files, attached here as Kernel Output data sources. No GPU needed; runs in
# seconds.
#
# **Kaggle setup requirements**
# - Add Data → Kernel Output → attach both `dulanamanjitha/aadnet-xai` and
#   `dulanamanjitha/vlaai-xai`
# - Enable Internet (to clone this repo for `config/dtu_channel_montage.csv`
#   and the shared `aad_xai.xai.composite_stability` module)
# - No GPU/accelerator required

# %% [markdown]
# ## 1. Clone repository (for the montage file + shared analysis module)

# %%
import os
import subprocess

REPO_DIR = "/kaggle/working/AAD_XAI"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "https://github.com/manjithadulana98/AAD_XAI.git", REPO_DIR], check=True)
else:
    print(f"Repository already cloned at {REPO_DIR}")
os.chdir(REPO_DIR)

# Lightweight install: this analysis only needs numpy/pandas/scipy, not the
# full torch/captum/shap/lime stack the GPU pipelines require. The package's
# xai/__init__.py imports those optional submodules defensively, so `pip
# install -e .` without the heavy extras is sufficient here.
subprocess.run(["pip", "install", "-q", "-e", "."], check=True)
print("Setup done.")

# %% [markdown]
# ## 2. Locate the attached kernel-output data
#
# Kernel-output data sources (attached via `kernel_sources` in
# kernel-metadata.json, or Add Data > Kernel Output in the UI) mount
# somewhere under `/kaggle/input/`, but the exact nesting has proven
# inconsistent in practice (observed once as a flat `/kaggle/input/notebooks/`
# with no per-kernel subfolder visible at the top level). Rather than guess
# a fixed path, this recursively searches `/kaggle/input/` for each model's
# distinguishing file and fails loudly (never silently) if neither is found.

# %%
import glob


def find_results_dir(required_file):
    matches = glob.glob(f"/kaggle/input/**/{required_file}", recursive=True)
    return os.path.dirname(matches[0]) if matches else None


aadnet_dir = find_results_dir("occ_subj_ch.npy")
vlaai_dir = find_results_dir("subject_channel_importance.csv")

if aadnet_dir is None or vlaai_dir is None:
    all_files = glob.glob("/kaggle/input/**/*", recursive=True)
    missing = []
    if aadnet_dir is None:
        missing.append("AADNet results (occ_subj_ch.npy) not found anywhere under /kaggle/input/")
    if vlaai_dir is None:
        missing.append("VLAAI results (subject_channel_importance.csv) not found anywhere under /kaggle/input/")
    raise FileNotFoundError(
        "Could not locate one or both models' results.\n" + "\n".join(missing) +
        f"\nFull contents of /kaggle/input ({len(all_files)} entries): {all_files[:200]}\n"
        "Check that both dulanamanjitha/aadnet-xai and dulanamanjitha/vlaai-xai "
        "are attached as Kernel Output data sources (Add Data > Kernel Output)."
    )

print("AADNet results dir:", aadnet_dir)
print("VLAAI results dir:", vlaai_dir)

# %% [markdown]
# ## 3. Load per-subject matrices

# %%
import numpy as np
import pandas as pd
from aad_xai.xai import load_montage_rois, run_region_wise, run_top_channels, \
    cross_validate_selection, select_best_roi, select_top_k_channels

CH_ROI, CH_NAME, NAME_TO_IDX, ROIS = load_montage_rois("config/dtu_channel_montage.csv")

aad_occ = np.load(os.path.join(aadnet_dir, "occ_subj_ch.npy"))
aad_perm = np.load(os.path.join(aadnet_dir, "perm_subj_ch.npy"))

vla_long = pd.read_csv(os.path.join(vlaai_dir, "subject_channel_importance.csv"))
vla_occ = vla_long.pivot(index="subject_id", columns="channel_index", values="occ_mean_dP").sort_index(axis=1).values
vla_perm = vla_long.pivot(index="subject_id", columns="channel_index", values="perm_mean_dP").sort_index(axis=1).values

MODELS = {
    "AADNet": {"occ": aad_occ, "perm": aad_perm},
    "VLAAI": {"occ": vla_occ, "perm": vla_perm},
}
print("Loaded per-subject matrices:", {m: v["occ"].shape for m, v in MODELS.items()})

# %% [markdown]
# ## 4 & 5. Region-wise composite test — occlusion and permutation

# %%
print("\n" + "=" * 100)
print("REGION-WISE COMPOSITE -- OCCLUSION")
print("=" * 100)
region_occ_results = run_region_wise(MODELS, ROIS, "occ")
print(region_occ_results.to_string(index=False))

# %%
print("\n" + "=" * 100)
print("REGION-WISE COMPOSITE -- PERMUTATION")
print("=" * 100)
region_perm_results = run_region_wise(MODELS, ROIS, "perm")
print(region_perm_results.to_string(index=False))

# %% [markdown]
# ## 6 & 7. Top-channel composite test — each model's own top-K, plus the fixed cross-model-agreement set

# %%
SHARED_TOP6 = ["Cz", "FC4", "CP4", "CPz", "P1", "TP10"]
shared_top6_idx = [NAME_TO_IDX[c] for c in SHARED_TOP6]

print("\n" + "=" * 100)
print("TOP-CHANNEL COMPOSITE -- OCCLUSION")
print("=" * 100)
top_occ_results = run_top_channels(MODELS, CH_NAME, shared_top6_idx, SHARED_TOP6, "occ")
print(top_occ_results.drop(columns="channels").to_string(index=False))

# %%
print("\n" + "=" * 100)
print("TOP-CHANNEL COMPOSITE -- PERMUTATION")
print("=" * 100)
top_perm_results = run_top_channels(MODELS, CH_NAME, shared_top6_idx, SHARED_TOP6, "perm")
print(top_perm_results.drop(columns="channels").to_string(index=False))

# %% [markdown]
# ## 8. Subject-wise cross-validation — do the same regions/channels replicate on held-out subjects?

# %%
print("\n" + "=" * 100)
print("SUBJECT-WISE CROSS-VALIDATION (500 random half-splits each)")
print("=" * 100)
cv_rows = []
for model_name, mats in MODELS.items():
    for method in ["occ", "perm"]:
        mat = mats[method]
        rate_roi, _ = cross_validate_selection(mat, lambda m: select_best_roi(m, ROIS))
        cv_rows.append({"model": model_name, "method": method, "selection": "best_roi_reselected_per_split", "replication_rate": rate_roi})
        rate_top6, _ = cross_validate_selection(mat, lambda m: select_top_k_channels(m, 6))
        cv_rows.append({"model": model_name, "method": method, "selection": "top_6_channels_reselected_per_split", "replication_rate": rate_top6})
        rate_fixed, _ = cross_validate_selection(mat, lambda m: shared_top6_idx)
        cv_rows.append({"model": model_name, "method": method, "selection": "fixed_cross_model_6 (no reselection)", "replication_rate": rate_fixed})

cv_df = pd.DataFrame(cv_rows)
print(cv_df.to_string(index=False))

# %% [markdown]
# ## 9. Save results

# %%
OUT_DIR = "/kaggle/working/cross_model_analysis"
os.makedirs(OUT_DIR, exist_ok=True)
region_occ_results.to_csv(os.path.join(OUT_DIR, "region_wise_occlusion.csv"), index=False)
region_perm_results.to_csv(os.path.join(OUT_DIR, "region_wise_permutation.csv"), index=False)
top_occ_results.to_csv(os.path.join(OUT_DIR, "top_channel_occlusion.csv"), index=False)
top_perm_results.to_csv(os.path.join(OUT_DIR, "top_channel_permutation.csv"), index=False)
cv_df.to_csv(os.path.join(OUT_DIR, "subject_wise_cross_validation.csv"), index=False)
print(f"Saved 5 CSVs to {OUT_DIR}")
