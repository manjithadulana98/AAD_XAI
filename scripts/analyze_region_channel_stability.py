# %% [markdown]
# # Composite Stability Analysis — Region-wise & Top-Channel, with Subject-Wise Cross-Validation
#
# Generalizes the ad-hoc 6-channel composite check (Cz, FC4, CP4, CPz, P1, TP10)
# into a systematic analysis, following the same subject-level methodology as
# the existing core-ROI composite test (`run_combined_roi_analysis` in
# `scripts/run_focused_xai.py`): average a channel group per subject, then
# Wilcoxon signed-rank + Cohen's d + bootstrap CI across the 18 subjects.
#
# The statistics themselves live in `aad_xai.xai.composite_stability` (shared
# with `notebooks/kaggle_analyze_cross_model.py`, the Kaggle version of this
# same analysis run against attached kernel outputs) -- this script only
# handles loading the local, already-downloaded per-subject matrices:
#   - AADNet: kaggle_output_AADnet/xai_results_aadnet/{occ,perm}_subj_ch.npy
#   - VLAAI:  kaggle_output_VLAAI/xai_results/subject_channel_importance.csv
#
# Section 6 directly addresses the circularity/double-dipping concern on the
# 6-channel composite: it repeatedly splits subjects in half, selects the
# "important" ROI/channels using only the train half, and tests whether that
# selection is STILL a significant composite on the held-out test half.

# %% [markdown]
# ## 1. Setup — load per-subject channel matrices for both models

# %%
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from aad_xai.xai import (
    load_montage_rois, run_region_wise, run_top_channels,
    cross_validate_selection, select_best_roi, select_top_k_channels,
)

CH_ROI, CH_NAME, NAME_TO_IDX, ROIS = load_montage_rois("config/dtu_channel_montage.csv")

# AADNet: (18, 64) subject x channel, indexed by channel_idx
aad_occ = np.load("kaggle_output_AADnet/xai_results_aadnet/occ_subj_ch.npy")
aad_perm = np.load("kaggle_output_AADnet/xai_results_aadnet/perm_subj_ch.npy")

# VLAAI: long format -> pivot to (18, 64) subject x channel, channel_index-ordered
vla_long = pd.read_csv("kaggle_output_VLAAI/xai_results/subject_channel_importance.csv")
vla_occ = vla_long.pivot(index="subject_id", columns="channel_index", values="occ_mean_dP").sort_index(axis=1).values
vla_perm = vla_long.pivot(index="subject_id", columns="channel_index", values="perm_mean_dP").sort_index(axis=1).values

MODELS = {
    "AADNet": {"occ": aad_occ, "perm": aad_perm},
    "VLAAI": {"occ": vla_occ, "perm": vla_perm},
}
print("Loaded per-subject matrices:", {m: v["occ"].shape for m, v in MODELS.items()})

# %% [markdown]
# ## 2 & 3. Region-wise composite test — occlusion and permutation

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
# ## 4 & 5. Top-channel composite test — each model's own top-K, plus the fixed cross-model-agreement set

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
# ## 6. Subject-wise cross-validation — do the same regions/channels replicate on held-out subjects?
#
# Directly addresses the circularity concern: repeatedly split the 18 subjects
# in half, select the "important" ROI/channels using ONLY the train half, then
# test whether that train-selected set is STILL a significant composite on
# the held-out test half (which had no say in the selection). The replication
# rate across many random splits is the honest answer to "does this feature
# selection hold up out of sample."

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
