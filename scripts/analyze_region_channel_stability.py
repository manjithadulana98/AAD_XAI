# %% [markdown]
# # Composite Stability Analysis — Region-wise & Top-Channel, with Subject-Wise Cross-Validation
#
# Generalizes the ad-hoc 6-channel composite check (Cz, FC4, CP4, CPz, P1, TP10)
# into a systematic analysis, following the same subject-level methodology as
# the existing core-ROI composite test (`run_combined_roi_analysis` in
# `scripts/run_focused_xai.py`): average a channel group per subject, then
# Wilcoxon signed-rank + Cohen's d + bootstrap CI across the 18 subjects.
#
# Runs entirely on already-downloaded per-subject matrices -- no GPU/Kaggle
# needed:
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
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from collections import OrderedDict


def cohens_d(x):
    x = np.asarray(x)
    return float(np.mean(x) / np.std(x, ddof=1))


def bootstrap_ci(values, n_boot=2000, seed=42):
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    means = np.array([values[rng.randint(0, len(values), len(values))].mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(x):
    x = np.asarray(x)
    if np.allclose(x, 0):
        return 1.0
    _, p = wilcoxon(x)
    return float(p)


def fdr_correction(p_values, alpha=0.05):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted = np.empty(n)
    adjusted[order[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        rank = i + 1
        adjusted[order[i]] = min(sorted_p[i] * n / rank, adjusted[order[i + 1]])
    adjusted = np.clip(adjusted, 0, 1)
    return adjusted, adjusted < alpha


# Montage: channel -> ROI mapping (shared by both models)
montage = pd.read_csv("config/dtu_channel_montage.csv")
CH_ROI = dict(zip(montage.channel_index, montage.roi))
CH_NAME = dict(zip(montage.channel_index, montage.electrode_name))
NAME_TO_IDX = {v: k for k, v in CH_NAME.items()}
ROIS = OrderedDict()
for idx, roi in sorted(CH_ROI.items()):
    ROIS.setdefault(roi, []).append(idx)

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
# ## 2 & 3. Region-wise composite test — one function, run for occlusion and permutation separately

# %%
def region_composite_test(subj_ch_matrix, channel_idxs):
    composite = subj_ch_matrix[:, channel_idxs].mean(axis=1)
    mean, lo, hi = bootstrap_ci(composite)
    return {
        "n_channels": len(channel_idxs),
        "mean_dP": mean, "ci_lo": lo, "ci_hi": hi,
        "cohens_d": cohens_d(composite), "wilcoxon_p": wilcoxon_p(composite),
    }


def run_region_wise(method):
    rows = []
    for model_name, mats in MODELS.items():
        mat = mats[method]
        recs, p_raw = [], []
        for roi, chs in ROIS.items():
            r = region_composite_test(mat, chs)
            r.update({"model": model_name, "roi": roi})
            recs.append(r); p_raw.append(r["wilcoxon_p"])
        adj, sig = fdr_correction(np.array(p_raw))
        for r, a, s in zip(recs, adj, sig):
            r["fdr_p"], r["fdr_sig"] = float(a), bool(s)
        rows.extend(recs)
    cols = ["model", "roi", "n_channels", "mean_dP", "ci_lo", "ci_hi", "cohens_d", "wilcoxon_p", "fdr_p", "fdr_sig"]
    return pd.DataFrame(rows)[cols].sort_values(["model", "cohens_d"], ascending=[True, False])


print("\n" + "=" * 100)
print("REGION-WISE COMPOSITE -- OCCLUSION")
print("=" * 100)
region_occ_results = run_region_wise("occ")
print(region_occ_results.to_string(index=False))

# %%
print("\n" + "=" * 100)
print("REGION-WISE COMPOSITE -- PERMUTATION")
print("=" * 100)
region_perm_results = run_region_wise("perm")
print(region_perm_results.to_string(index=False))

# %% [markdown]
# ## 4 & 5. Top-channel composite test — each model's own top-K, plus the fixed cross-model-agreement set

# %%
SHARED_TOP6 = ["Cz", "FC4", "CP4", "CPz", "P1", "TP10"]
shared_top6_idx = [NAME_TO_IDX[c] for c in SHARED_TOP6]


def run_top_channels(method, k_values=(6, 10, 15)):
    rows = []
    for model_name, mats in MODELS.items():
        mat = mats[method]
        own_rank = np.argsort(-np.abs(mat).mean(axis=0))  # rank by this model's OWN mean |effect|
        recs, p_raw = [], []
        for k in k_values:
            top_idx = own_rank[:k].tolist()
            r = region_composite_test(mat, top_idx)
            r.update({"model": model_name, "selection": f"own_top_{k}",
                      "channels": ",".join(CH_NAME[c] for c in top_idx)})
            recs.append(r); p_raw.append(r["wilcoxon_p"])
        r = region_composite_test(mat, shared_top6_idx)
        r.update({"model": model_name, "selection": "cross_model_shared_6", "channels": ",".join(SHARED_TOP6)})
        recs.append(r); p_raw.append(r["wilcoxon_p"])
        adj, sig = fdr_correction(np.array(p_raw))
        for r, a, s in zip(recs, adj, sig):
            r["fdr_p"], r["fdr_sig"] = float(a), bool(s)
        rows.extend(recs)
    cols = ["model", "selection", "n_channels", "channels", "mean_dP", "ci_lo", "ci_hi", "cohens_d", "wilcoxon_p", "fdr_p", "fdr_sig"]
    return pd.DataFrame(rows)[cols]


print("\n" + "=" * 100)
print("TOP-CHANNEL COMPOSITE -- OCCLUSION")
print("=" * 100)
top_occ_results = run_top_channels("occ")
print(top_occ_results.drop(columns="channels").to_string(index=False))

# %%
print("\n" + "=" * 100)
print("TOP-CHANNEL COMPOSITE -- PERMUTATION")
print("=" * 100)
top_perm_results = run_top_channels("perm")
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
def cross_validate_selection(mat, select_fn, n_splits=500, seed=42, alpha=0.05):
    n_subj = mat.shape[0]
    rng = np.random.RandomState(seed)
    results = []
    for i in range(n_splits):
        perm = rng.permutation(n_subj)
        half = n_subj // 2
        train_idx, test_idx = perm[:half], perm[half:]
        train_mat, test_mat = mat[train_idx], mat[test_idx]
        selected = select_fn(train_mat)
        test_composite = test_mat[:, selected].mean(axis=1)
        if len(test_composite) < 3 or np.allclose(test_composite, 0):
            continue
        p = wilcoxon_p(test_composite)
        results.append({"split": i, "p": p, "cohens_d": cohens_d(test_composite), "sig": p < alpha})
    rate = float(np.mean([r["sig"] for r in results])) if results else float("nan")
    return rate, pd.DataFrame(results)


def select_best_roi(train_mat):
    roi_scores = {roi: np.abs(train_mat[:, chs]).mean() for roi, chs in ROIS.items()}
    return ROIS[max(roi_scores, key=roi_scores.get)]


def select_top_k_channels(train_mat, k=6):
    scores = np.abs(train_mat).mean(axis=0)
    return list(np.argsort(-scores)[:k])


print("\n" + "=" * 100)
print("SUBJECT-WISE CROSS-VALIDATION (500 random half-splits each)")
print("=" * 100)
cv_rows = []
for model_name, mats in MODELS.items():
    for method in ["occ", "perm"]:
        mat = mats[method]
        rate_roi, _ = cross_validate_selection(mat, select_best_roi)
        cv_rows.append({"model": model_name, "method": method, "selection": "best_roi_reselected_per_split", "replication_rate": rate_roi})
        rate_top6, _ = cross_validate_selection(mat, lambda m: select_top_k_channels(m, 6))
        cv_rows.append({"model": model_name, "method": method, "selection": "top_6_channels_reselected_per_split", "replication_rate": rate_top6})
        rate_fixed, _ = cross_validate_selection(mat, lambda m: shared_top6_idx)
        cv_rows.append({"model": model_name, "method": method, "selection": "fixed_cross_model_6 (no reselection)", "replication_rate": rate_fixed})

cv_df = pd.DataFrame(cv_rows)
print(cv_df.to_string(index=False))
