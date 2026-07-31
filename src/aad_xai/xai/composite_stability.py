"""Composite-channel/ROI significance testing and subject-wise cross-validation.

Generalizes the core-ROI composite methodology (subject-level Wilcoxon +
Cohen's d + bootstrap CI over an averaged channel group) to: any ROI, any
top-K channel selection, and a subject-wise cross-validation check for
whether a channel/ROI selection made on one half of subjects replicates as
significant on the other, held-out half.

Used by both `scripts/analyze_region_channel_stability.py` (local, reads
from downloaded `kaggle_output_*` snapshots) and
`notebooks/kaggle_analyze_cross_model.py` (Kaggle, reads from attached
kernel-output data sources) -- the statistics are identical, only the data
loading differs.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def cohens_d(x) -> float:
    x = np.asarray(x)
    return float(np.mean(x) / np.std(x, ddof=1))


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 42):
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    means = np.array([values[rng.randint(0, len(values), len(values))].mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(x) -> float:
    x = np.asarray(x)
    if np.allclose(x, 0):
        return 1.0
    _, p = wilcoxon(x)
    return float(p)


def fdr_correction(p_values, alpha: float = 0.05):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted = np.empty(n)
    adjusted[order[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        rank = i + 1
        adjusted[order[i]] = min(sorted_p[i] * n / rank, adjusted[order[i + 1]])
    adjusted = np.clip(adjusted, 0.0, 1.0)
    return adjusted, adjusted < alpha


def load_montage_rois(csv_path: str):
    """Returns (CH_ROI, CH_NAME, NAME_TO_IDX, ROIS) from the shared DTU montage CSV."""
    montage = pd.read_csv(csv_path)
    ch_roi = dict(zip(montage.channel_index, montage.roi))
    ch_name = dict(zip(montage.channel_index, montage.electrode_name))
    name_to_idx = {v: k for k, v in ch_name.items()}
    rois = OrderedDict()
    for idx, roi in sorted(ch_roi.items()):
        rois.setdefault(roi, []).append(idx)
    return ch_roi, ch_name, name_to_idx, rois


def region_composite_test(subj_ch_matrix, channel_idxs) -> dict:
    """subj_ch_matrix: (n_subjects, n_channels). channel_idxs: any channel subset.
    Averages the subset per subject, then tests the resulting (n_subjects,) vector."""
    composite = subj_ch_matrix[:, channel_idxs].mean(axis=1)
    mean, lo, hi = bootstrap_ci(composite)
    return {
        "n_channels": len(channel_idxs),
        "mean_dP": mean, "ci_lo": lo, "ci_hi": hi,
        "cohens_d": cohens_d(composite), "wilcoxon_p": wilcoxon_p(composite),
    }


def run_region_wise(models: dict, rois: "OrderedDict[str, list]", method: str) -> pd.DataFrame:
    """models: {model_name: {"occ": (n_subj,64) array, "perm": (n_subj,64) array}}."""
    rows = []
    for model_name, mats in models.items():
        mat = mats[method]
        recs, p_raw = [], []
        for roi, chs in rois.items():
            r = region_composite_test(mat, chs)
            r.update({"model": model_name, "roi": roi})
            recs.append(r); p_raw.append(r["wilcoxon_p"])
        adj, sig = fdr_correction(np.array(p_raw))
        for r, a, s in zip(recs, adj, sig):
            r["fdr_p"], r["fdr_sig"] = float(a), bool(s)
        rows.extend(recs)
    cols = ["model", "roi", "n_channels", "mean_dP", "ci_lo", "ci_hi", "cohens_d", "wilcoxon_p", "fdr_p", "fdr_sig"]
    return pd.DataFrame(rows)[cols].sort_values(["model", "cohens_d"], ascending=[True, False])


def run_top_channels(models: dict, ch_name: dict, shared_top_idx: list, shared_top_names: list,
                      method: str, k_values=(6, 10, 15)) -> pd.DataFrame:
    rows = []
    for model_name, mats in models.items():
        mat = mats[method]
        own_rank = np.argsort(-np.abs(mat).mean(axis=0))
        recs, p_raw = [], []
        for k in k_values:
            top_idx = own_rank[:k].tolist()
            r = region_composite_test(mat, top_idx)
            r.update({"model": model_name, "selection": f"own_top_{k}",
                      "channels": ",".join(ch_name[c] for c in top_idx)})
            recs.append(r); p_raw.append(r["wilcoxon_p"])
        r = region_composite_test(mat, shared_top_idx)
        r.update({"model": model_name, "selection": "cross_model_shared", "channels": ",".join(shared_top_names)})
        recs.append(r); p_raw.append(r["wilcoxon_p"])
        adj, sig = fdr_correction(np.array(p_raw))
        for r, a, s in zip(recs, adj, sig):
            r["fdr_p"], r["fdr_sig"] = float(a), bool(s)
        rows.extend(recs)
    cols = ["model", "selection", "n_channels", "channels", "mean_dP", "ci_lo", "ci_hi", "cohens_d", "wilcoxon_p", "fdr_p", "fdr_sig"]
    return pd.DataFrame(rows)[cols]


def select_best_roi(train_mat, rois: "OrderedDict[str, list]") -> list:
    roi_scores = {roi: np.abs(train_mat[:, chs]).mean() for roi, chs in rois.items()}
    return rois[max(roi_scores, key=roi_scores.get)]


def select_top_k_channels(train_mat, k: int = 6) -> list:
    scores = np.abs(train_mat).mean(axis=0)
    return list(np.argsort(-scores)[:k])


def cross_validate_selection(mat, select_fn, n_splits: int = 500, seed: int = 42, alpha: float = 0.05):
    """Repeatedly splits subjects in half; select_fn(train_mat) picks channels using
    ONLY the train half; tests whether that selection is still significant on the
    held-out test half. Returns (replication_rate, per-split DataFrame)."""
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


def safe_spearman(a, b):
    """Spearman correlation with NaN-safe fills (0.0 for rho, 1.0 for p).
    Consolidates the three slightly different inline idioms this repo had
    accumulated across scripts/run_focused_xai.py, notebooks/kaggle_run_xai_aadnet.py,
    and scripts/run_xai_trf_comparison.py's own `_safe_spearman` -- this is now
    the one shared version."""
    from scipy.stats import spearmanr
    r, p = spearmanr(a, b)
    r = 0.0 if np.isnan(r) else float(r)
    p = 1.0 if np.isnan(p) else float(p)
    return r, p


def leave_one_out_ranking_reliability(mat, n_boot: int = 2000, seed: int = 42):
    """Phase 4: held-out-subject channel-ranking validation.

    For each subject row, compares that subject's own profile against (a) the
    mean of the OTHER n-1 subjects (leave-one-out -- a ranking this subject
    had no say in) and (b) the mean of ALL subjects including themselves (the
    "subject-vs-group-mean" baseline several pipelines already report). If
    LOO and baseline rho are similar, weak cross-subject consistency is a
    genuine property of the data, not an artifact of each subject being
    included in the very group average they're compared against.

    Parameters
    ----------
    mat : (n_subjects, n_channels) array. Pass whatever ranking-criterion
        matrix the caller's own split-half reliability already uses (e.g.
        |occ|+|perm| combined, or occlusion-only), so these numbers are
        directly comparable to that existing statistic.

    Returns
    -------
    (per_subject_df, summary_dict) -- per_subject_df has one row per subject
    with rho_loo/p_loo/rho_baseline/p_baseline; summary_dict has the mean +/-
    bootstrap CI for both distributions plus a Wilcoxon test on the LOO rhos.
    """
    n_subj = mat.shape[0]
    group_mean_incl = mat.mean(axis=0)

    rows = []
    for i in range(n_subj):
        others_mean = np.delete(mat, i, axis=0).mean(axis=0)
        rho_loo, p_loo = safe_spearman(mat[i], others_mean)
        rho_baseline, p_baseline = safe_spearman(mat[i], group_mean_incl)
        rows.append({
            "subject_row": i,
            "rho_loo": rho_loo, "p_loo": p_loo,
            "rho_baseline": rho_baseline, "p_baseline": p_baseline,
        })
    df = pd.DataFrame(rows)

    loo_mean, loo_lo, loo_hi = bootstrap_ci(df["rho_loo"].values, n_boot, seed=seed)
    base_mean, base_lo, base_hi = bootstrap_ci(df["rho_baseline"].values, n_boot, seed=seed + 1)
    loo_wilcox_p = wilcoxon_p(df["rho_loo"].values)

    summary = {
        "n_subjects": n_subj,
        "loo_mean_rho": loo_mean, "loo_ci_lo": loo_lo, "loo_ci_hi": loo_hi,
        "loo_wilcoxon_p": loo_wilcox_p,
        "baseline_mean_rho": base_mean, "baseline_ci_lo": base_lo, "baseline_ci_hi": base_hi,
        "loo_vs_baseline_diff": loo_mean - base_mean,
    }
    return df, summary
