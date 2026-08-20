"""DTU LOSO TRF explainability: Haufe-transform activation patterns, window-
and subject-level channel importance (occlusion + permutation + BH-FDR), a
lag-block cascading-randomization sanity check, and deletion/insertion
faithfulness curves.

Brings the DTU LOSO TRF baseline (`aad_xai.evaluation.loso_runner`) up to
the same statistical rigor already built for the AADNet/VLAAI XAI
pipelines (`scripts/run_focused_xai.py`), adapted throughout for the one
structural fact that drives every design choice here: `TRFDecoder` is a
plain `sklearn.Ridge` wrapper, not an `nn.Module` -- there are 18
independent fold-specific decoders (one per held-out subject), not one
shared trained model.

Reuses `aad_xai.xai.composite_stability`'s `bootstrap_ci`/`wilcoxon_p`/
`cohens_d`/`fdr_correction`/`safe_spearman`/`region_composite_test` rather
than hand-rolling a fourth duplicate (see `run_focused_xai.py`,
`notebooks/kaggle_run_xai_aadnet.py`, and `merge_combined_contrast_results.py`
for the three existing duplicates this avoids adding to). The Haufe
identity itself is adapted from `scripts/run_xai_trf_comparison.py`'s
`haufe_pattern_for_windows` (lines 535-571 at the time this was written).
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from ..config import PreprocessConfig
from ..data.base import Trial
from ..data.cv_splits import CV_STRATEGIES, assert_loso_fold_integrity
from ..data.dtu_dataset import DTUDataset
from ..data.windowing import make_windows
from ..models.trf_baseline import TRFDecoder, lag_matrix
from ..models.trf_decision import TRFDecisionWrapper
from .composite_stability import (
    bootstrap_ci,
    cohens_d,
    fdr_correction,
    region_composite_test,
    safe_spearman,
    wilcoxon_p,
)

ROIS = OrderedDict([
    ("Frontal",        list(range(0, 12))),
    ("Fronto-Central", list(range(12, 18))),
    ("Central",        list(range(18, 30))),
    ("Temporal",       list(range(30, 42))),
    ("Parietal",       list(range(42, 54))),
    ("Occipital",      list(range(54, 64))),
])


# ══════════════════════════════════════════════════════════════════════
# Weight loading
# ══════════════════════════════════════════════════════════════════════
def load_trf_decoder(npz_path, sfreq: float) -> TRFDecoder:
    """Reconstruct a usable TRFDecoder from a saved (coef, lags, x_mean,
    x_std) checkpoint (written by loso_runner.py's _collect hook).

    Deliberately does NOT restore intercept_/_y_mean/_y_std -- they were
    never saved (see run_experiments.py's _train_trf_fold return dict).
    Every downstream consumer in this module reduces predictions to either
    a Pearson correlation (TRFDecisionWrapper's r_att/r_unatt -- invariant
    to an additive intercept shift and to a positive-scale rescale of the
    predicted envelope, since np.corrcoef normalizes both away) or a
    mean-centered covariance (the Haufe identity, which reads coef_
    directly and never calls .predict() at all). Restoring the other 3
    fields would only matter if something here used a raw, un-normalized
    prediction value, which nothing does.
    """
    npz = np.load(npz_path)
    decoder = TRFDecoder()
    decoder.model.coef_ = npz["coef"]
    decoder.model.intercept_ = 0.0
    decoder.lags_ = npz["lags"]
    decoder.sfreq_ = float(sfreq)
    decoder._X_mean = npz["x_mean"]
    decoder._X_std = npz["x_std"]
    return decoder


# ══════════════════════════════════════════════════════════════════════
# EEG sampling / windowing helpers
# ══════════════════════════════════════════════════════════════════════
def _sample_training_eeg(train_trials: list[Trial], *, seg_len_s: float,
                          max_total_s: float = 300.0, seed: int = 0) -> np.ndarray:
    """Memory-bounded random-segment sample of a fold's training pool,
    mirroring run_experiments.py's _train_trf_fold sampling loop (many
    short random crops rather than every trial in full) so a Haufe pattern
    computed over 17 pooled subjects can't blow up memory the same way the
    FAConformer SI port's 8-band precompute once did. Not required to be a
    bit-exact reproduction of the actual fit's own random sample -- Haufe's
    identity only needs a representative sample of X in the model's fitted
    feature space, not the literal training set.
    """
    valid = [t for t in train_trials if int(t.eeg.shape[-1]) >= int(round(seg_len_s * t.sfreq))]
    if not valid:
        raise ValueError("No training trial is long enough for the requested segment length.")
    sfreq = float(valid[0].sfreq)
    seg_len = int(round(seg_len_s * sfreq))
    n_segments = max(1, int(np.ceil((max_total_s * sfreq) / seg_len)))

    rng = np.random.default_rng(seed)
    segs = []
    attempts = 0
    while len(segs) < n_segments and attempts < n_segments * 20:
        attempts += 1
        t = valid[int(rng.integers(0, len(valid)))]
        n = int(t.eeg.shape[1])
        start = int(rng.integers(0, n - seg_len + 1))
        segs.append(t.eeg[:, start:start + seg_len])
    return np.concatenate(segs, axis=1)  # (n_channels, n_times)


def _windows_to_tensors(trials: list[Trial], window_s: float, overlap_s: float = 0.0):
    """Build (eeg, att, unatt, labels, subject_ids) tensors for every window
    in the given trials -- eeg: (N, T, C), att/unatt: (N, T, 1) -- matching
    the shapes TRFDecisionWrapper.forward / get_attended_prob expect.
    """
    eeg_list, att_list, unatt_list, labels, subj_ids = [], [], [], [], []
    for t in trials:
        if t.audio_a is None or t.audio_b is None:
            continue
        windows = make_windows(
            n_times=t.eeg.shape[-1], sfreq=t.sfreq, window_s=window_s,
            overlap_s=overlap_s, label=t.label, subject_id=t.subject_id, trial_id=t.trial_id,
        )
        att_full = t.audio_a if int(t.label) == 0 else t.audio_b
        unatt_full = t.audio_b if int(t.label) == 0 else t.audio_a
        for w in windows:
            eeg_list.append(t.eeg[:, w.start:w.stop].T)  # (T, C)
            att_list.append(np.asarray(att_full[w.start:w.stop], dtype=np.float32).reshape(-1, 1))
            unatt_list.append(np.asarray(unatt_full[w.start:w.stop], dtype=np.float32).reshape(-1, 1))
            labels.append(w.label)
            subj_ids.append(w.subject_id)

    if not eeg_list:
        raise ValueError("No windows produced -- check window_s against trial lengths.")

    eeg = torch.tensor(np.stack(eeg_list), dtype=torch.float32)
    att = torch.tensor(np.stack(att_list), dtype=torch.float32)
    unatt = torch.tensor(np.stack(unatt_list), dtype=torch.float32)
    return eeg, att, unatt, np.array(labels), np.array(subj_ids)


# ══════════════════════════════════════════════════════════════════════
# Haufe transform
# ══════════════════════════════════════════════════════════════════════
def _haufe_from_eeg(decoder_like, eeg_cat: np.ndarray):
    """Core Haufe et al. (2014) computation given a decoder-like object
    (needs .lags_, .model.coef_, ._X_mean, ._X_std) and a fixed (n_channels,
    n_times) EEG sample. Factored out of haufe_pattern_for_fold so
    sanity_check_lag_cascade can hold the EEG sample fixed while only the
    coefficients vary across cascade steps -- resampling EEG at every step
    would confound "how much of the model is randomized" with "which random
    subsample happened to be drawn this time."

    Identity: Cov(X) @ w = Cov(X, y_hat), so no (n_features x n_features)
    covariance matrix needs to be formed. X is used in the exact z-scored
    space the Ridge model was fit on.
    """
    X = lag_matrix(eeg_cat, decoder_like.lags_)
    X_z = (X - decoder_like._X_mean) / decoder_like._X_std
    y_hat = X_z @ decoder_like.model.coef_

    n_lags = len(decoder_like.lags_)
    n_channels = eeg_cat.shape[0]
    var_yhat = float(np.var(y_hat))
    if var_yhat < 1e-12:
        return np.zeros(n_channels), np.zeros((n_lags, n_channels))

    Xc = X_z - X_z.mean(axis=0, keepdims=True)
    yc = y_hat - y_hat.mean()
    pattern_full = (Xc * yc[:, None]).mean(axis=0) / var_yhat
    pattern_matrix = pattern_full.reshape(n_lags, n_channels)  # matches lag_matrix's per-lag column blocks
    channel_magnitude = np.sqrt((pattern_matrix ** 2).sum(axis=0))
    return channel_magnitude, pattern_matrix


def haufe_pattern_for_fold(decoder: TRFDecoder, train_trials: list[Trial], *,
                            seg_len_s: float = 5.0, max_total_s: float = 300.0,
                            seed: int = 0):
    """Haufe activation pattern for one LOSO fold's fitted TRF, evaluated
    over a fresh sample of that fold's own training pool (the 17
    non-held-out subjects) -- not the held-out subject's data, since
    Haufe's identity is defined relative to the covariance structure the
    decoder actually learned from.

    Returns (channel_magnitude (n_channels,), pattern_matrix (n_lags, n_channels)).
    """
    eeg_cat = _sample_training_eeg(train_trials, seg_len_s=seg_len_s, max_total_s=max_total_s, seed=seed)
    return _haufe_from_eeg(decoder, eeg_cat)


# ══════════════════════════════════════════════════════════════════════
# Window-level importance (occlusion + permutation + FDR)
# ══════════════════════════════════════════════════════════════════════
def _sign_flip_p_value(values: np.ndarray, n_perm: int = 5000, seed: int = 42) -> float:
    """Two-sided sign-flip permutation p-value for mean != 0.

    Used at the WINDOW level, where windows drawn from the same trial are
    correlated (violates the independence a parametric or rank test needs)
    -- matches run_focused_xai.py's window-level test exactly. Subject-level
    tests use wilcoxon_p (composite_stability) instead, since each of the
    n_subjects values there genuinely is an independent sample.
    """
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    obs = abs(values.mean())
    null = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=len(values))
        null[i] = abs((values * signs).mean())
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def window_level_importance(decoder: TRFDecoder, test_trials: list[Trial], *,
                             window_s: float, overlap_s: float = 0.0,
                             n_boot: int = 1000, seed: int = 42,
                             max_windows: Optional[int] = 200) -> dict:
    """Channel occlusion + permutation importance over one held-out
    subject's test windows. Mirrors run_focused_xai.py's
    compute_channel_occlusion/compute_channel_permutation.

    max_windows caps the number of windows used (deterministic subsample,
    fixed seed) -- NOT "no subsampling" as originally planned. A local
    benchmark showed TRFDecisionWrapper.forward()'s per-window Python loop
    costs ~9ms/window, and DTU subjects have ~600 test windows each at a 5s
    decision window, so "every window x every channel x 18 folds" is a
    multi-hour cost, not the cheap operation the plan assumed. 200 matches
    run_xai_trf_comparison.py's own --max-samples default -- this is not a
    new compromise relative to that pipeline, it is the same cap AADNet/
    VLAAI's own TRF XAI comparison already applies for the same reason.
    """
    eeg, att, unatt, labels, _ = _windows_to_tensors(test_trials, window_s, overlap_s)
    if max_windows is not None and eeg.shape[0] > max_windows:
        keep = np.random.RandomState(seed).choice(eeg.shape[0], size=max_windows, replace=False)
        keep = torch.from_numpy(np.sort(keep)).long()
        eeg, att, unatt = eeg[keep], att[keep], unatt[keep]
        labels = labels[keep.numpy()]
    n_channels = eeg.shape[-1]
    wrapper = TRFDecisionWrapper(decoder)
    wrapper.eval()

    def _prob(eeg_batch):
        wrapper.set_envelopes(att, unatt)
        with torch.no_grad():
            logits = wrapper(eeg_batch)
        return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    base_probs = _prob(eeg)
    occ_pw = np.zeros((eeg.shape[0], n_channels))
    perm_pw = np.zeros((eeg.shape[0], n_channels))
    rng = np.random.RandomState(seed)

    for ch in range(n_channels):
        eeg_occ = eeg.clone()
        eeg_occ[:, :, ch] = 0.0
        occ_pw[:, ch] = base_probs - _prob(eeg_occ)

        eeg_perm = eeg.clone()
        perm_idx = torch.from_numpy(rng.permutation(eeg.shape[0])).long()
        eeg_perm[:, :, ch] = eeg.index_select(0, perm_idx)[:, :, ch]
        perm_pw[:, ch] = base_probs - _prob(eeg_perm)

    def _summarize(pw, offset):
        out = []
        for ch in range(n_channels):
            mean, lo, hi = bootstrap_ci(pw[:, ch], seed=seed)
            p = _sign_flip_p_value(pw[:, ch], seed=seed + offset + ch)
            out.append({"channel": ch, "mean_dp": mean, "ci_lo": lo, "ci_hi": hi, "p_value": p})
        return out

    occ_results = _summarize(occ_pw, 0)
    perm_results = _summarize(perm_pw, 1000)
    occ_fdr_p, occ_fdr_sig = fdr_correction(np.array([r["p_value"] for r in occ_results]))
    perm_fdr_p, perm_fdr_sig = fdr_correction(np.array([r["p_value"] for r in perm_results]))
    for i in range(n_channels):
        occ_results[i]["fdr_p"], occ_results[i]["fdr_sig"] = float(occ_fdr_p[i]), bool(occ_fdr_sig[i])
        perm_results[i]["fdr_p"], perm_results[i]["fdr_sig"] = float(perm_fdr_p[i]), bool(perm_fdr_sig[i])

    return {
        "occ_results": occ_results, "perm_results": perm_results,
        "occ_pw": occ_pw, "perm_pw": perm_pw, "base_probs": base_probs,
        "n_windows": int(eeg.shape[0]),
    }


def window_level_combined_table(occ_pw_pooled: np.ndarray, perm_pw_pooled: np.ndarray,
                                 occ_subject_profile: np.ndarray, perm_subject_profile: np.ndarray,
                                 *, fdr_alpha: float = 0.05, stability_threshold: float = 0.5,
                                 seed: int = 42) -> list[dict]:
    """Pooled window-level channel-importance table: every LOSO fold's
    held-out-subject windows contribute rows to occ_pw_pooled/perm_pw_pooled
    (each (total_windows, n_channels)) -- analogous to AADNet/VLAAI pooling
    all subjects' windows under one shared model, except here each window's
    delta-P comes from that window's OWN subject's fold-specific decoder (no
    leakage: that decoder never saw this subject during training).

    occ_subject_profile/perm_subject_profile ((n_subjects, n_channels), the
    per-fold mean over that subject's own windows) are used only for the
    subject-wise stability fraction, matching run_focused_xai.py's H.1
    "combined" table (channel_importance.csv/.json) field-for-field.
    """
    n_channels = occ_pw_pooled.shape[1]
    n_subjects = occ_subject_profile.shape[0]

    def _summarize(pw, offset):
        out = []
        for ch in range(n_channels):
            mean, lo, hi = bootstrap_ci(pw[:, ch], seed=seed)
            p = _sign_flip_p_value(pw[:, ch], seed=seed + offset + ch)
            out.append({"mean_dp": mean, "ci_lo": lo, "ci_hi": hi, "p_value": p})
        return out

    occ_results = _summarize(occ_pw_pooled, 0)
    perm_results = _summarize(perm_pw_pooled, 1000)
    occ_fdr_p, occ_fdr_sig = fdr_correction(np.array([r["p_value"] for r in occ_results]), fdr_alpha)
    perm_fdr_p, perm_fdr_sig = fdr_correction(np.array([r["p_value"] for r in perm_results]), fdr_alpha)

    combined = []
    for ch in range(n_channels):
        occ_mean, perm_mean = occ_results[ch]["mean_dp"], perm_results[ch]["mean_dp"]
        if occ_mean > 0 and perm_mean > 0:
            contribution_type = "facilitatory"
        elif occ_mean < 0 and perm_mean < 0:
            contribution_type = "suppressive"
        else:
            contribution_type = "mixed"

        occ_col = occ_subject_profile[:, ch]
        majority_sign = np.sign(np.median(occ_col))
        if majority_sign == 0:
            majority_sign = np.sign(occ_mean)
        stab_frac = float(np.mean(np.sign(occ_col) == majority_sign)) if n_subjects > 0 else 0.0

        same_sign = contribution_type in ("facilitatory", "suppressive")
        at_least_one_fdr = bool(occ_fdr_sig[ch]) or bool(perm_fdr_sig[ch])
        robust_significant = same_sign and at_least_one_fdr and (stab_frac >= stability_threshold)

        combined.append({
            "channel": ch,
            "occ_score": occ_mean, "occ_ci_lo": occ_results[ch]["ci_lo"], "occ_ci_hi": occ_results[ch]["ci_hi"],
            "occ_p_value": occ_results[ch]["p_value"], "occ_fdr_p": float(occ_fdr_p[ch]), "occ_fdr_sig": bool(occ_fdr_sig[ch]),
            "perm_score": perm_mean, "perm_ci_lo": perm_results[ch]["ci_lo"], "perm_ci_hi": perm_results[ch]["ci_hi"],
            "perm_p_value": perm_results[ch]["p_value"], "perm_fdr_p": float(perm_fdr_p[ch]), "perm_fdr_sig": bool(perm_fdr_sig[ch]),
            "contribution_type": contribution_type,
            "subject_stability_frac": stab_frac,
            "subject_stability": f"{int(round(stab_frac * n_subjects))}/{n_subjects}",
            "robust_significant": robust_significant,
        })

    occ_abs = np.array([abs(c["occ_score"]) for c in combined])
    perm_abs = np.array([abs(c["perm_score"]) for c in combined])
    occ_z = (occ_abs - occ_abs.mean()) / occ_abs.std() if occ_abs.std() > 1e-10 else np.zeros(n_channels)
    perm_z = (perm_abs - perm_abs.mean()) / perm_abs.std() if perm_abs.std() > 1e-10 else np.zeros(n_channels)
    combined_score = (occ_z + perm_z) / 2.0
    for ch in range(n_channels):
        combined[ch]["combined_score"] = float(combined_score[ch])

    return combined


# ══════════════════════════════════════════════════════════════════════
# Subject-level stats + Tier classification
# ══════════════════════════════════════════════════════════════════════
def subject_level_stats(occ_matrix: np.ndarray, perm_matrix: np.ndarray, haufe_matrix: np.ndarray,
                         window_level_table: list[dict], *, fdr_alpha: float = 0.05,
                         tier1_stability_frac: Optional[float] = None,
                         tier2_stability_frac: Optional[float] = None, top_pct: float = 80) -> dict:
    """Subject-level channel statistics + 4-tier classification, mirroring
    run_focused_xai.py's H.2/H.3 (per-channel bootstrap CI/median/
    Wilcoxon-p/Cohen's d across the n_subjects fold-level values, BH-FDR
    over the channel tests, then a 4-tier classification).

    Tier thresholds default to fractions of n_subjects (12/18, 10/18) --
    written as fractions rather than run_focused_xai.py's hardcoded
    literals, since this generalizes to a subject count other than 18.
    haufe_matrix gets its own bootstrap/Wilcoxon/FDR stats as a third,
    TRF-specific importance signal, reported alongside but not gating the
    Tier system (Haufe isn't part of AADNet/VLAAI's Tier criteria either --
    this keeps Tier classification directly comparable while adding Haufe
    as a genuinely new angle unique to a linear model).
    """
    n_subj, n_channels = occ_matrix.shape
    tier1_frac = tier1_stability_frac if tier1_stability_frac is not None else 12 / 18
    tier2_frac = tier2_stability_frac if tier2_stability_frac is not None else 10 / 18

    ch_stats = []
    for ch in range(n_channels):
        occ_col, perm_col, haufe_col = occ_matrix[:, ch], perm_matrix[:, ch], haufe_matrix[:, ch]
        occ_mean, occ_lo, occ_hi = bootstrap_ci(occ_col)
        perm_mean, perm_lo, perm_hi = bootstrap_ci(perm_col)
        haufe_mean, haufe_lo, haufe_hi = bootstrap_ci(haufe_col)
        ch_stats.append({
            "channel": ch,
            "occ_subj_mean": occ_mean, "occ_subj_ci_lo": occ_lo, "occ_subj_ci_hi": occ_hi,
            "occ_subj_median": float(np.median(occ_col)),
            "occ_wilcox_p": wilcoxon_p(occ_col), "occ_cohens_d": cohens_d(occ_col),
            "perm_subj_mean": perm_mean, "perm_subj_ci_lo": perm_lo, "perm_subj_ci_hi": perm_hi,
            "perm_subj_median": float(np.median(perm_col)),
            "perm_wilcox_p": wilcoxon_p(perm_col), "perm_cohens_d": cohens_d(perm_col),
            "haufe_subj_mean": haufe_mean, "haufe_subj_ci_lo": haufe_lo, "haufe_subj_ci_hi": haufe_hi,
            "haufe_wilcox_p": wilcoxon_p(haufe_col), "haufe_cohens_d": cohens_d(haufe_col),
        })

    occ_fdr_p, occ_fdr_sig = fdr_correction(np.array([r["occ_wilcox_p"] for r in ch_stats]), fdr_alpha)
    perm_fdr_p, perm_fdr_sig = fdr_correction(np.array([r["perm_wilcox_p"] for r in ch_stats]), fdr_alpha)
    haufe_fdr_p, haufe_fdr_sig = fdr_correction(np.array([r["haufe_wilcox_p"] for r in ch_stats]), fdr_alpha)
    for i, row in enumerate(ch_stats):
        row["occ_fdr_p"], row["occ_fdr_sig"] = float(occ_fdr_p[i]), bool(occ_fdr_sig[i])
        row["perm_fdr_p"], row["perm_fdr_sig"] = float(perm_fdr_p[i]), bool(perm_fdr_sig[i])
        row["haufe_fdr_p"], row["haufe_fdr_sig"] = float(haufe_fdr_p[i]), bool(haufe_fdr_sig[i])
        row["both_fdr_sig"] = bool(row["occ_fdr_sig"] and row["perm_fdr_sig"])

    combined_abs = np.array([abs(r["occ_subj_mean"]) + abs(r["perm_subj_mean"]) for r in ch_stats])
    top_threshold = np.percentile(combined_abs, top_pct)
    wl_by_channel = {r["channel"]: r for r in window_level_table}

    for ch, row in enumerate(ch_stats):
        occ_sign, perm_sign = np.sign(row["occ_subj_mean"]), np.sign(row["perm_subj_mean"])
        same_sign = (occ_sign == perm_sign) and (occ_sign != 0)
        wl = wl_by_channel.get(ch, {})
        stab_frac = wl.get("subject_stability_frac", 0.0)
        in_top = combined_abs[ch] >= top_threshold

        if row["both_fdr_sig"] and same_sign and stab_frac >= tier1_frac and in_top:
            tier = "tier1_high_confidence"
        elif (row["occ_fdr_sig"] or row["perm_fdr_sig"]) and same_sign and stab_frac >= tier2_frac:
            tier = "tier2_candidate"
        elif wl.get("robust_significant", False):
            tier = "tier3_exploratory"
        else:
            tier = "tier4_not_robust"
        if tier in ("tier1_high_confidence", "tier2_candidate") and occ_sign < 0:
            tier += "_suppressive"
        row["tier"] = tier

    tier1 = sorted((r for r in ch_stats if r["tier"].startswith("tier1")),
                   key=lambda r: -(abs(r["occ_subj_mean"]) + abs(r["perm_subj_mean"])))
    tier2 = sorted((r for r in ch_stats if r["tier"].startswith("tier2")),
                   key=lambda r: -(abs(r["occ_subj_mean"]) + abs(r["perm_subj_mean"])))

    return {
        "channel_stats": ch_stats,
        "n_subjects": n_subj,
        "tier1_stability_frac": tier1_frac, "tier2_stability_frac": tier2_frac,
        "n_tier1_high_confidence": len(tier1), "n_tier2_candidate": len(tier2),
        "tier1_channels": tier1, "tier2_channels": tier2,
    }


def roi_level_stats(occ_matrix: np.ndarray, perm_matrix: np.ndarray, *,
                     rois: "OrderedDict[str, list]" = ROIS, fdr_alpha: float = 0.05) -> list[dict]:
    """ROI-level significance, reusing composite_stability.region_composite_test
    directly (already generic over any channel subset) rather than
    re-deriving the same bootstrap/Wilcoxon/Cohen's-d logic a second time.
    """
    rows = []
    for method_name, mat in [("occ", occ_matrix), ("perm", perm_matrix)]:
        recs, p_raw = [], []
        for roi, chs in rois.items():
            r = region_composite_test(mat, chs)
            r.update({"roi": roi, "method": method_name})
            recs.append(r)
            p_raw.append(r["wilcoxon_p"])
        adj, sig = fdr_correction(np.array(p_raw), fdr_alpha)
        for r, a, s in zip(recs, adj, sig):
            r["fdr_p"], r["fdr_sig"] = float(a), bool(s)
        rows.extend(recs)
    return rows


# ══════════════════════════════════════════════════════════════════════
# Sanity check: lag-block cascading randomization
# ══════════════════════════════════════════════════════════════════════
def sanity_check_lag_cascade(decoder: TRFDecoder, train_trials: list[Trial], *,
                              seg_len_s: float = 5.0, max_total_s: float = 300.0,
                              seed: int = 0) -> dict:
    """Cascading-randomization sanity check (Adebayo et al., 2018), adapted
    to TRF's structure: sanity_checks.py's cascading_randomization walks an
    nn.Module's named_children top-down, re-initializing one registered
    submodule at a time -- meaningless here, since TRFDecoder wraps a single
    flat Ridge coefficient vector, not a stack of nn.Module layers.

    Instead this progressively randomizes coef_ in LAG-BLOCK chunks --
    starting from the lag farthest from stimulus onset and working inward --
    and recomputes the Haufe channel-magnitude pattern after each step
    (holding the same EEG sample fixed throughout, so only the coefficients
    change between steps). A faithful attribution should diverge from the
    original as more of the model is destroyed; if it doesn't, that is
    itself a reportable finding, not a bug to hide.
    """
    eeg_sample = _sample_training_eeg(train_trials, seg_len_s=seg_len_s, max_total_s=max_total_s, seed=seed)
    n_lags = len(decoder.lags_)
    n_channels = decoder.model.coef_.shape[0] // n_lags
    coef_matrix = decoder.model.coef_.reshape(n_lags, n_channels).copy()
    coef_scale = float(coef_matrix.std()) or 1.0

    orig_pattern, _ = _haufe_from_eeg(decoder, eeg_sample)

    rng = np.random.RandomState(seed + 777)
    randomized = coef_matrix.copy()
    lag_order = list(range(n_lags - 1, -1, -1))  # farthest-from-stimulus lag first

    steps = []
    for step_i, lag_idx in enumerate(lag_order):
        randomized[lag_idx, :] = rng.normal(loc=0.0, scale=coef_scale, size=n_channels)
        probe = SimpleNamespace(
            lags_=decoder.lags_, _X_mean=decoder._X_mean, _X_std=decoder._X_std,
            model=SimpleNamespace(coef_=randomized.reshape(-1)),
        )
        pattern, _ = _haufe_from_eeg(probe, eeg_sample)
        rho, _ = safe_spearman(pattern, orig_pattern)
        steps.append({
            "n_lags_randomized": step_i + 1,
            "lag_index_just_randomized": int(lag_idx),
            "rho_vs_original": rho,
        })

    return {"original_pattern": orig_pattern.tolist(), "cascade_steps": steps}


# ══════════════════════════════════════════════════════════════════════
# Deletion / insertion faithfulness
# ══════════════════════════════════════════════════════════════════════
def faithfulness_curves(fold_decoders: dict, trials: list[Trial], folds: list,
                         combined_score_ranking: list[int], *, window_s: float,
                         overlap_s: float = 0.0, k_step: int = 4, n_random_perms: int = 20,
                         seed: int = 42, max_windows: Optional[int] = 200) -> dict:
    """Deletion/insertion faithfulness (Petsiuk et al. / Samek et al.),
    adapted from run_focused_xai.py's Section J for the LOSO setting: 18
    fold-specific decoders rather than one shared model. Each fold
    contributes its own held-out subject's accuracy-vs-K curve (via
    TRFDecisionWrapper); curves are then averaged across folds with a
    bootstrap CI over folds, so a channel ranking's faithfulness is
    assessed at the same "does this generalize across subjects" level as
    every other statistic in this module.

    Cost note: (n K-values) x (1 ranked + n_random_perms random) x
    (2 directions) forward passes PER FOLD -- with defaults (k_step=4 ->
    17 K-values, n_random_perms=20) that's 714 forward passes x 18 folds.
    A local benchmark found TRFDecisionWrapper.forward()'s per-window
    Python loop costs ~9ms/window, and DTU subjects have ~600 test windows
    each at a 5s decision window -- at full window counts and these
    defaults that is ~20 CPU-hours across 18 folds, not a cheap sweep.
    max_windows caps windows per fold the same way window_level_importance
    does (matching run_xai_trf_comparison.py's own --max-samples default);
    for a first run also pass a larger --faithfulness-k-step / smaller
    --faithfulness-random-perms to gauge wall-clock cost, matching
    run_focused_xai.py's own caution for this section.
    """
    k_values = list(range(0, 65, k_step))
    if k_values[-1] != 64:
        k_values.append(64)
    rng = np.random.RandomState(seed + 9000)

    per_fold_curves = {}
    for fold in folds:
        subj = trials[fold.test_idx[0]].subject_id
        decoder = fold_decoders[subj]
        test_trials = [trials[i] for i in fold.test_idx]
        eeg, att, unatt, _, _ = _windows_to_tensors(test_trials, window_s, overlap_s)
        if max_windows is not None and eeg.shape[0] > max_windows:
            keep = np.random.RandomState(seed).choice(eeg.shape[0], size=max_windows, replace=False)
            keep = torch.from_numpy(np.sort(keep)).long()
            eeg, att, unatt = eeg[keep], att[keep], unatt[keep]
        wrapper = TRFDecisionWrapper(decoder)
        wrapper.eval()

        def _prob(eeg_batch):
            wrapper.set_envelopes(att, unatt)
            with torch.no_grad():
                logits = wrapper(eeg_batch)
            return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        def _acc_for_mask(present):
            eeg_m = torch.zeros_like(eeg)
            present = sorted(present)
            if present:
                idx_t = torch.as_tensor(present, dtype=torch.long)
                eeg_m.index_copy_(2, idx_t, eeg.index_select(2, idx_t))
            probs = _prob(eeg_m)
            return float((probs > 0.5).mean())

        curves = {}
        for direction in ["deletion", "insertion"]:
            for ranking in ["combined_score", "random"]:
                curves[(direction, ranking)] = {}
                for k in k_values:
                    if ranking == "combined_score":
                        present = (combined_score_ranking[k:] if direction == "deletion"
                                   else combined_score_ranking[:k])
                        acc = _acc_for_mask(present)
                    else:
                        accs = []
                        for _ in range(n_random_perms):
                            perm = rng.permutation(64)
                            present = perm[k:] if direction == "deletion" else perm[:k]
                            accs.append(_acc_for_mask(present))
                        acc = float(np.mean(accs))
                    curves[(direction, ranking)][k] = acc
        per_fold_curves[fold.fold_id] = curves

    aggregate = {}
    for direction in ["deletion", "insertion"]:
        for ranking in ["combined_score", "random"]:
            for k in k_values:
                vals = np.array([per_fold_curves[fid][(direction, ranking)][k] for fid in per_fold_curves])
                mean, lo, hi = bootstrap_ci(vals, seed=seed)
                aggregate.setdefault((direction, ranking), {})[k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

    auc_summary = {}
    for (direction, ranking), curve in aggregate.items():
        ks_sorted = sorted(curve.keys())
        ys = [curve[k]["mean"] for k in ks_sorted]
        auc = float(np.trapz(ys, x=ks_sorted) / (ks_sorted[-1] - ks_sorted[0]))
        auc_summary.setdefault(direction, {})[ranking] = auc
    for direction in ["deletion", "insertion"]:
        auc_summary[direction]["gap"] = (
            auc_summary[direction]["combined_score"] - auc_summary[direction]["random"]
        )

    return {"per_fold_curves": per_fold_curves, "aggregate_curves": aggregate, "auc_summary": auc_summary}


# ══════════════════════════════════════════════════════════════════════
# Orchestration + CLI
# ══════════════════════════════════════════════════════════════════════
def _save_json(obj, path):
    def _default(x):
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.bool_):
            return bool(x)
        return str(x)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def run_trf_explain(*, data_dir: str, weights_dir: str, output_dir: str,
                     window_s: float = 5.0, seed: int = 42,
                     preprocess: Optional[PreprocessConfig] = None,
                     haufe_seg_len_s: float = 5.0, haufe_max_total_s: float = 300.0,
                     n_boot: int = 1000, fdr_alpha: float = 0.05,
                     faithfulness_k_step: int = 4, faithfulness_n_random_perms: int = 20,
                     max_windows_per_subject: Optional[int] = 200,
                     skip_faithfulness: bool = False, skip_sanity: bool = False) -> dict:
    """CLI-facing orchestrator: loads DTUDataset, rebuilds the same LOSO
    folds the accuracy sweep used (same seed -> same folds, deterministic),
    loads each fold's saved TRF weights, and runs the full explainability
    pipeline end to end.
    """
    preprocess = preprocess or PreprocessConfig()
    ds = DTUDataset(root=data_dir, load_audio=True, preprocess=preprocess)
    trials = list(ds.trials())
    folds = list(CV_STRATEGIES["loso"](trials, seed=seed))
    for fold in folds:
        assert_loso_fold_integrity(trials, fold)

    weights_dir = Path(weights_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfreq = float(trials[0].sfreq)

    haufe_by_subject, occ_profile, perm_profile = [], [], []
    occ_pw_pooled, perm_pw_pooled = [], []
    fold_decoders, subject_order = {}, []

    for fold in folds:
        subj = trials[fold.test_idx[0]].subject_id
        npz_path = weights_dir / f"{subj}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"No saved TRF weights for subject {subj} at {npz_path} -- "
                "run the LOSO fit with weight-saving enabled first (loso_runner.py)."
            )
        decoder = load_trf_decoder(npz_path, sfreq)
        fold_decoders[subj] = decoder
        subject_order.append(subj)

        train_trials = [trials[i] for i in fold.train_idx]
        test_trials = [trials[i] for i in fold.test_idx]

        haufe_mag, _ = haufe_pattern_for_fold(
            decoder, train_trials, seg_len_s=haufe_seg_len_s, max_total_s=haufe_max_total_s, seed=seed,
        )
        haufe_by_subject.append(haufe_mag)

        imp = window_level_importance(
            decoder, test_trials, window_s=window_s, n_boot=n_boot, seed=seed,
            max_windows=max_windows_per_subject,
        )
        occ_profile.append(imp["occ_pw"].mean(axis=0))
        perm_profile.append(imp["perm_pw"].mean(axis=0))
        occ_pw_pooled.append(imp["occ_pw"])
        perm_pw_pooled.append(imp["perm_pw"])
        print(f"  [{subj}] Haufe + window-level importance done ({imp['n_windows']} windows)")

    haufe_matrix = np.stack(haufe_by_subject)
    occ_matrix = np.stack(occ_profile)
    perm_matrix = np.stack(perm_profile)
    occ_pw_pooled = np.concatenate(occ_pw_pooled, axis=0)
    perm_pw_pooled = np.concatenate(perm_pw_pooled, axis=0)

    window_table = window_level_combined_table(
        occ_pw_pooled, perm_pw_pooled, occ_matrix, perm_matrix, fdr_alpha=fdr_alpha, seed=seed,
    )
    subject_stats = subject_level_stats(occ_matrix, perm_matrix, haufe_matrix, window_table, fdr_alpha=fdr_alpha)
    roi_stats = roi_level_stats(occ_matrix, perm_matrix, fdr_alpha=fdr_alpha)
    ranking = [c["channel"] for c in sorted(window_table, key=lambda c: abs(c["combined_score"]), reverse=True)]

    sanity_results = None
    if not skip_sanity:
        sanity_results = {}
        for fold in folds:
            subj = trials[fold.test_idx[0]].subject_id
            train_trials = [trials[i] for i in fold.train_idx]
            sanity_results[subj] = sanity_check_lag_cascade(
                fold_decoders[subj], train_trials, seg_len_s=haufe_seg_len_s,
                max_total_s=haufe_max_total_s, seed=seed,
            )
            print(f"  [{subj}] sanity check (lag cascade) done")

    faithfulness_results = None
    if not skip_faithfulness:
        faithfulness_results = faithfulness_curves(
            fold_decoders, trials, folds, ranking, window_s=window_s,
            k_step=faithfulness_k_step, n_random_perms=faithfulness_n_random_perms, seed=seed,
            max_windows=max_windows_per_subject,
        )
        print("  Faithfulness curves done")

    _save_json(window_table, out_dir / "window_level_channel_importance.json")
    _save_json(subject_stats, out_dir / "subject_level_channel_stats.json")
    _save_json(roi_stats, out_dir / "roi_level_stats.json")
    _save_json({"channel_magnitude_by_subject": haufe_matrix.tolist(), "subjects": subject_order},
               out_dir / "haufe_patterns.json")
    if sanity_results is not None:
        _save_json(sanity_results, out_dir / "sanity_check_lag_cascade.json")
    if faithfulness_results is not None:
        _save_json(
            {
                "auc_summary": faithfulness_results["auc_summary"],
                "aggregate_curves": {
                    f"{d}|{r}": v for (d, r), v in faithfulness_results["aggregate_curves"].items()
                },
            },
            out_dir / "faithfulness_summary.json",
        )

    return {
        "n_subjects": len(folds),
        "window_table": window_table,
        "subject_stats": subject_stats,
        "roi_stats": roi_stats,
        "sanity_results": sanity_results,
        "faithfulness_results": faithfulness_results,
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="DTU LOSO TRF explainability: Haufe patterns, window/subject-level "
                    "channel importance, lag-cascade sanity check, faithfulness curves.")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--weights-dir", type=str, required=True,
                     help="Directory of per-subject trf_weights/*.npz saved by loso_runner.py.")
    ap.add_argument("--output-dir", type=str, default="results_dtu_loso_trf_explain")
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    ap.add_argument("--skip-faithfulness", action="store_true")
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--faithfulness-k-step", type=int, default=4)
    ap.add_argument("--faithfulness-random-perms", type=int, default=20)
    ap.add_argument("--max-windows-per-subject", type=int, default=200,
                     help="Cap on test windows used per fold for window_level_importance and "
                          "faithfulness_curves (deterministic subsample). Matches "
                          "run_xai_trf_comparison.py's own --max-samples default. Pass 0 to disable.")
    args = ap.parse_args(argv)

    summary = run_trf_explain(
        data_dir=args.data_dir, weights_dir=args.weights_dir, output_dir=args.output_dir,
        window_s=args.window_s, seed=args.seed, n_boot=args.n_boot, fdr_alpha=args.fdr_alpha,
        skip_faithfulness=args.skip_faithfulness, skip_sanity=args.skip_sanity,
        faithfulness_k_step=args.faithfulness_k_step,
        faithfulness_n_random_perms=args.faithfulness_random_perms,
        max_windows_per_subject=(args.max_windows_per_subject or None),
    )
    print(f"Done. Tier-1 (high-confidence): {summary['subject_stats']['n_tier1_high_confidence']}, "
          f"Tier-2 (candidate): {summary['subject_stats']['n_tier2_candidate']}")


if __name__ == "__main__":
    main()
