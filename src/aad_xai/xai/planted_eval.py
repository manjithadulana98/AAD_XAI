"""Pre-registered evaluation of the planted-ground-truth benchmark (interpretability level 3).

Pure functions over the per-run records written by notebooks/kaggle_run_xai_level3*.py. Each
record holds "test_acc", "val" (class-balanced validation accuracy) and "metrics" (per-run
AUROCs, decoy ranks, band ranks, edge-removal drops).

The level-3 rules (RESULTS_LOG 2026-11-09):
- V1 median test accuracy >= 0.70; V2 band-stop ranks the planted band first in >= 90% of seeds;
  V3 removing the planted edges drops the margin more than removing random edges (one-sided
  Wilcoxon over seeds, p < 0.05).
- Electrodes: median AUROC >= 0.9 in every condition. Decoy: never in the top 6 of any seed.
- Bands: band attribution ranks the planted band first in >= 90% of seeds in every condition.
- Edges (only if V3 holds in every condition): median AUROC >= 0.9 in every condition.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

CH_METHODS = ("occlusion_zero", "occlusion_permute", "ig", "attention")
EDGE_METHODS = ("attention", "attn_x_grad")


def select_main_snr(acc_by_snr: dict, threshold: float = 0.70):
    """Lowest SNR whose median accuracy over seeds is >= threshold; None if no SNR reaches it.
    acc_by_snr: {snr_db: [test accuracy per seed]}. Non-monotone curves are not smoothed."""
    reached = [float(s) for s, accs in acc_by_snr.items() if len(accs) and np.median(accs) >= threshold]
    return min(reached) if reached else None


def is_learned(val: float, threshold: float = 0.60) -> bool:
    """A seed counts as learned if its class-balanced validation accuracy is >= threshold."""
    return bool(val >= threshold)


def regime(accs, lo: float = 0.60, hi: float = 0.85) -> str:
    """'realistic' if the median is <= hi and at least half the seeds lie in [lo, hi];
    'near ceiling' if the median is above hi; otherwise 'spread' (seeds split, e.g. learned or not)."""
    accs = np.asarray(accs, dtype=float)
    if np.median(accs) > hi:
        return "near ceiling"
    return "realistic" if np.mean((accs >= lo) & (accs <= hi)) >= 0.5 else "spread"


def _one_sided_greater(a, b) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) < 2 or not np.any(d):
        return 1.0
    return float(wilcoxon(d, alternative="greater").pvalue)


def verdicts(main: dict, *, main_acc: float = 0.70, auroc_pass: float = 0.9, top: int = 6,
             band_share: float = 0.9, alpha: float = 0.05) -> dict:
    """main: {condition: [records at that condition's main SNR]} -> the level-3 verdicts."""
    out = {"n": {c: len(rs) for c, rs in main.items()}, "validity": {}}
    for c, rs in main.items():
        M = [r["metrics"] for r in rs]
        p3 = _one_sided_greater([m["edge_gt_drop"] for m in M], [m["edge_random_drop"] for m in M])
        out["validity"][c] = {
            "median_acc": float(np.median([r["test_acc"] for r in rs])),
            "V1": bool(np.median([r["test_acc"] for r in rs]) >= main_acc),
            "V2": bool(np.mean([m["band_stop_rank"] == 1 for m in M]) >= band_share),
            "V3": bool(p3 < alpha), "p3": p3}
    out["electrodes"], out["decoy"] = {}, {}
    for m in CH_METHODS:
        med = {c: float(np.median([r["metrics"][f"auroc_{m}"] for r in rs])) for c, rs in main.items()}
        out["electrodes"][m] = {"median_auroc": med, "pass": bool(min(med.values()) >= auroc_pass)}
        best = min(r["metrics"][f"decoy_rank_{m}"] for rs in main.values() for r in rs)
        out["decoy"][m] = {"best_rank": int(best), "pass": bool(best > top)}
    share = {c: float(np.mean([r["metrics"]["band_attr_rank"] == 1 for r in rs])) for c, rs in main.items()}
    out["bands"] = {"share_first": share, "pass": bool(min(share.values()) >= band_share)}
    if all(v["V3"] for v in out["validity"].values()):
        out["edges"] = {}
        for m in EDGE_METHODS:
            med = {c: float(np.median([r["metrics"][f"edge_auroc_{m}"] for r in rs])) for c, rs in main.items()}
            out["edges"][m] = {"median_auroc": med, "pass": bool(min(med.values()) >= auroc_pass)}
    else:
        out["edges"] = None
    return out


def failure_kind(all_pass: bool, learned_pass: bool) -> str:
    """How to label a method verdict once it is computed on all seeds and on learned seeds only."""
    if all_pass:
        return "pass"
    return "model failure" if learned_pass else "method failure"
