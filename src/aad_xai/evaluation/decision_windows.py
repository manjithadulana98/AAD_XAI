"""Longer decision windows from consecutive short-window outputs.

Models are scored on fixed short windows (e.g. 5 s). A decision on a longer
window (10, 20, 40 s) combines the k = L / short consecutive, non-overlapping
short windows of the same trial. The remainder at the end of a trial that
does not fill a whole decision window is dropped.

Score convention (same as argmax over [score_A, score_B] with ties -> A):
score > 0 predicts candidate B (label 1), otherwise candidate A (label 0).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

STAT_NAMES = ("sum_p", "sum_y", "sum_pp", "sum_yy", "sum_py")


def group_windows(
    subjects: Sequence[str],
    trial_ids: Sequence[str],
    starts: Sequence[int],
    labels: Sequence[int],
    n_per_decision: int,
) -> list[tuple[str, int, np.ndarray]]:
    """Group short windows into decision windows.

    Parameters
    ----------
    subjects, trial_ids, starts, labels : sequences of length N (one per short window)
    n_per_decision : int
        Short windows per decision window (1 = the short windows themselves).

    Returns
    -------
    list of (subject_id, label, indices) -- ``indices`` (n_per_decision,) into
    the N short windows, consecutive in time within one trial.
    """
    if n_per_decision < 1:
        raise ValueError("n_per_decision must be >= 1")
    by_trial: dict[str, list[int]] = {}
    for i, t in enumerate(trial_ids):
        by_trial.setdefault(t, []).append(i)
    groups = []
    for idx in by_trial.values():
        idx = sorted(idx, key=lambda i: starts[i])
        if len({labels[i] for i in idx}) != 1 or len({subjects[i] for i in idx}) != 1:
            raise ValueError(f"trial {trial_ids[idx[0]]} mixes labels or subjects")
        for g in range(len(idx) // n_per_decision):
            sel = np.asarray(idx[g * n_per_decision:(g + 1) * n_per_decision])
            groups.append((subjects[sel[0]], int(labels[sel[0]]), sel))
    return groups


def accuracy_counts(
    groups: Sequence[tuple[str, int, np.ndarray]], decisions: Sequence[int],
) -> dict[str, list[int]]:
    """Per-subject [correct, total] given one 0/1 decision per group."""
    out: dict[str, list[int]] = {}
    for (subj, label, _), d in zip(groups, decisions):
        c = out.setdefault(subj, [0, 0])
        c[0] += int(int(d) == label)
        c[1] += 1
    return out


def summed_score_decisions(scores: np.ndarray, groups) -> np.ndarray:
    """Decision per group from the sum of its short-window scores (N,) -> (n_groups,)."""
    scores = np.asarray(scores, dtype=np.float64)
    return np.array([int(scores[idx].sum() > 0) for _, _, idx in groups], dtype=int)


def corr_from_stats(stats: np.ndarray, n: int | np.ndarray) -> np.ndarray:
    """Pearson correlation from summed sufficient statistics.

    Parameters
    ----------
    stats : array (..., 5) -- [sum_p, sum_y, sum_pp, sum_yy, sum_py] over n samples
    n : number of samples summed (scalar or broadcastable to stats[..., 0])

    Returns
    -------
    array (...) of correlations.
    """
    sp, sy, spp, syy, spy = np.moveaxis(np.asarray(stats, dtype=np.float64), -1, 0)
    cov = spy / n - (sp / n) * (sy / n)
    var_p = spp / n - (sp / n) ** 2
    var_y = syy / n - (sy / n) ** 2
    return cov / np.sqrt(np.clip(var_p * var_y, 1e-24, None))


def window_stats(pred: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Sufficient statistics of one short window.

    Parameters
    ----------
    pred : (T, K) reconstructed envelope bands
    env : (2, K, T) candidate envelope bands

    Returns
    -------
    (2, K, 5) float64 -- STAT_NAMES per candidate and band.
    """
    p = np.asarray(pred, dtype=np.float64).T[None]               # (1, K, T)
    y = np.asarray(env, dtype=np.float64)                        # (2, K, T)
    p = np.broadcast_to(p, y.shape)
    return np.stack([p.sum(-1), y.sum(-1), (p * p).sum(-1), (y * y).sum(-1), (p * y).sum(-1)], axis=-1)


def recomputed_corr_decisions(stats: np.ndarray, n_per_window: int, groups) -> np.ndarray:
    """Decision per group from correlations recomputed over the whole decision window.

    stats : (N, 2, K, 5) per short window. Decision = mean over bands of
    r_B - r_A > 0.
    """
    out = []
    for _, _, idx in groups:
        r = corr_from_stats(stats[idx].sum(0), n_per_window * len(idx))   # (2, K)
        out.append(int(r[1].mean() - r[0].mean() > 0))
    return np.array(out, dtype=int)
