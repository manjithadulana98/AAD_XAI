"""Longer decision windows from consecutive short windows: grouping, summed-score
decisions, and correlations recomputed from summed statistics."""
from __future__ import annotations

import numpy as np
import pytest

from aad_xai.evaluation.decision_windows import (
    accuracy_counts,
    corr_from_stats,
    group_windows,
    recomputed_corr_decisions,
    summed_score_decisions,
    window_stats,
)


def _meta():
    # Trial T1 (S1, label 1): 5 windows given out of order; trial T2 (S2, label 0): 3 windows.
    subjects = ["S1"] * 5 + ["S2"] * 3
    trials = ["T1"] * 5 + ["T2"] * 3
    starts = [256, 0, 128, 512, 384, 0, 64, 128]
    labels = [1] * 5 + [0] * 3
    return subjects, trials, starts, labels


def test_grouping_is_consecutive_and_drops_remainder():
    groups = group_windows(*_meta(), n_per_decision=2)
    assert [(s, l) for s, l, _ in groups] == [("S1", 1), ("S1", 1), ("S2", 0)]
    starts = _meta()[2]
    assert [[starts[i] for i in idx] for _, _, idx in groups] == [[0, 128], [256, 384], [0, 64]]


def test_single_window_groups_reproduce_window_decisions():
    subjects, trials, starts, labels = _meta()
    scores = np.array([0.3, -1.0, 2.0, 0.1, -0.2, -0.5, 0.4, -0.1])
    groups = group_windows(subjects, trials, starts, labels, 1)
    dec = summed_score_decisions(scores, groups)
    expect = {}
    for s, x, y in zip(subjects, scores, labels):
        c = expect.setdefault(s, [0, 0]); c[0] += int(int(x > 0) == y); c[1] += 1
    assert accuracy_counts(groups, dec) == expect


def test_summed_scores_and_tie_goes_to_a():
    subjects, trials, starts, labels = _meta()
    scores = np.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # T1 by start: -1, 1, 1, 0, 0
    groups = group_windows(subjects, trials, starts, labels, 2)
    np.testing.assert_array_equal(summed_score_decisions(scores, groups), [0, 1, 0])
    assert accuracy_counts(groups, [0, 1, 0]) == {"S1": [1, 2], "S2": [1, 1]}


def test_inconsistent_trial_raises():
    with pytest.raises(ValueError):
        group_windows(["S1", "S1"], ["T", "T"], [0, 1], [0, 1], 1)
    with pytest.raises(ValueError):
        group_windows(["S1"], ["T"], [0], [0], 0)


def test_corr_from_stats_matches_numpy():
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((300, 3))
    env = rng.standard_normal((2, 3, 300)) + 0.5 * pred.T[None]
    r = corr_from_stats(window_stats(pred, env), 300)                 # (2, 3)
    for c in range(2):
        for k in range(3):
            assert r[c, k] == pytest.approx(np.corrcoef(pred[:, k], env[c, k])[0, 1], abs=1e-10)


def test_recomputed_corr_equals_corr_of_concatenation():
    rng = np.random.default_rng(1)
    T, K = 64, 2
    preds = [rng.standard_normal((T, K)) for _ in range(4)]
    envs = [rng.standard_normal((2, K, T)) + 0.3 * p.T[None] * np.array([0.2, 1.0])[:, None, None] for p in preds]
    stats = np.stack([window_stats(p, e) for p, e in zip(preds, envs)])  # (4, 2, K, 5)
    groups = group_windows(["S1"] * 4, ["T"] * 4, [0, T, 2 * T, 3 * T], [1] * 4, 4)
    P = np.concatenate(preds)                                          # (4T, K)
    E = np.concatenate(envs, axis=-1)                                  # (2, K, 4T)
    r = np.array([[np.corrcoef(P[:, k], E[c, k])[0, 1] for k in range(K)] for c in range(2)])
    np.testing.assert_allclose(corr_from_stats(stats[groups[0][2]].sum(0), 4 * T), r, atol=1e-10)
    assert recomputed_corr_decisions(stats, T, groups)[0] == int(r[1].mean() > r[0].mean())
