from __future__ import annotations
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Simple classification accuracy."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float((y_true == y_pred).mean())


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval.

    Parameters
    ----------
    values : array-like
        1-D array of per-seed (or per-fold) metric values.
    confidence : float
        Confidence level (default 0.95 → 95 % CI).
    n_boot : int
        Number of bootstrap resamples.

    Returns
    -------
    mean, ci_low, ci_high : float
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2:
        m = float(values.mean())
        return m, m, m

    boot_means = np.array(
        [rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)]
    )
    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


def binary_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute confusion matrix counts for binary labels {0,1}."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total = int(y_true.size)
    acc = float((y_true == y_pred).mean()) if total > 0 else 0.0
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
        "accuracy": acc,
    }

