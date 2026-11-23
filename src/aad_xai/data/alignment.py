"""Per-subject Euclidean alignment (EA) of EEG trials (He & Wu, IEEE TBME 2020).

For each subject, the reference R is the mean of that subject's trial
covariance matrices. Every trial is mapped to R^{-1/2} X, so each subject's
mean covariance becomes (close to) the identity. This removes subject-specific
scale and spatial mixing before a subject-independent model sees the data.

Uses EEG only (no labels, no audio). For held-out subjects it uses their
unlabelled EEG from the same split, so it is transductive: say so when
reporting results.

Interpretability caveat: R^{-1/2} mixes channels. It is the symmetric (ZCA)
whitening, the one closest to the identity, but an aligned "channel" is still
a weighted sum of electrodes, not one electrode.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from .base import Trial


def euclidean_reference(eegs: Sequence[np.ndarray], shrinkage: float = 1e-3) -> np.ndarray:
    """Mean trial covariance with a small ridge, so it is invertible.

    Parameters
    ----------
    eegs : sequence of arrays, each (C, T_i)
        One subject's trials. Each trial is centred per channel and weighted
        equally, whatever its length.
    shrinkage : float
        Ridge added to the diagonal, as a fraction of the mean eigenvalue.
        Needed for average-referenced EEG, whose covariance has rank C - 1.

    Returns
    -------
    ndarray, shape (C, C), symmetric positive definite.
    """
    if len(eegs) == 0:
        raise ValueError("need at least one trial")
    covs = []
    for x in eegs:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(axis=1, keepdims=True)                 # (C, T)
        covs.append(x @ x.T / x.shape[1])                     # (C, C)
    R = np.mean(covs, axis=0)                                 # (C, C)
    C = R.shape[0]
    return R + shrinkage * (np.trace(R) / C) * np.eye(C)


def inverse_sqrtm(R: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root R^{-1/2} of an SPD matrix (C, C)."""
    evals, evecs = np.linalg.eigh(R)
    if evals.min() <= 0:
        raise ValueError(f"matrix not positive definite (min eigenvalue {evals.min():.3g})")
    return (evecs / np.sqrt(evals)) @ evecs.T                 # (C, C)


def euclidean_align(trials: Sequence[Trial], shrinkage: float = 1e-3) -> list[Trial]:
    """Align every subject's trials with that subject's own reference.

    The reference of a subject is computed only from that subject's trials in
    ``trials``. Call it separately for the train, val and test lists of a fold,
    so that no split informs another.

    Returns new Trial objects in the same order. Only ``eeg`` changes; it keeps
    its shape (C, T) and dtype. Labels and audio are untouched.
    """
    by_subject = _by_subject(trials)
    W = {s: inverse_sqrtm(euclidean_reference([trials[i].eeg for i in idx], shrinkage))   # (C, C)
         for s, idx in by_subject.items()}
    return _apply(trials, W)


def calibration_eeg(trials: Sequence[Trial], seconds: float | None = None) -> dict[str, np.ndarray]:
    """Per subject, the first ``seconds`` of EEG of ``trials``, taken in list order.

    Each trial is centred per channel and the pieces are joined in time, so the
    calibration covariance weights every sample equally. The trial that crosses
    the budget is cut. ``seconds=None`` takes all of the subject's EEG.

    Returns {subject_id: (C, T_cal)} float64. Raises ValueError if a subject has
    less EEG than ``seconds``.
    """
    out: dict[str, np.ndarray] = {}
    for s, idx in _by_subject(trials).items():
        pieces, n = [], 0
        need = None if seconds is None else int(round(seconds * trials[idx[0]].sfreq))
        for i in idx:
            if need is not None and n >= need:
                break
            x = np.asarray(trials[i].eeg, dtype=np.float64)
            x = x - x.mean(axis=1, keepdims=True)                 # (C, T_i)
            if need is not None:
                x = x[:, :need - n]
            pieces.append(x)
            n += x.shape[1]
        if need is not None and n < need:
            raise ValueError(f"{s}: {n} calibration samples, {need} needed")
        out[s] = np.concatenate(pieces, axis=1)                   # (C, T_cal)
    return out


def euclidean_align_to(trials: Sequence[Trial], calibration: dict[str, np.ndarray],
                       shrinkage: float = 1e-3) -> list[Trial]:
    """Align every subject's trials with a reference from that subject's calibration EEG only.

    ``calibration`` maps subject_id to (C, T_cal) EEG, e.g. from :func:`calibration_eeg`
    on recordings that are not scored. The trials themselves do not enter the
    reference, so unlike :func:`euclidean_align` this is not transductive for them.
    Returns new Trial objects in the same order; only ``eeg`` changes.
    """
    missing = sorted({t.subject_id for t in trials} - set(calibration))
    if missing:
        raise KeyError(f"no calibration EEG for {missing}")
    W = {s: inverse_sqrtm(euclidean_reference([calibration[s]], shrinkage))         # (C, C)
         for s in _by_subject(trials)}
    return _apply(trials, W)


def _by_subject(trials: Sequence[Trial]) -> dict[str, list[int]]:
    by_subject: dict[str, list[int]] = {}
    for i, t in enumerate(trials):
        by_subject.setdefault(t.subject_id, []).append(i)
    return by_subject


def _apply(trials: Sequence[Trial], W: dict[str, np.ndarray]) -> list[Trial]:
    """W[subject] @ eeg for every trial; shape (C, T) and dtype kept."""
    return [dataclasses.replace(t, eeg=(W[t.subject_id] @ t.eeg.astype(np.float64)).astype(t.eeg.dtype))
            for t in trials]
