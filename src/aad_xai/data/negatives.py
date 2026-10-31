"""Same-trial, same-talker negative candidates for match-mismatch training.

Each training window gets extra candidate envelopes cut from the *same trial*
at other times (at least ``min_gap_s`` away from the window): ``n_per_stream``
from stream A and ``n_per_stream`` from stream B. Because both talkers appear
among the negatives, talker identity cannot single out the attended candidate;
only the EEG-speech temporal match can.

Candidate order returned per window:
``[A, B, A_neg_1 .. A_neg_n, B_neg_1 .. B_neg_n]``, so the label (0 = A,
1 = B) is still the index of the target candidate.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .torch_dataset import WindowedEEGAudioDataset


def sample_negative_starts(
    rng: np.random.Generator, start: int, n_times: int, win_len: int, min_gap: int, n: int,
) -> np.ndarray:
    """Draw ``n`` window starts in ``[0, n_times - win_len]`` with ``|s - start| >= min_gap``.

    Drawn with replacement, uniformly over the allowed starts.

    Returns
    -------
    (n,) int array
    """
    allowed = np.arange(n_times - win_len + 1)
    allowed = allowed[np.abs(allowed - start) >= min_gap]
    if allowed.size == 0:
        raise ValueError(
            f"no start at least {min_gap} samples from {start} in a trial of {n_times} samples "
            f"(window {win_len})"
        )
    return rng.choice(allowed, size=n, replace=True)


class SameTrialNegativesDataset(Dataset):
    """Wrap a WindowedEEGAudioDataset and append same-trial negative envelopes.

    Parameters
    ----------
    base : WindowedEEGAudioDataset
    n_per_stream : int
        Negatives drawn from each of stream A and stream B.
    min_gap_s : float
        Minimum distance between a negative's start and the window's start.
    seed : int
        Seed of the dataset's own generator. Negatives are redrawn every time
        an item is read, so each epoch sees new ones (reproducibly).

    Items: ``(eeg (C, T), env (2 + 2n, T) or (2 + 2n, K, T), label)``.
    """

    def __init__(self, base: WindowedEEGAudioDataset, n_per_stream: int, min_gap_s: float, seed: int = 0):
        if n_per_stream < 1:
            raise ValueError("n_per_stream must be >= 1")
        self.base = base
        self.n_per_stream = n_per_stream
        self.rng = np.random.default_rng(seed)
        self.win_len = int(round(base.window_s * base.trials[0].sfreq)) if base.trials else 0
        self.min_gap = int(round(min_gap_s * base.trials[0].sfreq)) if base.trials else 0

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        eeg, env, label = self.base[idx]
        win = self.base.window_indices[idx]
        trial = self.base.trials[self.base._trial_idx[idx]]
        T = win.stop - win.start
        n_times = min(trial.eeg.shape[1], trial.audio_a.shape[-1], trial.audio_b.shape[-1])
        negs = []
        for stream in (trial.audio_a, trial.audio_b):
            for s in sample_negative_starts(self.rng, win.start, n_times, T, self.min_gap, self.n_per_stream):
                negs.append(stream[..., s:s + T])                     # (T,) or (K, T)
        neg = torch.as_tensor(np.stack(negs), dtype=env.dtype)      # (2n, T) or (2n, K, T)
        return eeg, torch.cat([env, neg], dim=0), label


def cross_trial_permutation(trial_ids) -> np.ndarray:
    """Permutation pairing every window with a window of a *different* trial.

    Used for the trial-swap control: score each EEG window against the
    candidate envelopes of another trial. Windows are ordered by trial and the
    order is rolled by the largest trial's window count, so no window maps
    into its own trial.

    Parameters
    ----------
    trial_ids : sequence of length N

    Returns
    -------
    (N,) int array ``perm``; window i is paired with window ``perm[i]``.
    """
    trial_ids = np.asarray(trial_ids)
    order = np.argsort(trial_ids, kind="stable")
    _, counts = np.unique(trial_ids, return_counts=True)
    if counts.max() > len(trial_ids) - counts.max():
        raise ValueError("one trial holds more than half the windows; no cross-trial pairing exists")
    perm = np.empty(len(trial_ids), dtype=int)
    perm[order] = np.roll(order, -int(counts.max()))
    return perm
