"""PyTorch Dataset wrapper that chains: Trial list → windowing → tensors.

This module takes pre-split trials (belonging to *one* partition only) and
creates a ``torch.utils.data.Dataset`` of ``(eeg_window, label)`` pairs.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import Trial
from .windowing import make_windows, WindowIndex


class WindowedEEGDataset(Dataset):
    """Map-style dataset of EEG windows from a list of Trial objects.

    Parameters
    ----------
    trials : Sequence[Trial]
        Trials belonging to **one** split partition (train / val / test).
    window_s : float
        Window length in seconds.
    overlap_s : float
        Overlap between consecutive windows in seconds.
        Overlap is allowed *within* this split only.
    """

    def __init__(
        self,
        trials: Sequence[Trial],
        window_s: float,
        overlap_s: float = 0.0,
    ):
        self.trials = list(trials)
        self.window_s = window_s
        self.overlap_s = overlap_s

        # Build flat index of all windows across trials
        self._windows: list[WindowIndex] = []
        self._trial_idx: list[int] = []  # maps window → trial
        for t_idx, trial in enumerate(self.trials):
            wins = make_windows(
                n_times=trial.eeg.shape[1],
                sfreq=trial.sfreq,
                window_s=window_s,
                overlap_s=overlap_s,
                label=trial.label,
                subject_id=trial.subject_id,
                trial_id=trial.trial_id,
            )
            self._windows.extend(wins)
            self._trial_idx.extend([t_idx] * len(wins))

    # ------------------------------------------------------------------
    @property
    def window_indices(self) -> list[WindowIndex]:
        """Expose window metadata for leakage assertions."""
        return self._windows

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        win = self._windows[idx]
        trial = self.trials[self._trial_idx[idx]]
        eeg_slice = trial.eeg[:, win.start : win.stop]  # (C, T)
        x = torch.as_tensor(eeg_slice, dtype=torch.float32)
        return x, win.label


class WindowedEEGAudioDataset(Dataset):
    """Map-style dataset of EEG + 2-stream audio windows from Trial objects.

    Returns ``(eeg_window, env_window, label)`` where:
      - eeg_window: (C, T)
      - env_window: (2, T)
      - label: int (0/1)
    """

    def __init__(
        self,
        trials: Sequence[Trial],
        window_s: float,
        overlap_s: float = 0.0,
    ):
        self.trials = [
            t for t in trials
            if (t.audio_a is not None and t.audio_b is not None)
        ]
        self.window_s = window_s
        self.overlap_s = overlap_s

        self._windows: list[WindowIndex] = []
        self._trial_idx: list[int] = []
        for t_idx, trial in enumerate(self.trials):
            n_times = min(
                trial.eeg.shape[1],
                len(trial.audio_a),
                len(trial.audio_b),
            )
            wins = make_windows(
                n_times=n_times,
                sfreq=trial.sfreq,
                window_s=window_s,
                overlap_s=overlap_s,
                label=trial.label,
                subject_id=trial.subject_id,
                trial_id=trial.trial_id,
            )
            self._windows.extend(wins)
            self._trial_idx.extend([t_idx] * len(wins))

    @property
    def window_indices(self) -> list[WindowIndex]:
        return self._windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        win = self._windows[idx]
        trial = self.trials[self._trial_idx[idx]]

        eeg = trial.eeg[:, win.start:win.stop]
        env = np.stack(
            [
                trial.audio_a[win.start:win.stop],
                trial.audio_b[win.start:win.stop],
            ],
            axis=0,
        )

        x = torch.as_tensor(eeg, dtype=torch.float32)
        e = torch.as_tensor(env, dtype=torch.float32)
        return x, e, win.label
