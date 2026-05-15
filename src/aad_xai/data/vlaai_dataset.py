"""DTU dataset adapter for VLAAI XAI experiments.

Loads ``.npz`` files from ``external/vlaai/evaluation_datasets/DTU/``,
windows them, and produces ``(eeg, env_attended, env_unattended, label)``
tuples for the AADDecisionWrapper.

Since the DTU single-speaker trials only contain the attended envelope,
the "unattended" envelope is constructed by pairing each trial with a
mismatched trial from the same subject.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def window_data(data: np.ndarray, window_length: int, hop: int) -> np.ndarray:
    """Window data into overlapping segments.

    Parameters
    ----------
    data : (n_samples, n_channels)
    window_length : int — samples per window
    hop : int — hop size in samples

    Returns
    -------
    (n_windows, window_length, n_channels)
    """
    n_windows = (data.shape[0] - window_length) // hop
    out = np.empty((n_windows, window_length, data.shape[1]), dtype=data.dtype)
    for i in range(n_windows):
        out[i] = data[i * hop: i * hop + window_length]
    return out


def load_dtu_trials(
    data_dir: str | Path,
    subjects: Sequence[str] | None = None,
) -> dict[str, list[dict]]:
    """Load all DTU ``.npz`` trial files, grouped by subject.

    Parameters
    ----------
    data_dir : path to ``evaluation_datasets/DTU/``
    subjects : subset of subject IDs (e.g. ['S1', 'S2']). None = all.

    Returns
    -------
    dict mapping subject_id → list of dicts with keys 'eeg', 'envelope', 'path'.
    """
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("*.npz"))

    trials: dict[str, list[dict]] = {}
    for f in npz_files:
        # Filenames like DTU_S1_000.npz
        parts = f.stem.split("_")
        subj = parts[1] if len(parts) >= 2 else f.stem

        if subjects is not None and subj not in subjects:
            continue

        data = np.load(str(f))
        trials.setdefault(subj, []).append({
            "eeg": data["eeg"],
            "envelope": data["envelope"],
            "path": str(f),
        })
    return trials


class VLAAIDTUDataset(Dataset):
    """PyTorch Dataset for VLAAI XAI on DTU data.

    Each sample is ``(eeg_window, env_attended, env_unattended, label)``
    where ``label = 1`` (attended class).

    The "unattended" envelope is taken from a different trial of the same
    subject (circular pairing).

    Parameters
    ----------
    data_dir : path to ``evaluation_datasets/DTU/``
    window_length : int — window size in samples (default 320 = 5s @ 64 Hz)
    hop : int — hop size in samples (default 64 = 80 % overlap = 1 s hop)
    subjects : subset of subjects to include (None = all)
    standardize : bool — z-score EEG and envelope per trial
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_length: int = 320,
        hop: int = 64,
        subjects: Sequence[str] | None = None,
        standardize: bool = True,
    ):
        self.window_length = window_length
        self.hop = hop

        trials_by_subj = load_dtu_trials(data_dir, subjects)

        # Build windowed samples with paired attended/unattended envelopes
        self._eeg: list[np.ndarray] = []
        self._env_att: list[np.ndarray] = []
        self._env_unatt: list[np.ndarray] = []
        self._subject_ids: list[str] = []

        for subj, trial_list in trials_by_subj.items():
            n_trials = len(trial_list)
            for t_idx, trial in enumerate(trial_list):
                eeg = trial["eeg"].astype(np.float32)
                env = trial["envelope"].astype(np.float32)

                if standardize:
                    eeg = ((eeg - eeg.mean(axis=0, keepdims=True)) / (eeg.std(axis=0, keepdims=True) + 1e-8)).astype(np.float32)
                    env = ((env - env.mean(axis=0, keepdims=True)) / (env.std(axis=0, keepdims=True) + 1e-8)).astype(np.float32)

                # "Unattended" envelope: circular time-shift by half the
                # signal length.  This breaks temporal alignment while
                # preserving spectral statistics — a standard proxy when
                # only the attended envelope is available.
                shift = env.shape[0] // 2
                env_unatt = np.roll(env, shift, axis=0)

                # Window all three signals
                eeg_w = window_data(eeg, window_length, hop)
                env_w = window_data(env, window_length, hop)
                env_unatt_w = window_data(env_unatt, window_length, hop)

                n = min(eeg_w.shape[0], env_w.shape[0], env_unatt_w.shape[0])
                self._eeg.append(eeg_w[:n])
                self._env_att.append(env_w[:n])
                self._env_unatt.append(env_unatt_w[:n])
                self._subject_ids.extend([subj] * n)

        self._eeg_arr = np.concatenate(self._eeg, axis=0)
        self._env_att_arr = np.concatenate(self._env_att, axis=0)
        self._env_unatt_arr = np.concatenate(self._env_unatt, axis=0)
        self.subject_ids = np.array(self._subject_ids)

    def __len__(self) -> int:
        return self._eeg_arr.shape[0]

    def __getitem__(self, idx: int):
        eeg = torch.from_numpy(self._eeg_arr[idx])           # (T, 64)
        env_att = torch.from_numpy(self._env_att_arr[idx])    # (T, 1)
        env_unatt = torch.from_numpy(self._env_unatt_arr[idx])  # (T, 1)
        label = torch.tensor(1, dtype=torch.long)  # always "attended"
        return eeg, env_att, env_unatt, label
