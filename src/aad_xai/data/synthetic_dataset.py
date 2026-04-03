"""Synthetic multi-subject dataset for smoke-testing the full pipeline.

Generates fake EEG trials with a weak per-class signal so that models can
learn *something*, while providing realistic Trial objects with unique
subject_ids, trial_ids, and group_ids suitable for leakage-safe splitting.
"""
from __future__ import annotations

import numpy as np
from .base import BaseDataset, Trial


class SyntheticDataset(BaseDataset):
    """Generate synthetic EEG trials for pipeline smoke-tests.

    Parameters
    ----------
    n_subjects : int
        Number of simulated subjects (≥3 for a valid 3-way split).
    trials_per_subject : int
        Number of trials each subject contributes.
    n_channels : int
        Number of EEG channels.
    duration_s : float
        Duration of each trial in seconds.
    sfreq : float
        Sampling frequency (Hz).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_subjects: int = 8,
        trials_per_subject: int = 20,
        n_channels: int = 64,
        duration_s: float = 10.0,
        sfreq: float = 128.0,
        seed: int = 0,
    ):
        self.n_subjects = n_subjects
        self.trials_per_subject = trials_per_subject
        self.n_channels = n_channels
        self.duration_s = duration_s
        self.sfreq = sfreq
        self.seed = seed

    def trials(self):
        rng = np.random.default_rng(self.seed)
        n_times = int(round(self.duration_s * self.sfreq))

        for subj_idx in range(self.n_subjects):
            subject_id = f"synth_S{subj_idx:02d}"
            for trial_idx in range(self.trials_per_subject):
                trial_id = f"{subject_id}_T{trial_idx:03d}"
                # Each subject gets unique group_ids so leakage checks pass
                group_id = f"G_{subject_id}_{trial_idx:03d}"

                label = rng.integers(0, 2)
                eeg = rng.standard_normal((self.n_channels, n_times)).astype(np.float32)

                # Inject a weak but learnable class-dependent signal
                if label == 1:
                    # Subtle sinusoidal modulation on frontal channels
                    t = np.arange(n_times) / self.sfreq
                    signal = 0.3 * np.sin(2 * np.pi * 4.0 * t)  # 4 Hz
                    eeg[0, :] += signal
                    eeg[1, :] += 0.5 * signal
                else:
                    t = np.arange(n_times) / self.sfreq
                    signal = 0.3 * np.sin(2 * np.pi * 6.0 * t)  # 6 Hz
                    eeg[0, :] += signal

                # Generate a fake speech envelope for TRF testing
                env_att = rng.standard_normal(n_times).astype(np.float32) * 0.1
                env_unatt = rng.standard_normal(n_times).astype(np.float32) * 0.1
                # The attended envelope correlates weakly with EEG channel 0
                env_att += 0.2 * eeg[0]

                yield Trial(
                    eeg=eeg,
                    sfreq=self.sfreq,
                    label=int(label),
                    subject_id=subject_id,
                    trial_id=trial_id,
                    group_id=group_id,
                    audio_a=env_att,       # always = attended
                    audio_b=env_unatt,     # always = unattended
                    audio_sr=int(self.sfreq),
                )
