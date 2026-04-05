from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import scipy.io
from .base import BaseDataset, Trial

# Number of EEG channels to keep (first N columns of the raw matrix).
# The DTU files have 73 columns: 64 EEG + 8 EXG + 1 Status channel.
_N_EEG_CHANNELS = 64


class DTUDataset(BaseDataset):
    """Loader for the DTU EEG+audio AAD dataset.

    Expected layout::

        root/
          eeg_new/
            S1.mat
            S2.mat
            ...
            S18.mat

    Each ``S*.mat`` file contains one subject's full continuous recording
    with the following MATLAB variables:

    - ``data.eeg``        – ``(n_samples, n_channels)`` float64
    - ``data.fsample.eeg``– sampling rate (512 Hz)
    - ``data.event.eeg``  – struct array with ``sample`` and ``value`` fields;
                            events come in pairs: even indices are trial onsets
                            (trigger matches ``expinfo[i].trigger``), odd
                            indices are trial offsets.
    - ``expinfo``         – ``(70,)`` struct array; ``attend_mf`` gives the
                            attended speaker (1=male, 2=female); used as label.

    Parameters
    ----------
    root : str | Path
        Path to the extracted DTU dataset folder containing ``eeg_new/``.
    n_eeg_channels : int
        How many leading EEG channels to keep (default 64; the remaining
        columns in the raw matrix are ExG + Status channels).
    """

    def __init__(self, root: str | Path, n_eeg_channels: int = _N_EEG_CHANNELS):
        self.root = Path(root)
        self.n_eeg_channels = n_eeg_channels

    def trials(self) -> Iterator[Trial]:
        eeg_dir = self.root / "eeg_new"
        if not eeg_dir.exists():
            raise FileNotFoundError(
                f"Expected EEG folder at {eeg_dir}. "
                "Update loader to match your extracted DTU dataset."
            )

        for mat_path in sorted(eeg_dir.glob("S*.mat")):
            yield from self._parse_subject_file(mat_path, self.n_eeg_channels)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_subject_file(mat_path: Path, n_eeg_channels: int) -> Iterator[Trial]:
        subject_id = mat_path.stem  # e.g. "S1"

        mat = scipy.io.loadmat(str(mat_path), squeeze_me=False)
        d0 = mat["data"][0, 0]

        # Sampling rate
        sfreq = float(d0["fsample"][0, 0]["eeg"][0, 0])

        # Continuous EEG: (n_samples, n_channels) → keep first n_eeg_channels
        eeg_cont = np.asarray(d0["eeg"][0, 0], dtype=np.float32)
        eeg_cont = eeg_cont[:, :n_eeg_channels]  # drop ExG / Status columns

        # Event array: struct rows with 'sample' and 'value' scalars
        ev = d0["event"][0, 0]["eeg"][0, 0]
        samples_col = ev["sample"].flatten()   # (n_events,) of object arrays
        values_col  = ev["value"].flatten()    # (n_events,) of object arrays
        n_events = len(samples_col)

        # Trial metadata
        expinfo = mat["expinfo"][:, 0]         # (n_trials,) struct array

        # Events come in pairs: index 2*i = onset, 2*i+1 = offset
        n_trials = min(len(expinfo), n_events // 2)

        for i in range(n_trials):
            onset  = int(samples_col[2 * i].flat[0])
            offset = int(samples_col[2 * i + 1].flat[0])

            eeg_segment = eeg_cont[onset:offset, :]  # (n_times, n_channels)
            eeg_segment = eeg_segment.T               # (n_channels, n_times)

            # Label: 0 = attend male, 1 = attend female
            attend_mf = int(expinfo[i]["attend_mf"][0, 0])
            label = attend_mf - 1  # 1→0, 2→1

            trial_id = f"{subject_id}_T{i:03d}"
            # Use wavfile as group_id for leakage control in splits
            wavfile = str(expinfo[i]["wavfile_male"][0]).strip() or f"trial{i}"
            group_id = f"{subject_id}_{wavfile}"

            yield Trial(
                eeg=eeg_segment,
                sfreq=sfreq,
                label=label,
                subject_id=subject_id,
                trial_id=trial_id,
                group_id=group_id,
            )


