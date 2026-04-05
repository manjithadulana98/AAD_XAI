from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
import scipy.io
from .base import BaseDataset, Trial
from .kul_dataset import _load_wav_envelope

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
          Audio/
            aske_story1_trial_1.wav
            marianne_story1_trial_1.wav
            dss.wav
            ...

    Each ``S*.mat`` file contains one subject's full continuous recording
    with the following MATLAB variables:

    - ``data.eeg``        – ``(n_samples, n_channels)`` float64
    - ``data.fsample.eeg``– sampling rate (512 Hz)
    - ``data.event.eeg``  – struct array with ``sample`` and ``value`` fields;
                            events come in pairs: even indices are trial onsets,
                            odd indices are trial offsets.
    - ``expinfo``         – ``(70,)`` struct array; ``attend_mf`` gives the
                            attended speaker (1=male, 2=female); ``wavfile_male``
                            and ``wavfile_female`` give audio filenames.

    Parameters
    ----------
    root : str | Path
        Path to the extracted DTU dataset folder containing ``eeg_new/``.
    audio_dir : str | Path | None
        Folder containing audio WAV files.  Defaults to ``root/Audio``.
        Pass *None* to skip audio loading (EEG-only mode).
    load_audio : bool
        If *True* (default), compute speech envelopes from WAV files when
        both male and female audio are available for a trial.
    n_eeg_channels : int
        How many leading EEG channels to keep (default 64).
    """

    def __init__(
        self,
        root: str | Path,
        audio_dir: Optional[str | Path] = None,
        load_audio: bool = True,
        n_eeg_channels: int = _N_EEG_CHANNELS,
    ):
        self.root = Path(root)
        if audio_dir is None:
            self.audio_dir: Optional[Path] = self.root / "Audio"
        elif audio_dir is not None:
            self.audio_dir = Path(audio_dir)
        else:
            self.audio_dir = None
        self.load_audio = load_audio
        self.n_eeg_channels = n_eeg_channels

    def trials(self) -> Iterator[Trial]:
        eeg_dir = self.root / "eeg_new"
        if not eeg_dir.exists():
            raise FileNotFoundError(
                f"Expected EEG folder at {eeg_dir}. "
                "Update loader to match your extracted DTU dataset."
            )

        for mat_path in sorted(eeg_dir.glob("S*.mat")):
            yield from self._parse_subject_file(
                mat_path, self.n_eeg_channels, self.audio_dir if self.load_audio else None
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_subject_file(
        mat_path: Path,
        n_eeg_channels: int,
        audio_dir: Optional[Path],
    ) -> Iterator[Trial]:
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
            wavfile_male_raw = expinfo[i]["wavfile_male"]
            wavfile_male = str(wavfile_male_raw.flat[0]).strip() if wavfile_male_raw.size > 0 else ""
            group_id = f"{subject_id}_{wavfile_male or f'trial{i}'}"

            # ── Audio envelopes (optional) ──────────────────────────────
            audio_a: Optional[np.ndarray] = None
            audio_b: Optional[np.ndarray] = None
            audio_sr: Optional[int] = None

            if audio_dir is not None and wavfile_male:
                wavfile_female_raw = expinfo[i]["wavfile_female"]
                wavfile_female = (
                    str(wavfile_female_raw.flat[0]).strip()
                    if wavfile_female_raw.size > 0
                    else ""
                )
                # Only load audio when both streams are present (2-speaker trials)
                if wavfile_female:
                    wav_a = audio_dir / wavfile_male
                    wav_b = audio_dir / wavfile_female
                    if wav_a.exists() and wav_b.exists():
                        audio_sr = 64
                        audio_a = _load_wav_envelope(wav_a, target_sfreq=float(audio_sr))
                        audio_b = _load_wav_envelope(wav_b, target_sfreq=float(audio_sr))

            yield Trial(
                eeg=eeg_segment,
                sfreq=sfreq,
                label=label,
                subject_id=subject_id,
                trial_id=trial_id,
                group_id=group_id,
                audio_a=audio_a,
                audio_b=audio_b,
                audio_sr=audio_sr,
            )



