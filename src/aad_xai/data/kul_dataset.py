from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, Literal, Sequence
import logging

import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy.signal import hilbert, butter, filtfilt, resample_poly
from math import gcd

from .base import BaseDataset, Trial
from ..config import PreprocessConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Audio envelope helpers
# --------------------------------------------------------------------------- #

_INTERMEDIATE_SR = 8000  # downsample audio here before Hilbert (efficiency)


def _load_wav_envelope(
    wav_path: Path,
    target_sfreq: float = 64.0,
    low_hz: float = 1.0,
    high_hz: float = 8.0,
    method: Literal["hilbert", "powerlaw_subbands"] = "hilbert",
) -> np.ndarray:
    """Read a WAV file and return a bandpass-filtered speech envelope.

        Pipeline depends on ``method``:

        - "hilbert": WAV -> mono -> downsample to 8 kHz -> Hilbert envelope ->
            downsample to ``target_sfreq`` -> bandpass ``[low_hz, high_hz]``.

        - "powerlaw_subbands": Biesmans-style powerlaw-subband envelope extraction
            using a fixed gammatone filterbank (see :func:`aad_xai.data.speech_features.powerlaw_subbands_envelope`).
    """
    sr, audio = scipy.io.wavfile.read(str(wav_path))

    # Normalise to float32 to keep peak memory bounded for long WAVs.
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.float32(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)

    # Stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if method == "powerlaw_subbands":
        from .speech_features import powerlaw_subbands_envelope

        return powerlaw_subbands_envelope(
            audio,
            int(sr),
            target_sr=int(target_sfreq),
            bandpass_hz=(float(low_hz), float(high_hz)),
        )

    # Default: simple Hilbert envelope
    # Step 1 — downsample to intermediate rate for efficiency
    if sr != _INTERMEDIATE_SR:
        g = gcd(int(sr), _INTERMEDIATE_SR)
        audio = resample_poly(audio, _INTERMEDIATE_SR // g, int(sr) // g).astype(np.float32, copy=False)

    # Step 2 — Hilbert amplitude envelope
    env = np.abs(hilbert(audio))

    # Step 3 — downsample to target (e.g. 20/64 Hz)
    target_int = int(target_sfreq)
    g2 = gcd(target_int, _INTERMEDIATE_SR)
    env = resample_poly(env, target_int // g2, _INTERMEDIATE_SR // g2).astype(np.float32, copy=False)

    # Step 4 — bandpass filter
    nyq = target_sfreq / 2.0
    b, a = butter(4, [low_hz / nyq, high_hz / nyq], btype="band")
    env = filtfilt(b, a, env)

    return env.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
#  KULeuven dataset
# --------------------------------------------------------------------------- #


class KULeuvenDataset(BaseDataset):
    """Loader for the KU Leuven Auditory Attention Detection dataset.

    Tested against the *Das et al. 2019* release with the layout::

        root/
          S1.mat          # or  EEG/S1.mat
          S2.mat
          ...
          stimuli/
            part1_track1_dry.wav
            part1_track1_hrtf.wav
            ...

    Each ``S*.mat`` file contains a MATLAB cell array ``trials`` with
    fields ``RawData.EegData``, ``FileHeader.SampleRate``,
    ``attended_ear``, ``stimuli``, etc.

    Parameters
    ----------
    root : str | Path
        Path to the extracted dataset folder containing S*.mat and stimuli/.
    preprocess : PreprocessConfig | None
        EEG preprocessing settings.  *None* keeps EEG as-is (128 Hz,
        bandpass already applied by the original MATLAB pipeline).
    load_audio : bool
        If *True* (default), compute speech envelopes from WAV files.
        Set to *False* for a fast load when only EEG is needed (e.g.
        CNN training where audio_a/audio_b are unused).
    """

    def __init__(
        self,
        root: str | Path,
        preprocess: Optional[PreprocessConfig] = None,
        load_audio: bool = True,
        envelope_method: Literal["hilbert", "powerlaw_subbands"] = "hilbert",
        include_experiments: Optional[Sequence[int]] = None,
    ):
        self.root = Path(root)
        self.preprocess = preprocess
        self.load_audio = load_audio
        self.envelope_method = envelope_method
        self.include_experiments = tuple(include_experiments) if include_experiments is not None else None
        # Envelope cache keyed by WAV filename (shared across subjects)
        self._env_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def trials(self) -> Iterator[Trial]:
        # Auto-detect MAT location: root/S*.mat  or  root/EEG/S*.mat
        mat_files = sorted(self.root.glob("S*.mat"))
        if not mat_files:
            eeg_dir = self.root / "EEG"
            mat_files = sorted(eeg_dir.glob("S*.mat"))
        if not mat_files:
            raise FileNotFoundError(
                f"No S*.mat files found in {self.root} or {self.root / 'EEG'}."
            )

        stimuli_dir = self.root / "stimuli"
        if self.load_audio and not stimuli_dir.exists():
            logger.warning(
                "stimuli/ folder not found at %s — audio envelopes will be None.",
                stimuli_dir,
            )

        for mat_path in mat_files:
            subject_id = mat_path.stem
            yield from self._parse_mat(mat_path, subject_id, stimuli_dir)

    # ------------------------------------------------------------------
    def _parse_mat(
        self, mat_path: Path, subject_id: str, stimuli_dir: Path
    ) -> Iterator[Trial]:
        """Parse a single-subject MAT file into :class:`Trial` objects."""
        try:
            mat = scipy.io.loadmat(str(mat_path), squeeze_me=False)
        except NotImplementedError:
            raise NotImplementedError(
                f"{mat_path.name} is a v7.3 (HDF5) MAT file.  "
                "Add h5py-based parsing if your release uses v7.3."
            )

        if "trials" not in mat:
            raise KeyError(
                f"Key 'trials' not found in {mat_path.name}.  "
                f"Available keys: {[k for k in mat if not k.startswith('__')]}"
            )

        trials_arr = mat["trials"]               # shape (1, n_trials), dtype=object
        n_trials = trials_arr.shape[1] if trials_arr.ndim == 2 else trials_arr.shape[0]

        for idx in range(n_trials):
            ts = trials_arr[0, idx] if trials_arr.ndim == 2 else trials_arr[idx]

            # ---- EEG ----
            raw_data = ts["RawData"][0, 0]
            eeg = np.asarray(raw_data["EegData"][0, 0], dtype=np.float64)  # (time, ch)
            sfreq_raw = float(
                ts["FileHeader"][0, 0]["SampleRate"][0, 0].flat[0]
            )

            # Transpose to (channels, time)
            if eeg.ndim == 2 and eeg.shape[0] > eeg.shape[1]:
                eeg = eeg.T  # now (64, ~50000)

            # ---- Preprocess EEG (optional) ----
            if self.preprocess is not None:
                from .preprocessing import preprocess_eeg
                eeg, sfreq_out = preprocess_eeg(eeg, sfreq_raw, self.preprocess)
            else:
                sfreq_out = sfreq_raw
                eeg = eeg.astype(np.float32)

            # ---- Label ----
            attended_ear = str(ts["attended_ear"][0, 0].flat[0])  # 'L' or 'R'
            label = 0 if attended_ear == "L" else 1

            # ---- Metadata for leakage control ----
            experiment = int(ts["experiment"][0, 0].flat[0])
            part = int(ts["part"][0, 0].flat[0])
            repetition = int(ts["repetition"][0, 0].flat[0])
            condition = str(ts["condition"][0, 0].flat[0])

            if self.include_experiments is not None and experiment not in self.include_experiments:
                continue

            trial_id = f"{subject_id}_T{idx:03d}"
            # Keep subject-prefixed group id for compatibility with existing
            # split assertions and logs.  Strict story-disjoint evaluation is
            # handled explicitly in CV strategies.
            group_id = f"{subject_id}_exp{experiment}_p{part}_rep{repetition}_{condition}"

            # ---- Audio envelopes ----
            audio_a: Optional[np.ndarray] = None
            audio_b: Optional[np.ndarray] = None

            if self.load_audio and stimuli_dir.exists():
                stim = ts["stimuli"][0, 0]  # (2, 1) object array
                left_wav_name = str(stim[0, 0].flat[0])   # left-ear stimulus
                right_wav_name = str(stim[1, 0].flat[0])   # right-ear stimulus

                n_times = eeg.shape[1]
                bp = self.preprocess.bandpass_hz if self.preprocess else (1.0, 8.0)

                left_env = self._get_envelope(
                    stimuli_dir / left_wav_name, sfreq_out, n_times, bp
                )
                right_env = self._get_envelope(
                    stimuli_dir / right_wav_name, sfreq_out, n_times, bp
                )

                if left_env is not None and right_env is not None:
                    # Task-1 candidate ordering is fixed across all trials:
                    #   audio_a = left candidate, audio_b = right candidate.
                    # label indicates which candidate is attended (0=left, 1=right).
                    audio_a, audio_b = left_env, right_env

            yield Trial(
                eeg=eeg,
                sfreq=sfreq_out,
                label=label,
                subject_id=subject_id,
                trial_id=trial_id,
                group_id=group_id,
                audio_a=audio_a,
                audio_b=audio_b,
                audio_sr=int(sfreq_out),
            )

    # ------------------------------------------------------------------
    def _get_envelope(
        self,
        wav_path: Path,
        target_sfreq: float,
        target_len: int,
        bandpass_hz: tuple[float, float] = (1.0, 8.0),
    ) -> Optional[np.ndarray]:
        """Return a cached speech envelope truncated/padded to *target_len*."""
        key = f"{wav_path.name}|{self.envelope_method}|sr{int(target_sfreq)}|bp{bandpass_hz[0]}-{bandpass_hz[1]}"
        if key not in self._env_cache:
            if not wav_path.exists():
                logger.warning("WAV file not found: %s", wav_path)
                return None
            logger.info("Computing envelope for %s ...", wav_path.name)
            self._env_cache[key] = _load_wav_envelope(
                wav_path, target_sfreq=target_sfreq,
                low_hz=bandpass_hz[0], high_hz=bandpass_hz[1],
                method=self.envelope_method,
            )

        env = self._env_cache[key]

        # Truncate or zero-pad to match EEG length
        if len(env) >= target_len:
            return env[:target_len]
        return np.pad(env, (0, target_len - len(env))).astype(np.float32)

