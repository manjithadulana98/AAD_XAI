from __future__ import annotations
import numpy as np
import mne
from ..config import PreprocessConfig

def preprocess_eeg(eeg: np.ndarray, sfreq_in: float, cfg: PreprocessConfig) -> tuple[np.ndarray, float]:
    """Preprocess EEG.

    Parameters
    ----------
    eeg : np.ndarray
        Shape (n_channels, n_times)
    sfreq_in : float
        Sampling rate of the input
    cfg : PreprocessConfig

    Returns
    -------
    eeg_pp : np.ndarray
        Shape (n_channels, n_times_pp)
    sfreq_out : float
    """
    info = mne.create_info(ch_names=[f"Ch{i}" for i in range(eeg.shape[0])], sfreq=sfreq_in, ch_types="eeg")
    raw = mne.io.RawArray(eeg, info, verbose=False)

    if cfg.reref == "average":
        raw = raw.set_eeg_reference("average", projection=False, verbose=False)

    raw = raw.filter(cfg.bandpass_hz[0], cfg.bandpass_hz[1], verbose=False)

    if int(sfreq_in) != int(cfg.sfreq_out):
        raw = raw.resample(cfg.sfreq_out, npad="auto", verbose=False)

    return raw.get_data(), float(raw.info["sfreq"])
