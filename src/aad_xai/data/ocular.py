"""Ocular (EOG) channels of the DTU dataset, and EOG regression.

The DTU EEG files (Fuglsang et al. 2018, Zenodo record 1199011) hold 73 columns:
64 scalp electrodes, 8 external (ExG) electrodes, then the BioSemi status
channel. The Zenodo record lists the external electrodes (1-based) as
  65 right mastoid, 66 left mastoid,
  67 vertical EOG below right eye, 68 horizontal EOG right eye,
  69 vertical EOG above right eye, 70 vertical EOG below left eye,
  71 horizontal EOG left eye, 72 vertical EOG above left eye.

Bipolar EOG derivations (above - below, right - left) cancel activity common to
both electrodes, most of the cortical signal included, and keep the eye's
corneo-retinal dipole and lid artefacts.
"""
from __future__ import annotations

import numpy as np

N_SCALP = 64
N_DTU_WITH_EXG = 72                 # scalp + ExG; the status column is dropped
# 0-based column indices into the DTU EEG matrix
DTU_EXG = {
    "M_right": 64, "M_left": 65,
    "VEOG_below_R": 66, "HEOG_R": 67, "VEOG_above_R": 68,
    "VEOG_below_L": 69, "HEOG_L": 70, "VEOG_above_L": 71,
}
EOG_NAMES = ("VEOG_R", "VEOG_L", "HEOG")
MASTOID_NAMES = ("M_right", "M_left")


def dtu_derivations(x: np.ndarray) -> dict[str, np.ndarray]:
    """Split one DTU recording into scalp EEG, mastoids and bipolar EOG.

    Parameters
    ----------
    x : (72, T) -- the 64 scalp and 8 ExG columns, on any common reference

    Returns
    -------
    dict with
      "eeg"     : (64, T) scalp EEG, average reference over the 64 scalp channels
      "mastoid" : (2, T)  right, left mastoid on the same average reference
      "eog"     : (3, T)  VEOG_R = above - below right eye,
                          VEOG_L = above - below left eye,
                          HEOG   = right - left outer canthus
    All three are unchanged by adding a signal common to every column.
    """
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] != N_DTU_WITH_EXG:
        raise ValueError(f"expected ({N_DTU_WITH_EXG}, T), got {x.shape}")
    e = DTU_EXG
    car = x[:N_SCALP].mean(axis=0, keepdims=True)                                # (1, T)
    eog = np.stack([x[e["VEOG_above_R"]] - x[e["VEOG_below_R"]],
                    x[e["VEOG_above_L"]] - x[e["VEOG_below_L"]],
                    x[e["HEOG_R"]] - x[e["HEOG_L"]]])                            # (3, T)
    return {"eeg": x[:N_SCALP] - car,                                            # (64, T)
            "mastoid": x[[e["M_right"], e["M_left"]]] - car,                     # (2, T)
            "eog": eog}


def fit_eog_regression(eeg: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Least-squares weights of every EEG channel on the reference channels.

    Parameters
    ----------
    eeg : (C, T) EEG
    ref : (R, T) reference (EOG) channels, same samples

    Returns
    -------
    B : (R, C) -- eeg[c] ~ sum_r B[r, c] * ref[r] + intercept[c]
    """
    eeg, ref = np.asarray(eeg, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    if eeg.ndim != 2 or ref.ndim != 2 or eeg.shape[1] != ref.shape[1]:
        raise ValueError(f"eeg {eeg.shape} and ref {ref.shape} must be (C, T) and (R, T)")
    X = np.vstack([ref, np.ones((1, ref.shape[1]))]).T                           # (T, R+1)
    coef, *_ = np.linalg.lstsq(X, eeg.T, rcond=None)                             # (R+1, C)
    return coef[:-1]                                                             # (R, C)


def regress_out(eeg: np.ndarray, ref: np.ndarray, B: np.ndarray) -> np.ndarray:
    """EEG minus its reference-explained part: (C, T) - B.T @ ref; the intercept stays."""
    eeg, ref = np.asarray(eeg), np.asarray(ref)
    return (eeg - (np.asarray(B).T @ ref)).astype(eeg.dtype, copy=False)          # (C, T)
