"""Round 9: electrode graph from the DTU (BioSemi-64) montage, and
length-normalised band power in the frequency encoder."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from aad_xai.models.sgat import DTU_MONTAGE_PATH, StimulusGAT, _load_default_adjacency
from aad_xai.models.sgat_frequency_encoder import PerChannelFrequencyEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stgcn"))
import adjacency as stgcn_adjacency  # noqa: E402

# Channel labels stored in the DTU files (data.dim.chan.eeg, first 64 of 73),
# identical for all 18 subjects (checked on Kaggle, 2026-11-03).
DTU_FILE_ORDER = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
    "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]


def _names():
    return stgcn_adjacency.load_montage(DTU_MONTAGE_PATH)["electrode_name"].tolist()


def _neighbours(name):
    names = _names()
    a = _load_default_adjacency().numpy() > 0
    i = names.index(name)
    return {names[j] for j in np.where(a[i])[0] if j != i}


# --- electrode graph ------------------------------------------------------------

def test_montage_is_in_dtu_file_order():
    assert _names() == DTU_FILE_ORDER


def test_default_adjacency_joins_physical_neighbours():
    assert _neighbours("Cz") <= {"C1", "C2", "FCz", "CPz", "FC1", "FC2", "CP1", "CP2"}
    assert _neighbours("T7") <= {"FT7", "TP7", "C5", "C3", "FC5", "CP5"}
    assert _neighbours("Fpz") <= {"Fp1", "Fp2", "AFz", "AF3", "AF4", "Fz"}


def test_default_adjacency_edges_are_short():
    m = stgcn_adjacency.load_montage(DTU_MONTAGE_PATH)
    xy = m[["x", "y"]].to_numpy()
    a = _load_default_adjacency().numpy() > 0
    np.fill_diagonal(a, False)
    i, j = np.where(a)
    assert np.linalg.norm(xy[i] - xy[j], axis=1).max() < 0.05  # metres; the old template graph reached 0.13+


def test_model_uses_the_dtu_graph_and_saves_it():
    model = StimulusGAT(n_channels=64, sfreq=64.0)
    expected = (_load_default_adjacency() + torch.eye(64)) > 0
    assert torch.equal(model.gat.adj_mask, expected)
    assert "gat.adj_mask" in model.state_dict()  # old checkpoints restore their own graph


# --- length-normalised band power -------------------------------------------------

def _bandpassed_noise(n_times, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, n_times)).cumsum(-1) * 0.05 + rng.standard_normal((4, n_times))
    return sosfiltfilt(butter(4, [1, 31], btype="band", fs=64, output="sos"), x)


def _mean_features(enc, x, win):
    w = np.stack([x[:, i:i + win] for i in range(0, x.shape[1] - win, win)])         # (N, C, T)
    w = (w - w.mean(-1, keepdims=True)) / w.std(-1, keepdims=True)                   # per-window z-score
    return enc._band_power(torch.tensor(w, dtype=torch.float32)).mean((0, 1)).numpy()  # (n_bands,)


def test_unnormalised_band_power_shifts_with_window_length():
    x = _bandpassed_noise(64 * 200)
    enc = PerChannelFrequencyEncoder(4, sfreq=64.0)
    shift = _mean_features(enc, x, 640) - _mean_features(enc, x, 320)
    assert np.all(np.abs(shift - np.log(2)) < 0.1)


def test_length_normalised_band_power_does_not_shift():
    x = _bandpassed_noise(64 * 200)
    enc = PerChannelFrequencyEncoder(4, sfreq=64.0, length_normalize=True)
    for short in (64, 128, 320):
        shift = _mean_features(enc, x, 640) - _mean_features(enc, x, short)
        assert np.all(np.abs(shift) < 0.15), (short, shift)


def test_model_flag_reaches_encoder_and_scores_1s_windows():
    model = StimulusGAT(n_channels=64, sfreq=64.0, film_pooling="corr", speech_kernel_sizes=(33, 33),
                        freq_length_normalize=True).eval()
    assert model.frequency_encoder.length_normalize
    logits = model(torch.randn(3, 64, 64), torch.randn(3, 2, 64))   # 1 s windows: (B, 2)
    assert logits.shape == (3, 2) and torch.isfinite(logits).all()
