"""Oracle 5-s decoding accuracy of the planted benchmark vs SNR (level 3b range choice).

The oracle knows the planted electrodes, band and each subject's TRF. Per subject it builds
the matched spatial filter Sigma^-1 a (Sigma: covariance of the subject's band-passed EEG,
a: indicator of the planted electrodes), then per 5-s window correlates the filtered EEG with
the TRF-filtered candidate envelopes and picks the higher. With Gaussian background this is close to the best linear detector, so it is
an approximate upper bound: a trained model's 0.70 crossing should lie at or above the oracle's.
Output used to choose the level-3b sweep (6 subjects, 2 trials per story, seed 0):
  delta_T7  -70 0.525  -60 0.533  -55 0.549  -50 0.582  -45 0.625  -40 0.693  -30 0.922  -20 0.999
  theta_P8  -70 0.503  -60 0.525  -55 0.546  -50 0.588  -45 0.664  -40 0.787  -30 0.986  -20 1.000
usage: python scripts/planted_oracle.py <repo>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from aad_xai.data.planted_sources import (PlantedSpec, bandpass, make_planted_trials,  # noqa: E402
                                          neural_response, trf_kernel)

W = 320
for cond, center, band in (("delta_T7", "T7", (1.0, 4.0)), ("theta_P8", "P8", (4.0, 8.0))):
    for snr in (-70, -60, -55, -50, -45, -40, -30, -20):
        spec = PlantedSpec(center=center, band_hz=band, snr_db=snr, decoy="Cz", decoy_snr_db=0.0)
        trials, truth = make_planted_trials(spec, n_subjects=6, trials_per_story=2, seed=0)
        hits = []
        a = np.zeros(64)
        a[truth.planted_idx] = 1.0                                              # (C,)
        filt = {}
        for sid in {t.subject_id for t in trials}:
            X = np.concatenate([bandpass(t.eeg, 64.0, band) for t in trials if t.subject_id == sid], axis=1)
            S = np.cov(X) + 1e-6 * np.eye(64)
            filt[sid] = np.linalg.solve(S, a)                                  # matched filter Sigma^-1 a
        for t in trials:
            k = trf_kernel(truth.lags_s[t.subject_id], 64.0)
            x = filt[t.subject_id] @ bandpass(t.eeg, 64.0, band)
            ra, rb = (neural_response(e, k, 64.0, band) for e in (t.audio_a, t.audio_b))
            for i in range(0, len(x) - W + 1, W):
                ca = np.corrcoef(x[i:i + W], ra[i:i + W])[0, 1]
                cb = np.corrcoef(x[i:i + W], rb[i:i + W])[0, 1]
                hits.append((ca > cb) == (t.label == 0))
        print(f"{cond} {snr:+d} dB oracle acc {np.mean(hits):.3f} (n={len(hits)})", flush=True)
