# TRF — Vandecappelle et al. (eLife 2021) — Reproduction

Protocol:
- cv: `within_subject_story_speaker_out` (2 folds per subject; only stories with unique speakers)
- envelope: `powerlaw_subbands` (gammatone + powerlaw 0.6 + sum)
- EEG preprocessing: bandpass 1–9 Hz, sfreq_out=20 Hz, average reference
- include_experiments: 1,2 (exclude experiment-3 repeats)
- TRF lags: 0–250 ms
- window overlap: 50% (overlap = window/2)

Settings:
- bandpass_hz: (1.0, 9.0)
- sfreq_out: 20 Hz
- trf_tmin: 0.000s
- trf_tmax: 0.250s
- max_train_seconds: 1200s
- tune_alpha: True (metric=corr)
- alpha_grid: logspace(-7..7,15)

| Decision window (s) | Train window (s) | Overlap (s) | Folds | Acc (mean +/- std) | Time (s) |
|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 2.50 | 32 | 0.6328 +/- 0.0609 | 1373 |
| 10 | 10 | 5.00 | 32 | 0.6809 +/- 0.0849 | 1238 |
| 20 | 20 | 10.00 | 32 | 0.7239 +/- 0.1001 | 1218 |
| 40 | 40 | 20.00 | 32 | 0.8056 +/- 0.1083 | 1396 |
| 60 | 60 | 30.00 | 32 | 0.8587 +/- 0.1296 | 394 |
