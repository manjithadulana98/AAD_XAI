# TRF — Vandecappelle et al. (eLife 2021) Linear Baseline — Single Table

This table is intended for quick reporting. All requested decision windows are complete.

Protocol preset: `vandecappelle2021_linear`
- CV: `within_subject_story_speaker_out` (2 folds/subject; only story+speaker-disjoint folds)
- EEG preprocessing: bandpass 1–9 Hz, `sfreq_out=20 Hz`, average reference
- Trials: experiments 1 & 2 only (exclude experiment-3 repeats)
- Envelope: `powerlaw_subbands` (gammatone subbands + powerlaw 0.6 + sum)
- TRF lags: `tmin=0.0s`, `tmax=0.25s`
- Window overlap: 50% (`overlap = window/2`)
- Alpha tuning: `corr` (RidgeCV + GroupKFold + Pearson scoring)

| Decision window (s) | Train window (s) | Overlap (s) | Folds | Acc (mean +/- std) | Time (s) |
|---:|---:|---:|---:|---:|---:|
| 5  | 5  | 2.50 | 32 | 0.6328 +/- 0.0609 | 1373 |
| 10 | 10 | 5.00 | 32 | 0.6809 +/- 0.0849 | 1238 |
| 20 | 20 | 10.00 | 32 | 0.7239 +/- 0.1001 | 1218 |
| 40 | 40 | 20.00 | 32 | 0.8056 +/- 0.1083 | 1396 |
| 60 | 60 | 30.00 | 32 | 0.8587 +/- 0.1296 | 394 |
