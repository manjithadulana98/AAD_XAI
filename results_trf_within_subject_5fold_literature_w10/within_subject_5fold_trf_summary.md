# TRF — Within-subject 5-fold Accuracy vs Window

Protocol: `cv=within_subject_5fold`, model=`trf`, KULeuven.
Default stepping (literature-style): step=max(0.5s, L/5) so overlap=L-step (unless --overlap is provided).

Settings:
- lags: tmin=0.000s, tmax=0.500s
- max_train_seconds: 1200s
- tune_alpha: True (metric=aad_acc)
- alpha_grid: 0.01,0.1,1,10,100,1000,10000,100000

| Decision window (s) | Train window (s) | Overlap (s) | Folds | Acc (mean +/- std) | Time (s) |
|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 4.00 | 80 | 0.5754 +/- 0.0456 | 13941 |
| 10 | 10 | 8.00 | 80 | 0.6160 +/- 0.0704 | 5860 |
| 20 | 20 | 16.00 | 80 | 0.6487 +/- 0.0856 | 4750 |
| 40 | 40 | 32.00 | 80 | 0.6922 +/- 0.1097 | 371 |
| 60 | 60 | 48.00 | 80 | 0.7166 +/- 0.1287 | 6280 |
| 60 | 60 | 48.00 | 80 | 0.7166 +/- 0.1287 | 6280 |
