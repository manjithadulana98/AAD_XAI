"""Within-subject TRF with alpha tuning + multiple window lengths.

Combination of literature best practices:
  1. Within-subject training (separate TRF per subject)
  2. Ridge alpha tuning via RidgeCV (efficient SVD-based LOO)
  3. Multiple decision window lengths (5s, 10s, 20s, 40s)

Usage:
    python scripts/_trf_subject_acc.py
    python scripts/_trf_subject_acc.py --windows 5 10 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aad_xai.data.vlaai_dataset import load_dtu_trials, window_data
from aad_xai.models.trf_baseline import lag_matrix

FS = 64  # Hz
ALPHAS = np.logspace(-1, 5, 7)  # 7 values for RidgeCV
MAX_TRAIN_SAMPLES = 10000  # cap total training samples per fold (~156s)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "external" / "vlaai" / "evaluation_datasets" / "DTU"))
    p.add_argument("--windows", nargs="+", type=int, default=[5, 10, 20, 40],
                   help="Decision window lengths in seconds")
    p.add_argument("--trf-tmin", type=float, default=0.0)
    p.add_argument("--trf-tmax", type=float, default=0.5)
    p.add_argument("--n-folds", type=int, default=5, help="K-fold CV per subject")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(ROOT / "xai_results_trf_comparison" / "trf_within_subject_results.json"))
    return p.parse_args()


def pearson_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class FastTRF:
    """Minimal TRF that builds lag matrix once, uses RidgeCV for alpha."""

    def __init__(self, tmin_s=0.0, tmax_s=0.5):
        self.tmin_s = tmin_s
        self.tmax_s = tmax_s
        self.model = None
        self.lags = None
        self._X_mean = None
        self._X_std = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, eeg, env, sfreq):
        """eeg: (C, T), env: (T,)"""
        self.lags = np.arange(
            int(round(self.tmin_s * sfreq)),
            int(round(self.tmax_s * sfreq)) + 1)

        # Cap BEFORE building lag matrix to avoid huge memory allocation
        T = eeg.shape[1]
        if T > MAX_TRAIN_SAMPLES:
            eeg = eeg[:, :MAX_TRAIN_SAMPLES]
            env = env[:MAX_TRAIN_SAMPLES]

        X = lag_matrix(eeg, self.lags)
        y = env[:X.shape[0]].astype(np.float32)

        # Z-score
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        self._X_std[self._X_std < 1e-8] = 1.0
        X = (X - self._X_mean) / self._X_std
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y = (y - self._y_mean) / self._y_std

        # Fast alpha search: train/val split on time (last 20% for validation)
        from sklearn.linear_model import Ridge
        n_tr = int(X.shape[0] * 0.8)
        X_tr, X_va = X[:n_tr], X[n_tr:]
        y_tr, y_va = y[:n_tr], y[n_tr:]

        best_alpha, best_score = ALPHAS[0], -np.inf
        for a in ALPHAS:
            mdl = Ridge(alpha=a, solver="cholesky", fit_intercept=True)
            mdl.fit(X_tr, y_tr)
            pred = mdl.predict(X_va)
            r = np.corrcoef(pred, y_va)[0, 1] if pred.std() > 1e-12 else 0.0
            if np.isfinite(r) and r > best_score:
                best_score, best_alpha = r, a

        # Retrain on full data with best alpha
        self.model = Ridge(alpha=best_alpha, solver="cholesky", fit_intercept=True)
        self.model.fit(X, y)
        self.best_alpha = float(best_alpha)
        return self

    def predict(self, eeg):
        """eeg: (C, T) -> envelope (T,)"""
        X = lag_matrix(np.asarray(eeg, dtype=np.float32), self.lags)
        X = (X - self._X_mean) / self._X_std
        y_sc = self.model.predict(X)
        return y_sc * self._y_std + self._y_mean


def _prepare_raw_trials(trials_by_subj):
    """Normalize trial data once (window-length agnostic)."""
    raw = {}
    for subj, trial_list in trials_by_subj.items():
        subj_trials = []
        for trial in trial_list:
            eeg = trial["eeg"].astype(np.float32)
            env = trial["envelope"].astype(np.float32)
            eeg = (eeg - eeg.mean(0, keepdims=True)) / (eeg.std(0, keepdims=True) + 1e-8)
            env = (env - env.mean(0, keepdims=True)) / (env.std(0, keepdims=True) + 1e-8)
            env_unatt = np.roll(env, env.shape[0] // 2, axis=0)
            subj_trials.append({"eeg": eeg, "env": env.ravel(), "env_unatt": env_unatt})
        raw[subj] = subj_trials
    return raw


def _eval_windows(trf, raw_trials, test_set, window_len, hop):
    """Evaluate a trained TRF on test trials with a given window length."""
    n_correct, n_total = 0, 0
    for t in test_set:
        eeg = raw_trials[t]["eeg"]
        env_att = raw_trials[t]["env"]
        env_unatt = raw_trials[t]["env_unatt"].ravel()
        eeg_w = window_data(eeg, window_len, hop)
        env_att_w = window_data(env_att.reshape(-1, 1), window_len, hop)
        env_unatt_w = window_data(env_unatt.reshape(-1, 1), window_len, hop)
        n = min(len(eeg_w), len(env_att_w), len(env_unatt_w))
        for i in range(n):
            pred = trf.predict(eeg_w[i].T)
            l = min(len(pred), window_len)
            if pearson_r(pred[:l], env_att_w[i][:l, 0]) > pearson_r(pred[:l], env_unatt_w[i][:l, 0]):
                n_correct += 1
        n_total += n
    return n_correct / n_total if n_total else 0.0, n_total


def run_all_windows(data_dir, window_list, tmin, tmax, n_folds, seed):
    """Within-subject K-fold CV: train once per fold, evaluate all window lengths."""
    rng = np.random.RandomState(seed)

    print("  Loading data...", end=" ", flush=True)
    t0 = time.time()
    trials_by_subj = load_dtu_trials(data_dir)
    print(f"done ({time.time()-t0:.1f}s)")

    print("  Normalizing trials...", end=" ", flush=True)
    t0 = time.time()
    raw = _prepare_raw_trials(trials_by_subj)
    print(f"done ({time.time()-t0:.1f}s)")

    subjects = sorted(raw.keys())
    # results[win_s][subj] = {...}
    results = {w: {} for w in window_list}

    for si, subj in enumerate(subjects):
        subj_trials = raw[subj]
        n_trials = len(subj_trials)
        if n_trials < 2:
            print(f"  {subj}: <2 trials, skipping")
            continue

        trial_idx = np.arange(n_trials)
        rng.shuffle(trial_idx)
        actual_folds = min(n_folds, n_trials)

        # Per-window accumulators
        fold_accs = {w: [] for w in window_list}
        fold_alphas = []

        t_subj = time.time()
        for fold_i in range(actual_folds):
            test_set = trial_idx[fold_i::actual_folds]
            train_set = np.array([t for t in trial_idx if t not in test_set])

            # Concatenate raw training data
            eeg_tr = np.concatenate([subj_trials[t]["eeg"] for t in train_set], axis=0)
            env_tr = np.concatenate([subj_trials[t]["env"] for t in train_set], axis=0)

            # Train TRF ONCE per fold
            trf = FastTRF(tmin_s=tmin, tmax_s=tmax)
            trf.fit(eeg_tr.T, env_tr, FS)
            fold_alphas.append(trf.best_alpha)

            # Evaluate at each window length
            for w in window_list:
                wlen = int(w * FS)
                hop = max(wlen // 5, FS)
                acc, _ = _eval_windows(trf, subj_trials, test_set, wlen, hop)
                fold_accs[w].append(acc)

            print(f"    {subj} fold {fold_i+1}/{actual_folds}: "
                  + " | ".join(f"{w}s={fold_accs[w][-1]:.0%}" for w in window_list)
                  + f"  alpha={trf.best_alpha:.0f}", flush=True)

        elapsed = time.time() - t_subj
        for w in window_list:
            m, s = np.mean(fold_accs[w]), np.std(fold_accs[w])
            results[w][subj] = {
                "accuracy_mean": float(m), "accuracy_std": float(s),
                "fold_accs": [float(a) for a in fold_accs[w]],
                "best_alphas": [float(a) for a in fold_alphas],
                "n_folds": actual_folds,
            }

        accs_str = " | ".join(f"{w}s={np.mean(fold_accs[w]):.1%}" for w in window_list)
        print(f"  [{si+1}/{len(subjects)}] {subj}: {accs_str}  ({elapsed:.0f}s)")

    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("WITHIN-SUBJECT TRF WITH ALPHA TUNING + MULTI-WINDOW")
    print(f"  Windows: {args.windows}s, TRF lags: {args.trf_tmin}-{args.trf_tmax}s")
    print(f"  Folds: {args.n_folds}, Seed: {args.seed}")
    print(f"  Alphas: {len(ALPHAS)} values from {ALPHAS[0]:.0e} to {ALPHAS[-1]:.0e}")
    print("=" * 70)

    t_total = time.time()
    results_by_win = run_all_windows(
        args.data_dir, args.windows, args.trf_tmin, args.trf_tmax,
        args.n_folds, args.seed,
    )
    total_elapsed = time.time() - t_total

    # Build all_results dict for output
    all_results = {}
    for win_s in args.windows:
        ws = f"{win_s}s"
        accs = [r["accuracy_mean"] for r in results_by_win[win_s].values()]
        all_results[ws] = {
            "window_s": win_s,
            "overall_mean": float(np.mean(accs)) if accs else 0,
            "overall_std": float(np.std(accs)) if accs else 0,
            "subjects": results_by_win[win_s],
        }
    # ── Final comparison table ────────────────────────────────────
    print(f"\nTotal time: {total_elapsed:.0f}s")
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: WITHIN-SUBJECT TRF (alpha-tuned)")
    print("=" * 70)
    subjects_all = sorted(set().union(*[all_results[k]["subjects"].keys() for k in all_results]))

    # Header
    win_strs = [f"{w}s" for w in args.windows]
    hdr = f"{'Subject':>8s}" + "".join(f" {ws:>8s}" for ws in win_strs) + f" {'Prev 5s':>8s}"
    print(hdr)
    print("-" * len(hdr))

    # Previous global-TRF results for comparison
    prev_global = {
        "S1": 32.0, "S2": 46.0, "S3": 78.0, "S4": 48.0, "S5": 52.0,
        "S6": 38.0, "S7": 56.0, "S8": 38.0, "S9": 44.0, "S10": 94.0,
        "S11": 54.0, "S12": 52.0, "S13": 66.0, "S14": 46.0, "S15": 64.0,
        "S16": 64.0, "S17": 60.0, "S18": 68.0,
    }

    for subj in subjects_all:
        row = f"{subj:>8s}"
        for ws in win_strs:
            if ws in all_results and subj in all_results[ws]["subjects"]:
                acc = all_results[ws]["subjects"][subj]["accuracy_mean"]
                row += f" {acc:>7.1%}"
            else:
                row += f" {'N/A':>8s}"
        prev = prev_global.get(subj, 0)
        row += f" {prev:>7.1f}%"
        print(row)

    print("-" * len(hdr))
    row = f"{'OVERALL':>8s}"
    for ws in win_strs:
        if ws in all_results:
            row += f" {all_results[ws]['overall_mean']:>7.1%}"
        else:
            row += f" {'N/A':>8s}"
    row += f" {'55.6':>7s}%"
    print(row)

    # Literature comparison
    lit = {5: 57.5, 10: 61.6, 20: 64.9, 40: 69.2}
    row = f"{'LIT.REF':>8s}"
    for w in args.windows:
        row += f" {lit.get(w, 0):>7.1f}%"
    row += f" {'57.5':>7s}%"
    print(row)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
