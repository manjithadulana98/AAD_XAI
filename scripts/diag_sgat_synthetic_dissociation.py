"""Double-dissociation test: can StimulusGAT represent EEG-envelope tracking?

Hypothesis under test: StimulusGAT time-pools EEG (temporal encoder,
band power) and speech (speech encoder) *before* they interact via FiLM, so
it can only compare marginal statistics of the two streams. Auditory
attention information lives in their temporal covariance (what TRF uses).

Two synthetic tasks with known ground truth:

  T (tracking): EEG = spatial pattern x (TRF kernel * attended envelope) +
     noise. Both envelopes are drawn from the same random process, so the
     label is carried only by EEG-envelope temporal covariance.
  S (spectral): candidate envelopes are modulated at 3 Hz and 6 Hz (random
     assignment); EEG oscillates at the attended rate with random phase, so
     the label is carried only by time-pooled spectra, not by covariance.

Prediction if the hypothesis holds: TRF solves T but not S; StimulusGAT
solves S but not T.

Usage:
    python scripts/diag_sgat_synthetic_dissociation.py --seeds 3 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aad_xai.models.sgat import StimulusGAT  # noqa: E402

SFREQ = 64
WINDOW_S = 5.0
T = int(SFREQ * WINDOW_S)  # 320 samples
N_CHANNELS = 16
KERNEL_LEN = 16  # 250 ms


def _ring_adjacency(n: int, k: int = 2) -> torch.Tensor:
    A = torch.zeros(n, n)
    for i in range(n):
        for d in range(1, k + 1):
            A[i, (i + d) % n] = A[(i + d) % n, i] = 1.0
    return A


def _trf_kernel() -> np.ndarray:
    t = np.arange(KERNEL_LEN, dtype=float)
    tau = 6.0  # peak ~94 ms
    return (t / tau) * np.exp(1.0 - t / tau)


def _zscore(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-8)


def _speech_like_envelopes(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, T) positive envelopes, all from one stationary process."""
    sos_bp = butter(4, [1.0, 8.0], btype="bandpass", fs=SFREQ, output="sos")
    sos_lp = butter(4, 8.0, btype="lowpass", fs=SFREQ, output="sos")
    pad = 64
    x = rng.standard_normal((n, T + 2 * pad))
    env = sosfiltfilt(sos_lp, np.abs(sosfiltfilt(sos_bp, x, axis=-1)), axis=-1)[:, pad:-pad]
    env = np.maximum(env, 0.0)
    return env / env.mean(axis=-1, keepdims=True)


def make_task_tracking(rng, n, snr_db, spatial, delay=0):
    """Label carried only by temporal covariance (phase-locked response).

    ``delay`` (samples) shifts the whole response later, probing how far
    apart in time EEG and speech can be and still be matched.
    """
    env = np.stack([_speech_like_envelopes(rng, n), _speech_like_envelopes(rng, n)], axis=1)  # (n, 2, T)
    y = rng.integers(0, 2, size=n)
    att = env[np.arange(n), y]                                           # (n, T)
    k = np.concatenate([np.zeros(delay), _trf_kernel()])                 # (delay + KERNEL_LEN,)
    resp = np.stack([np.convolve(a - a.mean(), k)[:T] for a in att])     # (n, T), causal
    resp = _zscore(resp)
    noise_std = 10 ** (-snr_db / 20)
    eeg = spatial[None, :, None] * resp[:, None, :] + noise_std * rng.standard_normal((n, N_CHANNELS, T))
    return _zscore(eeg).astype(np.float32), env.astype(np.float32), y


def make_task_spectral(rng, n, snr_db, spatial):
    """Label carried only by time-pooled spectra (random-phase response)."""
    t = np.arange(T) / SFREQ
    freqs = np.array([3.0, 6.0])
    f_a_idx = rng.integers(0, 2, size=n)
    f = np.stack([freqs[f_a_idx], freqs[1 - f_a_idx]], axis=1)          # (n, 2)
    phi = rng.uniform(0, 2 * np.pi, size=(n, 2))
    env = 1.0 + 0.5 * np.sin(2 * np.pi * f[:, :, None] * t + phi[:, :, None])
    env = env + 0.1 * _speech_like_envelopes(rng, 2 * n).reshape(n, 2, T)
    y = rng.integers(0, 2, size=n)
    f_att = f[np.arange(n), y]
    psi = rng.uniform(0, 2 * np.pi, size=n)                              # independent of envelope phase
    resp = np.sin(2 * np.pi * f_att[:, None] * t + psi[:, None])
    resp = _zscore(resp)
    noise_std = 10 ** (-snr_db / 20)
    eeg = spatial[None, :, None] * resp[:, None, :] + noise_std * rng.standard_normal((n, N_CHANNELS, T))
    return _zscore(eeg).astype(np.float32), env.astype(np.float32), y


TASKS = {"tracking": make_task_tracking, "spectral": make_task_spectral}


# ---- TRF baseline: backward ridge decoder, lagged EEG -> envelope ----

def _lagged(eeg: np.ndarray, n_lags: int = KERNEL_LEN) -> np.ndarray:
    """(n, C, T) -> (n, T, C*n_lags); EEG lags the stimulus, so use eeg[t+lag]."""
    n, C, _ = eeg.shape
    out = np.zeros((n, T, C * n_lags), dtype=np.float64)
    for lag in range(n_lags):
        out[:, : T - lag, lag::n_lags] = eeg[:, :, lag:].transpose(0, 2, 1)
    return out


def trf_accuracy(train, test, alphas=(1e-1, 1e1, 1e3, 1e5), n_lags=KERNEL_LEN) -> float:
    eeg_tr, env_tr, y_tr = train
    eeg_te, env_te, y_te = test
    n_val = len(y_tr) // 5
    X = _lagged(eeg_tr, n_lags)
    target = _zscore(env_tr[np.arange(len(y_tr)), y_tr])

    def fit(Xf, tf, alpha):
        Xf2, tf2 = Xf.reshape(-1, Xf.shape[-1]), tf.reshape(-1)
        return np.linalg.solve(Xf2.T @ Xf2 + alpha * np.eye(Xf2.shape[1]), Xf2.T @ tf2)

    def acc(w, Xe, env, y):
        rec = Xe @ w                                                       # (n, T)
        r = [[np.corrcoef(rec[i], env[i, c])[0, 1] for c in range(2)] for i in range(len(y))]
        return float((np.argmax(r, axis=1) == y).mean())

    best_alpha = max(
        alphas,
        key=lambda a: acc(fit(X[n_val:], target[n_val:], a), X[:n_val], env_tr[:n_val], y_tr[:n_val]),
    )
    return acc(fit(X, target, best_alpha), _lagged(eeg_te, n_lags), env_te, y_te)


# ---- StimulusGAT ----

def sgat_accuracy(train, test, *, seed, epochs, lr, batch_size=64, device="cpu"):
    torch.manual_seed(seed)
    model = StimulusGAT(n_channels=N_CHANNELS, sfreq=SFREQ, adjacency=_ring_adjacency(N_CHANNELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    Xtr, Etr, Ytr = (torch.as_tensor(a) for a in train)
    Xte, Ete, Yte = (torch.as_tensor(a) for a in test)
    Ytr, Yte = Ytr.long(), Yte.long()
    n = len(Ytr)
    train_acc = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xtr[idx].to(device), Etr[idx].to(device))
            loss = loss_fn(logits, Ytr[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1).cpu() == Ytr[idx]).sum().item()
        train_acc.append(correct / n)
    model.eval()
    with torch.no_grad():
        test_acc = (model(Xte.to(device), Ete.to(device)).argmax(1).cpu() == Yte).float().mean().item()
    return test_acc, train_acc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--snr-db", type=float, nargs="+", default=[0.0, -10.0])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--delay-ms", type=float, nargs="+", default=[0.0],
                   help="tracking task only: extra response latency; TRF lags are widened to cover it")
    p.add_argument("--out", type=Path, default=Path("results/diag_sgat_synthetic_dissociation.json"))
    args = p.parse_args()

    results = []
    for task in args.tasks:
        for delay_ms in (args.delay_ms if task == "tracking" else [0.0]):
            delay = int(round(delay_ms * SFREQ / 1000))
            kw = {"delay": delay} if task == "tracking" else {}
            for snr in args.snr_db:
                rng = np.random.default_rng(0)
                spatial = rng.standard_normal(N_CHANNELS)
                train = TASKS[task](rng, args.n_train, snr, spatial, **kw)
                test = TASKS[task](rng, args.n_test, snr, spatial, **kw)
                results.append(_run_condition(task, snr, delay_ms, train, test, delay + KERNEL_LEN, args))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                                    "results": results}, indent=2))
    print(f"Wrote {args.out}")
    return 0


def _run_condition(task, snr, delay_ms, train, test, n_lags, args):
    t0 = time.time()
    trf = trf_accuracy(train, test, n_lags=n_lags)
    sgat_runs = [sgat_accuracy(train, test, seed=s, epochs=args.epochs, lr=args.lr) for s in range(args.seeds)]
    sgat_test = [r[0] for r in sgat_runs]
    sgat_train_final = [r[1][-1] for r in sgat_runs]
    row = {
        "task": task, "snr_db": snr, "delay_ms": delay_ms, "trf_test_acc": trf,
        "sgat_test_acc_mean": float(np.mean(sgat_test)), "sgat_test_acc_std": float(np.std(sgat_test)),
        "sgat_train_acc_final_mean": float(np.mean(sgat_train_final)),
        "sgat_test_acc_per_seed": sgat_test,
        "sgat_train_curves": [r[1] for r in sgat_runs],
    }
    print(f"[{task:8s} snr={snr:+.0f}dB delay={delay_ms:.0f}ms]  TRF={trf:.3f}  "
          f"SGAT test={row['sgat_test_acc_mean']:.3f}+/-{row['sgat_test_acc_std']:.3f}  "
          f"SGAT train(final)={row['sgat_train_acc_final_mean']:.3f}  ({time.time()-t0:.0f}s)",
          flush=True)
    return row


if __name__ == "__main__":
    raise SystemExit(main())
