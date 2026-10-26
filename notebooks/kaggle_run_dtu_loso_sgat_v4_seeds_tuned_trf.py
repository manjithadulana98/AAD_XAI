# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT v4 -- Extra Seeds and a Tuned TRF Baseline on the Same LOSO Folds (Kaggle)
#
# v4 LOSO (seed 42): StimulusGAT 0.6286 +/- 0.0487 vs. fixed-alpha TRF
# 0.5997 +/- 0.0408 (16/18 subjects, Wilcoxon p=0.002). Two gaps before that
# claim is solid:
# 1. The TRF used one fixed alpha (100) and no input-normalisation choice.
#    Here TRF selects alpha (19 values, 1..1e9) AND per-window EEG
#    standardisation (off/on) on each fold's validation subject, by
#    validation envelope correlation (the repo's existing tune_alpha metric).
# 2. One seed. Here StimulusGAT is retrained with seeds 43 and 44; seed 42
#    comes from the v4 run and is combined offline.
#
# The TRF is solved from accumulated normal equations (X^T X built once per
# fold on GPU), which is the same ridge model as TRFDecoder (standardised
# lagged EEG, centred target, lags 0..0.4 s) but makes a 38-point grid cheap.
# alpha=100 without standardisation is reported separately as a check that it
# reproduces the v4 TRF numbers.
#
# Folds: leave_one_subject_out(all_trials, seed=42), identical to v4.
# Results are written after every fold, so a timeout still leaves complete,
# paired folds.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - GPU required (~7 h)

# %% [markdown]
# ## 1. Clone repository and install dependencies

# %%
import os
import subprocess
import sys

REPO_DIR = "/kaggle/working/AAD_XAI"

if not os.path.exists(REPO_DIR):
    subprocess.run(
        ["git", "clone", "https://github.com/manjithadulana98/AAD_XAI.git", REPO_DIR],
        check=True,
    )
else:
    print(f"Repository already present at {REPO_DIR} -- pulling latest.")
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)

os.chdir(REPO_DIR)

try:
    import torch as _torch_preinstalled
    print(f"Pre-installed torch {_torch_preinstalled.__version__} found -- "
          "keeping it; installing the rest of requirements.txt without touching torch.")
    with open("requirements.txt") as _f:
        _reqs_no_torch = [ln for ln in _f if ln.strip() and not ln.strip().lower().startswith("torch")]
    with open("/tmp/requirements_no_torch.txt", "w") as _f:
        _f.writelines(_reqs_no_torch)
    subprocess.run(["pip", "install", "-q", "-r", "/tmp/requirements_no_torch.txt"], check=True)
except ImportError:
    print("No pre-installed torch found -- installing requirements.txt as-is.")
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

subprocess.run(["pip", "install", "-q", "-e", "."], check=True)

_src_dir = os.path.join(REPO_DIR, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import aad_xai  # noqa: F401
GIT_COMMIT = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
print(f"Commit: {GIT_COMMIT}")

# %% [markdown]
# ## 2. Confirm the time-resolved FiLM model is live

# %%
import torch
from aad_xai.models.sgat import StimulusGAT
from aad_xai.models.sgat_film import TimeResolvedFiLM

_probe = StimulusGAT(n_channels=4, sfreq=64.0, adjacency=torch.eye(4))
assert isinstance(getattr(_probe, "film", None), TimeResolvedFiLM), "TimeResolvedFiLM NOT live."
assert _probe.readout.bias is None, "Bias-free readout NOT live."
print("Time-resolved FiLM model confirmed live.")

# %% [markdown]
# ## 3. Load all subjects

# %%
import time
from pathlib import Path

from aad_xai.config import PreprocessConfig
from aad_xai.data.dtu_dataset import DTUDataset

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DTU dataset: {DTU_ROOT}   device: {DEVICE}")

WINDOW_S = 5.0
FOLD_SEED = 42
SGAT_SEEDS = [43, 44]
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
SHIFT_S = 2.5
TRF_TMIN_S, TRF_TMAX_S = 0.0, 0.4
TRF_ALPHAS = 10.0 ** (__import__("numpy").arange(19) / 2)   # 1 .. 1e9, includes exactly 100
OUT_DIR = Path("/kaggle/working/results_sgat_v4_seeds_tuned_trf")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

preprocess = PreprocessConfig()
SFREQ = preprocess.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)

_t0 = time.time()
all_trials = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=preprocess).trials())
print(f"Loaded {len(all_trials)} trials, {len({t.subject_id for t in all_trials})} subjects "
      f"in {time.time() - _t0:.0f}s")

# %% [markdown]
# ## 4. Tuned TRF (normal equations) and StimulusGAT helpers

# %%
import copy

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.data.cv_splits import leave_one_subject_out
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.models.trf_baseline import lag_matrix, _safe_corrcoef
from aad_xai.utils.seed import seed_everything

TRF_LAGS = np.arange(int(round(TRF_TMIN_S * SFREQ)), int(round(TRF_TMAX_S * SFREQ)) + 1)


def _windows(trials, standardize):
    """Yield (eeg (C,T), env (2,T), label) as numpy for every window."""
    ds = WindowedEEGAudioDataset(trials, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=standardize)
    for i in range(len(ds)):
        eeg, env, label = ds[i]
        yield eeg.numpy(), env.numpy(), int(label)


def _accumulate(trials, standardize, chunk=200):
    """Sufficient statistics of the lagged design matrix for ridge regression."""
    xtx = xsum = xty = None
    ysum, n = 0.0, 0
    buf_x, buf_y = [], []

    def flush():
        nonlocal xtx, xsum, xty, ysum, n
        X = torch.from_numpy(np.concatenate(buf_x)).to(DEVICE)               # (rows, C*L) float32
        y = torch.from_numpy(np.concatenate(buf_y)).to(DEVICE)               # (rows,)
        if xtx is None:
            d = X.shape[1]
            xtx = torch.zeros(d, d, dtype=torch.float64, device=DEVICE)
            xsum = torch.zeros(d, dtype=torch.float64, device=DEVICE)
            xty = torch.zeros(d, dtype=torch.float64, device=DEVICE)
        xtx += (X.T @ X).double()
        xsum += X.sum(0).double()
        xty += (X.T @ y).double()
        ysum += y.double().sum().item()
        n += X.shape[0]
        buf_x.clear(); buf_y.clear()

    for eeg, env, label in _windows(trials, standardize):
        buf_x.append(lag_matrix(eeg, TRF_LAGS))
        buf_y.append(env[label].astype(np.float32))
        if len(buf_x) == chunk:
            flush()
    if buf_x:
        flush()
    return {k: v.cpu().numpy() for k, v in (("xtx", xtx), ("xsum", xsum), ("xty", xty))} | {"ysum": ysum, "n": n}


def _ridge_path(stats, alphas):
    """Weights for every alpha on standardised X and centred y: (d, n_alpha), plus mu, sd."""
    n = stats["n"]
    mu = stats["xsum"] / n
    cov = stats["xtx"] / n - np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    gram = (stats["xtx"] - n * np.outer(mu, mu)) / np.outer(sd, sd)
    b = (stats["xty"] - n * mu * (stats["ysum"] / n)) / sd
    evals, evecs = np.linalg.eigh(gram)
    proj = evecs.T @ b
    W = evecs @ (proj[:, None] / (evals[:, None] + np.asarray(alphas)[None, :]))
    return W, mu, sd


def _predict_all(eeg, W, mu, sd):
    Xs = (lag_matrix(eeg, TRF_LAGS) - mu) / sd                               # (T, d)
    return Xs @ W                                                            # (T, n_alpha)


def _evaluate_path(trials, standardize, W, mu, sd):
    """Per alpha: decoding accuracy and attended-envelope correlation (concatenated)."""
    n_alpha = W.shape[1]
    correct = np.zeros(n_alpha)
    preds, targets = [], []
    n_win = 0
    for eeg, env, label in _windows(trials, standardize):
        P = _predict_all(eeg, W, mu, sd)
        for k in range(n_alpha):
            r_a, r_b = _safe_corrcoef(P[:, k], env[0]), _safe_corrcoef(P[:, k], env[1])
            correct[k] += int((0 if r_a > r_b else 1) == label)
        preds.append(P)
        targets.append(env[label])
        n_win += 1
    P_all, y_all = np.concatenate(preds), np.concatenate(targets)
    corr = np.array([_safe_corrcoef(P_all[:, k], y_all) for k in range(n_alpha)])
    return correct / max(n_win, 1), corr


def tuned_trf_fold(tr, va, te):
    out = {}
    best = None
    for standardize in (False, True):
        W, mu, sd = _ridge_path(_accumulate(tr, standardize), TRF_ALPHAS)
        val_acc, val_corr = _evaluate_path(va, standardize, W, mu, sd)
        test_acc, _ = _evaluate_path(te, standardize, W, mu, sd)
        key = "std" if standardize else "raw"
        out[key] = {"val_corr": val_corr.tolist(), "val_acc": val_acc.tolist(), "test_acc": test_acc.tolist()}
        k = int(np.argmax(val_corr))
        if best is None or val_corr[k] > best["val_corr"]:
            best = {"standardize": standardize, "alpha": float(TRF_ALPHAS[k]),
                    "val_corr": float(val_corr[k]), "test_acc": float(test_acc[k])}
    alpha100 = int(np.argmin(np.abs(TRF_ALPHAS - 100.0)))
    out["fixed_alpha100_raw_test_acc"] = out["raw"]["test_acc"][alpha100]
    out["selected"] = best
    return out


def _sgat_accuracy(model, loader, shift_samples=0):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, envb, yb in loader:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            if shift_samples:
                envb = torch.roll(envb, shifts=shift_samples, dims=-1)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            correct += (model(xb, envb).argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / max(total, 1)


def train_sgat_fold(ds_tr, ds_va, ds_te, n_channels, seed):
    seed_everything(seed)
    model = StimulusGAT(n_channels=n_channels, sfreq=SFREQ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best = {"val_acc": -1.0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, envb, yb in dl_tr:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb, envb), yb).backward()
            opt.step()
        val_acc = _sgat_accuracy(model, dl_va)
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}

    model.load_state_dict(best["state"])
    return model, {
        "best_epoch": best["epoch"], "val_acc": best["val_acc"],
        "test_acc": _sgat_accuracy(model, dl_te),
        "test_acc_shifted": _sgat_accuracy(model, dl_te, shift_samples=SHIFT_SAMPLES),
    }

# %% [markdown]
# ## 5. Full LOSO: tuned TRF + StimulusGAT seeds per fold

# %%
import json

folds = list(leave_one_subject_out(all_trials, seed=FOLD_SEED))
print(f"{len(folds)} LOSO folds, TRF grid {len(TRF_ALPHAS)} alphas x 2 input normalisations, "
      f"SGAT seeds {SGAT_SEEDS}")

fold_results = []
for fi, fold in enumerate(folds):
    t0 = time.time()
    test_subject = fold.meta["test_subject"]
    tr = [all_trials[i] for i in fold.train_idx]
    va = [all_trials[i] for i in fold.val_idx]
    te = [all_trials[i] for i in fold.test_idx]

    r = {"test_subject": test_subject, "val_subject": va[0].subject_id, "trf": tuned_trf_fold(tr, va, te)}
    t_trf = time.time() - t0

    ds_tr, ds_va, ds_te = (WindowedEEGAudioDataset(t, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=True)
                           for t in (tr, va, te))
    r["sgat"] = {}
    for seed in SGAT_SEEDS:
        model, rs = train_sgat_fold(ds_tr, ds_va, ds_te, n_channels=ds_tr[0][0].shape[0], seed=seed)
        torch.save(model.state_dict(), CKPT_DIR / f"sgat_v4_seed{seed}_{test_subject}.pt")
        r["sgat"][str(seed)] = rs
    fold_results.append(r)

    sel = r["trf"]["selected"]
    sg = "  ".join(f"s{s}={r['sgat'][str(s)]['test_acc']:.4f}(shift {r['sgat'][str(s)]['test_acc_shifted']:.3f})"
                   for s in SGAT_SEEDS)
    print(f"[{fi+1:>2}/{len(folds)}] {test_subject:>4}: TRF tuned={sel['test_acc']:.4f} "
          f"(alpha={sel['alpha']:.0e}, std={sel['standardize']})  TRF a100={r['trf']['fixed_alpha100_raw_test_acc']:.4f}  "
          f"{sg}  (TRF {t_trf:.0f}s, total {time.time()-t0:.0f}s)", flush=True)

    with open(OUT_DIR / "loso_v4_seeds_tuned_trf.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {
                       "window_s": WINDOW_S, "fold_seed": FOLD_SEED, "sgat_seeds": SGAT_SEEDS, "epochs": EPOCHS,
                       "lr": LR, "weight_decay": WEIGHT_DECAY, "shift_s": SHIFT_S,
                       "trf_lags_s": [TRF_TMIN_S, TRF_TMAX_S], "trf_alphas": TRF_ALPHAS.tolist(),
                       "trf_selection": "max validation correlation over alpha x {raw, standardized}"},
                   "folds": fold_results}, f, indent=2)

# %% [markdown]
# ## 6. Summary (seed 42 is combined offline with the v4 results)

# %%
from scipy.stats import wilcoxon

trf_tuned = np.array([r["trf"]["selected"]["test_acc"] for r in fold_results])
trf_a100 = np.array([r["trf"]["fixed_alpha100_raw_test_acc"] for r in fold_results])
sgat = np.array([[r["sgat"][str(s)]["test_acc"] for s in SGAT_SEEDS] for r in fold_results])
sgat_mean = sgat.mean(1)

print(f"\n{len(fold_results)} folds")
print(f"  TRF tuned (val-selected)   {trf_tuned.mean():.4f} +/- {trf_tuned.std():.4f}")
print(f"  TRF alpha=100 (v4 check)   {trf_a100.mean():.4f} +/- {trf_a100.std():.4f}   (v4 TRF: 0.5997 +/- 0.0408)")
for j, s in enumerate(SGAT_SEEDS):
    print(f"  StimulusGAT seed {s}       {sgat[:, j].mean():.4f} +/- {sgat[:, j].std():.4f}")
print(f"  StimulusGAT seed-mean      {sgat_mean.mean():.4f} +/- {sgat_mean.std():.4f}")
d = sgat_mean - trf_tuned
print(f"SGAT(seed-mean) - TRF tuned: {d.mean():+.4f}, better on {int((d > 0).sum())}/{len(d)}, "
      f"Wilcoxon p={wilcoxon(sgat_mean, trf_tuned).pvalue:.4g}")
print("Selected TRF configs:", [(r["trf"]["selected"]["alpha"], r["trf"]["selected"]["standardize"]) for r in fold_results])
