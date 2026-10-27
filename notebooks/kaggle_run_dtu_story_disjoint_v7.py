# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Story-Disjoint Protocol, Round 3: power-law sub-band speech envelope (Kaggle)
#
# Single factor changed vs. the current best (round 2): the speech envelope is
# extracted Biesmans-style (15-band gammatone filterbank at 8 kHz, |x|^0.6
# per band, summed, band-passed 1-8 Hz, z-scored per trial;
# `envelope_method="powerlaw_subbands"`) instead of the broadband Hilbert
# envelope. Still one envelope per talker, so StimulusGAT's input shape and
# architecture are unchanged.
#
# Arms, on the same story-disjoint folds as rounds 1-2:
# - TRF on 1-8 Hz EEG with the power-law envelope (same ridge grid and
#   selection as round 1). Compared offline, per subject, with round 1's
#   Hilbert TRF (same EEG, same folds).
# - StimulusGAT on 1-31 Hz EEG (kept in round 2), 2.5 s training hop, seed 42,
#   power-law envelope. Compared offline with round 2's `wide_overlap`.
#
# Pre-registered decisions (margin rule adopted after round 2):
# - StimulusGAT keeps the power-law envelope iff its mean pooled val accuracy
#   beats round 2 (0.6500) by >= 0.005 AND the shifted-envelope control is not
#   above chance (subject-level Wilcoxon vs 0.5, p >= 0.05).
# - The TRF baseline for later rounds uses whichever envelope gives the higher
#   TRF validation accuracy (tuned-by-val-corr config), regardless of the
#   StimulusGAT decision, so the baseline stays as strong as possible.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - GPU required (~3 h)

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
# ## 2. Confirm the model, the split and the decoupled envelope band are live

# %%
import torch
from aad_xai.config import PreprocessConfig
from aad_xai.models.sgat import StimulusGAT
from aad_xai.models.sgat_film import TimeResolvedFiLM
from aad_xai.data.cv_splits import (
    assert_story_disjoint_fold_integrity,
    dtu_content_group,
    subject_group_story_out,
)

_probe = StimulusGAT(n_channels=4, sfreq=64.0, adjacency=torch.eye(4))
assert isinstance(getattr(_probe, "film", None), TimeResolvedFiLM), "TimeResolvedFiLM NOT live."
assert _probe.readout.bias is None, "Bias-free readout NOT live."
assert PreprocessConfig(bandpass_hz=(1.0, 31.0), envelope_bandpass_hz=(1.0, 8.0)).envelope_band == (1.0, 8.0), \
    "envelope_bandpass_hz NOT live."
import inspect
from aad_xai.data.dtu_dataset import DTUDataset
assert "envelope_method" in inspect.signature(DTUDataset).parameters, "DTUDataset envelope_method NOT live."
print("Model, story-disjoint split, envelope_bandpass_hz and envelope_method confirmed live.")

# %% [markdown]
# ## 3. Load all subjects twice (TRF: EEG 1-8 Hz; StimulusGAT: EEG 1-31 Hz), power-law envelope

# %%
import time
from collections import Counter
from pathlib import Path

import numpy as np

from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.data.kul_dataset import _load_wav_envelope

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DTU dataset: {DTU_ROOT}   device: {DEVICE}")

WINDOW_S = 5.0
TRAIN_OVERLAP_S = 2.5
FOLD_SEED = 42
SGAT_SEED = 42
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
SHIFT_S = 2.5
TRF_TMIN_S, TRF_TMAX_S, TRF_TMAX_SHORT_S = 0.0, 0.4, 0.25
TRF_ALPHAS = 10.0 ** (np.arange(19) / 2)   # 1 .. 1e9
ENVELOPE_METHOD = "powerlaw_subbands"
ENVELOPE_BAND_HZ = (1.0, 8.0)
TRF_EEG_BAND_HZ = (1.0, 8.0)
SGAT_EEG_BAND_HZ = (1.0, 31.0)
SGAT_ARMS = {"wide_overlap_powerlaw": TRAIN_OVERLAP_S}
OUT_DIR = Path("/kaggle/working/results_story_disjoint_v7")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

pp_trf = PreprocessConfig(bandpass_hz=TRF_EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
pp_sgat = PreprocessConfig(bandpass_hz=SGAT_EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
assert pp_trf.sfreq_out == pp_sgat.sfreq_out
SFREQ = pp_sgat.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)

# Envelope sanity check on one stimulus: power-law vs Hilbert (same band/rate).
_wav = sorted(Path(DTU_ROOT, "Audio").glob("aske_story1_trial_1.wav"))[0]
_t0 = time.time()
_e_pl = _load_wav_envelope(_wav, target_sfreq=SFREQ, low_hz=1.0, high_hz=8.0, method=ENVELOPE_METHOD)
_t_pl = time.time() - _t0
_e_hi = _load_wav_envelope(_wav, target_sfreq=SFREQ, low_hz=1.0, high_hz=8.0, method="hilbert")
_n = min(len(_e_pl), len(_e_hi))
print(f"{_wav.name}: power-law {len(_e_pl)} samples ({_t_pl:.1f}s), Hilbert {len(_e_hi)} samples, "
      f"r = {np.corrcoef(_e_pl[:_n], _e_hi[:_n])[0, 1]:.3f}")
assert abs(len(_e_pl) - len(_e_hi)) <= 2, "envelope lengths disagree"

_t0 = time.time()
_ds_trf = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp_trf, envelope_method=ENVELOPE_METHOD)
trials_trf = list(_ds_trf.trials())
print(f"TRF trials: {len(trials_trf)} in {time.time() - _t0:.0f}s (EEG {TRF_EEG_BAND_HZ} Hz)")
_t0 = time.time()
_ds_sgat = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp_sgat, envelope_method=ENVELOPE_METHOD)
_ds_sgat._env_cache = _ds_trf._env_cache   # same method/rate/band -> reuse the envelopes
trials_sgat = list(_ds_sgat.trials())
print(f"SGAT trials: {len(trials_sgat)} in {time.time() - _t0:.0f}s (EEG {SGAT_EEG_BAND_HZ} Hz)")

assert [t.trial_id for t in trials_trf] == [t.trial_id for t in trials_sgat]
assert all(t.label == u.label and t.group_id == u.group_id for t, u in zip(trials_trf, trials_sgat))
for t, u in zip(trials_trf, trials_sgat):
    if t.audio_a is not None:
        assert np.array_equal(t.audio_a, u.audio_a) and np.array_equal(t.audio_b, u.audio_b)
all_trials = trials_sgat
print("Content groups:", sorted(Counter(dtu_content_group(t) for t in all_trials).items(), key=str))

# %% [markdown]
# ## 4. TRF (normal equations) and StimulusGAT helpers, with per-subject counts

# %%
import copy

import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.models.trf_baseline import lag_matrix, _safe_corrcoef
from aad_xai.utils.seed import seed_everything


def _lags(tmax_s):
    return np.arange(int(round(TRF_TMIN_S * SFREQ)), int(round(tmax_s * SFREQ)) + 1)


def _windows(trials, standardize):
    """Yield (eeg (C,T), env (2,T), label, subject_id) as numpy for every window."""
    ds = WindowedEEGAudioDataset(trials, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=standardize)
    for i, w in enumerate(ds.window_indices):
        eeg, env, label = ds[i]
        yield eeg.numpy(), env.numpy(), int(label), w.subject_id


def _accumulate(trials, standardize, lags, chunk=200):
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

    for eeg, env, label, _ in _windows(trials, standardize):
        buf_x.append(lag_matrix(eeg, lags))
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


def _evaluate_path(trials, standardize, lags, W, mu, sd):
    """Per alpha: per-subject correct counts, window counts, and pooled attended-envelope correlation."""
    n_alpha = W.shape[1]
    correct, total = {}, Counter()
    preds, targets = [], []
    for eeg, env, label, subj in _windows(trials, standardize):
        P = ((lag_matrix(eeg, lags) - mu) / sd) @ W                          # (T, n_alpha)
        c = correct.setdefault(subj, np.zeros(n_alpha))
        for k in range(n_alpha):
            r_a, r_b = _safe_corrcoef(P[:, k], env[0]), _safe_corrcoef(P[:, k], env[1])
            c[k] += int((0 if r_a > r_b else 1) == label)
        total[subj] += 1
        preds.append(P)
        targets.append(env[label])
    P_all, y_all = np.concatenate(preds), np.concatenate(targets)
    corr = np.array([_safe_corrcoef(P_all[:, k], y_all) for k in range(n_alpha)])
    return correct, total, corr


def _trf_variant(tr, va, te, standardize, lags):
    W, mu, sd = _ridge_path(_accumulate(tr, standardize, lags), TRF_ALPHAS)
    vc, vt, vcorr = _evaluate_path(va, standardize, lags, W, mu, sd)
    tc, tt, _ = _evaluate_path(te, standardize, lags, W, mu, sd)
    val_acc = sum(vc.values()) / sum(vt.values())                            # (n_alpha,)
    return {"val_corr": vcorr, "val_acc": val_acc, "test_correct": tc, "test_total": tt}


def _pick(variants, criterion):
    """Best (variant key, alpha index) by a validation criterion."""
    return max(((key, int(np.argmax(v[criterion])), float(np.max(v[criterion]))) for key, v in variants.items()),
               key=lambda x: x[2])[:2]


def _counts_at(v, k):
    return {s: [int(v["test_correct"][s][k]), int(v["test_total"][s])] for s in v["test_total"]}


def trf_fold(tr, va, te):
    variants = {
        "raw": _trf_variant(tr, va, te, False, _lags(TRF_TMAX_S)),
        "std": _trf_variant(tr, va, te, True, _lags(TRF_TMAX_S)),
    }
    short = _trf_variant(tr, va, te, False, _lags(TRF_TMAX_SHORT_S))
    out = {}
    for crit in ("val_corr", "val_acc"):
        key, k = _pick(variants, crit)
        out[f"tuned_by_{crit}"] = {"variant": key, "alpha": float(TRF_ALPHAS[k]),
                                   "val_acc": float(variants[key]["val_acc"][k]),
                                   "per_subject": _counts_at(variants[key], k)}
    k = int(np.argmax(variants["raw"]["val_corr"]))
    ks = int(np.argmax(short["val_corr"]))
    out["raw_lag400_by_val_corr"] = {"alpha": float(TRF_ALPHAS[k]), "per_subject": _counts_at(variants["raw"], k)}
    out["raw_lag250_by_val_corr"] = {"alpha": float(TRF_ALPHAS[ks]), "per_subject": _counts_at(short, ks)}
    return out


def _sgat_counts(model, ds, shift_samples=0):
    """Per-subject [correct, total] on a non-shuffled dataset."""
    model.eval()
    hits = []
    with torch.no_grad():
        for xb, envb, yb in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            if shift_samples:
                envb = torch.roll(envb, shifts=shift_samples, dims=-1)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            hits.append((model(xb, envb).argmax(1) == yb).cpu())
    hits = torch.cat(hits).numpy()
    out = {}
    for h, w in zip(hits, ds.window_indices):
        c = out.setdefault(w.subject_id, [0, 0])
        c[0] += int(h); c[1] += 1
    return out


def _pooled(counts):
    return sum(c for c, _ in counts.values()) / max(sum(n for _, n in counts.values()), 1)


def train_sgat_fold(ds_tr, ds_va, ds_te, n_channels, seed):
    seed_everything(seed)
    model = StimulusGAT(n_channels=n_channels, sfreq=SFREQ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    best = {"val_acc": -1.0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, envb, yb in dl_tr:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb, envb), yb).backward()
            opt.step()
        val_acc = _pooled(_sgat_counts(model, ds_va))
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}

    model.load_state_dict(best["state"])
    return model, {
        "best_epoch": best["epoch"],
        "val": _sgat_counts(model, ds_va),
        "test": _sgat_counts(model, ds_te),
        "test_shifted": _sgat_counts(model, ds_te, shift_samples=SHIFT_SAMPLES),
        "n_train_windows": len(ds_tr),
    }

# %% [markdown]
# ## 5. Story-disjoint CV (same folds as rounds 1-2): TRF + StimulusGAT per fold

# %%
import json

folds = list(subject_group_story_out(all_trials, seed=FOLD_SEED))
for fold in folds:
    assert_story_disjoint_fold_integrity(all_trials, fold)
print(f"{len(folds)} story-disjoint folds (integrity checked); SGAT arms {SGAT_ARMS}")

fold_results = []
for fi, fold in enumerate(folds):
    t0 = time.time()
    trf = trf_fold(*([trials_trf[i] for i in idx] for idx in (fold.train_idx, fold.val_idx, fold.test_idx)))
    t_trf = time.time() - t0
    tr, va, te = ([trials_sgat[i] for i in idx] for idx in (fold.train_idx, fold.val_idx, fold.test_idx))

    r = {"fold_id": fold.fold_id, "meta": fold.meta, "trf": trf, "sgat": {}}
    ds_va, ds_te = (WindowedEEGAudioDataset(t, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=True)
                    for t in (va, te))
    for arm, overlap_s in SGAT_ARMS.items():
        ds_tr = WindowedEEGAudioDataset(tr, window_s=WINDOW_S, overlap_s=overlap_s, standardize_eeg=True)
        model, rs = train_sgat_fold(ds_tr, ds_va, ds_te, n_channels=ds_tr[0][0].shape[0], seed=SGAT_SEED)
        torch.save(model.state_dict(), CKPT_DIR / f"sgat_v7_{arm}_{fold.fold_id}.pt")
        r["sgat"][arm] = rs
    fold_results.append(r)

    trf_sel = r["trf"]["tuned_by_val_corr"]
    sg = "  ".join(f"{arm}: val={_pooled(r['sgat'][arm]['val']):.3f} test={_pooled(r['sgat'][arm]['test']):.3f} "
                   f"(shift {_pooled(r['sgat'][arm]['test_shifted']):.3f}, ep {r['sgat'][arm]['best_epoch']})"
                   for arm in SGAT_ARMS)
    print(f"[{fi+1:>2}/{len(folds)}] {fold.fold_id} test={fold.meta['test_subjects']} "
          f"stories={fold.meta['held_out_content_groups']}: TRF={_pooled(trf_sel['per_subject']):.3f} "
          f"({trf_sel['variant']}, a={trf_sel['alpha']:.0e}, val {trf_sel['val_acc']:.3f})  {sg}  "
          f"(TRF {t_trf:.0f}s, total {time.time()-t0:.0f}s)", flush=True)

    with open(OUT_DIR / "story_disjoint_v7.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {
                       "protocol": "subject_group_story_out (6 subject groups x 3 story folds, 2 val subjects)",
                       "envelope_method": ENVELOPE_METHOD, "envelope_band_hz": list(ENVELOPE_BAND_HZ),
                       "trf_eeg_band_hz": list(TRF_EEG_BAND_HZ), "sgat_eeg_band_hz": list(SGAT_EEG_BAND_HZ),
                       "sfreq": SFREQ, "window_s": WINDOW_S, "fold_seed": FOLD_SEED, "sgat_seed": SGAT_SEED,
                       "sgat_arms": SGAT_ARMS, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
                       "shift_s": SHIFT_S, "trf_lags_s": [TRF_TMIN_S, TRF_TMAX_S],
                       "trf_short_lags_s": [TRF_TMIN_S, TRF_TMAX_SHORT_S], "trf_alphas": TRF_ALPHAS.tolist()},
                   "folds": fold_results}, f, indent=2)

# %% [markdown]
# ## 6. Summary (comparisons with rounds 1-2 are done offline, per subject)

# %%
from scipy.stats import wilcoxon


def per_subject(get):
    acc = {}
    for r in fold_results:
        for s, (c, n) in get(r).items():
            a = acc.setdefault(s, [0, 0]); a[0] += c; a[1] += n
    subjects = sorted(acc, key=lambda s: int(s[1:]))
    return subjects, np.array([acc[s][0] / acc[s][1] for s in subjects])


_, trf_test = per_subject(lambda r: r["trf"]["tuned_by_val_corr"]["per_subject"])
trf_val = np.mean([r["trf"]["tuned_by_val_corr"]["val_acc"] for r in fold_results])
print(f"TRF power-law (tuned by val corr): test {trf_test.mean():.4f} +/- {trf_test.std():.4f}  "
      f"mean val acc {trf_val:.4f}")
for arm in SGAT_ARMS:
    _, test = per_subject(lambda r: r["sgat"][arm]["test"])
    _, shifted = per_subject(lambda r: r["sgat"][arm]["test_shifted"])
    val = np.mean([_pooled(r["sgat"][arm]["val"]) for r in fold_results])
    d = test - trf_test
    print(f"SGAT {arm}: test {test.mean():.4f} +/- {test.std():.4f}  "
          f"shifted {shifted.mean():.4f} (above 0.5 on {int((shifted > 0.5).sum())}/{len(shifted)}, "
          f"Wilcoxon vs 0.5 p={wilcoxon(shifted - 0.5).pvalue:.4g})  mean pooled val acc {val:.4f}")
    print(f"  SGAT - TRF (both power-law): {d.mean():+.4f}, better on {int((d > 0).sum())}/{len(d)}, "
          f"Wilcoxon p={wilcoxon(test, trf_test).pvalue:.4g}")
