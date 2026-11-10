# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


# %% [markdown]
# # TRF + EA baseline with and without EOG regression (Kaggle, CPU)
#
# Companion of story-disjoint round 11 (notebook v16: StimulusGAT on EOG-regressed EEG).
# The round-11 SGAT result must be compared with an equally preprocessed linear baseline.
#
# Baseline: the TRF + EA of rounds 5-9 (1-8 Hz EEG, 64 Hz, backward model with lags 0-400 ms,
# 15-band power-law envelope, ridge alpha and raw/standardised variant chosen by validation
# correlation, EA per subject per split), on the same 18 story-disjoint folds.
# Arms:
# - trf_ea: scalp EEG (64 channels, average reference after filtering), EA per split.
#   Reproduction check: rounds 5-9 report 0.667 at 5 s (CAR before filtering there, so tiny
#   numerical differences are expected; a difference > 0.5 points means a pipeline error).
# - trf_ea_eogreg: the 3 bipolar EOG derivations regressed out per subject per split (the
#   function used by v16, aad_xai.data.ocular.regress_eog_trials), then EA per split.
# EA references are per subject per split, so training statistics are accumulated per fold
# (the class-sum shortcut of the ocular control does not hold with EA).
#
# Pre-registered:
# - Baseline rule (round 5): the round-11 TRF baseline is whichever arm has the higher mean
#   validation accuracy over the 18 folds.
# - Descriptive: trf_ea_eogreg minus trf_ea at 5/10/20/40 s, per-subject Wilcoxon, Holm over
#   the 4 lengths; shifted-envelope control of both arms (subject Wilcoxon vs 0.5).
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - CPU only (no GPU needed)

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
# ## 2. Configuration

# %%
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aad_xai.config import PreprocessConfig
from aad_xai.data.alignment import euclidean_align
from aad_xai.data.cv_splits import assert_story_disjoint_fold_integrity, subject_group_story_out
from aad_xai.data.ocular import N_DTU_WITH_EXG, dtu_derivations, regress_eog_trials
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.evaluation.decision_windows import (corr_from_stats, group_windows, recomputed_corr_decisions,
                                                 window_stats)
from aad_xai.models.sgat import DTU_MONTAGE_PATH
from aad_xai.models.trf_baseline import lag_matrix

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
DEVICE = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 1)

WINDOW_S = 5.0
FOLD_SEED = 42
SHIFT_S = 2.5
TRF_TMIN_S, TRF_TMAX_S = 0.0, 0.4
TRF_ALPHAS = 10.0 ** (np.arange(19) / 2)   # 1 .. 1e9
TRF_ENVELOPE_METHOD = "powerlaw_subbands_multi"
EEG_BAND_HZ = (1.0, 8.0)
ENVELOPE_BAND_HZ = (1.0, 8.0)
EA_SHRINKAGE = 1e-3
DECISION_S = (10, 20, 40)
ARMS = ("trf_ea", "trf_ea_eogreg")
OUT_DIR = Path("/kaggle/working/results_trf_ea_eogreg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pp = PreprocessConfig(bandpass_hz=EEG_BAND_HZ, reref=None, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
SFREQ = pp.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)
NAMES = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")["electrode_name"].tolist()
print(f"DTU dataset: {DTU_ROOT}; CPU threads {torch.get_num_threads()}")

# %% [markdown]
# ## 3. Load the 64 scalp + 8 external channels (no re-reference yet; 1-8 Hz, 64 Hz)

# %%
import scipy.io

from aad_xai.data.dtu_dataset import DTUDataset

assert DTU_ROOT is not None, "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'."
_raw = scipy.io.loadmat(str(Path(DTU_ROOT) / "eeg_new" / "S1.mat"), variable_names=["data"], squeeze_me=True)["data"]
_file_labels = [str(x) for x in _raw["dim"].item()["chan"].item()["eeg"].item().tolist()]
del _raw
assert _file_labels[:64] == NAMES, "scalp channel order differs from the montage"
print(f"File labels of columns 65-{len(_file_labels)}: {_file_labels[64:]}")

_t0 = time.time()
raw_trials = list(DTUDataset(root=DTU_ROOT, load_audio=True, n_eeg_channels=N_DTU_WITH_EXG, preprocess=pp,
                             envelope_method=TRF_ENVELOPE_METHOD).trials())
assert all(t.eeg.shape[0] == N_DTU_WITH_EXG for t in raw_trials)
print(f"Trials: {len(raw_trials)} in {time.time() - _t0:.0f}s")

# %% [markdown]
# ## 4. Scalp EEG and EOG derivations

# %%
scalp_trials, EOG = [], {}
for t in raw_trials:
    d = dtu_derivations(t.eeg)
    scalp_trials.append(dataclasses.replace(t, eeg=d["eeg"].astype(np.float32)))      # (64, T)
    EOG[t.trial_id] = d["eog"].astype(np.float32)                                      # (3, T)
del raw_trials
print(f"{len(scalp_trials)} trials: EEG {scalp_trials[0].eeg.shape}, EOG {EOG[scalp_trials[0].trial_id].shape}")


def arm_splits(arm, fold):
    """Train, val and test trials of one fold for one arm; every fitted step (EOG
    regression, EA) uses only the trials of its own split."""
    out = []
    for idx in (fold.train_idx, fold.val_idx, fold.test_idx):
        ts = [scalp_trials[i] for i in idx]
        if arm == "trf_ea_eogreg":
            ts = regress_eog_trials(ts, EOG)
        out.append(euclidean_align(ts, shrinkage=EA_SHRINKAGE))
    return out

# %% [markdown]
# ## 5. TRF helpers (round 5-9 baseline code, plus the shifted-envelope control)

# %%
from collections import Counter

TRF_LAGS = np.arange(int(round(TRF_TMIN_S * SFREQ)), int(round(TRF_TMAX_S * SFREQ)) + 1)


def _meta(ds):
    """Window metadata, one entry per window, in dataset order."""
    w = ds.window_indices
    return {"subject": np.array([x.subject_id for x in w]), "trial": np.array([x.trial_id for x in w]),
            "start": np.array([x.start for x in w]), "label": np.array([x.label for x in w])}


def _windows(trials, standardize):
    """Yield (eeg (C,T), env (2,K,T), label, subject_id) as numpy for every window."""
    ds = WindowedEEGAudioDataset(trials, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=standardize)
    for i, w in enumerate(ds.window_indices):
        eeg, env, label = ds[i]
        yield eeg.numpy(), env.numpy(), int(label), w.subject_id


def _accumulate(trials, standardize, chunk=200):
    """Sufficient statistics for ridge with K targets (the attended talker's K bands)."""
    st = {}
    buf_x, buf_y = [], []

    def flush():
        X = torch.from_numpy(np.concatenate(buf_x)).to(DEVICE)               # (rows, C*L)
        Y = torch.from_numpy(np.concatenate(buf_y)).to(DEVICE)               # (rows, K)
        if not st:
            d, k = X.shape[1], Y.shape[1]
            st.update(xtx=torch.zeros(d, d, dtype=torch.float64, device=DEVICE),
                      xsum=torch.zeros(d, dtype=torch.float64, device=DEVICE),
                      xty=torch.zeros(d, k, dtype=torch.float64, device=DEVICE),
                      ysum=torch.zeros(k, dtype=torch.float64, device=DEVICE), n=0)
        st["xtx"] += (X.T @ X).double(); st["xsum"] += X.sum(0).double()
        st["xty"] += (X.T @ Y).double(); st["ysum"] += Y.sum(0).double(); st["n"] += X.shape[0]
        buf_x.clear(); buf_y.clear()

    for eeg, env, label, _ in _windows(trials, standardize):
        buf_x.append(lag_matrix(eeg, TRF_LAGS))
        buf_y.append(env[label].T.astype(np.float32))                          # (T, K)
        if len(buf_x) == chunk:
            flush()
    if buf_x:
        flush()
    return {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in st.items()}


def _ridge_path(stats, alphas):
    """Weights for every alpha and band on standardised X and centred Y: (d, n_alpha, K), plus mu, sd."""
    n = stats["n"]
    mu = stats["xsum"] / n
    cov = stats["xtx"] / n - np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    gram = (stats["xtx"] - n * np.outer(mu, mu)) / np.outer(sd, sd)
    b = (stats["xty"] - n * mu[:, None] * (stats["ysum"] / n)[None, :]) / sd[:, None]     # (d, K)
    evals, evecs = np.linalg.eigh(gram)
    proj = evecs.T @ b                                                                    # (d, K)
    W = np.einsum("ij,jak->iak", evecs, proj[:, None, :] / (evals[:, None, None] + np.asarray(alphas)[None, :, None]))
    return W, mu, sd


def _z(x, axis):
    x = x - x.mean(axis=axis, keepdims=True)
    return x / (np.sqrt((x ** 2).mean(axis=axis, keepdims=True)) + 1e-12)


def _evaluate_path(trials, standardize, W, mu, sd):
    """Per alpha: per-subject correct counts, window counts, and the pooled attended
    correlation averaged over bands (selection criterion)."""
    d, n_alpha, K = W.shape
    Wf = torch.from_numpy(W.reshape(d, n_alpha * K)).float().to(DEVICE)
    correct, total = {}, Counter()
    s = {k: np.zeros((n_alpha, K)) for k in ("p", "y", "pp", "yy", "py")}
    n_rows = 0
    for eeg, env, label, subj in _windows(trials, standardize):
        X = torch.from_numpy(((lag_matrix(eeg, TRF_LAGS) - mu) / sd).astype(np.float32)).to(DEVICE)
        P = (X @ Wf).double().cpu().numpy().reshape(-1, n_alpha, K)          # (T, A, K)
        r = np.einsum("tak,ckt->cak", _z(P, 0), _z(env.astype(np.float64), -1)) / P.shape[0]   # (2, A, K)
        pred = (r.mean(-1)[1] > r.mean(-1)[0]).astype(int)                  # (A,)
        c = correct.setdefault(subj, np.zeros(n_alpha))
        c += pred == label
        total[subj] += 1
        y = env[label].T.astype(np.float64)[:, None, :]                       # (T, 1, K)
        s["p"] += P.sum(0); s["y"] += y.sum(0); s["pp"] += (P ** 2).sum(0)
        s["yy"] += (y ** 2).sum(0); s["py"] += (P * y).sum(0); n_rows += P.shape[0]
    cov = s["py"] / n_rows - (s["p"] / n_rows) * (s["y"] / n_rows)
    var_p = s["pp"] / n_rows - (s["p"] / n_rows) ** 2
    var_y = s["yy"] / n_rows - (s["y"] / n_rows) ** 2
    corr = (cov / np.sqrt(np.clip(var_p * var_y, 1e-24, None))).mean(-1)     # (A,)
    return correct, total, corr


def _trf_variant(stats, va, te, standardize):
    W, mu, sd = _ridge_path(stats, TRF_ALPHAS)
    vc, vt, vcorr = _evaluate_path(va, standardize, W, mu, sd)
    tc, tt, _ = _evaluate_path(te, standardize, W, mu, sd)
    val_acc = sum(vc.values()) / sum(vt.values())                            # (n_alpha,)
    return {"val_corr": vcorr, "val_acc": val_acc, "test_correct": tc, "test_total": tt,
            "W": W, "mu": mu, "sd": sd}


def _trf_outputs(trials, standardize, w, mu, sd):
    """At the selected alpha: per-window score (mean over bands of r_B - r_A), (N,), and
    sufficient statistics for longer windows, (N, 2, K, 5); the same with both
    envelopes circularly shifted by SHIFT_S (control)."""
    wt = torch.from_numpy(w).float().to(DEVICE)                               # (d, K)
    out = {"score": [], "stats": [], "shift_score": [], "shift_stats": []}
    for eeg, env, _, _ in _windows(trials, standardize):
        X = torch.from_numpy(((lag_matrix(eeg, TRF_LAGS) - mu) / sd).astype(np.float32)).to(DEVICE)
        P = (X @ wt).double().cpu().numpy()                                    # (T, K)
        for key, e in (("", env), ("shift_", np.roll(env, SHIFT_SAMPLES, axis=-1))):
            st = window_stats(P, e)                                            # (2, K, 5)
            r = corr_from_stats(st, P.shape[0])                                # (2, K)
            out[key + "score"].append(r[1].mean() - r[0].mean())
            out[key + "stats"].append(st)
    return {k: np.array(v) for k, v in out.items()}


def trf_fold(tr_stats, va, te):
    """TRF tuned by val correlation (the baseline rule since round 1), plus per-window outputs.
    tr_stats: {standardize (bool): training ridge statistics}."""
    variants = {"raw": _trf_variant(tr_stats[False], va, te, False), "std": _trf_variant(tr_stats[True], va, te, True)}
    key, k = max(((key, int(np.argmax(v["val_corr"])), float(np.max(v["val_corr"]))) for key, v in variants.items()),
                 key=lambda x: x[2])[:2]
    v = variants[key]
    out = {"variant": key, "alpha": float(TRF_ALPHAS[k]), "val_acc": float(v["val_acc"][k]),
           "val_corr": float(v["val_corr"][k]),
           "per_subject": {s: [int(v["test_correct"][s][k]), int(v["test_total"][s])] for s in v["test_total"]}}
    ds = WindowedEEGAudioDataset(te, window_s=WINDOW_S, overlap_s=0.0)
    arrays = {f"test_{name}": val for name, val in _meta(ds).items()}
    arrays.update({f"test_{name}": val for name, val in
                   _trf_outputs(te, key == "std", v["W"][:, k, :], v["mu"], v["sd"]).items()})
    arrays["weights"] = v["W"][:, k, :]                                        # (C*L, K), standardised X
    # The per-window scores must reproduce the counts used for selection.
    agree = (arrays["test_score"] > 0).astype(int) == arrays["test_label"]
    n_ok = sum(c for c, _ in out["per_subject"].values())
    out["score_count_mismatch"] = int(abs(int(agree.sum()) - n_ok))
    assert out["score_count_mismatch"] <= max(2, 0.005 * len(agree)), "TRF window scores inconsistent with counts"
    return out, arrays

# %% [markdown]
# ## 6. Story-disjoint CV, both arms per fold

# %%
folds = list(subject_group_story_out(scalp_trials, seed=FOLD_SEED))
for fold in folds:
    assert_story_disjoint_fold_integrity(scalp_trials, fold)
print(f"{len(folds)} story-disjoint folds (integrity checked)")

RESULTS = {arm: {} for arm in ARMS}


def save_json():
    with open(OUT_DIR / "trf_ea_eogreg.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {
                       "protocol": "subject_group_story_out (6 subject groups x 3 story folds, 2 val subjects)",
                       "alignment": f"euclidean per subject per split (shrinkage {EA_SHRINKAGE})",
                       "eog_regression": "trf_ea_eogreg: 3 bipolar EOG regressed out per subject per split, before EA",
                       "eeg_band_hz": list(EEG_BAND_HZ), "envelope_band_hz": list(ENVELOPE_BAND_HZ),
                       "envelope_method": TRF_ENVELOPE_METHOD, "sfreq": SFREQ, "window_s": WINDOW_S,
                       "trf_lags_s": [TRF_TMIN_S, TRF_TMAX_S], "trf_alphas": TRF_ALPHAS.tolist(),
                       "shift_s": SHIFT_S, "fold_seed": FOLD_SEED},
                   "folds": {arm: {fid: r["rec"] for fid, r in res.items()} for arm, res in RESULTS.items()}},
                  f, indent=2)


for fi, fold in enumerate(folds):
    for arm in ARMS:
        t0 = time.time()
        tr, va, te = arm_splits(arm, fold)
        tr_stats = {std: _accumulate(tr, std) for std in (False, True)}
        rec, arrays = trf_fold(tr_stats, va, te)
        rec["shift_correct"] = int(((arrays["test_shift_score"] > 0).astype(int) == arrays["test_label"]).sum())
        rec["n_test"] = int(len(arrays["test_label"]))
        np.savez_compressed(OUT_DIR / f"{arm}_{fold.fold_id}.npz", **arrays)
        RESULTS[arm][fold.fold_id] = {"rec": rec, "arrays": arrays}
        acc = sum(c for c, _ in rec["per_subject"].values()) / rec["n_test"]
        print(f"[{fi + 1:>2}/{len(folds)}] {arm} {fold.fold_id} test={fold.meta['test_subjects']}: {acc:.3f} "
              f"(val {rec['val_acc']:.3f}, shift {rec['shift_correct'] / rec['n_test']:.3f}, {rec['variant']}, "
              f"a={rec['alpha']:.0e}, {time.time() - t0:.0f}s)", flush=True)
    save_json()

# %% [markdown]
# ## 7. Pre-registered analysis

# %%
from scipy.stats import wilcoxon


def subject_acc(arm, L=5, kind=""):
    """Per-subject accuracy pooled over the subject's test folds, decision window L s."""
    counts = {}
    for r in RESULTS[arm].values():
        a = r["arrays"]
        if L == 5:
            groups = [(s, int(y), np.array([i])) for i, (s, y) in enumerate(zip(a["test_subject"], a["test_label"]))]
            dec = (a[f"test_{kind}score"] > 0).astype(int)
        else:
            groups = group_windows(a["test_subject"], a["test_trial"], a["test_start"], a["test_label"],
                                   int(L // WINDOW_S))
            dec = recomputed_corr_decisions(a[f"test_{kind}stats"], int(WINDOW_S * SFREQ), groups)
        for (s, y, _), d in zip(groups, dec):
            c = counts.setdefault(str(s), [0, 0])
            c[0] += int(int(d) == y); c[1] += 1
    subj = sorted(counts, key=lambda s: int(s[1:]))
    return subj, np.array([counts[s][0] / counts[s][1] for s in subj])


def _wil(d, alternative="two-sided"):
    d = np.asarray(d, dtype=float)
    return float(wilcoxon(d, alternative=alternative).pvalue) if np.any(d != 0) else 1.0


val = {arm: float(np.mean([r["rec"]["val_acc"] for r in RESULTS[arm].values()])) for arm in ARMS}
print(f"\n{'arm':14s} {'val':>6s}  " + " ".join(f"{L:>3}s acc (shift, p)" for L in (5,) + DECISION_S))
for arm in ARMS:
    cells = []
    for L in (5,) + DECISION_S:
        _, a = subject_acc(arm, L)
        _, sh = subject_acc(arm, L, "shift_")
        cells.append(f"{a.mean():.3f} ({sh.mean():.3f}, {_wil(sh - 0.5):.2g})")
    print(f"{arm:14s} {val[arm]:.4f}  " + "   ".join(cells))

baseline = max(ARMS, key=lambda a: val[a])
print(f"\nBaseline rule (higher mean val): {baseline}  (val {val['trf_ea']:.4f} vs {val['trf_ea_eogreg']:.4f})")
_, a5 = subject_acc("trf_ea")
print(f"Reproduction: trf_ea at 5 s {a5.mean():.4f} (rounds 5-9: 0.667)")

diffs = {}
for L in (5,) + DECISION_S:
    _, a = subject_acc("trf_ea", L)
    _, b = subject_acc("trf_ea_eogreg", L)
    diffs[L] = (float((b - a).mean()), int((b > a).sum()), _wil(b - a))
order = sorted(diffs, key=lambda k: diffs[k][2])
holm, running = {}, 0.0
for rank, k in enumerate(order):
    running = max(running, min(1.0, (len(order) - rank) * diffs[k][2]))
    holm[k] = running
print("\ntrf_ea_eogreg minus trf_ea:")
for L, (d, n_hi, p) in diffs.items():
    print(f"  {L:>2} s  {d:+.4f}  eogreg higher in {n_hi}/18  p={p:.3g}  Holm p={holm[L]:.3g}")
