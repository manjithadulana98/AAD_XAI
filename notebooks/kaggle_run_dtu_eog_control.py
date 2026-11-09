# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ocular control: is the attended talker decodable from the eye channels? (Kaggle, CPU)
#
# Why: in the interpretability run X1 on the round-9 StimulusGAT, occlusion ranked
# Fp2/Fp1/Fpz/AF7/AF8 highest, and our pipeline does no EOG removal. Eye movements are
# reported to track attended speech, so the model may partly read ocular activity. This
# run asks the prior question with linear decoders (no GPU): does the eye signal in DTU
# carry the attended talker?
#
# Channels (Zenodo 1199011 channel list; aad_xai.data.ocular): 3 bipolar EOG derivations
# (VEOG right = above - below right eye, VEOG left, HEOG = right - left outer canthus);
# 2 mastoids; 64 scalp electrodes, average reference.
#
# Decoder: the story-disjoint TRF baseline of rounds 5-9 (backward ridge, lags 0-400 ms,
# EEG 1-8 Hz, 15-band power-law envelope, ridge alpha and raw/standardised variant chosen
# by validation correlation), WITHOUT Euclidean alignment; the same 18 folds
# (subject_group_story_out, seed 42); 5 s test windows.
#
# Arms
# - eog3: the 3 bipolar EOG channels only (primary)
# - mastoid2: the 2 mastoids only (descriptive: a non-ocular 2-channel reference)
# - eeg64: all scalp electrodes
# - eeg64_eogreg: scalp electrodes with the 3 EOG channels regressed out (least squares
#   per subject over all of its trials; uses no labels)
# - eeg56_nofront: scalp electrodes without Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8
#
# Pre-registered (written before any result; subject = unit, n = 18, each subject's 3
# test folds pooled):
# - Gate (is the channel map right?): G1 median over subjects of corr(VEOG_R, VEOG_L)
#   >= 0.5; G2 median |corr(mean VEOG, mean of Fp1/Fpz/Fp2)| >= 0.3; G3 median
#   |corr(HEOG, F8 - F7)| >= 0.2; G4 each mastoid's most correlated scalp electrode is on
#   its own side's posterior-temporal set (right TP8/T8/P8/P10, left TP7/T7/P7/P9) in
#   >= 14 of 18 subjects. If the gate fails, no ocular claim is made.
# - Primary: eog3 accuracy at 5 s above 0.5 (one-sided Wilcoxon over subjects, p < 0.05)
#   AND its shifted-envelope control (2.5 s) not different from 0.5 (two-sided p >= 0.05)
#   -> "the attended talker is decodable from the eye channels". Otherwise "no evidence
#   with a linear decoder" (not evidence of absence).
# - Secondary (two-sided paired Wilcoxon at 5 s, Holm over the 2): eeg64 vs eeg64_eogreg;
#   eeg64 vs eeg56_nofront.
# - Descriptive: mastoid2; every arm at 10/20/40 s (correlation recomputed over the
#   decision window) and its shift control; variance of each electrode explained by the
#   EOG channels; selected alpha and variant.
#
# Caveats written in advance: bipolar EOG cancels most but not all cortical activity, so a
# positive eog3 result is strong but not conclusive evidence of an ocular source. This run
# tests linear decoders, not StimulusGAT itself.
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
from aad_xai.data.cv_splits import assert_story_disjoint_fold_integrity, subject_group_story_out
from aad_xai.data.ocular import N_DTU_WITH_EXG, dtu_derivations, fit_eog_regression, regress_out
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
DECISION_S = (10, 20, 40)
FRONTAL = ("Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8")
MASTOID_NEIGHBOURS = {"right": ("TP8", "T8", "P8", "P10"), "left": ("TP7", "T7", "P7", "P9")}
ARMS_CHEAP = ("eog3", "mastoid2")
ARMS_EEG = ("eeg64", "eeg64_eogreg", "eeg56_nofront")
ARMS = ARMS_CHEAP + ARMS_EEG
OUT_DIR = Path("/kaggle/working/results_eog_control")
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
# ## 4. Derivations, EOG regression, channel-map gate

# %%
idx = {n: i for i, n in enumerate(NAMES)}
keep56 = [i for i, n in enumerate(NAMES) if n not in FRONTAL]
assert len(keep56) == 56
subjects = sorted({t.subject_id for t in raw_trials}, key=lambda s: int(s[1:]))
der = {t.trial_id: dtu_derivations(t.eeg) for t in raw_trials}

B, gate_rows, eog_r2 = {}, [], []
_r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
for s in subjects:
    ids = [t.trial_id for t in raw_trials if t.subject_id == s]
    eeg = np.concatenate([der[i]["eeg"] for i in ids], axis=1)                  # (64, T_s)
    eog = np.concatenate([der[i]["eog"] for i in ids], axis=1)                  # (3, T_s)
    mast = np.concatenate([der[i]["mastoid"] for i in ids], axis=1)             # (2, T_s)
    B[s] = fit_eog_regression(eeg, eog)                                         # (3, 64)
    eog_r2.append(1 - regress_out(eeg, eog, B[s]).var(axis=1) / eeg.var(axis=1))
    cm = np.corrcoef(np.vstack([mast, eeg]))[:2, 2:]                            # (2, 64)
    best = [NAMES[int(np.argmax(c))] for c in cm]
    gate_rows.append({
        "subject": s,
        "g1_veog_r_l": _r(eog[0], eog[1]),
        "g2_veog_fp": abs(_r(eog[:2].mean(0), eeg[[idx["Fp1"], idx["Fpz"], idx["Fp2"]]].mean(0))),
        "g3_heog_f8_f7": abs(_r(eog[2], eeg[idx["F8"]] - eeg[idx["F7"]])),
        "mastoid_best": best,
        "g4_ok": best[0] in MASTOID_NEIGHBOURS["right"] and best[1] in MASTOID_NEIGHBOURS["left"],
    })
gate = {
    "G1": float(np.median([g["g1_veog_r_l"] for g in gate_rows])),
    "G2": float(np.median([g["g2_veog_fp"] for g in gate_rows])),
    "G3": float(np.median([g["g3_heog_f8_f7"] for g in gate_rows])),
    "G4": int(sum(g["g4_ok"] for g in gate_rows)),
}
gate["pass"] = bool(gate["G1"] >= 0.5 and gate["G2"] >= 0.3 and gate["G3"] >= 0.2 and gate["G4"] >= 14)
print(f"Gate: G1 {gate['G1']:.2f} (>=0.5)  G2 {gate['G2']:.2f} (>=0.3)  G3 {gate['G3']:.2f} (>=0.2)  "
      f"G4 {gate['G4']}/{len(subjects)} (>=14)  -> {'PASS' if gate['pass'] else 'FAIL'}")
eog_r2 = np.median(np.array(eog_r2), axis=0)                                     # (64,)
print("EEG variance explained by the EOG channels (median over subjects), top 10: "
      + ", ".join(f"{NAMES[i]} {eog_r2[i]:.2f}" for i in np.argsort(-eog_r2)[:10]))


def arm_eeg(t, arm):
    d = der[t.trial_id]
    if arm == "eog3":
        return d["eog"]
    if arm == "mastoid2":
        return d["mastoid"]
    if arm == "eeg64":
        return d["eeg"]
    if arm == "eeg64_eogreg":
        return regress_out(d["eeg"], d["eog"], B[t.subject_id])
    if arm == "eeg56_nofront":
        return d["eeg"][keep56]
    raise ValueError(arm)


ARM_TRIALS = {arm: [dataclasses.replace(t, eeg=arm_eeg(t, arm).astype(np.float32)) for t in raw_trials]
              for arm in ARMS}
del der
print({arm: ARM_TRIALS[arm][0].eeg.shape for arm in ARMS})

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


def train_classes(folds):
    """Trials grouped by their training-set membership over all folds: {frozenset of fold
    numbers: trial indices}. Every fold's training set is a union of these classes, so the
    ridge statistics, which are sums over windows, are computed once per class and summed."""
    member = {}
    for f, fold in enumerate(folds):
        for i in fold.train_idx:
            member.setdefault(i, set()).add(f)
    classes = {}
    for i, fs in member.items():
        classes.setdefault(frozenset(fs), []).append(i)
    return classes


def _sum_stats(parts):
    parts = [p for p in parts if p]
    return {k: sum(p[k] for p in parts) for k in parts[0]}


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
# ## 6. Story-disjoint CV, one arm at a time (cheap arms first)

# %%
folds = list(subject_group_story_out(raw_trials, seed=FOLD_SEED))
for fold in folds:
    assert_story_disjoint_fold_integrity(raw_trials, fold)
print(f"{len(folds)} story-disjoint folds (integrity checked)")

RESULTS = {arm: {} for arm in ARMS}


def save_json():
    with open(OUT_DIR / "eog_control.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {
                       "protocol": "subject_group_story_out (6 subject groups x 3 story folds, 2 val subjects), no EA",
                       "eeg_band_hz": list(EEG_BAND_HZ), "envelope_band_hz": list(ENVELOPE_BAND_HZ),
                       "envelope_method": TRF_ENVELOPE_METHOD, "sfreq": SFREQ, "window_s": WINDOW_S,
                       "trf_lags_s": [TRF_TMIN_S, TRF_TMAX_S], "trf_alphas": TRF_ALPHAS.tolist(),
                       "shift_s": SHIFT_S, "fold_seed": FOLD_SEED, "frontal_removed": list(FRONTAL)},
                   "file_labels_exg": _file_labels[64:] if "_file_labels" in globals() else None,
                   "gate": gate, "gate_rows": gate_rows,
                   "eog_r2_median": dict(zip(NAMES, map(float, eog_r2))),
                   "folds": {arm: {fid: r["rec"] for fid, r in res.items()} for arm, res in RESULTS.items()}},
                  f, indent=2)


CLASSES = train_classes(folds)
print(f"{len(CLASSES)} training classes; every fold's training set is a union of them")
for arm in ARMS:
    t_arm = time.time()
    ts = ARM_TRIALS[arm]
    cls = {std: {key: _accumulate([ts[i] for i in ix], std) for key, ix in CLASSES.items()} for std in (False, True)}
    print(f"{arm}: class statistics in {time.time() - t_arm:.0f}s", flush=True)
    for fi, fold in enumerate(folds):
        t0 = time.time()
        tr_stats = {std: _sum_stats([s for key, s in cls[std].items() if fi in key]) for std in (False, True)}
        if arm in ARMS_CHEAP:
            # Exactness check: summed class statistics == direct accumulation over the fold.
            for std in (False, True):
                direct = _accumulate([ts[i] for i in fold.train_idx], std)
                for k, v in direct.items():
                    assert np.allclose(v, tr_stats[std][k], rtol=1e-4, atol=1e-4 * np.abs(v).max()), (arm, k)
        rec, arrays = trf_fold(tr_stats, *[[ts[i] for i in ix] for ix in (fold.val_idx, fold.test_idx)])
        rec["shift_correct"] = int(((arrays["test_shift_score"] > 0).astype(int) == arrays["test_label"]).sum())
        rec["n_test"] = int(len(arrays["test_label"]))
        np.savez_compressed(OUT_DIR / f"{arm}_{fold.fold_id}.npz", **arrays)
        RESULTS[arm][fold.fold_id] = {"rec": rec, "arrays": arrays}
        acc = sum(c for c, _ in rec["per_subject"].values()) / rec["n_test"]
        print(f"[{fi + 1:>2}/{len(folds)}] {arm} {fold.fold_id} test={fold.meta['test_subjects']}: {acc:.3f} "
              f"(shift {rec['shift_correct'] / rec['n_test']:.3f}, {rec['variant']}, a={rec['alpha']:.0e}, "
              f"{time.time() - t0:.0f}s)", flush=True)
        save_json()
    del cls

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


print(f"Gate: {gate}")
print(f"\n{'arm':14s} " + " ".join(f"{L:>3}s acc (shift)" for L in (5,) + DECISION_S))
for arm in ARMS:
    cells = []
    for L in (5,) + DECISION_S:
        _, a = subject_acc(arm, L)
        _, sh = subject_acc(arm, L, "shift_")
        cells.append(f"{a.mean():.3f} ({sh.mean():.3f})")
    print(f"{arm:14s} " + "   ".join(cells))

subj, eog = subject_acc("eog3")
_, eog_sh = subject_acc("eog3", kind="shift_")
p_primary, p_shift = _wil(eog - 0.5, "greater"), _wil(eog_sh - 0.5)
primary = bool(p_primary < 0.05 and p_shift >= 0.05)
print(f"\nPrimary: eog3 at 5 s {eog.mean():.4f} (SD {eog.std(ddof=1):.3f}), > 0.5 p={p_primary:.3g}; "
      f"shift {eog_sh.mean():.4f} p={p_shift:.3g}")
print("  -> " + ("the attended talker is decodable from the eye channels" if primary
                 else "no evidence with a linear decoder"),
      "" if gate["pass"] else "(GATE FAILED: channel map not confirmed, no ocular claim)")
print("  per subject: " + ", ".join(f"{s} {v:.3f}" for s, v in zip(subj, eog)))

_, e64 = subject_acc("eeg64")
sec = {}
for arm in ("eeg64_eogreg", "eeg56_nofront"):
    _, a = subject_acc(arm)
    sec[arm] = (float((e64 - a).mean()), int((e64 > a).sum()), _wil(e64 - a))
order = sorted(sec, key=lambda k: sec[k][2])
holm = {}
running = 0.0
for rank, k in enumerate(order):
    running = max(running, min(1.0, (len(order) - rank) * sec[k][2]))
    holm[k] = running
print(f"\nSecondary (eeg64 {e64.mean():.4f} minus arm, 5 s):")
for k, (d, n_hi, p) in sec.items():
    print(f"  {k:14s} diff {d:+.4f}  eeg64 higher in {n_hi}/18  p={p:.3g}  Holm p={holm[k]:.3g}")
_, m2 = subject_acc("mastoid2")
print(f"\nDescriptive: mastoid2 at 5 s {m2.mean():.4f} (> 0.5 p={_wil(m2 - 0.5, 'greater'):.3g})")
