# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # KUL external held-out test: TRF + EA baseline (Kaggle, CPU)
#
# Pre-registration: `docs/preregistration_kul_external_test.md` (frozen before this run).
# The linear baseline of the DTU rounds 5-11, unchanged: backward ridge decoder on 1-8 Hz EEG
# at 64 Hz, lags 0-400 ms, 15-band power-law envelope targets, ridge alpha (1 .. 1e9) and the
# raw / standardised variant chosen per fold by validation correlation, EA per subject per
# split. Same data, folds and 5 s test windows as the StimulusGAT notebook
# (kaggle_run_kul_external_sgat.py). Longer decisions are computed offline from the saved
# per-window sufficient statistics (correlation recomputed over the whole window).
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/kul-dataset`
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
import glob
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch

from aad_xai.config import PreprocessConfig
from aad_xai.data.alignment import euclidean_align
from aad_xai.data.cv_splits import (KUL_STORY_FOLDS, assert_story_disjoint_fold_integrity, kul_story_part,
                                    subject_group_story_out)
from aad_xai.data.kul_dataset import KULeuvenDataset
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.evaluation.decision_windows import corr_from_stats, window_stats
from aad_xai.models.sgat import DTU_MONTAGE_PATH
from aad_xai.models.trf_baseline import lag_matrix

assert "envelope_source" in KULeuvenDataset.__init__.__code__.co_varnames, "repo predates the KUL loader update"
assert KUL_STORY_FOLDS == ((1, 2), (3, 4))
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
N_SUBJECT_GROUPS, N_VAL_SUBJECTS = 8, 2
STIMULI_FINGERPRINT = "853dd5093a3d88b2209fc72db741067f"   # kernel kul-aad-structure-check v2
OUT_DIR = Path("/kaggle/working/results_kul_external_trf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pp = PreprocessConfig(bandpass_hz=EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)   # average reference
SFREQ = pp.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)
print(f"CPU threads {torch.get_num_threads()}")

# %% [markdown]
# ## 3. KUL data: integrity, design checks, load (EEG 1-8 Hz, 15-band power-law envelope)

# %%
hits = glob.glob("/kaggle/input/**/KULeuven/S1.mat", recursive=True)
assert len(hits) == 1, f"attach the dataset dulanamanjitha/kul-dataset: {hits}"
KUL_ROOT = Path(hits[0]).parent
manifest = json.load(open(KUL_ROOT / "manifest.json"))
assert (manifest["record"], manifest["version"]) == ("4004271", "2.0"), manifest["version"]
for name, info in sorted(manifest["files"].items()):
    if name == "stimuli.zip":
        continue
    h = hashlib.md5()
    with open(KUL_ROOT / name, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == info["md5"], f"{name}: md5 differs from Zenodo"
wav_md5 = {w.name: hashlib.md5(w.read_bytes()).hexdigest() for w in sorted((KUL_ROOT / "stimuli").glob("*.wav"))}
fingerprint = hashlib.md5("".join(f"{k}:{v};" for k, v in sorted(wav_md5.items())).encode()).hexdigest()
assert fingerprint == STIMULI_FINGERPRINT, f"stimuli differ from the checked release: {fingerprint}"
print(f"KUL at {KUL_ROOT}: subject files match the Zenodo MD5s; stimuli fingerprint {fingerprint}")


def _unwrap(x):
    a = np.asarray(x)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.flat[0])
    return a


_tr = scipy.io.loadmat(str(KUL_ROOT / "S1.mat"))["trials"]
_t0 = _unwrap(_tr[0, 0] if _tr.ndim == 2 else _tr[0])
_labels = [str(np.asarray(lab).flat[0]) for lab in _unwrap(_unwrap(_t0["FileHeader"])["Channels"])["Label"].flat]
MONTAGE = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")["electrode_name"].tolist()
assert _labels == MONTAGE, f"KUL channel order differs from the montage: {_labels[:6]} vs {MONTAGE[:6]}"
del _tr, _t0

_t0 = time.time()
kul = KULeuvenDataset(KUL_ROOT, preprocess=pp, load_audio=True, envelope_method=TRF_ENVELOPE_METHOD,
                      include_experiments=(1, 2), envelope_source="dry")
all_trials = list(kul.trials())
TRIAL_INFO = {t.trial_id: kul.trial_info[t.trial_id] for t in all_trials}
print(f"Trials: {len(all_trials)} in {time.time() - _t0:.0f}s")

assert len({t.subject_id for t in all_trials}) == 16 and len(all_trials) == 128
for t in all_trials:
    info = TRIAL_INFO[t.trial_id]
    assert t.eeg.shape[0] == 64 and t.sfreq == SFREQ
    assert t.audio_a.ndim == 2 and t.audio_a.shape == t.audio_b.shape and t.audio_a.shape[1] == t.eeg.shape[1]
    assert info["left_envelope_wav"].endswith("_dry.wav") and info["right_envelope_wav"].endswith("_dry.wav")
    assert info["experiment"] in (1, 2) and kul_story_part(t) == info["story_part"]
    assert t.label == int(info["attended_ear"] == "R")
print(f"Design checks passed; envelope bands K = {all_trials[0].audio_a.shape[0]}")

# %% [markdown]
# ## 4. TRF helpers (DTU rounds 5-11 baseline code, unchanged)

# %%
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
# ## 5. KUL story-disjoint CV (same folds as the StimulusGAT notebook)

# %%
folds = list(subject_group_story_out(all_trials, content_group=kul_story_part, story_folds=KUL_STORY_FOLDS,
                                     n_subject_groups=N_SUBJECT_GROUPS, n_val_subjects=N_VAL_SUBJECTS,
                                     seed=FOLD_SEED))
for fold in folds:
    assert_story_disjoint_fold_integrity(all_trials, fold, content_group=kul_story_part)
_tested = Counter(i for f in folds for i in f.test_idx)
assert len(folds) == 16 and set(_tested) == set(range(len(all_trials))) and set(_tested.values()) == {1}
print(f"{len(folds)} story-disjoint folds (integrity checked; every trial tested once)")

RESULTS = {}


def save_json():
    with open(OUT_DIR / "kul_external_trf.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {
                       "preregistration": "docs/preregistration_kul_external_test.md",
                       "data": {"zenodo": "4004271 v2.0", "kaggle_dataset": "dulanamanjitha/kul-dataset",
                                "stimuli_fingerprint": fingerprint, "experiments": [1, 2],
                                "envelope_source": "dry"},
                       "protocol": f"subject_group_story_out, story folds {KUL_STORY_FOLDS}, "
                                   f"{N_SUBJECT_GROUPS} subject groups, {N_VAL_SUBJECTS} val subjects",
                       "alignment": f"euclidean per subject per split (shrinkage {EA_SHRINKAGE})",
                       "eeg_band_hz": list(EEG_BAND_HZ), "envelope_band_hz": list(ENVELOPE_BAND_HZ),
                       "envelope_method": TRF_ENVELOPE_METHOD, "sfreq": SFREQ, "window_s": WINDOW_S,
                       "trf_lags_s": [TRF_TMIN_S, TRF_TMAX_S], "trf_alphas": TRF_ALPHAS.tolist(),
                       "shift_s": SHIFT_S, "fold_seed": FOLD_SEED},
                   "trial_info": TRIAL_INFO,
                   "folds": {fid: r["rec"] for fid, r in RESULTS.items()}}, f, indent=2)


for fi, fold in enumerate(folds):
    t0 = time.time()
    tr, va, te = (euclidean_align([all_trials[i] for i in idx], shrinkage=EA_SHRINKAGE)
                  for idx in (fold.train_idx, fold.val_idx, fold.test_idx))
    tr_stats = {std: _accumulate(tr, std) for std in (False, True)}
    rec, arrays = trf_fold(tr_stats, va, te)
    rec["shift_correct"] = int(((arrays["test_shift_score"] > 0).astype(int) == arrays["test_label"]).sum())
    rec["n_test"] = int(len(arrays["test_label"]))
    rec["meta"] = fold.meta
    np.savez_compressed(OUT_DIR / f"trf_ea_{fold.fold_id}.npz", **arrays)
    RESULTS[fold.fold_id] = {"rec": rec, "arrays": arrays}
    acc = sum(c for c, _ in rec["per_subject"].values()) / rec["n_test"]
    print(f"[{fi + 1:>2}/{len(folds)}] trf_ea {fold.fold_id} test={fold.meta['test_subjects']}: {acc:.3f} "
          f"(val {rec['val_acc']:.3f}, shift {rec['shift_correct'] / rec['n_test']:.3f}, {rec['variant']}, "
          f"a={rec['alpha']:.0e}, {time.time() - t0:.0f}s)", flush=True)
    save_json()

# %% [markdown]
# ## 6. Summary (the pre-registered analysis runs offline)

# %%
from scipy.stats import wilcoxon

counts, shift = {}, {}
for r in RESULTS.values():
    a = r["arrays"]
    for s, y, sc, ss in zip(a["test_subject"], a["test_label"], a["test_score"], a["test_shift_score"]):
        c = counts.setdefault(str(s), [0, 0]); c[0] += int((sc > 0) == bool(y)); c[1] += 1
        c = shift.setdefault(str(s), [0, 0]); c[0] += int((ss > 0) == bool(y)); c[1] += 1
subj = sorted(counts, key=lambda s: int(s[1:]))
acc = np.array([counts[s][0] / counts[s][1] for s in subj])
sh = np.array([shift[s][0] / shift[s][1] for s in subj])
val = np.mean([r["rec"]["val_acc"] for r in RESULTS.values()])
print(f"TRF + EA, 5 s: val {val:.4f}  test {acc.mean():.4f} +/- {acc.std():.4f}  "
      f"shift {sh.mean():.4f} (p={wilcoxon(sh - 0.5).pvalue:.3g})")
