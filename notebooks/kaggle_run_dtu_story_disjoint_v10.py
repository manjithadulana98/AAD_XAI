# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Story-Disjoint Protocol, Round 6: 5-seed ensemble and longer decision windows (Kaggle)
#
# Final configuration of each model (round 5), no new factor:
# - TRF: 1-8 Hz EEG, 15-band power-law envelope, multi-output ridge, Euclidean
#   alignment (EA). Deterministic, so it is run once.
# - StimulusGAT: 1-31 Hz EEG, summed power-law envelope, 2.5 s hop, EA.
#   Seeds {42, 1, 2, 3, 4}. Seed 42 is reloaded from the round-5 checkpoints
#   (no retraining); the others are trained here.
#
# The work is split over two kernels run in parallel (PART below):
# - A: TRF, seed-42 reload (checked against round 5), seeds 1 and 2.
# - B: seeds 3 and 4.
# Every fold saves per-window outputs (SGAT logits, TRF scores and sufficient
# statistics, window metadata) so that the ensemble and the longer decision
# windows are computed offline.
#
# Pre-registered analysis (offline, both parts combined):
# - Ensemble = mean of the 5 seeds' logits. Longer windows (10/20/40 s) combine
#   consecutive 5 s windows of one trial (aad_xai.evaluation.decision_windows):
#   SGAT sums the logit differences.
# - TRF at each length uses whichever rule has the higher mean val accuracy:
#   summed 5 s correlation differences, or the correlation recomputed over the
#   whole decision window (the usual TRF practice).
# - Primary: ensemble SGAT vs TRF at 5 s, per-subject Wilcoxon (two-sided).
#   Secondary: the same at 10/20/40 s; Holm correction over the 4 lengths.
# - Shift control (envelopes rolled by 2.5 s) for the ensemble at every length:
#   subject-level Wilcoxon vs 0.5, and no fold >= 0.6 at 5 s.
# - Descriptive: ensemble vs the mean single seed; seed-to-seed spread.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - Part A also attaches the output of kernel `dulanamanjitha/story-disjoint-v9-round5`
# - GPU required (~4.5 h per part)

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
# ## 2. Part, and confirm the modules are live

# %%
PART = "A"
PARTS = {
    "A": {"train_seeds": [1, 2], "run_trf": True, "reload_seed42": True},
    "B": {"train_seeds": [3, 4], "run_trf": False, "reload_seed42": False},
}
CFG = PARTS[PART]
print(f"Part {PART}: {CFG}")

import torch
from aad_xai.config import PreprocessConfig
from aad_xai.models.sgat import StimulusGAT
from aad_xai.data.alignment import euclidean_align
from aad_xai.data.cv_splits import assert_story_disjoint_fold_integrity, subject_group_story_out
from aad_xai.evaluation.decision_windows import corr_from_stats, window_stats
print("StimulusGAT, euclidean_align, story-disjoint split and decision_windows confirmed live.")

# %% [markdown]
# ## 3. Load subjects (StimulusGAT: EEG 1-31 Hz, summed envelope; part A also TRF: EEG 1-8 Hz, 15-band envelope)

# %%
import time
from pathlib import Path

import numpy as np

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
TRAIN_OVERLAP_S = 2.5
FOLD_SEED = 42
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
SHIFT_S = 2.5
TRF_TMIN_S, TRF_TMAX_S = 0.0, 0.4
TRF_ALPHAS = 10.0 ** (np.arange(19) / 2)   # 1 .. 1e9
TRF_ENVELOPE_METHOD = "powerlaw_subbands_multi"
SGAT_ENVELOPE_METHOD = "powerlaw_subbands"
N_BANDS = 15
ENVELOPE_BAND_HZ = (1.0, 8.0)
TRF_EEG_BAND_HZ = (1.0, 8.0)
SGAT_EEG_BAND_HZ = (1.0, 31.0)
EA_SHRINKAGE = 1e-3
OUT_DIR = Path(f"/kaggle/working/results_story_disjoint_v10{PART}")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

pp_trf = PreprocessConfig(bandpass_hz=TRF_EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
pp_sgat = PreprocessConfig(bandpass_hz=SGAT_EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
assert pp_trf.sfreq_out == pp_sgat.sfreq_out
SFREQ = pp_sgat.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)

_t0 = time.time()
trials_sgat = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp_sgat,
                              envelope_method=SGAT_ENVELOPE_METHOD).trials())
print(f"SGAT trials: {len(trials_sgat)} in {time.time() - _t0:.0f}s (EEG {SGAT_EEG_BAND_HZ} Hz, {SGAT_ENVELOPE_METHOD})")
trials_trf = None
if CFG["run_trf"]:
    _t0 = time.time()
    trials_trf = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp_trf,
                                 envelope_method=TRF_ENVELOPE_METHOD).trials())
    print(f"TRF trials: {len(trials_trf)} in {time.time() - _t0:.0f}s (EEG {TRF_EEG_BAND_HZ} Hz, {TRF_ENVELOPE_METHOD})")
    assert [t.trial_id for t in trials_trf] == [t.trial_id for t in trials_sgat]
    for t, u in zip(trials_trf, trials_sgat):
        assert t.label == u.label and t.group_id == u.group_id
all_trials = trials_sgat

V9_CKPT, V9_FOLDS = {}, {}
if CFG["reload_seed42"]:
    import json
    V9_CKPT = {p.name: p for p in Path("/kaggle/input").rglob("sgat_v9_wide_overlap_powerlaw_ea_*.pt")}
    _v9_json = next(Path("/kaggle/input").rglob("story_disjoint_v9.json"))
    V9_FOLDS = {f["fold_id"]: f for f in json.load(open(_v9_json))["folds"]}
    print(f"Round-5 checkpoints found: {len(V9_CKPT)}; round-5 results: {_v9_json}")
    assert len(V9_CKPT) == 18 and len(V9_FOLDS) == 18, "attach the output of story-disjoint-v9-round5"

# %% [markdown]
# ## 4. Helpers: TRF (selected by val correlation, with per-window outputs) and StimulusGAT logits

# %%
import copy
from collections import Counter

import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.models.trf_baseline import lag_matrix
from aad_xai.utils.seed import seed_everything

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


def _trf_variant(tr, va, te, standardize):
    W, mu, sd = _ridge_path(_accumulate(tr, standardize), TRF_ALPHAS)
    vc, vt, vcorr = _evaluate_path(va, standardize, W, mu, sd)
    tc, tt, _ = _evaluate_path(te, standardize, W, mu, sd)
    val_acc = sum(vc.values()) / sum(vt.values())                            # (n_alpha,)
    return {"val_corr": vcorr, "val_acc": val_acc, "test_correct": tc, "test_total": tt,
            "W": W, "mu": mu, "sd": sd}


def _trf_outputs(trials, standardize, w, mu, sd):
    """At the selected alpha: per-window score (mean over bands of r_B - r_A), (N,),
    and sufficient statistics for recomputing correlations on longer windows, (N, 2, K, 5)."""
    wt = torch.from_numpy(w).float().to(DEVICE)                               # (d, K)
    scores, stats = [], []
    for eeg, env, _, _ in _windows(trials, standardize):
        X = torch.from_numpy(((lag_matrix(eeg, TRF_LAGS) - mu) / sd).astype(np.float32)).to(DEVICE)
        P = (X @ wt).double().cpu().numpy()                                    # (T, K)
        st = window_stats(P, env)                                              # (2, K, 5)
        r = corr_from_stats(st, P.shape[0])                                    # (2, K)
        scores.append(r[1].mean() - r[0].mean())
        stats.append(st)
    return np.array(scores), np.stack(stats)


def trf_fold(tr, va, te):
    """TRF tuned by val correlation (the baseline rule since round 1), plus per-window outputs."""
    variants = {"raw": _trf_variant(tr, va, te, False), "std": _trf_variant(tr, va, te, True)}
    key, k = max(((key, int(np.argmax(v["val_corr"])), float(np.max(v["val_corr"]))) for key, v in variants.items()),
                 key=lambda x: x[2])[:2]
    v = variants[key]
    out = {"variant": key, "alpha": float(TRF_ALPHAS[k]), "val_acc": float(v["val_acc"][k]),
           "per_subject": {s: [int(v["test_correct"][s][k]), int(v["test_total"][s])] for s in v["test_total"]}}
    arrays = {}
    for split, trials in (("val", va), ("test", te)):
        ds = WindowedEEGAudioDataset(trials, window_s=WINDOW_S, overlap_s=0.0)
        for name, val in _meta(ds).items():
            arrays[f"trf_{split}_{name}"] = val
        arrays[f"trf_{split}_score"], arrays[f"trf_{split}_stats"] = \
            _trf_outputs(trials, key == "std", v["W"][:, k, :], v["mu"], v["sd"])
    # The per-window scores must reproduce the counts used for selection.
    agree = (arrays["trf_test_score"] > 0).astype(int) == arrays["trf_test_label"]
    n_ok = sum(c for c, _ in out["per_subject"].values())
    out["score_count_mismatch"] = int(abs(int(agree.sum()) - n_ok))
    assert out["score_count_mismatch"] <= max(2, 0.005 * len(agree)), "TRF window scores inconsistent with counts"
    return out, arrays


def _sgat_logits(model, ds, shift_samples=0):
    """(N, 2) logits on a non-shuffled dataset."""
    model.eval()
    out = []
    with torch.no_grad():
        for xb, envb, _ in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            if shift_samples:
                envb = torch.roll(envb, shifts=shift_samples, dims=-1)
            out.append(model(xb, envb).float().cpu())
    return torch.cat(out).numpy()


def _counts(logits, meta):
    """Per-subject [correct, total] from (N, 2) logits."""
    out = {}
    for pred, s, y in zip(logits.argmax(1), meta["subject"], meta["label"]):
        c = out.setdefault(str(s), [0, 0])
        c[0] += int(pred == y); c[1] += 1
    return out


def _pooled(counts):
    return sum(c for c, _ in counts.values()) / max(sum(n for _, n in counts.values()), 1)


def train_sgat_fold(ds_tr, ds_va, meta_va, n_channels, seed):
    """Same training as rounds 1-5; best epoch by pooled val accuracy."""
    seed_everything(seed)
    model = StimulusGAT(n_channels=n_channels, sfreq=SFREQ).to(DEVICE)   # summed envelope, env (B, 2, T)
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
        val_acc = _pooled(_counts(_sgat_logits(model, ds_va), meta_va))
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    return model, best["epoch"]

# %% [markdown]
# ## 5. Story-disjoint CV (same folds as rounds 1-5), EA per split; save per-window outputs

# %%
import json

folds = list(subject_group_story_out(all_trials, seed=FOLD_SEED))
for fold in folds:
    assert_story_disjoint_fold_integrity(all_trials, fold)
print(f"{len(folds)} story-disjoint folds (integrity checked)")


def aligned_splits(trials, fold):
    """EA with references computed separately inside train, val and test."""
    return [euclidean_align([trials[i] for i in idx], shrinkage=EA_SHRINKAGE)
            for idx in (fold.train_idx, fold.val_idx, fold.test_idx)]


fold_results = []
for fi, fold in enumerate(folds):
    t0 = time.time()
    rec = {"fold_id": fold.fold_id, "meta": fold.meta, "sgat": {}}
    tr, va, te = aligned_splits(trials_sgat, fold)
    ds_va, ds_te = (WindowedEEGAudioDataset(t, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=True)
                    for t in (va, te))
    meta_va, meta_te = _meta(ds_va), _meta(ds_te)
    arrays = {f"{split}_{k}": v for split, m in (("val", meta_va), ("test", meta_te)) for k, v in m.items()}
    n_channels = ds_va[0][0].shape[0]

    if CFG["run_trf"]:
        rec["trf"], trf_arrays = trf_fold(*aligned_splits(trials_trf, fold))
        arrays.update(trf_arrays)

    models = []
    if CFG["reload_seed42"]:
        m42 = StimulusGAT(n_channels=n_channels, sfreq=SFREQ).to(DEVICE)
        m42.load_state_dict(torch.load(V9_CKPT[f"sgat_v9_wide_overlap_powerlaw_ea_{fold.fold_id}.pt"],
                                       map_location=DEVICE))
        models.append((42, m42, V9_FOLDS[fold.fold_id]["sgat"]["wide_overlap_powerlaw_ea"]["best_epoch"]))
    ds_tr = WindowedEEGAudioDataset(tr, window_s=WINDOW_S, overlap_s=TRAIN_OVERLAP_S, standardize_eeg=True)
    for seed in CFG["train_seeds"]:
        model, ep = train_sgat_fold(ds_tr, ds_va, meta_va, n_channels, seed)
        torch.save(model.state_dict(), CKPT_DIR / f"sgat_v10_s{seed}_{fold.fold_id}.pt")
        models.append((seed, model, ep))

    for seed, model, ep in models:
        lv, lt = _sgat_logits(model, ds_va), _sgat_logits(model, ds_te)
        ls = _sgat_logits(model, ds_te, shift_samples=SHIFT_SAMPLES)
        arrays.update({f"sgat_s{seed}_val_logits": lv, f"sgat_s{seed}_test_logits": lt,
                       f"sgat_s{seed}_test_shifted_logits": ls})
        rec["sgat"][str(seed)] = {"best_epoch": ep, "val": _counts(lv, meta_va), "test": _counts(lt, meta_te),
                                  "test_shifted": _counts(ls, meta_te)}
    if CFG["reload_seed42"]:
        # The reloaded seed-42 model must reproduce round 5 (up to GPU rounding).
        v9_test = V9_FOLDS[fold.fold_id]["sgat"]["wide_overlap_powerlaw_ea"]["test"]
        diff = sum(abs(rec["sgat"]["42"]["test"][s][0] - v9_test[s][0]) for s in v9_test)
        rec["seed42_reload_diff"] = int(diff)
        assert diff <= max(2, 0.005 * len(meta_te["label"])), f"seed-42 reload differs from round 5 by {diff}"
    del models
    np.savez_compressed(OUT_DIR / f"{fold.fold_id}.npz", **arrays)
    fold_results.append(rec)

    trf_txt = (f"TRF={_pooled(rec['trf']['per_subject']):.3f} ({rec['trf']['variant']}, a={rec['trf']['alpha']:.0e}, "
               f"mismatch {rec['trf']['score_count_mismatch']})  " if CFG["run_trf"] else "")
    sg_txt = "  ".join(f"s{s}: val={_pooled(r['val']):.3f} test={_pooled(r['test']):.3f} "
                       f"(shift {_pooled(r['test_shifted']):.3f}, ep {r['best_epoch']})" for s, r in rec["sgat"].items())
    reload_txt = f"  reload diff {rec['seed42_reload_diff']}" if CFG["reload_seed42"] else ""
    print(f"[{fi+1:>2}/{len(folds)}] {fold.fold_id} test={fold.meta['test_subjects']}: {trf_txt}{sg_txt}{reload_txt}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    with open(OUT_DIR / f"story_disjoint_v10{PART}.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "part": PART, "config": {
                       "protocol": "subject_group_story_out (6 subject groups x 3 story folds, 2 val subjects)",
                       "alignment": f"euclidean per subject per split (shrinkage {EA_SHRINKAGE})",
                       "trf_envelope_method": TRF_ENVELOPE_METHOD, "sgat_envelope_method": SGAT_ENVELOPE_METHOD,
                       "n_bands_trf": N_BANDS, "envelope_band_hz": list(ENVELOPE_BAND_HZ),
                       "trf_eeg_band_hz": list(TRF_EEG_BAND_HZ), "sgat_eeg_band_hz": list(SGAT_EEG_BAND_HZ),
                       "sfreq": SFREQ, "window_s": WINDOW_S, "train_overlap_s": TRAIN_OVERLAP_S,
                       "fold_seed": FOLD_SEED, "part_cfg": CFG, "epochs": EPOCHS, "lr": LR,
                       "weight_decay": WEIGHT_DECAY, "shift_s": SHIFT_S, "trf_lags_s": [TRF_TMIN_S, TRF_TMAX_S],
                       "trf_alphas": TRF_ALPHAS.tolist()},
                   "folds": fold_results}, f, indent=2)

# %% [markdown]
# ## 6. Summary of this part (the ensemble and longer windows are analysed offline)

# %%
from scipy.stats import wilcoxon


def per_subject(get):
    acc = {}
    for r in fold_results:
        for s, (c, n) in get(r).items():
            a = acc.setdefault(s, [0, 0]); a[0] += c; a[1] += n
    subjects = sorted(acc, key=lambda s: int(s[1:]))
    return subjects, np.array([acc[s][0] / acc[s][1] for s in subjects])


if CFG["run_trf"]:
    _, trf_test = per_subject(lambda r: r["trf"]["per_subject"])
    print(f"TRF + EA: test {trf_test.mean():.4f} +/- {trf_test.std():.4f}")
for seed in fold_results[0]["sgat"]:
    _, test = per_subject(lambda r, s=seed: r["sgat"][s]["test"])
    _, shifted = per_subject(lambda r, s=seed: r["sgat"][s]["test_shifted"])
    flagged = [r["fold_id"] for r in fold_results if _pooled(r["sgat"][seed]["test_shifted"]) >= 0.6]
    print(f"SGAT seed {seed}: test {test.mean():.4f} +/- {test.std():.4f}  shifted {shifted.mean():.4f} "
          f"(Wilcoxon vs 0.5 p={wilcoxon(shifted - 0.5).pvalue:.3g}; folds >= 0.6: {flagged or 'none'})")
