# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Story-Disjoint Protocol, Round 9: corrected baseline (Kaggle)
#
# Model: round-7b correlation pooling + shared speech kernels (33, 33) (round-8 screen
# KEEP) + the fixes from the 2026-11-03 code review:
# 1. Electrode graph from the DTU (BioSemi-64) electrode positions; earlier rounds used a
#    mismatched template (26% of edges were real neighbours). Checked below against the
#    channel labels stored in the DTU files.
# 2. Length-normalised band power in the frequency encoder, so the 5 s model can also
#    score 1 s and 2 s windows (saved here for the field-standard comparison).
# 3. Class-balanced cross-entropy, and epoch selection by class-balanced validation
#    accuracy: on these folds an EEG-free "training-majority talker" rule scores 0.67, so
#    neither training nor selection should reward a talker preference.
# Seeds 42, 1, 2 (PART fixA) and 3, 4 (PART fixB).
#
# Pre-registered analysis (offline, written before the runs):
# - The corrected model becomes the baseline regardless (these are bug fixes). Its 5-seed
#   ensemble is compared with the 5-seed round-7b corr ensemble at 5/10/20/40 s,
#   per-subject Wilcoxon, Holm over the 4 lengths. Claim "better" iff positive with Holm
#   p < 0.05 at some length; "no loss" iff the mean difference is >= -0.5 points at every
#   length; otherwise "loss", to be investigated before the interpretability phase.
# - Controls (5-seed ensemble): shift and trial-swap at chance (subject Wilcoxon vs 0.5
#   p >= 0.05 at every length, no fold >= 0.6 at 5 s); the shift-control rate of picking
#   the fold's training-majority talker is reported per fold.
# - Descriptive: single-model mean vs CA-TCN (0.583 / 0.634 / 0.703 / 0.767 at
#   1 / 2 / 5 / 10 s); 1 s and 2 s accuracy of the 5 s-trained model; same-seed
#   comparison with the round-8 skern seeds 42 and 1 (effect of the fixes).
# - The round-8 skern confirmation (seeds 2-4 under the old code) is superseded: skern is
#   part of this baseline on the strength of its screen, and is confirmed only jointly
#   with the fixes by the primary comparison above.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - GPU required

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
PART = "fixA"
PARTS = {
    "fixA": {"arm": "fix", "seeds": [42, 1, 2], "speech_kernel_sizes": [33, 33], "train_window_s": 5.0},
    "fixB": {"arm": "fix", "seeds": [3, 4], "speech_kernel_sizes": [33, 33], "train_window_s": 5.0},
}
CFG = PARTS[PART]
print(f"Part {PART}: {CFG}")

import torch
from aad_xai.config import PreprocessConfig
from aad_xai.models.sgat import DTU_MONTAGE_PATH, StimulusGAT
from aad_xai.data.alignment import euclidean_align
from aad_xai.data.cv_splits import assert_story_disjoint_fold_integrity, subject_group_story_out
from aad_xai.data.negatives import cross_trial_permutation
assert "freq_length_normalize" in StimulusGAT.__init__.__code__.co_varnames, "repo predates round 9"
print("StimulusGAT (round-9 fixes) and controls confirmed live.")

# %% [markdown]
# ## 3. Load subjects (EEG 1-31 Hz, summed power-law envelope)

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
TRAIN_HOP_S = 2.5
TRAIN_WINDOW_S = CFG["train_window_s"]
TRAIN_OVERLAP_S = TRAIN_WINDOW_S - TRAIN_HOP_S
FOLD_SEED = 42
SEED = 42
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
SHIFT_S = 2.5
SGAT_ENVELOPE_METHOD = "powerlaw_subbands"
ENVELOPE_BAND_HZ = (1.0, 8.0)
SGAT_EEG_BAND_HZ = (1.0, 31.0)
EA_SHRINKAGE = 1e-3
OUT_DIR = Path(f"/kaggle/working/results_story_disjoint_v14{PART}")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

pp_sgat = PreprocessConfig(bandpass_hz=SGAT_EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
SFREQ = pp_sgat.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)

_t0 = time.time()
trials_sgat = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp_sgat,
                              envelope_method=SGAT_ENVELOPE_METHOD).trials())
print(f"Trials: {len(trials_sgat)} in {time.time() - _t0:.0f}s (EEG {SGAT_EEG_BAND_HZ} Hz, {SGAT_ENVELOPE_METHOD})")
all_trials = trials_sgat

# Fix 1 check: the graph's electrode order must equal the channel labels in the DTU files.
import pandas as pd
import scipy.io
_raw = scipy.io.loadmat(str(Path(DTU_ROOT) / "eeg_new" / "S1.mat"), variable_names=["data"], squeeze_me=True)["data"]
_file_labels = [str(x) for x in _raw["dim"].item()["chan"].item()["eeg"].item().tolist()][:64]
_montage = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")["electrode_name"].tolist()
assert _file_labels == _montage, f"montage order differs from the DTU files: {_file_labels[:6]} vs {_montage[:6]}"
print("Electrode graph order matches the DTU channel labels (64/64).")
del _raw

# %% [markdown]
# ## 4. Helpers: stacked windows, logits, controls, StimulusGAT training

# %%
import copy

import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.utils.seed import seed_everything


def _meta(ds):
    """Window metadata, one entry per window, in dataset order."""
    w = ds.window_indices
    return {"subject": np.array([x.subject_id for x in w]), "trial": np.array([x.trial_id for x in w]),
            "start": np.array([x.start for x in w]), "label": np.array([x.label for x in w])}


def _stack(ds):
    """All windows of a dataset: X (N, C, T), E (N, 2, T) float32 tensors."""
    xs, es = zip(*[(x, e) for x, e, _ in (ds[i] for i in range(len(ds)))])
    return torch.stack(xs), torch.stack(es)


def _logits(model, X, E):
    """(N, 2) logits, model in eval mode."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            out.append(model(X[i:i + BATCH_SIZE].to(DEVICE), E[i:i + BATCH_SIZE].to(DEVICE)).float().cpu())
    return torch.cat(out).numpy()


def _counts(logits, labels, subjects):
    """Per-subject [correct, total] from (N, 2) logits."""
    out = {}
    for pred, s, y in zip(logits.argmax(1), subjects, labels):
        c = out.setdefault(str(s), [0, 0])
        c[0] += int(pred == y); c[1] += 1
    return out


def _pooled(counts):
    return sum(c for c, _ in counts.values()) / max(sum(n for _, n in counts.values()), 1)


def eval_suite(model, prefix, X_va, E_va, meta_va, X_te, E_te, meta_te):
    """Val, test, shift-control and trial-swap-control logits and counts."""
    perm = cross_trial_permutation(meta_te["trial"])
    logits = {"val": _logits(model, X_va, E_va), "test": _logits(model, X_te, E_te),
              "test_shifted": _logits(model, X_te, torch.roll(E_te, SHIFT_SAMPLES, dims=-1)),
              "test_swapped": _logits(model, X_te, E_te[perm])}
    counts = {k: _counts(v, (meta_va if k == "val" else meta_te)["label"],
                         (meta_va if k == "val" else meta_te)["subject"]) for k, v in logits.items()}
    return {f"{prefix}_{k}_logits": v for k, v in logits.items()}, counts


def _balanced(logits, labels):
    """Mean over the classes present of per-class accuracy (= accuracy if one class)."""
    pred = logits.argmax(1)
    return float(np.mean([np.mean(pred[labels == c] == c) for c in np.unique(labels)]))


def train_sgat(ds_tr, X_va, E_va, meta_va, n_channels, seed=SEED):
    """AdamW, 25 epochs, correlation pooling, this part's speech kernels, DTU electrode graph,
    length-normalised band power; class-balanced loss; best epoch by class-balanced
    pooled 5 s val accuracy."""
    seed_everything(seed)
    model = StimulusGAT(n_channels=n_channels, sfreq=SFREQ, film_pooling="corr",
                        speech_kernel_sizes=tuple(CFG["speech_kernel_sizes"]),
                        freq_length_normalize=True).to(DEVICE)
    ds = ds_tr
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_tr = np.array([w.label for w in ds_tr.window_indices])
    class_w = torch.tensor([len(y_tr) / (2.0 * max((y_tr == c).sum(), 1)) for c in (0, 1)],
                           dtype=torch.float32, device=DEVICE)   # (2,) inverse class frequency
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    dl_tr = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    best = {"val_acc": -1.0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, envb, yb in dl_tr:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)                  # (B, C, T_train), (B, 2, T_train)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            loss_fn(model.score_candidates(xb, envb), yb).backward()   # (B, N_cand) logits
            opt.step()
        val_acc = _balanced(_logits(model, X_va, E_va), meta_va["label"])
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    return model, best["epoch"], best["val_acc"], class_w.tolist()


SHORT_WINDOWS_S = (1.0, 2.0)


def short_window_suite(model, prefix, short):
    """Test logits on non-overlapping 1 s / 2 s windows, plus a half-window shift control."""
    out = {}
    for w_s, (X, E) in short.items():
        tag = f"test{int(w_s)}s"
        out[f"{prefix}_{tag}_logits"] = _logits(model, X, E)
        out[f"{prefix}_{tag}_shifted_logits"] = _logits(model, X, torch.roll(E, X.shape[-1] // 2, dims=-1))
    return out

# %% [markdown]
# ## 5. Story-disjoint CV (same folds as rounds 1-6), EA per split; save per-window outputs

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


ARM = CFG["arm"]
fold_results = []
for fi, fold in enumerate(folds):
    t0 = time.time()
    rec = {"fold_id": fold.fold_id, "meta": fold.meta, "models": {}}
    tr, va, te = aligned_splits(trials_sgat, fold)
    ds_va, ds_te = (WindowedEEGAudioDataset(t, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=True)
                    for t in (va, te))
    meta_va, meta_te = _meta(ds_va), _meta(ds_te)
    (X_va, E_va), (X_te, E_te) = _stack(ds_va), _stack(ds_te)
    arrays = {f"{split}_{k}": v for split, m in (("val", meta_va), ("test", meta_te)) for k, v in m.items()}
    n_channels = X_va.shape[1]
    suite = (X_va, E_va, meta_va, X_te, E_te, meta_te)

    short = {}
    for w_s in SHORT_WINDOWS_S:
        ds_s = WindowedEEGAudioDataset(te, window_s=w_s, overlap_s=0.0, standardize_eeg=True)
        short[w_s] = _stack(ds_s)
        arrays.update({f"test{int(w_s)}s_{k}": v for k, v in _meta(ds_s).items()})

    ds_tr = WindowedEEGAudioDataset(tr, window_s=TRAIN_WINDOW_S, overlap_s=TRAIN_OVERLAP_S, standardize_eeg=True)
    for seed in CFG["seeds"]:
        model, ep, val_bal, class_w = train_sgat(ds_tr, X_va, E_va, meta_va, n_channels, seed)
        torch.save(model.state_dict(), CKPT_DIR / f"sgat_v14_{ARM}_s{seed}_{fold.fold_id}.pt")
        a, counts = eval_suite(model, f"{ARM}_s{seed}", *suite)
        arrays.update(a)
        arrays.update(short_window_suite(model, f"{ARM}_s{seed}", short))
        rec["models"][f"{ARM}_s{seed}"] = {"best_epoch": ep, "val_balanced": val_bal, "class_weights": class_w,
                                           **counts}

    np.savez_compressed(OUT_DIR / f"{fold.fold_id}.npz", **arrays)
    fold_results.append(rec)

    txt = "  ".join(f"{m}: val={_pooled(r['val']):.3f} test={_pooled(r['test']):.3f} "
                    f"(shift {_pooled(r['test_shifted']):.3f}, swap {_pooled(r['test_swapped']):.3f}, "
                    f"ep {r['best_epoch']})" for m, r in rec["models"].items())
    print(f"[{fi+1:>2}/{len(folds)}] {fold.fold_id} test={fold.meta['test_subjects']}: {txt}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    with open(OUT_DIR / f"story_disjoint_v14{PART}.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "part": PART, "config": {
                       "protocol": "subject_group_story_out (6 subject groups x 3 story folds, 2 val subjects)",
                       "alignment": f"euclidean per subject per split (shrinkage {EA_SHRINKAGE})",
                       "envelope_method": SGAT_ENVELOPE_METHOD, "envelope_band_hz": list(ENVELOPE_BAND_HZ),
                       "eeg_band_hz": list(SGAT_EEG_BAND_HZ), "sfreq": SFREQ, "window_s": WINDOW_S,
                       "train_window_s": TRAIN_WINDOW_S, "train_overlap_s": TRAIN_OVERLAP_S, "fold_seed": FOLD_SEED, "seed": SEED,
                       "part_cfg": CFG, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
                       "shift_s": SHIFT_S, "film_pooling": "corr", "freq_length_normalize": True,
                       "class_balanced": True, "montage": str(DTU_MONTAGE_PATH.name),
                       "short_windows_s": list(SHORT_WINDOWS_S),
                       },
                   "folds": fold_results}, f, indent=2)

# %% [markdown]
# ## 6. Summary of this part (decisions and comparisons are analysed offline)

# %%
from scipy.stats import wilcoxon


def per_subject(get):
    acc = {}
    for r in fold_results:
        for s, (c, n) in get(r).items():
            a = acc.setdefault(s, [0, 0]); a[0] += c; a[1] += n
    subjects = sorted(acc, key=lambda s: int(s[1:]))
    return subjects, np.array([acc[s][0] / acc[s][1] for s in subjects])


for m in fold_results[0]["models"]:
    val = np.mean([_pooled(r["models"][m]["val"]) for r in fold_results])
    _, test = per_subject(lambda r, m=m: r["models"][m]["test"])
    line = f"{m}: val {val:.4f}  test {test.mean():.4f} +/- {test.std():.4f}"
    for ctrl in ("test_shifted", "test_swapped"):
        _, c = per_subject(lambda r, m=m, k=ctrl: r["models"][m][k])
        flagged = [r["fold_id"] for r in fold_results if _pooled(r["models"][m][ctrl]) >= 0.6]
        line += f"  {ctrl[5:]} {c.mean():.4f} (p={wilcoxon(c - 0.5).pvalue:.3g}; folds >= 0.6: {flagged or 'none'})"
    print(line)
