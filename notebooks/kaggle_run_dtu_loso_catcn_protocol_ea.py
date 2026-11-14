# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT + Euclidean alignment under the AADNet / CA-TCN subject-independent protocol (Kaggle)
#
# EA ARM. Identical to notebooks/kaggle_run_dtu_loso_catcn_protocol.py (the no-EA run, RESULTS_LOG
# 2026-11-08: 0.678 at 5 s vs CA-TCN 0.703) except for one factor: per-subject Euclidean
# alignment (aad_xai.data.alignment.euclidean_align, shrinkage 1e-3), with references computed
# separately inside the training trials, the validation trials and the test fold (the test
# subject's 7-8 held-out trials, about 6 min of EEG). EEG only, no labels or audio; for the test
# subject it is transductive (it uses the unlabelled test-fold EEG). CA-TCN and AADNet use no
# alignment, so any comparison with them must state this. On the story-disjoint protocol EA gave
# StimulusGAT +2.7 points at 5 s (round 5, p = 0.010).
# Same seed (42), folds, training and scoring as the no-EA run, so the two are paired by subject.
#
# Pre-registered analysis of this arm (offline, written before the runs):
# - Versus the published numbers: the bootstrap-CI rule below ("above" / "below" / "level" at
#   1 / 2 / 5 / 10 / 25 / 50 s), reported with the EA caveat.
# - Versus the no-EA run: per-subject paired Wilcoxon of EA minus no-EA at the six lengths, Holm
#   over 6. EA "helps" iff positive with Holm p < 0.05 at some length.
# - Controls at 5 s as below (shift and trial swap at chance).
#
# Original description of the protocol (unchanged):
#
# Purpose: a direct comparison with the published subject-independent DTU results
# (CA-TCN, arXiv 2603.26394, Table III: CA-TCN 58.3 / 63.4 / 70.3 / 76.7 / 85.8 / 92.5 %,
# AADNet 57.5 / 62.1 / 69.4 / 75.8 / 84.3 / 90.9 %, Ridge 56.0 / 60.1 / 65.6 / 70.2 /
# 78.1 / 83.6 % at 1 / 2 / 5 / 10 / 25 / 50 s), with no protocol caveats.
#
# Protocol = AADNet's public code (DTUDataset.createSICrossValidation), which CA-TCN
# describes: leave one subject out; the test subject's 60 two-talker trials split by
# KFold(8, shuffle, random_state = subject index); training on the other 17 subjects
# minus every trial whose attended audio file is attended in a test trial; validation =
# a random 20% of those trials (train_test_split, random_state = subject index);
# 18 x 8 = 144 models per seed. Implemented in cv_splits.loso_attended_track_out and
# checked against a literal copy of AADNet's loop (tests/test_loso_attended_track.py).
# Test decision windows of 1, 2, 5, 10, 25, 50 s with 50% overlap (AADNet's
# setWindowSize); accuracy per fold, mean over the 8 folds per subject, mean over subjects.
#
# Model: the round-9 StimulusGAT recipe (corr pooling, speech kernels (33, 33), DTU
# electrode graph, length-normalised band power, class-balanced loss and selection), with
# NO Euclidean alignment (CA-TCN uses none). Our preprocessing is kept (EEG 1-31 Hz,
# power-law sub-band envelope 1-8 Hz, 64 Hz, per-window z-scoring). Training windows 5 s,
# hop 2.5 s. One deviation to fit the GPU budget: training stops when class-balanced
# validation accuracy has not improved for 5 epochs (max 25; CA-TCN also stops early, on
# validation loss with patience 5). Windows longer than 5 s are scored by summing the
# logits of their non-overlapping 5 s sub-windows; 1 s and 2 s windows are scored
# directly. Seed 42, PARTs A (S1-S9) and B (S10-S18).
#
# Pre-registered analysis (offline, written before the runs):
# - Report our per-subject mean accuracy at each CA-TCN length, the mean over subjects,
#   and a 95% bootstrap CI over subjects (10,000 resamples). Our result is "above" /
#   "below" CA-TCN at a length if its mean lies outside our CI, otherwise "level". No
#   significance test against CA-TCN is possible (per-subject numbers are not published).
# - Controls at 5 s: envelope shift (2.5 s roll) and trial swap must be at chance (subject
#   Wilcoxon vs 0.5, p >= 0.05; no subject >= 0.6).
# - This is one training seed (CA-TCN reports one run); more seeds can be added later.
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
# ## 2. Part and configuration

# %%
PART = "A"
PARTS = {"A": [f"S{i}" for i in range(1, 10)], "B": [f"S{i}" for i in range(10, 19)]}
SEEDS = [42]
ARM = "loso_ea"
EA_SHRINKAGE = 1e-3

import time
from pathlib import Path

import numpy as np
import scipy.io
import torch

from aad_xai.config import PreprocessConfig
from aad_xai.data.alignment import euclidean_align
from aad_xai.data.cv_splits import assert_loso_attended_track_integrity, loso_attended_track_out
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.data.negatives import cross_trial_permutation
from aad_xai.models.sgat import DTU_MONTAGE_PATH, StimulusGAT

DTU_ROOT = next((p for p in ["/kaggle/input/aad-xai-artifacts/datasets/DTU",
                             "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU"]
                 if os.path.isdir(p)), None)
assert DTU_ROOT is not None, "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_S = 5.0
TRAIN_HOP_S = 2.5
SHORT_WINDOWS_S = (1.0, 2.0)
EPOCHS = 25
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
SHIFT_S = 2.5
N_TRIAL_FOLDS = 8
VAL_FRACTION = 0.2
MAX_FOLDS = None                      # smoke tests only
OUT_DIR = Path(f"/kaggle/working/results_loso_catcn_ea_{PART}")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

pp = PreprocessConfig(bandpass_hz=(1.0, 31.0), envelope_bandpass_hz=(1.0, 8.0))
SFREQ = pp.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)
print(f"Part {PART}: {PARTS[PART]}  seeds {SEEDS}  device {DEVICE}")

# %% [markdown]
# ## 3. Load subjects; attended audio file of every two-talker trial

# %%
_t0 = time.time()
all_trials = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp,
                             envelope_method="powerlaw_subbands").trials())
print(f"Trials: {len(all_trials)} in {time.time() - _t0:.0f}s")

ATTENDED = {}
for sid in sorted({t.subject_id for t in all_trials}, key=lambda s: int(s[1:])):
    exp = scipy.io.loadmat(str(Path(DTU_ROOT) / "eeg_new" / f"{sid}.mat"), variable_names=["expinfo"],
                           squeeze_me=True)["expinfo"]
    for t in (t for t in all_trials if t.subject_id == sid and t.audio_a is not None):
        i = int(t.trial_id.split("_T")[1])
        male, female = str(exp["wavfile_male"][i]).strip(), str(exp["wavfile_female"][i]).strip()
        assert t.group_id == f"{sid}_{male}" and female, (t.trial_id, male, female)
        assert t.label == int(exp["attend_mf"][i]) - 1
        ATTENDED[t.trial_id] = male if t.label == 0 else female
counts = {s: sum(1 for k in ATTENDED if k.startswith(f"{s}_")) for s in {t.subject_id for t in all_trials}}
print(f"two-talker trials per subject: {sorted(set(counts.values()))}; distinct attended files: "
      f"{len(set(ATTENDED.values()))}")
assert set(counts.values()) == {60}, counts

# %% [markdown]
# ## 4. Helpers: windows, logits, training with early stopping

# %%
import copy
import json

import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.utils.seed import seed_everything


def _meta(ds):
    w = ds.window_indices
    return {"subject": np.array([x.subject_id for x in w]), "trial": np.array([x.trial_id for x in w]),
            "start": np.array([x.start for x in w]), "label": np.array([x.label for x in w])}


def _stack(ds):
    """X (N, C, T), E (N, 2, T) float32."""
    xs, es = zip(*[(x, e) for x, e, _ in (ds[i] for i in range(len(ds)))])
    return torch.stack(xs), torch.stack(es)


def _logits(model, X, E):
    """(N, 2) logits, eval mode."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(model(X[i:i + 256].to(DEVICE), E[i:i + 256].to(DEVICE)).float().cpu())
    return torch.cat(out).numpy()


def _balanced(logits, labels):
    pred = logits.argmax(1)
    return float(np.mean([np.mean(pred[labels == c] == c) for c in np.unique(labels)]))


def _windows(trials, window_s, hop_s):
    return WindowedEEGAudioDataset(trials, window_s=window_s, overlap_s=window_s - hop_s, standardize_eeg=True)


def train_sgat(ds_tr, X_va, E_va, y_va, n_channels, seed):
    """Round-9 recipe without alignment; best epoch by class-balanced val accuracy; stop
    after PATIENCE epochs without improvement (max EPOCHS)."""
    seed_everything(seed)
    model = StimulusGAT(n_channels=n_channels, sfreq=SFREQ, film_pooling="corr", speech_kernel_sizes=(33, 33),
                        freq_length_normalize=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_tr = np.array([w.label for w in ds_tr.window_indices])
    class_w = torch.tensor([len(y_tr) / (2.0 * max((y_tr == c).sum(), 1)) for c in (0, 1)],
                           dtype=torch.float32, device=DEVICE)          # (2,) inverse class frequency
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    best, history = {"val": -1.0}, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, eb, yb in dl_tr:
            xb, eb = xb.to(DEVICE), eb.to(DEVICE)                       # (B, C, T), (B, 2, T)
            opt.zero_grad(set_to_none=True)
            loss_fn(model.score_candidates(xb, eb), torch.as_tensor(yb, dtype=torch.long, device=DEVICE)).backward()
            opt.step()
        v = _balanced(_logits(model, X_va, E_va), y_va)
        history.append(v)
        if v > best["val"]:
            best = {"val": v, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= PATIENCE:
            break
    model.load_state_dict(best["state"])
    return model, best["epoch"], best["val"], history, class_w.tolist()

# %% [markdown]
# ## 5. 144 folds (this part's test subjects), per-window outputs saved

# %%
folds = [f for f in loso_attended_track_out(all_trials, ATTENDED, n_trial_folds=N_TRIAL_FOLDS,
                                            val_fraction=VAL_FRACTION)
         if f.meta["test_subjects"][0] in PARTS[PART]]
for f in folds:
    assert_loso_attended_track_integrity(all_trials, f, ATTENDED)
folds = folds[:MAX_FOLDS] if MAX_FOLDS else folds
print(f"{len(folds)} folds in part {PART} (integrity checked); excluded per fold: "
      f"{min(f.meta['n_excluded'] for f in folds)}-{max(f.meta['n_excluded'] for f in folds)} trials")

fold_results = []
t_all = time.time()
for fi, fold in enumerate(folds):
    t0 = time.time()
    tr, va, te = ([all_trials[i] for i in idx] for idx in (fold.train_idx, fold.val_idx, fold.test_idx))
    # EA: references per subject, computed separately inside train, val and the test fold.
    tr, va, te = (euclidean_align(x, shrinkage=EA_SHRINKAGE) for x in (tr, va, te))
    ds_tr = _windows(tr, WINDOW_S, TRAIN_HOP_S)
    ds_va = _windows(va, WINDOW_S, WINDOW_S)
    X_va, E_va = _stack(ds_va)
    y_va = _meta(ds_va)["label"]
    grids = {"test5": _windows(te, WINDOW_S, WINDOW_S / 2)}
    grids.update({f"test{int(w)}": _windows(te, w, w / 2) for w in SHORT_WINDOWS_S})
    stacked = {g: _stack(ds) for g, ds in grids.items()}
    arrays = {f"{g}_{k}": v for g, ds in grids.items() for k, v in _meta(ds).items()}
    arrays["trial_ids"] = np.array([t.trial_id for t in te])
    arrays["trial_lens"] = np.array([t.eeg.shape[1] for t in te])
    X5, E5 = stacked["test5"]
    perm = cross_trial_permutation(arrays["test5_trial"])
    rec = {"fold_id": fold.fold_id, "meta": fold.meta, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "n_train_windows": len(ds_tr), "models": {}}
    for seed in SEEDS:
        name = f"{ARM}_s{seed}"
        model, ep, val, hist, class_w = train_sgat(ds_tr, X_va, E_va, y_va, X_va.shape[1], seed)
        torch.save(model.state_dict(), CKPT_DIR / f"sgat_loso_{name}_{fold.fold_id}.pt")
        for g, (X, E) in stacked.items():
            arrays[f"{name}_{g}_logits"] = _logits(model, X, E)
        arrays[f"{name}_test5_shifted_logits"] = _logits(model, X5, torch.roll(E5, SHIFT_SAMPLES, dims=-1))
        arrays[f"{name}_test5_swapped_logits"] = _logits(model, X5, E5[perm])
        acc = lambda key: float(np.mean(arrays[key].argmax(1) == arrays["test5_label"]))
        rec["models"][name] = {"best_epoch": ep, "epochs_run": len(hist), "val_balanced": val, "val_history": hist,
                               "class_weights": class_w, "test5_acc": acc(f"{name}_test5_logits"),
                               "shift_acc": acc(f"{name}_test5_shifted_logits"),
                               "swap_acc": acc(f"{name}_test5_swapped_logits")}
    np.savez_compressed(OUT_DIR / f"{fold.fold_id}.npz", **arrays)
    fold_results.append(rec)
    m = rec["models"][f"{ARM}_s{SEEDS[0]}"]
    print(f"[{fi + 1:>2}/{len(folds)}] {fold.fold_id}: train {len(tr)} trials ({len(ds_tr)} windows), "
          f"excluded {fold.meta['n_excluded']}; val {m['val_balanced']:.3f} (ep {m['best_epoch']}/{m['epochs_run']}) "
          f"test5 {m['test5_acc']:.3f} shift {m['shift_acc']:.3f} swap {m['swap_acc']:.3f} "
          f"({time.time() - t0:.0f}s, total {(time.time() - t_all) / 60:.0f} min)", flush=True)
    with open(OUT_DIR / f"loso_catcn_ea_{PART}.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "part": PART, "config": {
            "protocol": "AADNet/CA-TCN SI: LOSO x KFold(8) trial folds, attended-file exclusion, 20% val",
            "alignment": f"euclidean per subject per split (train, val, test fold), shrinkage {EA_SHRINKAGE}", "eeg_band_hz": [1.0, 31.0], "envelope": "powerlaw_subbands 1-8 Hz",
            "sfreq": SFREQ, "window_s": WINDOW_S, "train_hop_s": TRAIN_HOP_S, "test_overlap": 0.5,
            "short_windows_s": list(SHORT_WINDOWS_S), "epochs_max": EPOCHS, "patience": PATIENCE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "seeds": SEEDS, "film_pooling": "corr",
            "speech_kernel_sizes": [33, 33], "freq_length_normalize": True, "class_balanced": True,
            "montage": DTU_MONTAGE_PATH.name, "shift_s": SHIFT_S}, "folds": fold_results}, f, indent=2)

# %% [markdown]
# ## 6. Summary of this part at 5 s (all lengths analysed offline)

# %%
from scipy.stats import wilcoxon

for seed in SEEDS:
    name = f"{ARM}_s{seed}"
    per = {}
    for r in fold_results:
        s = r["meta"]["test_subjects"][0]
        per.setdefault(s, []).append([r["models"][name][k] for k in ("test5_acc", "shift_acc", "swap_acc")])
    subj = sorted(per, key=lambda s: int(s[1:]))
    A = np.array([np.mean(per[s], axis=0) for s in subj])                # (n_subj, 3)
    for j, k in enumerate(("test 5 s", "shift", "swap")):
        p = wilcoxon(A[:, j] - 0.5).pvalue if len(subj) > 1 and np.any(A[:, j] != 0.5) else float("nan")
        print(f"{name} {k}: mean {A[:, j].mean():.4f} (subjects {len(subj)}, vs 0.5 p={p:.3g})")
    print("per subject 5 s:", {s: round(float(a), 3) for s, a in zip(subj, A[:, 0])})
