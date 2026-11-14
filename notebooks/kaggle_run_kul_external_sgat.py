# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # KUL external held-out test: StimulusGAT, round-9 configuration frozen (Kaggle, GPU)
#
# Pre-registration: `docs/preregistration_kul_external_test.md` (frozen before this run).
# Nothing below was tuned on KUL; the model, preprocessing and training are the DTU round-9
# baseline (notebook v14) without EOG regression (KUL has no EOG channels).
#
# - Data: KU Leuven AAD (Zenodo 4004271 v2.0, private Kaggle dataset `dulanamanjitha/kul-dataset`,
#   MD5-checked below), experiments 1 and 2 only (8 trials per subject, 16 subjects); envelopes
#   from the dry stimulus files for both conditions.
# - Split: subject_group_story_out, content group = story part, story folds {1, 2} / {3, 4},
#   8 groups of 2 test subjects, 2 validation subjects, seed 42: 16 folds.
# - Controls: envelopes rolled by 2.5 s; trial swap with the same subject's trial of the other
#   held-out part in the same experiment (never the same audio).
# - Seeds 42, 1, 2 (PART A) and 3, 4 (PART B). Analysis offline (scripts/analyse_kul_external.py).
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/kul-dataset`
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
PART = "A"
PARTS = {"A": {"seeds": [42, 1, 2]}, "B": {"seeds": [3, 4]}}
CFG = PARTS[PART]
print(f"Part {PART}: {CFG}")

import torch
from aad_xai.config import PreprocessConfig
from aad_xai.models.sgat import DTU_MONTAGE_PATH, StimulusGAT
from aad_xai.data.alignment import euclidean_align
from aad_xai.data.cv_splits import (KUL_STORY_FOLDS, assert_story_disjoint_fold_integrity, kul_story_part,
                                    subject_group_story_out)
from aad_xai.data.kul_dataset import KULeuvenDataset
from aad_xai.data.negatives import partner_trial_permutation, same_stratum_other_content_partners
assert "freq_length_normalize" in StimulusGAT.__init__.__code__.co_varnames, "repo predates round 9"
assert "envelope_source" in KULeuvenDataset.__init__.__code__.co_varnames, "repo predates the KUL loader update"
assert KUL_STORY_FOLDS == ((1, 2), (3, 4))
print("StimulusGAT (round 9), KUL loader, KUL folds and swap control confirmed live.")

# %% [markdown]
# ## 3. KUL data: integrity, design checks, load (EEG 1-31 Hz, summed power-law envelope)

# %%
import glob
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

STIMULI_FINGERPRINT = "853dd5093a3d88b2209fc72db741067f"   # kernel kul-aad-structure-check v2

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


# Electrode graph: KUL channel labels must equal the montage order the graph is built from.
_tr = scipy.io.loadmat(str(KUL_ROOT / "S1.mat"))["trials"]
_t0 = _unwrap(_tr[0, 0] if _tr.ndim == 2 else _tr[0])
_labels = [str(np.asarray(lab).flat[0]) for lab in _unwrap(_unwrap(_t0["FileHeader"])["Channels"])["Label"].flat]
MONTAGE = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")["electrode_name"].tolist()
assert _labels == MONTAGE, f"KUL channel order differs from the montage: {_labels[:6]} vs {MONTAGE[:6]}"
print("Electrode graph order matches the KUL channel labels (64/64).")
del _tr, _t0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_S = 5.0
TRAIN_HOP_S = 2.5
TRAIN_WINDOW_S = 5.0
TRAIN_OVERLAP_S = TRAIN_WINDOW_S - TRAIN_HOP_S
FOLD_SEED = 42
SEED = 42
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
SHIFT_S = 2.5
SPEECH_KERNEL_SIZES = (33, 33)
SGAT_ENVELOPE_METHOD = "powerlaw_subbands"
ENVELOPE_BAND_HZ = (1.0, 8.0)
SGAT_EEG_BAND_HZ = (1.0, 31.0)
EA_SHRINKAGE = 1e-3
N_SUBJECT_GROUPS, N_VAL_SUBJECTS = 8, 2
OUT_DIR = Path(f"/kaggle/working/results_kul_external_sgat_{PART}")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

pp_sgat = PreprocessConfig(bandpass_hz=SGAT_EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
SFREQ = pp_sgat.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)

_t0 = time.time()
kul = KULeuvenDataset(KUL_ROOT, preprocess=pp_sgat, load_audio=True, envelope_method=SGAT_ENVELOPE_METHOD,
                      include_experiments=(1, 2), envelope_source="dry")
all_trials = list(kul.trials())
TRIAL_INFO = {t.trial_id: kul.trial_info[t.trial_id] for t in all_trials}
print(f"Trials: {len(all_trials)} in {time.time() - _t0:.0f}s (EEG {SGAT_EEG_BAND_HZ} Hz, {SGAT_ENVELOPE_METHOD}, dry)")

# Design checks (pre-registration, section 2).
subjects = sorted({t.subject_id for t in all_trials}, key=lambda s: int(s[1:]))
assert len(subjects) == 16 and len(all_trials) == 128, (len(subjects), len(all_trials))
for t in all_trials:
    info = TRIAL_INFO[t.trial_id]
    assert t.eeg.shape[0] == 64 and t.sfreq == SFREQ, (t.trial_id, t.eeg.shape, t.sfreq)
    assert t.audio_a is not None and t.audio_a.shape == t.audio_b.shape == (t.eeg.shape[1],), t.trial_id
    assert info["left_envelope_wav"].endswith("_dry.wav") and info["right_envelope_wav"].endswith("_dry.wav")
    assert info["experiment"] in (1, 2) and kul_story_part(t) == info["story_part"] == info["part"]
    att = info["right_track"] if info["attended_ear"] == "R" else info["left_track"]
    assert att == info["attended_track"] == info["experiment"], t.trial_id
    assert t.label == int(info["attended_ear"] == "R")
pairs = Counter((i["subject"], i["story_part"], i["attended_track"]) for i in TRIAL_INFO.values())
assert set(pairs.values()) == {1} and len(pairs) == 16 * 4 * 2, "each part twice per subject, once per track"
mins = Counter()
for t in all_trials:
    mins[t.subject_id] += t.eeg.shape[1] / SFREQ / 60
print(f"Design checks passed: 16 subjects x 8 trials; EEG minutes per subject "
      f"{min(mins.values()):.1f}-{max(mins.values()):.1f}; labels (right ear) "
      f"{Counter(t.label for t in all_trials)}; conditions "
      f"{Counter(i['condition'] for i in TRIAL_INFO.values())}")

# %% [markdown]
# ## 4. Helpers: stacked windows, logits, controls, StimulusGAT training (v14, unchanged)

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


def swap_permutation(meta_te):
    """Content-disjoint trial swap: each test window gets the envelopes of the window at the same
    position in the same subject's trial of the other held-out part, same experiment."""
    tids = sorted(set(meta_te["trial"]))
    keys = {t: (TRIAL_INFO[t]["subject"], TRIAL_INFO[t]["experiment"], TRIAL_INFO[t]["story_part"]) for t in tids}
    partner = same_stratum_other_content_partners(keys)
    perm = partner_trial_permutation(meta_te["trial"], meta_te["start"], partner)
    assert all(TRIAL_INFO[meta_te["trial"][j]]["story_part"] != TRIAL_INFO[meta_te["trial"][i]]["story_part"]
               for i, j in enumerate(perm)), "swap pairs the same audio"
    return perm


def eval_suite(model, prefix, perm, X_va, E_va, meta_va, X_te, E_te, meta_te):
    """Val, test, shift-control and trial-swap-control logits and counts."""
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
    """AdamW, 25 epochs, correlation pooling, speech kernels (33, 33), BioSemi-64 electrode graph,
    length-normalised band power; class-balanced loss; best epoch by class-balanced
    pooled 5 s val accuracy."""
    seed_everything(seed)
    model = StimulusGAT(n_channels=n_channels, sfreq=SFREQ, film_pooling="corr",
                        speech_kernel_sizes=SPEECH_KERNEL_SIZES, freq_length_normalize=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_tr = np.array([w.label for w in ds_tr.window_indices])
    class_w = torch.tensor([len(y_tr) / (2.0 * max((y_tr == c).sum(), 1)) for c in (0, 1)],
                           dtype=torch.float32, device=DEVICE)   # (2,) inverse class frequency
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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
# ## 5. KUL story-disjoint CV, EA per split; save per-window outputs

# %%
folds = list(subject_group_story_out(all_trials, content_group=kul_story_part, story_folds=KUL_STORY_FOLDS,
                                     n_subject_groups=N_SUBJECT_GROUPS, n_val_subjects=N_VAL_SUBJECTS,
                                     seed=FOLD_SEED))
for fold in folds:
    assert_story_disjoint_fold_integrity(all_trials, fold, content_group=kul_story_part)
_tested = Counter(i for f in folds for i in f.test_idx)
assert len(folds) == 16 and set(_tested) == set(range(len(all_trials))) and set(_tested.values()) == {1}
print(f"{len(folds)} story-disjoint folds (integrity checked; every trial tested once)")


def aligned_splits(trials, fold):
    """EA with references computed separately inside train, val and test."""
    return [euclidean_align([trials[i] for i in idx], shrinkage=EA_SHRINKAGE)
            for idx in (fold.train_idx, fold.val_idx, fold.test_idx)]


def save_json(fold_results):
    with open(OUT_DIR / f"kul_external_sgat_{PART}.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "part": PART, "config": {
                       "preregistration": "docs/preregistration_kul_external_test.md",
                       "data": {"zenodo": "4004271 v2.0", "kaggle_dataset": "dulanamanjitha/kul-dataset",
                                "stimuli_fingerprint": fingerprint, "experiments": [1, 2],
                                "envelope_source": "dry"},
                       "protocol": f"subject_group_story_out, story folds {KUL_STORY_FOLDS}, "
                                   f"{N_SUBJECT_GROUPS} subject groups, {N_VAL_SUBJECTS} val subjects",
                       "alignment": f"euclidean per subject per split (shrinkage {EA_SHRINKAGE})",
                       "envelope_method": SGAT_ENVELOPE_METHOD, "envelope_band_hz": list(ENVELOPE_BAND_HZ),
                       "eeg_band_hz": list(SGAT_EEG_BAND_HZ), "sfreq": SFREQ, "window_s": WINDOW_S,
                       "train_window_s": TRAIN_WINDOW_S, "train_overlap_s": TRAIN_OVERLAP_S,
                       "fold_seed": FOLD_SEED, "seeds": CFG["seeds"], "epochs": EPOCHS, "lr": LR,
                       "weight_decay": WEIGHT_DECAY, "shift_s": SHIFT_S, "film_pooling": "corr",
                       "speech_kernel_sizes": list(SPEECH_KERNEL_SIZES), "freq_length_normalize": True,
                       "class_balanced": True, "montage": str(DTU_MONTAGE_PATH.name),
                       "short_windows_s": list(SHORT_WINDOWS_S),
                       "swap": "same subject, same experiment, other held-out part, same window position"},
                   "trial_info": TRIAL_INFO, "folds": fold_results}, f, indent=2)


fold_results = []
for fi, fold in enumerate(folds):
    t0 = time.time()
    rec = {"fold_id": fold.fold_id, "meta": fold.meta, "models": {}}
    tr, va, te = aligned_splits(all_trials, fold)
    ds_va, ds_te = (WindowedEEGAudioDataset(t, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=True)
                    for t in (va, te))
    meta_va, meta_te = _meta(ds_va), _meta(ds_te)
    (X_va, E_va), (X_te, E_te) = _stack(ds_va), _stack(ds_te)
    perm = swap_permutation(meta_te)
    arrays = {f"{split}_{k}": v for split, m in (("val", meta_va), ("test", meta_te)) for k, v in m.items()}
    arrays["test_swap_perm"] = perm
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
        torch.save(model.state_dict(), CKPT_DIR / f"sgat_kul_s{seed}_{fold.fold_id}.pt")
        a, counts = eval_suite(model, f"kul_s{seed}", perm, *suite)
        arrays.update(a)
        arrays.update(short_window_suite(model, f"kul_s{seed}", short))
        rec["models"][f"kul_s{seed}"] = {"best_epoch": ep, "val_balanced": val_bal, "class_weights": class_w,
                                         **counts}

    np.savez_compressed(OUT_DIR / f"{fold.fold_id}.npz", **arrays)
    fold_results.append(rec)

    txt = "  ".join(f"{m}: val={_pooled(r['val']):.3f} test={_pooled(r['test']):.3f} "
                    f"(shift {_pooled(r['test_shifted']):.3f}, swap {_pooled(r['test_swapped']):.3f}, "
                    f"ep {r['best_epoch']})" for m, r in rec["models"].items())
    print(f"[{fi+1:>2}/{len(folds)}] {fold.fold_id} test={fold.meta['test_subjects']}: {txt}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    save_json(fold_results)

# %% [markdown]
# ## 6. Summary of this part (the pre-registered analysis runs offline on both parts)

# %%
from scipy.stats import wilcoxon


def per_subject(get):
    acc = {}
    for r in fold_results:
        for s, (c, n) in get(r).items():
            a = acc.setdefault(s, [0, 0]); a[0] += c; a[1] += n
    subj = sorted(acc, key=lambda s: int(s[1:]))
    return subj, np.array([acc[s][0] / acc[s][1] for s in subj])


for m in fold_results[0]["models"]:
    val = np.mean([_pooled(r["models"][m]["val"]) for r in fold_results])
    _, test = per_subject(lambda r, m=m: r["models"][m]["test"])
    line = f"{m}: val {val:.4f}  test {test.mean():.4f} +/- {test.std():.4f}"
    for ctrl in ("test_shifted", "test_swapped"):
        _, c = per_subject(lambda r, m=m, k=ctrl: r["models"][m][k])
        flagged = [r["fold_id"] for r in fold_results if _pooled(r["models"][m][ctrl]) >= 0.6]
        line += f"  {ctrl[5:]} {c.mean():.4f} (p={wilcoxon(c - 0.5).pvalue:.3g}; folds >= 0.6: {flagged or 'none'})"
    print(line)
