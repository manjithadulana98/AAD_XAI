# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DTU-trained StimulusGAT on KUL without any KUL training (Kaggle, GPU, inference only)
#
# Exploratory (section 8 of docs/preregistration_kul_external_test.md): KUL has already been
# used for the external test, so nothing here is confirmatory. Rules are fixed in
# scripts/analyse_kul_zero_shot.py before this notebook runs.
#
# Question: the KUL external test trained new models on KUL with the frozen DTU recipe. Do the
# DTU models themselves decode KUL? DTU candidates are a male and a female talker; KUL
# candidates are two male readers in the left and right ear, in another language (Flemish).
# StimulusGAT scores each candidate with the same network, so the candidate definition does
# not enter the model; whether its speech tracking carries over to new voices is the question.
#
# - Models: the 90 round-9 DTU checkpoints (18 story-disjoint folds x seeds 42, 1, 2, 3, 4),
#   outputs of story-disjoint-v14-round9-fixa / -fixb. None has seen KUL data.
# - KUL inputs: exactly the test windows of the KUL external test. For each of its 16 folds,
#   the test split (2 subjects, 4 trials each) is aligned on its own (Euclidean alignment per
#   subject, shrinkage 1e-3), EEG 1-31 Hz, summed power-law envelope of the dry stimuli,
#   non-overlapping 5 s windows, per-window channel z-score. Every KUL trial is scored once.
# - Controls: envelopes rolled by 2.5 s; trial swap as in the KUL external test.
# - Saved per KUL fold: window metadata and the logits of all 90 models (test, shifted,
#   swapped). Analysis offline (scripts/analyse_kul_zero_shot.py).
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/kul-dataset` and the outputs of
#   `story-disjoint-v14-round9-fixa` and `-fixb`
# - GPU required (about 2.7 million window forward passes)

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
# ## 2. Confirm the modules are live

# %%
import torch
from aad_xai.config import PreprocessConfig
from aad_xai.models.sgat import DTU_MONTAGE_PATH, StimulusGAT
from aad_xai.data.alignment import euclidean_align
from aad_xai.data.cv_splits import (KUL_STORY_FOLDS, assert_story_disjoint_fold_integrity, kul_story_part,
                                    subject_group_story_out)
from aad_xai.data.kul_dataset import KULeuvenDataset
from aad_xai.data.negatives import partner_trial_permutation, same_stratum_other_content_partners
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
assert "freq_length_normalize" in StimulusGAT.__init__.__code__.co_varnames, "repo predates round 9"
assert "envelope_source" in KULeuvenDataset.__init__.__code__.co_varnames, "repo predates the KUL loader update"
assert KUL_STORY_FOLDS == ((1, 2), (3, 4))
print("StimulusGAT (round 9), KUL loader and KUL folds confirmed live.")

# %% [markdown]
# ## 3. KUL data: integrity, design checks, load (as in the KUL external test)

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


# Electrode graph: KUL channel labels must equal the montage order the DTU models were built on.
_tr = scipy.io.loadmat(str(KUL_ROOT / "S1.mat"))["trials"]
_t0 = _unwrap(_tr[0, 0] if _tr.ndim == 2 else _tr[0])
_labels = [str(np.asarray(lab).flat[0]) for lab in _unwrap(_unwrap(_t0["FileHeader"])["Channels"])["Label"].flat]
MONTAGE = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")["electrode_name"].tolist()
assert _labels == MONTAGE, f"KUL channel order differs from the montage: {_labels[:6]} vs {MONTAGE[:6]}"
print("Electrode graph order matches the KUL channel labels (64/64).")
del _tr, _t0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_S = 5.0
FOLD_SEED = 42
BATCH_SIZE = 256
SHIFT_S = 2.5
SPEECH_KERNEL_SIZES = (33, 33)
SGAT_ENVELOPE_METHOD = "powerlaw_subbands"
ENVELOPE_BAND_HZ = (1.0, 8.0)
SGAT_EEG_BAND_HZ = (1.0, 31.0)
EA_SHRINKAGE = 1e-3
N_SUBJECT_GROUPS, N_VAL_SUBJECTS = 8, 2
DTU_SEEDS = [42, 1, 2, 3, 4]
N_DTU_FOLDS = 18
OUT_DIR = Path("/kaggle/working/results_kul_zero_shot_sgat")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pp_sgat = PreprocessConfig(bandpass_hz=SGAT_EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
SFREQ = pp_sgat.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)

_t0 = time.time()
kul = KULeuvenDataset(KUL_ROOT, preprocess=pp_sgat, load_audio=True, envelope_method=SGAT_ENVELOPE_METHOD,
                      include_experiments=(1, 2), envelope_source="dry")
all_trials = list(kul.trials())
TRIAL_INFO = {t.trial_id: kul.trial_info[t.trial_id] for t in all_trials}
print(f"Trials: {len(all_trials)} in {time.time() - _t0:.0f}s (EEG {SGAT_EEG_BAND_HZ} Hz, {SGAT_ENVELOPE_METHOD}, dry)")

subjects = sorted({t.subject_id for t in all_trials}, key=lambda s: int(s[1:]))
assert len(subjects) == 16 and len(all_trials) == 128, (len(subjects), len(all_trials))
for t in all_trials:
    info = TRIAL_INFO[t.trial_id]
    assert t.eeg.shape[0] == 64 and t.sfreq == SFREQ, (t.trial_id, t.eeg.shape, t.sfreq)
    assert t.audio_a is not None and t.audio_a.shape == t.audio_b.shape == (t.eeg.shape[1],), t.trial_id
    assert info["left_envelope_wav"].endswith("_dry.wav") and info["right_envelope_wav"].endswith("_dry.wav")
    assert t.label == int(info["attended_ear"] == "R")
print(f"Design checks passed; labels (right ear) {Counter(t.label for t in all_trials)}")

# %% [markdown]
# ## 4. The 90 DTU round-9 models

# %%
CKPT = {p.name: p for p in Path("/kaggle/input").rglob("sgat_v14_fix_s*_sgso_*.pt")}
assert len(CKPT) == N_DTU_FOLDS * len(DTU_SEEDS), (
    f"{len(CKPT)} checkpoints: attach the outputs of story-disjoint-v14-round9-fixa and -fixb")
DTU_FOLDS = sorted({name.split("_", 4)[-1][:-3] for name in CKPT})      # e.g. sgso_g0_k0
assert len(DTU_FOLDS) == N_DTU_FOLDS and all(f.startswith("sgso_g") for f in DTU_FOLDS), DTU_FOLDS


def load_model(seed, fold_id):
    """Round-9 StimulusGAT, strict state-dict load (the architecture must match exactly)."""
    m = StimulusGAT(n_channels=64, sfreq=SFREQ, film_pooling="corr", speech_kernel_sizes=SPEECH_KERNEL_SIZES,
                    freq_length_normalize=True).to(DEVICE)
    m.load_state_dict(torch.load(CKPT[f"sgat_v14_fix_s{seed}_{fold_id}.pt"], map_location=DEVICE))
    return m.eval()


MODELS = {(seed, fid): load_model(seed, fid) for fid in DTU_FOLDS for seed in DTU_SEEDS}
print(f"{len(MODELS)} DTU models loaded on {DEVICE}; DTU folds: {DTU_FOLDS}")

# %% [markdown]
# ## 5. Helpers (window stacking and swap control as in the KUL external test)

# %%


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
    """(N, 2) logits, model in eval mode; X, E already on DEVICE."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            out.append(model(X[i:i + BATCH_SIZE], E[i:i + BATCH_SIZE]).float().cpu())
    return torch.cat(out).numpy()


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


# %% [markdown]
# ## 6. Score every KUL test window with every DTU model

# %%
folds = list(subject_group_story_out(all_trials, content_group=kul_story_part, story_folds=KUL_STORY_FOLDS,
                                     n_subject_groups=N_SUBJECT_GROUPS, n_val_subjects=N_VAL_SUBJECTS,
                                     seed=FOLD_SEED))
for fold in folds:
    assert_story_disjoint_fold_integrity(all_trials, fold, content_group=kul_story_part)
_tested = Counter(i for f in folds for i in f.test_idx)
assert len(folds) == 16 and set(_tested) == set(range(len(all_trials))) and set(_tested.values()) == {1}
print(f"{len(folds)} KUL folds; only their test splits are used (every trial scored once)")


def save_json(fold_results):
    with open(OUT_DIR / "kul_zero_shot_sgat.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {
                       "status": "exploratory (preregistration section 8)",
                       "models": "round-9 DTU StimulusGAT, outputs of story-disjoint-v14-round9-fixa/-fixb",
                       "dtu_folds": DTU_FOLDS, "dtu_seeds": DTU_SEEDS,
                       "data": {"zenodo": "4004271 v2.0", "kaggle_dataset": "dulanamanjitha/kul-dataset",
                                "stimuli_fingerprint": fingerprint, "experiments": [1, 2],
                                "envelope_source": "dry"},
                       "windows": "test splits of the 16 KUL external-test folds, non-overlapping 5 s",
                       "alignment": f"euclidean per subject per KUL test split (shrinkage {EA_SHRINKAGE})",
                       "envelope_method": SGAT_ENVELOPE_METHOD, "envelope_band_hz": list(ENVELOPE_BAND_HZ),
                       "eeg_band_hz": list(SGAT_EEG_BAND_HZ), "sfreq": SFREQ, "window_s": WINDOW_S,
                       "shift_s": SHIFT_S,
                       "swap": "same subject, same experiment, other held-out part, same window position"},
                   "trial_info": TRIAL_INFO, "folds": fold_results}, f, indent=2)


fold_results = []
for fi, fold in enumerate(folds):
    t0 = time.time()
    te = euclidean_align([all_trials[i] for i in fold.test_idx], shrinkage=EA_SHRINKAGE)
    ds_te = WindowedEEGAudioDataset(te, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=True)
    meta_te = _meta(ds_te)
    X_te, E_te = _stack(ds_te)                                   # (N, 64, 320), (N, 2, 320)
    perm = swap_permutation(meta_te)
    X_te, E_te = X_te.to(DEVICE), E_te.to(DEVICE)
    inputs = {"test": E_te, "test_shifted": torch.roll(E_te, SHIFT_SAMPLES, dims=-1),
              "test_swapped": E_te[torch.as_tensor(perm, device=DEVICE)]}
    arrays = {f"test_{k}": v for k, v in meta_te.items()}
    arrays["test_swap_perm"] = perm
    acc = {}
    for (seed, fid), model in MODELS.items():
        for kind, E in inputs.items():
            lg = _logits(model, X_te, E)                         # (N, 2)
            arrays[f"dtu_s{seed}_{fid}_{kind}_logits"] = lg
    for kind in inputs:
        mean = np.mean([arrays[f"dtu_s{s}_{f}_{kind}_logits"] for s, f in MODELS], axis=0)   # (N, 2)
        acc[kind] = float(np.mean(mean.argmax(1) == meta_te["label"]))
    np.savez_compressed(OUT_DIR / f"{fold.fold_id}.npz", **arrays)
    fold_results.append({"fold_id": fold.fold_id, "meta": fold.meta, "n_test": int(len(meta_te["label"])),
                         "grand_ensemble_window_acc": acc})
    print(f"[{fi + 1:>2}/{len(folds)}] {fold.fold_id} test={fold.meta['test_subjects']}: "
          f"test {acc['test']:.3f}  shift {acc['test_shifted']:.3f}  swap {acc['test_swapped']:.3f}  "
          f"({len(meta_te['label'])} windows, {time.time() - t0:.0f}s)", flush=True)
    save_json(fold_results)

# %% [markdown]
# ## 7. Summary (the analysis with its decision rules runs offline)

# %%
from scipy.stats import wilcoxon

counts = {k: {} for k in ("test", "test_shifted", "test_swapped")}
for fold in folds:
    z = np.load(OUT_DIR / f"{fold.fold_id}.npz")
    for kind, c in counts.items():
        mean = np.mean([z[f"dtu_s{s}_{f}_{kind}_logits"] for s, f in MODELS], axis=0)
        for s, y, p in zip(z["test_subject"], z["test_label"], mean.argmax(1)):
            a = c.setdefault(str(s), [0, 0]); a[0] += int(p == y); a[1] += 1
for kind, c in counts.items():
    acc = np.array([c[s][0] / c[s][1] for s in subjects])
    print(f"grand ensemble (90 DTU models), 5 s, {kind}: {acc.mean():.4f} +/- {acc.std():.4f}  "
          f"above 0.5 in {(acc > 0.5).sum()}/16  p(two-sided vs 0.5)={wilcoxon(acc - 0.5).pvalue:.3g}")
