# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Euclidean alignment from a calibration recording instead of the test session (Kaggle, GPU)
#
# Inference only: no model is trained. Euclidean alignment (EA) has so far been computed for a
# test subject from all of that subject's scored EEG, so the reported accuracies are
# transductive. Here the test subject's reference comes only from EEG that is never scored:
# - st{10,30,60}: the first 10 / 30 / 60 s of the subject's single-talker DTU trials, taken in
#   recording order. These trials are never part of any test set. 60 s is the primary arm (a
#   one-minute calibration recording).
# - stall: all of the subject's single-talker EEG.
# - os60 (story-disjoint only): the first 60 s of the subject's two-talker trials that are not
#   in the fold's test split (other stories, never scored in that fold; their labels are not used).
# - full: the transductive reference used so far (reproduction gate, see below).
# - none: no alignment of the test EEG (lower bound; the models were trained on aligned EEG).
# Training and validation alignment are unchanged: they belong to the training side.
#
# Models (no retraining):
# - Part SD, story-disjoint protocol: the 90 round-9 checkpoints (18 folds x seeds 42, 1, 2, 3, 4;
#   outputs of story-disjoint-v14-round9-fixa / -fixb). Non-overlapping 5 s test windows;
#   shift (2.5 s roll) and trial-swap controls for the full and st60 arms.
# - Part CT, AADNet / CA-TCN protocol: the 144 EA-arm checkpoints (seed 42; outputs of
#   loso-ca-tcn-protocol-ea-a / -b). Test grids as in that run (5 s with hop 2.5 s, 1 s and
#   2 s with 50% overlap); shift and swap controls at 5 s for the full and st60 arms.
#
# Gate (checked here, before anything else is reported): the full arm must reproduce the
# saved test logits of the original runs (same decision in >= 99.9% of windows, per part).
# Each subject must have >= 60 s of single-talker EEG, otherwise the run stops.
#
# Pre-registered analysis (scripts/analyse_calibration_ea.py, written and tested on synthetic
# outputs before this run; the rules are not to be changed after it):
# - C1 (SD, primary): 5-seed ensemble, st60 minus full, at 5/10/20/40 s (summed 5 s logit
#   differences), per-subject Wilcoxon, Holm over 4. "No loss" iff the mean difference is
#   >= -0.5 points at every length and no length shows a Holm-significant drop; else "loss".
# - C2 (SD): st60 ensemble vs TRF + EA with the same st60 calibration
#   (notebooks/kaggle_run_dtu_trf_calibration_ea.py), the H2 outcome rule of the KUL test.
# - Controls (SD, st60 ensemble): shift and swap, subject Wilcoxon vs 0.5 p >= 0.05 at every
#   length and no fold >= 0.6 at 5 s.
# - Claim "calibration EA keeps the story-disjoint result" iff C1 "no loss", C2 "better" and
#   the controls are clean.
# - L1 (CT): st60 vs the published CA-TCN numbers with the bootstrap rule of the CA-TCN-protocol
#   runs ("above" / "level" / "below" at 1/2/5/10/25/50 s). Claim "parity with CA-TCN without
#   transductive alignment" iff the st60 verdict is "level" or "above" at every length where the
#   full-EA run was "level" or "above" (2-50 s), and the st60 controls are clean.
# - L2 (CT): st60 minus the no-EA run (same folds, seed 42), per-subject Wilcoxon, Holm over 6;
#   calibration EA "helps" iff positive with Holm p < 0.05 at some length.
# - Descriptive: every arm (dose response 10 / 30 / 60 s / all, os60, none) on both protocols.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts` and the outputs of
#   `story-disjoint-v14-round9-fixa`, `-fixb`, `loso-ca-tcn-protocol-ea-a` and `-ea-b`
# - GPU

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
from aad_xai.models.sgat import StimulusGAT
from aad_xai.data.alignment import calibration_eeg, euclidean_align, euclidean_align_to
from aad_xai.data.cv_splits import (assert_loso_attended_track_integrity, assert_story_disjoint_fold_integrity,
                                    loso_attended_track_out, subject_group_story_out)
from aad_xai.data.negatives import cross_trial_permutation
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
assert "freq_length_normalize" in StimulusGAT.__init__.__code__.co_varnames, "repo predates round 9"
print("StimulusGAT (round 9) and calibration alignment confirmed live.")

# %% [markdown]
# ## 3. Load DTU (as in rounds 9 and the CA-TCN-protocol EA run)

# %%
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.io

from aad_xai.data.dtu_dataset import DTUDataset

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'."
INPUT = Path("/kaggle/input")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_S = 5.0
SHORT_WINDOWS_S = (1.0, 2.0)
FOLD_SEED = 42
BATCH_SIZE = 256
SHIFT_S = 2.5
EA_SHRINKAGE = 1e-3
SD_SEEDS = [42, 1, 2, 3, 4]
SD_PART_OF_SEED = {42: "fixA", 1: "fixA", 2: "fixA", 3: "fixB", 4: "fixB"}
N_TRIAL_FOLDS, VAL_FRACTION = 8, 0.2
CAL_ARMS = {"st10": ("single", 10.0), "st30": ("single", 30.0), "st60": ("single", 60.0),
            "stall": ("single", None), "os60": ("other_story", 60.0)}
PRIMARY = "st60"
SD_ARMS = ["full", "none", "st10", "st30", "st60", "stall", "os60"]
CT_ARMS = ["full", "none", "st10", "st30", "st60", "stall"]
CONTROL_ARMS = ("full", PRIMARY)
GATE_AGREEMENT = 0.999
MAX_SD_FOLDS = MAX_CT_FOLDS = None                     # smoke tests only
OUT_DIR = Path("/kaggle/working/results_calibration_ea")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pp = PreprocessConfig(bandpass_hz=(1.0, 31.0), envelope_bandpass_hz=(1.0, 8.0))
SFREQ = pp.sfreq_out
SHIFT_SAMPLES = int(SHIFT_S * SFREQ)

_t0 = time.time()
all_trials = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp,
                             envelope_method="powerlaw_subbands").trials())
SUBJECTS = sorted({t.subject_id for t in all_trials}, key=lambda s: int(s[1:]))
print(f"Trials: {len(all_trials)} in {time.time() - _t0:.0f}s; {len(SUBJECTS)} subjects; device {DEVICE}")

# %% [markdown]
# ## 4. Single-talker trials: the calibration recordings

# %%
SINGLE = {s: [t for t in all_trials if t.subject_id == s and t.audio_a is None] for s in SUBJECTS}
SINGLE_INFO = {}
for s in SUBJECTS:
    idx = [int(t.trial_id.split("_T")[1]) for t in SINGLE[s]]
    assert idx == sorted(idx), "single-talker trials out of recording order"
    n_two = sum(1 for t in all_trials if t.subject_id == s and t.audio_a is not None)
    SINGLE_INFO[s] = {"trial_index": idx, "seconds": [round(t.eeg.shape[1] / SFREQ, 2) for t in SINGLE[s]],
                      "total_s": round(sum(t.eeg.shape[1] for t in SINGLE[s]) / SFREQ, 2), "n_two_talker": n_two}
    print(f"{s}: {len(idx)} single-talker trials at recording positions {idx} "
          f"({SINGLE_INFO[s]['total_s']:.0f} s), {n_two} two-talker trials")
assert {v["n_two_talker"] for v in SINGLE_INFO.values()} == {60}, "expected 60 two-talker trials per subject"
short = [s for s in SUBJECTS if SINGLE_INFO[s]["total_s"] < 60.0]
assert not short, f"less than 60 s of single-talker EEG for {short}: the primary arm cannot run"
CAL_SINGLE = {arm: calibration_eeg([t for s in SUBJECTS for t in SINGLE[s]], seconds=sec)
              for arm, (pool, sec) in CAL_ARMS.items() if pool == "single"}
print("single-talker calibration samples per subject: "
      + ", ".join(f"{a} {min(v[s].shape[1] for s in v)}-{max(v[s].shape[1] for s in v)}" for a, v in CAL_SINGLE.items()))


def aligned_test(arm, te, excluded_ids=()):
    """The test trials of one fold aligned for one arm. ``excluded_ids``: trial ids that may not
    enter an other-story calibration pool (the fold's own test trials)."""
    if arm == "full":
        return euclidean_align(te, shrinkage=EA_SHRINKAGE)
    if arm == "none":
        return list(te)
    pool, sec = CAL_ARMS[arm]
    if pool == "single":
        cal = CAL_SINGLE[arm]
    else:
        subj, excluded = {t.subject_id for t in te}, set(excluded_ids)
        other = [t for t in all_trials if t.subject_id in subj and t.audio_a is not None
                 and t.trial_id not in excluded]
        cal = calibration_eeg(other, seconds=sec)
    return euclidean_align_to(te, cal, shrinkage=EA_SHRINKAGE)

# %% [markdown]
# ## 5. Helpers

# %%


def _meta(ds):
    w = ds.window_indices
    return {"subject": np.array([x.subject_id for x in w]), "trial": np.array([x.trial_id for x in w]),
            "start": np.array([x.start for x in w]), "label": np.array([x.label for x in w])}


def _stack(ds):
    """All windows of a dataset: X (N, C, T), E (N, 2, T) float32 tensors."""
    xs, es = zip(*[(x, e) for x, e, _ in (ds[i] for i in range(len(ds)))])
    return torch.stack(xs), torch.stack(es)


def _windows(trials, window_s, hop_s):
    return WindowedEEGAudioDataset(trials, window_s=window_s, overlap_s=window_s - hop_s, standardize_eeg=True)


def _logits(model, X, E):
    """(N, 2) logits; X, E on DEVICE."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            out.append(model(X[i:i + BATCH_SIZE], E[i:i + BATCH_SIZE]).float().cpu())
    return torch.cat(out).numpy()


def load_model(path):
    """Round-9 StimulusGAT, strict state-dict load."""
    m = StimulusGAT(n_channels=64, sfreq=SFREQ, film_pooling="corr", speech_kernel_sizes=(33, 33),
                    freq_length_normalize=True).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.eval()


def _find(pattern):
    return {p.name: p for p in INPUT.rglob(pattern)}


GATE = {}


def gate_update(part, new, saved):
    g = GATE.setdefault(part, {"windows": 0, "agree": 0, "max_abs_diff": 0.0})
    g["windows"] += len(new)
    g["agree"] += int((new.argmax(1) == saved.argmax(1)).sum())
    g["max_abs_diff"] = max(g["max_abs_diff"], float(np.abs(new - saved).max()))


def save_json(records):
    with open(OUT_DIR / "calibration_ea.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {
                       "status": "exploratory; rules in scripts/analyse_calibration_ea.py",
                       "cal_arms": {a: {"pool": p, "seconds": s} for a, (p, s) in CAL_ARMS.items()},
                       "primary": PRIMARY, "sd_arms": SD_ARMS, "ct_arms": CT_ARMS,
                       "control_arms": list(CONTROL_ARMS), "sd_seeds": SD_SEEDS, "ct_seed": 42,
                       "ea_shrinkage": EA_SHRINKAGE, "sfreq": SFREQ, "window_s": WINDOW_S,
                       "short_windows_s": list(SHORT_WINDOWS_S), "shift_s": SHIFT_S,
                       "gate_agreement": GATE_AGREEMENT},
                   "single_talker": SINGLE_INFO, "gate": GATE, "folds": records}, f, indent=2)

# %% [markdown]
# ## 6. Part SD: the 90 round-9 models on the story-disjoint test splits

# %%
SD_CKPT = _find("sgat_v14_fix_s*_sgso_*.pt")
assert len(SD_CKPT) == 90, f"{len(SD_CKPT)} round-9 checkpoints: attach story-disjoint-v14-round9-fixa / -fixb"
SD_NPZ = {part: {p.stem: p for p in INPUT.rglob(f"results_story_disjoint_v14{part}/sgso_*.npz")}
          for part in ("fixA", "fixB")}
assert all(len(v) == 18 for v in SD_NPZ.values()), {k: len(v) for k, v in SD_NPZ.items()}

sd_folds = list(subject_group_story_out(all_trials, seed=FOLD_SEED))
for fold in sd_folds:
    assert_story_disjoint_fold_integrity(all_trials, fold)
    assert all(all_trials[i].audio_a is not None for i in fold.test_idx), "single-talker trial in a test split"
sd_folds = sd_folds[:MAX_SD_FOLDS] if MAX_SD_FOLDS else sd_folds
print(f"{len(sd_folds)} story-disjoint folds (integrity checked)")

records = {"SD": [], "CT": []}
for fi, fold in enumerate(sd_folds):
    t0 = time.time()
    te = [all_trials[i] for i in fold.test_idx]
    te_ids = [t.trial_id for t in te]
    models = {s: load_model(SD_CKPT[f"sgat_v14_fix_s{s}_{fold.fold_id}.pt"]) for s in SD_SEEDS}
    arrays, acc = {}, {}
    for arm in SD_ARMS:
        ds = _windows(aligned_test(arm, te, te_ids), WINDOW_S, WINDOW_S)
        meta = _meta(ds)
        if arm == "full":
            arrays.update({f"test_{k}": v for k, v in meta.items()})
            perm = cross_trial_permutation(meta["trial"])
            arrays["test_swap_perm"] = perm
        else:
            assert all(np.array_equal(meta[k], arrays[f"test_{k}"]) for k in meta), "windows differ between arms"
        X, E = (a.to(DEVICE) for a in _stack(ds))                    # (N, 64, 320), (N, 2, 320)
        inputs = {"test": E}
        if arm in CONTROL_ARMS:
            inputs.update({"test_shifted": torch.roll(E, SHIFT_SAMPLES, dims=-1),
                           "test_swapped": E[torch.as_tensor(perm, device=DEVICE)]})
        for s, model in models.items():
            for kind, Ek in inputs.items():
                arrays[f"{arm}_s{s}_{kind}_logits"] = _logits(model, X, Ek)
        ens = np.mean([arrays[f"{arm}_s{s}_test_logits"] for s in SD_SEEDS], axis=0)
        acc[arm] = float(np.mean(ens.argmax(1) == meta["label"]))
    for s in SD_SEEDS:                                               # reproduction gate
        with np.load(SD_NPZ[SD_PART_OF_SEED[s]][fold.fold_id]) as z:
            assert all(np.array_equal(z[f"test_{k}"], arrays[f"test_{k}"]) for k in ("subject", "trial", "start", "label"))
            for kind in ("test", "test_shifted", "test_swapped"):
                gate_update("SD", arrays[f"full_s{s}_{kind}_logits"], z[f"fix_s{s}_{kind}_logits"])
    np.savez_compressed(OUT_DIR / f"sd_{fold.fold_id}.npz", **arrays)
    records["SD"].append({"fold_id": fold.fold_id, "meta": fold.meta, "n_test": int(len(arrays["test_label"])),
                          "ensemble_window_acc": acc})
    save_json(records)
    print(f"[SD {fi + 1:>2}/{len(sd_folds)}] {fold.fold_id}: " + "  ".join(f"{a} {v:.3f}" for a, v in acc.items())
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    del models

g = GATE["SD"]
g["agreement"] = g["agree"] / g["windows"]
save_json(records)
print(f"SD gate: full arm reproduces the round-9 decisions in {g['agreement']:.5f} of windows "
      f"(max |logit diff| {g['max_abs_diff']:.2e})")
assert g["agreement"] >= GATE_AGREEMENT, "SD reproduction gate failed"

# %% [markdown]
# ## 7. Part CT: the 144 EA-arm models under the AADNet / CA-TCN protocol

# %%
ATTENDED = {}
for sid in SUBJECTS:
    exp = scipy.io.loadmat(str(Path(DTU_ROOT) / "eeg_new" / f"{sid}.mat"), variable_names=["expinfo"],
                           squeeze_me=True)["expinfo"]
    for t in (t for t in all_trials if t.subject_id == sid and t.audio_a is not None):
        i = int(t.trial_id.split("_T")[1])
        male, female = str(exp["wavfile_male"][i]).strip(), str(exp["wavfile_female"][i]).strip()
        assert t.group_id == f"{sid}_{male}" and female, (t.trial_id, male, female)
        assert t.label == int(exp["attend_mf"][i]) - 1
        ATTENDED[t.trial_id] = male if t.label == 0 else female

CT_CKPT = _find("sgat_loso_loso_ea_s42_*.pt")
CT_NPZ = {p.stem: p for p in INPUT.rglob("results_loso_catcn_ea_*/*.npz")}
ct_folds = list(loso_attended_track_out(all_trials, ATTENDED, n_trial_folds=N_TRIAL_FOLDS, val_fraction=VAL_FRACTION))
for f in ct_folds:
    assert_loso_attended_track_integrity(all_trials, f, ATTENDED)
assert len(ct_folds) == 144 and len(CT_CKPT) == 144 and len(CT_NPZ) == 144, (len(ct_folds), len(CT_CKPT), len(CT_NPZ))
ct_folds = ct_folds[:MAX_CT_FOLDS] if MAX_CT_FOLDS else ct_folds
print(f"{len(ct_folds)} CA-TCN-protocol folds (integrity checked), 144 EA-arm checkpoints")

t_all = time.time()
for fi, fold in enumerate(ct_folds):
    te = [all_trials[i] for i in fold.test_idx]
    model = load_model(CT_CKPT[f"sgat_loso_loso_ea_s42_{fold.fold_id}.pt"])
    arrays, acc = {"trial_ids": np.array([t.trial_id for t in te]), "trial_lens": np.array([t.eeg.shape[1] for t in te])}, {}
    for arm in CT_ARMS:
        te_a = aligned_test(arm, te)
        grids = {"test5": _windows(te_a, WINDOW_S, WINDOW_S / 2)}
        grids.update({f"test{int(w)}": _windows(te_a, w, w / 2) for w in SHORT_WINDOWS_S})
        for g_name, ds in grids.items():
            meta = _meta(ds)
            if arm == "full":
                arrays.update({f"{g_name}_{k}": v for k, v in meta.items()})
            else:
                assert all(np.array_equal(meta[k], arrays[f"{g_name}_{k}"]) for k in meta), "windows differ"
            X, E = (a.to(DEVICE) for a in _stack(ds))
            arrays[f"{arm}_s42_{g_name}_logits"] = _logits(model, X, E)
            if g_name == "test5" and arm in CONTROL_ARMS:
                perm = cross_trial_permutation(meta["trial"])
                arrays[f"{arm}_s42_test5_shifted_logits"] = _logits(model, X, torch.roll(E, SHIFT_SAMPLES, dims=-1))
                arrays[f"{arm}_s42_test5_swapped_logits"] = _logits(model, X, E[torch.as_tensor(perm, device=DEVICE)])
        acc[arm] = float(np.mean(arrays[f"{arm}_s42_test5_logits"].argmax(1) == arrays["test5_label"]))
    with np.load(CT_NPZ[fold.fold_id]) as z:                         # reproduction gate
        for g_name in ("test5", "test1", "test2"):
            assert np.array_equal(z[f"{g_name}_label"], arrays[f"{g_name}_label"])
            assert np.array_equal(z[f"{g_name}_trial"], arrays[f"{g_name}_trial"])
            gate_update("CT", arrays[f"full_s42_{g_name}_logits"], z[f"loso_ea_s42_{g_name}_logits"])
        for kind in ("shifted", "swapped"):
            gate_update("CT", arrays[f"full_s42_test5_{kind}_logits"], z[f"loso_ea_s42_test5_{kind}_logits"])
        assert np.array_equal(z["trial_ids"], arrays["trial_ids"])
    np.savez_compressed(OUT_DIR / f"ct_{fold.fold_id}.npz", **arrays)
    records["CT"].append({"fold_id": fold.fold_id, "meta": fold.meta, "test5_acc": acc})
    if fi % 8 == 7 or fi == len(ct_folds) - 1:
        save_json(records)
        print(f"[CT {fi + 1:>3}/{len(ct_folds)}] {fold.fold_id}: " + "  ".join(f"{a} {v:.3f}" for a, v in acc.items())
              + f"  (total {(time.time() - t_all) / 60:.0f} min)", flush=True)

g = GATE["CT"]
g["agreement"] = g["agree"] / g["windows"]
save_json(records)
print(f"CT gate: full arm reproduces the saved EA-run decisions in {g['agreement']:.5f} of windows "
      f"(max |logit diff| {g['max_abs_diff']:.2e})")
assert g["agreement"] >= GATE_AGREEMENT, "CT reproduction gate failed"

# %% [markdown]
# ## 8. Summary at 5 s (decisions and tests run offline)

# %%
for part, recs, key in (("SD", records["SD"], "ensemble_window_acc"), ("CT", records["CT"], "test5_acc")):
    arms = SD_ARMS if part == "SD" else CT_ARMS
    print(f"\n{part}: mean over folds of the 5 s window accuracy")
    for arm in arms:
        a = np.array([r[key][arm] for r in recs])
        print(f"  {arm:6s} {a.mean():.4f}")
