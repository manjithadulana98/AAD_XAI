# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Is the frontal-polar reliance specific? Cluster-matched group deletion (Kaggle)
#
# The X1-after-eye-removal run (RESULTS_LOG 2026-11-26) found that deleting the 8 frontal-polar
# electrodes (Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8) costs the round-11 models 4.5 accuracy
# points more than deleting 8 random electrodes (round 9: 5.5), and that eye removal did not
# measurably reduce this. The random sets were scattered over the cap, while Fp/AF is one
# contiguous band, so deleting it also removes spatially redundant signal that scattered
# deletions leave. This notebook asks whether Fp/AF costs more than other contiguous sets of
# 8 electrodes.
#
# Models: the round-11 checkpoints (EOG-cleaned input) and the round-9 checkpoints (uncleaned),
# 5 seeds x 18 folds each, on exactly the windows, electrode space and deletion of that run
# (permute: each window's electrode signal swapped for another window's, generator seed 0).
#
# Electrode sets, fixed before the run (aad_xai.xai.electrode_sets, 2-D montage coordinates):
# - FP: the 8 Fp/AF electrodes.
# - SCATTER: the 20 random sets of 8 non-Fp/AF electrodes of the previous run (rng 13), kept to
#   check that this run reproduces its numbers.
# - CLUSTERS: compact clusters, each electrode plus its 7 nearest neighbours; 55 distinct.
#   Comparison family: the 40 that share no electrode with FP.
# - BANDS: the FP band rotated about Cz by 45, 90, ..., 315 degrees and assigned one-to-one to
#   the nearest electrodes, so each copy keeps the band shape and its position at the cap edge.
#   Comparison family: the 5 that share no electrode with FP (90-270 degrees: left temporal,
#   left parieto-occipital, occipital, right parieto-occipital, right temporal).
#
# Pre-registered rules (written before any cluster deletion was computed; the FP and SCATTER
# numbers of the previous run were known). Subject = unit, n = 18, each subject's 3 folds and
# 5 seeds pooled. Analysis: scripts/analyse_cluster_deletion.py, run once.
# - Gate: every model reproduces its saved test decisions (agreement >= 0.99), and the FP and
#   SCATTER accuracies reproduce the previous run (mean absolute difference <= 0.005).
# - Reliance of FP over a family = mean accuracy with a family set deleted minus accuracy with
#   FP deleted (the extra drop caused by FP), per subject, mean over folds and seeds.
# - C1 (primary, round 11): FP reliance over CLUSTERS > 0.
# - C2 (round 11): FP reliance over BANDS > 0.
#   One-sided Wilcoxon against 0, p < 0.05 each. Outcome: "specific" if C1 and C2 (both must
#   hold, so no multiplicity correction), "not specific" if neither, "mixed" otherwise.
# - C3 (secondary): C1 and C2 on the round-9 models, same outcome rule.
# - C4 (the caveat itself, round 11): mean accuracy with a SCATTER set deleted minus mean
#   accuracy with a CLUSTERS set deleted > 0, i.e. contiguous deletions cost more than scattered.
# - Descriptive: margin versions; per subject, the share of CLUSTERS sets that cost less than
#   FP; the cluster-deletion map (accuracy drop when the cluster centred on each electrode is
#   deleted, all 64 centres) with its top 10; the accuracy drop of each band angle.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts` and the outputs of
#   `story-disjoint-v16-round11-eoga`, `-eogb`, `story-disjoint-v14-round9-fixa` and `-fixb`
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
# ## 2. Configuration and inputs: trials, EOG derivations, checkpoints

# %%
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aad_xai.config import PreprocessConfig
from aad_xai.data.alignment import euclidean_reference, inverse_sqrtm
from aad_xai.data.cv_splits import subject_group_story_out
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.data.ocular import N_DTU_WITH_EXG, dtu_derivations, regress_eog_trials
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.models.sgat import DTU_MONTAGE_PATH, StimulusGAT
from aad_xai.xai import sgat_faithfulness as sf
from aad_xai.xai.electrode_sets import centre_clusters, disjoint_from, knn_clusters, rotated_sets

DTU_ROOT = next(p for p in ["/kaggle/input/aad-xai-artifacts/datasets/DTU",
                            "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU"]
                if os.path.isdir(p))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_S = 5.0
FOLD_SEED = 42
EA_SHRINKAGE = 1e-3
SEEDS = [42, 1, 2, 3, 4]
N_RANDOM = 20
_mont = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")
MONTAGE = _mont["electrode_name"].tolist()
XY = _mont[["x", "y"]].to_numpy()                                                  # (64, 2)
FP_NAMES = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]
FP_IDX = [MONTAGE.index(n) for n in FP_NAMES]
assert sorted(n for n in MONTAGE if n.startswith(("Fp", "AF"))) == sorted(FP_NAMES)
# The electrode sets (header): SCATTER as in the previous run, compact clusters, rotated bands.
_rest = [c for c in range(64) if c not in FP_IDX]
_rng = np.random.default_rng(13)
SCATTER = [_rng.choice(_rest, len(FP_IDX), replace=False).tolist() for _ in range(N_RANDOM)]
CENTRE_CLUSTERS = centre_clusters(XY, len(FP_IDX))                                 # 64 x 8
CLUSTERS = knn_clusters(XY, len(FP_IDX))                                           # 55 x 8
BANDS = rotated_sets(XY, FP_IDX, range(0, 360, 45), centre=XY[MONTAGE.index("Cz")])  # {angle: 8}
assert len(CLUSTERS) == 55 and len(disjoint_from(CLUSTERS, FP_IDX)) == 40
assert BANDS[0] == sorted(FP_IDX) and len(disjoint_from([BANDS[a] for a in BANDS if a], FP_IDX)) == 5
print("bands:", {a: [MONTAGE[i] for i in s] for a, s in BANDS.items() if a})
OUT_DIR = Path("/kaggle/working/results_cluster_deletion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT = {p.name: p for p in Path("/kaggle/input").rglob("sgat_v16_eog_s*_sgso_*.pt")}
CKPT9 = {p.name: p for p in Path("/kaggle/input").rglob("sgat_v14_fix_s*_sgso_*.pt")}
NPZ, NPZ9 = {}, {}
for p in Path("/kaggle/input").rglob("sgso_*.npz"):
    if "v16" in str(p):
        NPZ.setdefault(p.stem, []).append(p)
    elif "v14" in str(p):
        NPZ9.setdefault(p.stem, []).append(p)
print(f"round-11 checkpoints {len(CKPT)}, npz {sum(map(len, NPZ.values()))}; round-9 checkpoints {len(CKPT9)}, "
      f"npz {sum(map(len, NPZ9.values()))}; device {DEVICE}")
assert len(CKPT) == 90, "attach the outputs of story-disjoint-v16-round11-eoga and -eogb"
assert len(CKPT9) == 90, "attach the outputs of story-disjoint-v14-round9-fixa and -fixb"

pp = PreprocessConfig(bandpass_hz=(1.0, 31.0), envelope_bandpass_hz=(1.0, 8.0))
SFREQ = pp.sfreq_out
_t0 = time.time()
trials = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp, envelope_method="powerlaw_subbands").trials())
print(f"Trials: {len(trials)} in {time.time() - _t0:.0f}s")
folds = list(subject_group_story_out(trials, seed=FOLD_SEED))

# EOG block (as in round 11): the 3 bipolar EOG derivations of every trial, same band and rate.
pp_eog = PreprocessConfig(bandpass_hz=(1.0, 31.0), reref=None, envelope_bandpass_hz=(1.0, 8.0))
_by_id = {t.trial_id: t for t in trials}
_t0 = time.time()
EOG, _rel = {}, []
for t72 in DTUDataset(root=DTU_ROOT, load_audio=False, n_eeg_channels=N_DTU_WITH_EXG, preprocess=pp_eog).trials():
    d = dtu_derivations(t72.eeg)
    ref = _by_id[t72.trial_id].eeg                                            # (64, T)
    assert d["eeg"].shape == ref.shape, (t72.trial_id, d["eeg"].shape, ref.shape)
    _rel.append(float(np.linalg.norm(d["eeg"] - ref) / np.linalg.norm(ref)))
    EOG[t72.trial_id] = d["eog"].astype(ref.dtype)                            # (3, T)
assert set(EOG) == set(_by_id), "EOG load and EEG load hold different trials"
assert max(_rel) < 1e-4, "the 72-column load does not reproduce the scalp EEG"
print(f"EOG for {len(EOG)} trials in {time.time() - _t0:.0f}s (scalp max relative difference {max(_rel):.2e})")
del _by_id
# END EOG block


# %% [markdown]
# ## 3. Group deletion of FP, scattered sets, compact clusters and rotated bands

# %%
def load_model(path):
    m = StimulusGAT(n_channels=64, sfreq=SFREQ, film_pooling="corr", speech_kernel_sizes=(33, 33),
                    freq_length_normalize=True).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.eval()


def saved_logits(npz, key, fold_id):
    for p in npz[fold_id]:
        z = np.load(p)
        if key in z.files:
            return z["test_subject"], z[key]
    raise KeyError((fold_id, key))


def _stack(ds):
    xs, es, ys = zip(*[ds[i] for i in range(len(ds))])
    return (torch.stack(xs).to(DEVICE), torch.stack(es).to(DEVICE),
            torch.as_tensor(ys, dtype=torch.long, device=DEVICE))


def reproduction(scorer, x, e, subj, saved):
    with torch.no_grad():
        ours = torch.cat([scorer(x[i:i + 256], e[i:i + 256]) for i in range(0, len(x), 256)]).cpu().numpy()
    subj_arr, logits = saved
    ref = logits[subj_arr == subj]
    return {"agree": float(np.mean(ours.argmax(1) == ref.argmax(1))), "max_abs_diff": float(np.abs(ours - ref).max())}


def set_deletion(scorer, x, e, y):
    """[accuracy, margin] intact and with each fixed electrode set deleted (permute, seed 0)."""
    ev = lambda s: list(sf.evaluate(scorer, sf.delete_electrodes(x, s, "permute"), e, y))
    return {"intact": list(sf.evaluate(scorer, x, e, y)),
            "fp": ev(FP_IDX),
            "scatter": [ev(s) for s in SCATTER],
            "clusters": [ev(s) for s in CLUSTERS],
            "bands": {str(a): ev(s) for a, s in BANDS.items() if a != 0}}


def _mean_acc(dels, key):
    """Mean accuracy over seeds (and over the sets of a family) for the progress line."""
    vals = []
    for d in dels.values():
        v = d[key]
        if key in ("intact", "fp"):
            vals.append(v[0])
        else:
            vals.append(np.mean([q[0] for q in (v.values() if isinstance(v, dict) else v)]))
    return float(np.mean(vals))


results = []
t_all = time.time()
for fi, fold in enumerate(folds):
    t0 = time.time()
    test_trials = [trials[i] for i in fold.test_idx]
    saved = {s: saved_logits(NPZ, f"eog_s{s}_test_logits", fold.fold_id) for s in SEEDS}
    saved9 = {s: saved_logits(NPZ9, f"fix_s{s}_test_logits", fold.fold_id) for s in SEEDS}
    models = {s: load_model(CKPT[f"sgat_v16_eog_s{s}_{fold.fold_id}.pt"]) for s in SEEDS}
    models9 = {s: load_model(CKPT9[f"sgat_v14_fix_s{s}_{fold.fold_id}.pt"]) for s in SEEDS}
    for subj in fold.meta["test_subjects"]:
        st_raw = [t for t in test_trials if t.subject_id == subj]
        st = regress_eog_trials(st_raw, EOG)                                              # EOG-cleaned
        W = inverse_sqrtm(euclidean_reference([t.eeg for t in st], EA_SHRINKAGE))        # (C, C)
        W9 = inverse_sqrtm(euclidean_reference([t.eeg for t in st_raw], EA_SHRINKAGE))   # (C, C)
        ds = WindowedEEGAudioDataset(st, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=False)
        ds9 = WindowedEEGAudioDataset(st_raw, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=False)
        x, e, y = _stack(ds)                                                              # (B, C, T) cleaned
        x9, e9, y9 = _stack(ds9)                                                          # (B, C, T) uncleaned
        assert torch.equal(e, e9) and torch.equal(y, y9)
        mk = lambda m, Wm=W: sf.ElectrodeSpaceScorer(m, torch.as_tensor(Wm, dtype=torch.float32)).to(DEVICE)
        rec = {"fold_id": fold.fold_id, "subject": subj, "n_windows": len(y)}
        rec["reproduction"] = {s: reproduction(mk(models[s]), x, e, subj, saved[s]) for s in SEEDS}
        rec["reproduction_r9"] = {s: reproduction(mk(models9[s], W9), x9, e9, subj, saved9[s]) for s in SEEDS}
        assert min(r["agree"] for r in rec["reproduction"].values()) >= 0.99, rec["reproduction"]
        assert min(r["agree"] for r in rec["reproduction_r9"].values()) >= 0.99, rec["reproduction_r9"]
        rec["deletion"] = {s: set_deletion(mk(models[s]), x, e, y) for s in SEEDS}
        rec["deletion_r9"] = {s: set_deletion(mk(models9[s], W9), x9, e9, y9) for s in SEEDS}
        results.append(rec)
        d = rec["deletion"]
        print(f"[{fi+1:>2}/{len(folds)}] {fold.fold_id} {subj}: n={len(y)} repro "
              f"{min(r['agree'] for r in rec['reproduction'].values()):.3f}/"
              f"{min(r['agree'] for r in rec['reproduction_r9'].values()):.3f}; r11 accuracy intact "
              f"{_mean_acc(d, 'intact'):.3f}, deleted: FP {_mean_acc(d, 'fp'):.3f} scatter "
              f"{_mean_acc(d, 'scatter'):.3f} clusters {_mean_acc(d, 'clusters'):.3f} bands "
              f"{_mean_acc(d, 'bands'):.3f}", flush=True)
    print(f"   fold done in {time.time() - t0:.0f}s (total {(time.time() - t_all) / 60:.0f} min)", flush=True)
    with open(OUT_DIR / "cluster_deletion.json", "w") as f:
        json.dump({"commit": GIT_COMMIT,
                   "config": {"seeds": SEEDS, "montage": MONTAGE, "fp_names": FP_NAMES, "deletion": "permute, seed 0",
                              "scatter": SCATTER, "clusters": CLUSTERS, "centre_clusters": CENTRE_CLUSTERS,
                              "bands": {str(a): s for a, s in BANDS.items()},
                              "electrode_space": "round 11: EOG-cleaned scalp EEG; round 9: uncleaned"},
                   "results": results}, f)

print(f"Done: {len(results)} subject-folds in {(time.time() - t_all) / 60:.0f} min")
