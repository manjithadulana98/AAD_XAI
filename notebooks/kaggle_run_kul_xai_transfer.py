# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interpretability on KUL: do the X1 conclusions and explanation maps transfer? (Kaggle, GPU)
#
# Models: the 80 round-9 StimulusGAT checkpoints of the KUL external test (kernels
# kul-external-sgat-a / -b; 16 folds x 5 seeds), unchanged. The explanation code is the X1
# harness (notebooks/kaggle_run_dtu_xai_round9.py) with the same settings. Only the data
# loading (identical to notebooks/kaggle_run_kul_external_sgat.py) and two extra records differ:
# - the per-electrode variance of each subject's raw test windows (a trivial reference map);
# - the directed edge list of the electrode graph (for the in-degree reference map).
#
# Status: EXPLORATORY under section 8 of docs/preregistration_kul_external_test.md. The analysis
# below was written before any KUL explanation was computed.
#
# Replication (the X1 rules, unchanged; subject = unit, n = 16, each subject's 2 folds pooled).
# An X1 verdict "replicates" if the same rule gives the same verdict on KUL:
# - R1 sanity (|median Spearman with the fully randomised model| < 0.3), per channel method and bands;
# - R2 channel faithfulness (deletion by permutation, delete below and keep above random,
#   Holm over 3 methods);
# - R3 edge faithfulness (edge removal, Holm over 2 methods);
# - R4 H2: attention's deletion area above occlusion's and above IG's;
# - R5 band attribution vs band-stop drop (median Spearman >= 0.7);
# - R6 seed stability (median Spearman over seed pairs >= 0.7).
#
# Transfer (DTU = the X1 output; same BioSemi-64 order and the same electrode graph):
# - T1 bands, by band-stop margin drop (the band report X1 found defensible): per KUL subject,
#   Spearman over the 8 bands between its profile and the DTU group profile; median and one-sided
#   Wilcoxon > 0 over 16 subjects. Prediction from DTU: theta and alpha1 are the two largest drops.
# - T2 channels (occlusion, IG, attention; X1 group-map recipe: per subject the mean over folds and
#   seeds of the all-window scores, z-scored, then averaged over subjects): Spearman between the
#   DTU and KUL group maps, and the partial Spearman controlling for both datasets' electrode
#   log-variance group maps (raw windows, same recipe) and the electrode in-degree of the graph
#   (attention received is a column sum); 95% bootstrap CIs (2000 resamples of
#   subjects within each dataset). Ceiling: sqrt of the product of the two split-half
#   reliabilities (1000 random halves, Spearman-Brown corrected). A method's map "transfers" if
#   the CI of the partial correlation excludes 0.
# - T3 edges (attention, attention x gradient): DTU vs KUL group-map Spearman over the directed
#   edges with a bootstrap CI (descriptive).
# - Descriptive: top-10 electrodes per method on each dataset.
# Analysis: scripts/analyse_xai_transfer.py (DTU electrode variance from
# notebooks/kaggle_dtu_electrode_variance.py, CPU).
#
# PART A explains folds 0-7 (test subjects of subject groups 0-3), PART B folds 8-15.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/kul-dataset` and the outputs of `kul-external-sgat-a`
#   and `kul-external-sgat-b`
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
# ## 2. Part and configuration (X1 settings)

# %%
PART = "A"
PARTS = {"A": {"folds": list(range(0, 8))}, "B": {"folds": list(range(8, 16))}}
CFG = PARTS[PART]
print(f"Part {PART}: {CFG}")

import torch
from aad_xai.config import PreprocessConfig
from aad_xai.models.sgat import DTU_MONTAGE_PATH, StimulusGAT
from aad_xai.data.alignment import euclidean_reference, inverse_sqrtm
from aad_xai.data.cv_splits import (KUL_STORY_FOLDS, assert_story_disjoint_fold_integrity, kul_story_part,
                                    subject_group_story_out)
from aad_xai.data.kul_dataset import KULeuvenDataset
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.xai import sgat_faithfulness as sf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_S = 5.0
FOLD_SEED = 42
EA_SHRINKAGE = 1e-3
SEEDS = [42, 1, 2, 3, 4]
FAITH_SEED = 42
N_RANDOM = 20
IG_STEPS = 32
BAND_STEPS = 16
CH_KS = [0, 1, 2, 4, 8, 16, 32, 48, 64]
N_CKPT = 80
SPEECH_KERNEL_SIZES = (33, 33)
SGAT_ENVELOPE_METHOD = "powerlaw_subbands"
ENVELOPE_BAND_HZ = (1.0, 8.0)
SGAT_EEG_BAND_HZ = (1.0, 31.0)
N_SUBJECT_GROUPS, N_VAL_SUBJECTS = 8, 2

# %% [markdown]
# ## 3. KUL data (integrity checks and load identical to the external-test notebook), folds

# %%
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

OUT_DIR = Path(f"/kaggle/working/results_xai_kul_{PART}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
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


_tr = scipy.io.loadmat(str(KUL_ROOT / "S1.mat"))["trials"]
_t0 = _unwrap(_tr[0, 0] if _tr.ndim == 2 else _tr[0])
_labels = [str(np.asarray(lab).flat[0]) for lab in _unwrap(_unwrap(_t0["FileHeader"])["Channels"])["Label"].flat]
MONTAGE = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")["electrode_name"].tolist()
assert _labels == MONTAGE, f"KUL channel order differs from the montage: {_labels[:6]} vs {MONTAGE[:6]}"
print("Electrode graph order matches the KUL channel labels (64/64).")
del _tr, _t0

pp = PreprocessConfig(bandpass_hz=SGAT_EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
SFREQ = pp.sfreq_out
_t0 = time.time()
kul = KULeuvenDataset(KUL_ROOT, preprocess=pp, load_audio=True, envelope_method=SGAT_ENVELOPE_METHOD,
                      include_experiments=(1, 2), envelope_source="dry")
trials = list(kul.trials())
assert len(trials) == 128 and len({t.subject_id for t in trials}) == 16, len(trials)
print(f"Trials: {len(trials)} in {time.time() - _t0:.0f}s")

folds = list(subject_group_story_out(trials, content_group=kul_story_part, story_folds=KUL_STORY_FOLDS,
                                     n_subject_groups=N_SUBJECT_GROUPS, n_val_subjects=N_VAL_SUBJECTS,
                                     seed=FOLD_SEED))
for fold in folds:
    assert_story_disjoint_fold_integrity(trials, fold, content_group=kul_story_part)
assert len(folds) == 16
run_folds = [folds[i] for i in CFG["folds"]]
print(f"{len(folds)} folds; this part explains {[f.fold_id for f in run_folds]}")

CKPT = {p.name: p for p in Path("/kaggle/input").rglob("sgat_kul_s*_sgso_*.pt")}
NPZ = {}
for p in Path("/kaggle/input").rglob("sgso_*.npz"):
    if "results_kul_external_sgat" in str(p):
        NPZ.setdefault(p.stem, []).append(p)
print(f"checkpoints: {len(CKPT)}  KUL npz files: {sum(len(v) for v in NPZ.values())}  device: {DEVICE}")
assert len(CKPT) == N_CKPT, "attach the outputs of kul-external-sgat-a and kul-external-sgat-b"


# %% [markdown]
# ## 4. Explanations and faithfulness tests, per fold and test subject (X1 code)

# %%
def load_model(seed, fold_id):
    m = StimulusGAT(n_channels=64, sfreq=SFREQ, film_pooling="corr", speech_kernel_sizes=SPEECH_KERNEL_SIZES,
                    freq_length_normalize=True).to(DEVICE)
    m.load_state_dict(torch.load(CKPT[f"sgat_kul_s{seed}_{fold_id}.pt"], map_location=DEVICE))
    return m.eval()


def saved_logits(fold_id, seed):
    for p in NPZ[fold_id]:
        z = np.load(p)
        if f"kul_s{seed}_test_logits" in z.files:
            return z["test_subject"], z[f"kul_s{seed}_test_logits"]
    raise KeyError((fold_id, seed))


def _stack(ds):
    xs, es, ys = zip(*[ds[i] for i in range(len(ds))])
    return (torch.stack(xs).to(DEVICE), torch.stack(es).to(DEVICE),
            torch.as_tensor(ys, dtype=torch.long, device=DEVICE))


def channel_block(scorer, xr, er, yr, xe, ee, ye):
    """Rank electrodes on (xr, er, yr); evaluate deletion / keep curves on (xe, ee, ye)."""
    scores = {"occlusion": sf.occlusion_scores(scorer, xr, er, yr, mode="zero"),
              "ig": sf.ig_channel_scores(scorer, xr, er, yr, steps=IG_STEPS),
              "attention": sf.attention_node_scores(scorer, xr, er, yr)}
    rankings = {k: sf.ranking_from_scores(v) for k, v in scores.items()}
    rand = sf.random_rankings(64, N_RANDOM, seed=7)
    curves = {}
    for mode in ("permute", "zero"):
        curves[mode] = {}
        for name, r in list(rankings.items()) + [(f"random{i}", r) for i, r in enumerate(rand)]:
            curves[mode][name] = {
                "delete": sf.deletion_curve(scorer, xe, ee, ye, r, CH_KS, mode=mode),
                "keep": sf.deletion_curve(scorer, xe, ee, ye, r, CH_KS, keep=True, mode=mode)}
    return {k: v.tolist() for k, v in scores.items()}, curves


def edge_block(scorer, edges, xr, er, yr, xe, ee, ye):
    E = len(edges)
    ks = sorted({0, 10, 25, 50, 100, 200, 300, E // 2, E})
    scores = {"attention": sf.edge_scores(scorer, xr, er, yr, edges, method="attention"),
              "attn_x_grad": sf.edge_scores(scorer, xr, er, yr, edges, method="attn_x_grad")}
    rankings = {k: sf.ranking_from_scores(v) for k, v in scores.items()}
    curves = {name: sf.edge_deletion_curve(scorer, xe, ee, ye, edges, r, ks)
              for name, r in list(rankings.items())
              + [(f"random{i}", r) for i, r in enumerate(sf.random_rankings(E, N_RANDOM, seed=11))]}
    return {k: v.tolist() for k, v in scores.items()}, curves


def band_block(scorer, xr, er, yr, xe, ee, ye):
    attr = sf.band_feature_attribution(scorer, xr, er, yr, steps=BAND_STEPS)
    drops = sf.band_stop_drops(scorer, xe, ee, ye, SFREQ)
    return {"attribution": attr.tolist(), "drops": [[list(b), a, m] for b, a, m in drops]}


EDGES = None
results = []
t_all = time.time()
for fi, fold in enumerate(run_folds):
    t0 = time.time()
    test_trials = [trials[i] for i in fold.test_idx]
    sub_saved = {s: saved_logits(fold.fold_id, s) for s in SEEDS}
    models = {s: load_model(s, fold.fold_id) for s in SEEDS}
    edges = sf.directed_edges(models[FAITH_SEED])
    EDGES = edges if EDGES is None else EDGES
    assert edges == EDGES, "electrode graph differs between folds"
    for subj in fold.meta["test_subjects"]:
        st = [t for t in test_trials if t.subject_id == subj]
        W = inverse_sqrtm(euclidean_reference([t.eeg for t in st], EA_SHRINKAGE))        # (C, C)
        ds = WindowedEEGAudioDataset(st, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=False)
        x, e, y = _stack(ds)                                                              # raw windows
        trial_ids = np.array([w.trial_id for w in ds.window_indices])
        rec = {"fold_id": fold.fold_id, "subject": subj, "n_windows": len(y),
               "locality": sf.alignment_locality(W),
               "variance": x.var(dim=-1, unbiased=False).mean(dim=0).cpu().tolist()}    # (C,) raw window variance

        # Reproduction: the electrode-space scorer must reproduce the saved KUL test decisions.
        rec["reproduction"] = {}
        for s in SEEDS:
            scorer = sf.ElectrodeSpaceScorer(models[s], torch.as_tensor(W, dtype=torch.float32)).to(DEVICE)
            with torch.no_grad():
                ours = torch.cat([scorer(x[i:i + 256], e[i:i + 256]) for i in range(0, len(x), 256)]).cpu().numpy()
            subj_arr, logits = sub_saved[s]
            ref = logits[subj_arr == subj]
            rec["reproduction"][s] = {"agree": float(np.mean(ours.argmax(1) == ref.argmax(1))),
                                      "max_abs_diff": float(np.abs(ours - ref).max())}
        assert min(r["agree"] for r in rec["reproduction"].values()) >= 0.99, rec["reproduction"]

        # Stability across seeds: channel scores on all windows.
        rec["seed_scores"] = {}
        for s in SEEDS:
            scorer = sf.ElectrodeSpaceScorer(models[s], torch.as_tensor(W, dtype=torch.float32)).to(DEVICE)
            rec["seed_scores"][s] = {"occlusion": sf.occlusion_scores(scorer, x, e, y).tolist(),
                                     "ig": sf.ig_channel_scores(scorer, x, e, y, steps=IG_STEPS).tolist(),
                                     "attention": sf.attention_node_scores(scorer, x, e, y).tolist()}

        # Faithfulness on the seed-42 model, cross-fitted over two halves of the trials.
        scorer = sf.ElectrodeSpaceScorer(models[FAITH_SEED], torch.as_tensor(W, dtype=torch.float32)).to(DEVICE)
        hA, hB = sf.cross_fit_halves(trial_ids)
        rec["halves"] = []
        for rank_mask, eval_mask in ((hA, hB), (hB, hA)):
            r_idx = torch.as_tensor(np.where(rank_mask)[0], device=DEVICE)
            e_idx = torch.as_tensor(np.where(eval_mask)[0], device=DEVICE)
            args = (x[r_idx], e[r_idx], y[r_idx], x[e_idx], e[e_idx], y[e_idx])
            ch_scores, ch_curves = channel_block(scorer, *args)
            ed_scores, ed_curves = edge_block(scorer, edges, *args)
            rec["halves"].append({"n_rank": len(r_idx), "n_eval": len(e_idx),
                                  "channel_scores": ch_scores, "channel_curves": ch_curves,
                                  "edge_scores": ed_scores, "edge_curves": ed_curves,
                                  "band": band_block(scorer, *args)})

        # Sanity (level 1): cascading randomisation, seed-42 model, first half.
        ia = torch.as_tensor(np.where(hA)[0], device=DEVICE)
        xa, ea, ya = x[ia], e[ia], y[ia]
        Wt = torch.as_tensor(W, dtype=torch.float32)
        mk = lambda m: sf.ElectrodeSpaceScorer(m, Wt).to(DEVICE)
        rec["sanity"] = {
            "occlusion": sf.cascade_sanity(models[FAITH_SEED], lambda m: sf.occlusion_scores(mk(m), xa, ea, ya)),
            "ig": sf.cascade_sanity(models[FAITH_SEED], lambda m: sf.ig_channel_scores(mk(m), xa, ea, ya, steps=IG_STEPS)),
            "attention": sf.cascade_sanity(models[FAITH_SEED], lambda m: sf.attention_node_scores(mk(m), xa, ea, ya)),
            "band": sf.cascade_sanity(models[FAITH_SEED], lambda m: sf.band_feature_attribution(mk(m), xa, ea, ya, steps=BAND_STEPS)),
        }
        results.append(rec)
        acc = rec["halves"][0]["channel_curves"]["permute"]["occlusion"]["delete"][0][1]
        print(f"[{fi+1:>2}/{len(run_folds)}] {fold.fold_id} {subj}: n={len(y)} locality={rec['locality']:.2f} "
              f"repro={min(r['agree'] for r in rec['reproduction'].values()):.3f} acc(half B)={acc:.3f}", flush=True)
    print(f"   fold done in {time.time() - t0:.0f}s (total {(time.time() - t_all) / 60:.0f} min)", flush=True)
    with open(OUT_DIR / f"xai_kul_{PART}.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "part": PART,
                   "config": {"seeds": SEEDS, "faith_seed": FAITH_SEED, "n_random": N_RANDOM,
                              "ig_steps": IG_STEPS, "band_steps": BAND_STEPS, "ch_ks": CH_KS,
                              "folds": [f.fold_id for f in run_folds], "montage": MONTAGE,
                              "stimuli_fingerprint": fingerprint},
                   "edges": [list(e_) for e_ in EDGES], "results": results}, f)

print(f"Done: {len(results)} subject-folds in {(time.time() - t_all) / 60:.0f} min")
