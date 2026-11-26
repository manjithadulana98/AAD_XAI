# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interpretability X1 after eye removal: the round-11 models (Kaggle)
#
# Round 9's X1 (RESULTS_LOG 2026-11-08) found that occlusion, the most faithful channel method,
# ranks the eight frontal-polar electrodes (Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8) first. Those
# electrodes carry most of the eye activity (S1: 80-94% of their variance is explained by the
# EOG). Round 11 regresses the three bipolar EOG derivations out of the scalp EEG before EA and
# loses no accuracy. This notebook asks whether the frontal-polar ranking goes away with it.
#
# Models: the round-11 checkpoints (story-disjoint-v16-round11-eoga / -eogb, 5 seeds x 18
# folds), explained exactly as X1 explained round 9: same harness (notebooks/
# kaggle_run_dtu_xai_round9.py), same windows, same cross-fitted halves and same settings.
# Electrode space is the EOG-cleaned scalp EEG, i.e. the input the round-11 model receives
# before EA (EOG regression fitted per subject on that subject's test trials, as in round 11).
# The round-9 checkpoints (story-disjoint-v14-round9-fixa / -fixb) are also loaded, only for
# the group-deletion test E3, on the same windows of the uncleaned EEG.
#
# Pre-registered rules (written before any round-11 explanation was computed; the round-9 X1
# values were known). Subject = unit, n = 18, each subject's 3 folds pooled. Analysis:
# scripts/analyse_xai_round11.py, run once.
# - Gate: the electrode-space scorer reproduces the saved test decisions of every round-11
#   and round-9 model on every subject-fold (agreement >= 0.99), as in X1.
# - E1 (primary): frontal-polar occlusion index = per subject, the mean z-score of the 8 Fp/AF
#   electrodes in the subject's occlusion map (mean over folds and the 5 seeds of the
#   all-window scores, z-scored over the 64 electrodes, the X1 group-map recipe). E1 holds if
#   the round-11 index is lower than the round-9 index (X1 output), one-sided paired Wilcoxon,
#   p < 0.05.
# - E2: the round-11 index is above 0 (one-sided Wilcoxon against 0, p < 0.05).
#   Outcome: "removed" if E1 and not E2; "reduced" if E1 and E2; "not reduced" if not E1.
# - E3 (attribution-free): frontal-polar reliance = accuracy drop when the 8 Fp/AF electrodes
#   are deleted (another window's signal swapped in, X1's primary deletion) minus the mean drop
#   over 20 random sets of 8 other electrodes; per subject, mean over folds and 5 seeds. E3a:
#   round 11 lower than round 9 (one-sided paired Wilcoxon, p < 0.05). E3b: round 11 above 0
#   (one-sided Wilcoxon, p < 0.05). Margin version reported, not tested.
# - Secondary, descriptive: the X1 rules (sanity, deletion faithfulness, edges, H2, bands,
#   seed stability) on round 11 next to round 9; frontal-polar index for IG and attention;
#   top-10 electrodes per method; Spearman of each occlusion group map with the log-variance
#   group map of its own input (cleaned EEG for round 11, uncleaned for round 9).
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

DTU_ROOT = next(p for p in ["/kaggle/input/aad-xai-artifacts/datasets/DTU",
                            "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU"]
                if os.path.isdir(p))
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
MONTAGE = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")["electrode_name"].tolist()
FP_NAMES = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]
FP_IDX = [MONTAGE.index(n) for n in FP_NAMES]
assert sorted(n for n in MONTAGE if n.startswith(("Fp", "AF"))) == sorted(FP_NAMES)
OUT_DIR = Path("/kaggle/working/results_xai_round11")
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
# ## 3. Explanations, faithfulness tests and group deletion, per fold and test subject

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


def group_deletion(scorer, x, e, y):
    """E3: [accuracy, margin] intact, with the 8 Fp/AF electrodes deleted (permute), and with each
    of N_RANDOM random sets of 8 other electrodes deleted (same sets for every model and window)."""
    rest = [c for c in range(64) if c not in FP_IDX]
    rng = np.random.default_rng(13)
    sets = [rng.choice(rest, len(FP_IDX), replace=False).tolist() for _ in range(N_RANDOM)]
    return {"intact": list(sf.evaluate(scorer, x, e, y)),
            "fp": list(sf.evaluate(scorer, sf.delete_electrodes(x, FP_IDX, "permute"), e, y)),
            "random": [list(sf.evaluate(scorer, sf.delete_electrodes(x, s, "permute"), e, y)) for s in sets]}


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
        x, e, y = _stack(ds)                                                              # cleaned windows
        x9, e9, y9 = _stack(ds9)                                                          # uncleaned windows
        assert torch.equal(e, e9) and torch.equal(y, y9)
        trial_ids = np.array([w.trial_id for w in ds.window_indices])
        mk = lambda m, Wm=W: sf.ElectrodeSpaceScorer(m, torch.as_tensor(Wm, dtype=torch.float32)).to(DEVICE)
        rec = {"fold_id": fold.fold_id, "subject": subj, "n_windows": len(y),
               "locality": sf.alignment_locality(W),
               "variance": x.var(dim=-1, unbiased=False).mean(dim=0).tolist(),                # (C,) cleaned
               "variance_raw": x9.var(dim=-1, unbiased=False).mean(dim=0).tolist()}           # (C,) uncleaned

        rec["reproduction"] = {s: reproduction(mk(models[s]), x, e, subj, saved[s]) for s in SEEDS}
        rec["reproduction_r9"] = {s: reproduction(mk(models9[s], W9), x9, e9, subj, saved9[s]) for s in SEEDS}
        assert min(r["agree"] for r in rec["reproduction"].values()) >= 0.99, rec["reproduction"]
        assert min(r["agree"] for r in rec["reproduction_r9"].values()) >= 0.99, rec["reproduction_r9"]

        # E3: group deletion of the frontal-polar electrodes, both rounds, every seed.
        rec["group_deletion"] = {s: group_deletion(mk(models[s]), x, e, y) for s in SEEDS}
        rec["group_deletion_r9"] = {s: group_deletion(mk(models9[s], W9), x9, e9, y9) for s in SEEDS}

        # Stability across seeds: channel scores on all windows.
        rec["seed_scores"] = {}
        for s in SEEDS:
            scorer = mk(models[s])
            rec["seed_scores"][s] = {"occlusion": sf.occlusion_scores(scorer, x, e, y).tolist(),
                                     "ig": sf.ig_channel_scores(scorer, x, e, y, steps=IG_STEPS).tolist(),
                                     "attention": sf.attention_node_scores(scorer, x, e, y).tolist()}

        # Faithfulness on the seed-42 model, cross-fitted over two halves of the trials.
        scorer = mk(models[FAITH_SEED])
        edges = sf.directed_edges(models[FAITH_SEED])
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
        a_idx = torch.as_tensor(np.where(hA)[0], device=DEVICE)
        xa, ea, ya = x[a_idx], e[a_idx], y[a_idx]
        rec["sanity"] = {
            "occlusion": sf.cascade_sanity(models[FAITH_SEED], lambda m: sf.occlusion_scores(mk(m), xa, ea, ya)),
            "ig": sf.cascade_sanity(models[FAITH_SEED], lambda m: sf.ig_channel_scores(mk(m), xa, ea, ya, steps=IG_STEPS)),
            "attention": sf.cascade_sanity(models[FAITH_SEED], lambda m: sf.attention_node_scores(mk(m), xa, ea, ya)),
            "band": sf.cascade_sanity(models[FAITH_SEED], lambda m: sf.band_feature_attribution(mk(m), xa, ea, ya, steps=BAND_STEPS)),
        }
        results.append(rec)
        gd, gd9 = rec["group_deletion"][FAITH_SEED], rec["group_deletion_r9"][FAITH_SEED]
        print(f"[{fi+1:>2}/{len(folds)}] {fold.fold_id} {subj}: n={len(y)} locality={rec['locality']:.2f} "
              f"repro r11 {min(r['agree'] for r in rec['reproduction'].values()):.3f} "
              f"r9 {min(r['agree'] for r in rec['reproduction_r9'].values()):.3f}; acc intact / Fp-AF deleted: "
              f"r11 {gd['intact'][0]:.3f}/{gd['fp'][0]:.3f}  r9 {gd9['intact'][0]:.3f}/{gd9['fp'][0]:.3f}", flush=True)
    print(f"   fold done in {time.time() - t0:.0f}s (total {(time.time() - t_all) / 60:.0f} min)", flush=True)
    with open(OUT_DIR / "xai_round11.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {"seeds": SEEDS, "faith_seed": FAITH_SEED, "n_random": N_RANDOM,
                                                     "ig_steps": IG_STEPS, "band_steps": BAND_STEPS, "ch_ks": CH_KS,
                                                     "montage": MONTAGE, "fp_names": FP_NAMES,
                                                     "electrode_space": "EOG-cleaned scalp EEG (round 11)"},
                   "edges": [list(map(int, ed)) for ed in edges], "results": results}, f)

print(f"Done: {len(results)} subject-folds in {(time.time() - t_all) / 60:.0f} min")
