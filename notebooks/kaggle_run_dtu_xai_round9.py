# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interpretability X1: faithfulness of StimulusGAT explanations (Kaggle)
#
# Models: the round-9 corrected baseline (story-disjoint folds, DTU electrode graph,
# length-normalised band power, class-balanced training), checkpoints from the kernels
# story-disjoint-v14-round9-fixa / -fixb. Explanations of the seed-42 model of every
# fold, for each test subject, computed on one half of the subject's test trials and
# evaluated on the other half (and vice versa). Channel explanations are in electrode
# space (the alignment matrix is applied inside the differentiable forward pass);
# graph nodes and edges are in aligned space, and alignment locality is reported.
#
# Explanations
# - Channels: zero-occlusion, integrated gradients (phase-randomised baseline), and
#   attention received (attended candidate).
# - Edges: attention weight, and attention x gradient.
# - Bands: integrated gradients on the frequency encoder's band-power features.
#
# Pre-registered rules (written before any explanation results; subject = unit, n = 18,
# each subject's 3 folds pooled):
# - Level 1, sanity: a channel method passes if the median over subjects of the Spearman
#   correlation between its trained-model scores and its scores after randomising every
#   module is below 0.3 in absolute value.
# - Level 2, faithfulness (primary evaluation: deleting electrodes by swapping in another
#   window's signal, on the held-out half): a channel method passes if its per-subject
#   deletion-curve area (accuracy) is below the mean of 20 random rankings AND its
#   keep-only curve area is above it, each by a one-sided Wilcoxon over subjects with
#   Holm over the 3 methods, p < 0.05. Same rule for the 2 edge methods (edge removal).
#   Zero-deletion is reported as secondary.
# - Bands: per subject, Spearman correlation over the 8 bands between the band
#   attribution and the margin drop when that band is filtered out of all electrodes;
#   passes if the median over subjects is at least 0.7.
# - H2: attention is less faithful than occlusion and integrated gradients if its
#   deletion area is higher than each of theirs (one-sided Wilcoxon, p < 0.05).
# - Level 4, stability: median Spearman correlation of channel scores across the 10
#   pairs of the 5 seeds at least 0.7.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts` and the outputs of
#   `story-disjoint-v14-round9-fixa` and `-fixb`
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
# ## 2. Configuration and inputs

# %%
import json
import time
from pathlib import Path

import numpy as np
import torch

from aad_xai.config import PreprocessConfig
from aad_xai.data.alignment import euclidean_reference, inverse_sqrtm
from aad_xai.data.cv_splits import subject_group_story_out
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.models.sgat import StimulusGAT
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
OUT_DIR = Path("/kaggle/working/results_xai_round9")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT = {p.name: p for p in Path("/kaggle/input").rglob("sgat_v14_fix_s*_sgso_*.pt")}
NPZ = {}
for p in Path("/kaggle/input").rglob("sgso_*.npz"):
    if "v14" in str(p):
        NPZ.setdefault(p.stem, []).append(p)
print(f"checkpoints: {len(CKPT)}  round-9 npz files: {sum(len(v) for v in NPZ.values())}  device: {DEVICE}")
assert len(CKPT) == 90, "attach the outputs of story-disjoint-v14-round9-fixa and -fixb"

pp = PreprocessConfig(bandpass_hz=(1.0, 31.0), envelope_bandpass_hz=(1.0, 8.0))
SFREQ = pp.sfreq_out
_t0 = time.time()
trials = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp, envelope_method="powerlaw_subbands").trials())
print(f"Trials: {len(trials)} in {time.time() - _t0:.0f}s")
folds = list(subject_group_story_out(trials, seed=FOLD_SEED))


# %% [markdown]
# ## 3. Explanations and faithfulness tests, per fold and test subject

# %%
def load_model(seed, fold_id):
    m = StimulusGAT(n_channels=64, sfreq=SFREQ, film_pooling="corr", speech_kernel_sizes=(33, 33),
                    freq_length_normalize=True).to(DEVICE)
    m.load_state_dict(torch.load(CKPT[f"sgat_v14_fix_s{seed}_{fold_id}.pt"], map_location=DEVICE))
    return m.eval()


def saved_logits(fold_id, seed):
    for p in NPZ[fold_id]:
        z = np.load(p)
        if f"fix_s{seed}_test_logits" in z.files:
            return z["test_subject"], z[f"fix_s{seed}_test_logits"]
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


results = []
t_all = time.time()
for fi, fold in enumerate(folds):
    t0 = time.time()
    test_trials = [trials[i] for i in fold.test_idx]
    sub_saved = {s: saved_logits(fold.fold_id, s) for s in SEEDS}
    models = {s: load_model(s, fold.fold_id) for s in SEEDS}
    for subj in fold.meta["test_subjects"]:
        st = [t for t in test_trials if t.subject_id == subj]
        W = inverse_sqrtm(euclidean_reference([t.eeg for t in st], EA_SHRINKAGE))        # (C, C)
        ds = WindowedEEGAudioDataset(st, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=False)
        x, e, y = _stack(ds)                                                              # raw windows
        trial_ids = np.array([w.trial_id for w in ds.window_indices])
        rec = {"fold_id": fold.fold_id, "subject": subj, "n_windows": len(y),
               "locality": sf.alignment_locality(W)}

        # Reproduction: the electrode-space scorer must reproduce the round-9 test decisions.
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
        xa, ea, ya = x[torch.as_tensor(np.where(hA)[0], device=DEVICE)], e[torch.as_tensor(np.where(hA)[0], device=DEVICE)], y[torch.as_tensor(np.where(hA)[0], device=DEVICE)]
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
        print(f"[{fi+1:>2}/{len(folds)}] {fold.fold_id} {subj}: n={len(y)} locality={rec['locality']:.2f} "
              f"repro={min(r['agree'] for r in rec['reproduction'].values()):.3f} acc(half B)={acc:.3f}", flush=True)
    print(f"   fold done in {time.time() - t0:.0f}s (total {(time.time() - t_all) / 60:.0f} min)", flush=True)
    with open(OUT_DIR / "xai_round9.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "config": {"seeds": SEEDS, "faith_seed": FAITH_SEED, "n_random": N_RANDOM,
                                                     "ig_steps": IG_STEPS, "band_steps": BAND_STEPS, "ch_ks": CH_KS},
                   "results": results}, f)

print(f"Done: {len(results)} subject-folds in {(time.time() - t_all) / 60:.0f} min")
