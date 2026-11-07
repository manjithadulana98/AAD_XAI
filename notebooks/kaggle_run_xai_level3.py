# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interpretability level 3: planted ground truth (Kaggle)
#
# Real EEG has no ground truth for explanations, so this run plants one
# (src/aad_xai/data/planted_sources.py) and scores every explanation method on whether
# it recovers it. Each simulated subject's EEG = realistic background (1/f, volume
# conduction over the DTU electrode positions, posterior alpha, average reference)
# + the attended envelope through a TRF (peak lag 100-250 ms, per subject), band-passed
# to one band, on a planted electrode set at a set SNR + a decoy electrode (Cz) that
# responds to both envelopes equally at 0 dB (stimulus-driven, attention-blind).
#
# Model and protocol: the round-9 StimulusGAT recipe unchanged (corr pooling, speech
# kernels (33, 33), DTU electrode graph, length-normalised band power, class-balanced
# loss and selection, 25 epochs), 18 subjects, story-disjoint fold sgso_g0_k0, EA per
# subject per split. Explanations in electrode space on the 3 test subjects' windows
# (per-subject z-scored scores averaged over the subjects); graph nodes and edges are
# in aligned space. One dataset draw and one training seed per run.
#
# Conditions: delta_T7 (delta 1-4 Hz on T7 and its 5 nearest electrodes) and theta_P8
# (theta 4-8 Hz on P8 and its 5 nearest), plus a null condition (no planted response,
# decoy only).
#
# Pre-registered (proposal deck B17; written before any results):
# - Stage 1, noise sweep: SNR -20, -15, -10, -5 dB per planted electrode, seeds 0-2,
#   both conditions. The main SNR is the lowest SNR at which the median test accuracy
#   (5 s windows) is >= 0.70 in both conditions. If none reaches it, the benchmark is
#   invalid in this range and no pass/fail claim is made.
# - Stage 2: seeds 0-9 at the main SNR (0-2 reused). Null: seeds 0-2.
# - Validity gates (main SNR, per condition): V1 median test accuracy >= 0.70;
#   V2 filtering the planted band out of all electrodes gives the largest margin drop
#   of the 8 bands in >= 90% of seeds (the model really uses the planted band);
#   V3 removing the planted edges (both ends in the planted set) drops the margin more
#   than removing as many random other edges (one-sided Wilcoxon over seeds,
#   p < 0.05). If V3 fails, edge ground truth is not identifiable for this model and
#   edge methods are not scored.
# - Electrodes: a channel method (zero-occlusion, permutation-occlusion, integrated
#   gradients, attention received) passes if its median AUROC for the planted set is
#   >= 0.9 in both conditions.
# - Decoy: a channel method passes if Cz is never in its top 10% (ranks 1-6 of 64) in
#   any main-SNR seed of either condition.
# - Bands: band attribution passes if the planted band ranks first in >= 90% of seeds
#   in both conditions.
# - Edges (if V3 holds): an edge method (attention weight, attention x gradient)
#   passes if its median AUROC for the planted edges is >= 0.9 in both conditions.
# - Descriptive: the sweep (accuracy, AUROC, decoy rank vs SNR); null-condition
#   accuracy and decoy ranks.
#
# **Kaggle setup requirements**
# - Enable Internet; GPU required (no dataset needed)

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
# ## 2. Configuration

# %%
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata, wilcoxon
from torch.utils.data import DataLoader

from aad_xai.data.alignment import euclidean_align, euclidean_reference, inverse_sqrtm
from aad_xai.data.cv_splits import assert_story_disjoint_fold_integrity, subject_group_story_out
from aad_xai.data.planted_sources import PlantedSpec, make_planted_trials
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.models.sgat import StimulusGAT
from aad_xai.models.sgat_frequency_encoder import DEFAULT_BANDS_HZ
from aad_xai.utils.seed import seed_everything
from aad_xai.xai import sgat_faithfulness as sf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SFREQ = 64.0
WINDOW_S = 5.0
TRAIN_HOP_S = 2.5
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EA_SHRINKAGE = 1e-3
FOLD_SEED = 42
N_SUBJECTS, TRIALS_PER_STORY, TRIAL_S = 18, 4, 50.0
CONDITIONS = {"delta_T7": {"center": "T7", "band_hz": (1.0, 4.0)},
              "theta_P8": {"center": "P8", "band_hz": (4.0, 8.0)}}
DECOY, DECOY_SNR_DB = "Cz", 0.0
SNR_SWEEP_DB = [-20.0, -15.0, -10.0, -5.0]
SWEEP_SEEDS = [0, 1, 2]
MAIN_SEEDS = list(range(10))
NULL_SEEDS = [0, 1, 2]
MAIN_ACC = 0.70
IG_STEPS = 32
BAND_STEPS = 16
N_EDGE_SETS = 20
CH_METHODS = ("occlusion_zero", "occlusion_permute", "ig", "attention")
EDGE_METHODS = ("attention", "attn_x_grad")
TOP10 = 6                      # top 10% of 64 electrodes
OUT_DIR = Path("/kaggle/working/results_xai_level3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"device {DEVICE}; conditions {list(CONDITIONS)}; sweep {SNR_SWEEP_DB} dB")

# %% [markdown]
# ## 3. Helpers: training (round-9 recipe), metrics, one run

# %%
def _stack(ds):
    """All windows: X (N, C, T), E (N, 2, T), y (N,)."""
    xs, es, ys = zip(*[ds[i] for i in range(len(ds))])
    return torch.stack(xs), torch.stack(es), torch.as_tensor(ys, dtype=torch.long)


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


def new_model():
    return StimulusGAT(n_channels=64, sfreq=SFREQ, film_pooling="corr", speech_kernel_sizes=(33, 33),
                       freq_length_normalize=True).to(DEVICE)


def train_sgat(ds_tr, X_va, E_va, y_va, seed):
    """Round-9 recipe: AdamW, class-balanced loss, best epoch by class-balanced val accuracy."""
    seed_everything(seed)
    model = new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_tr = np.array([w.label for w in ds_tr.window_indices])
    class_w = torch.tensor([len(y_tr) / (2.0 * max((y_tr == c).sum(), 1)) for c in (0, 1)],
                           dtype=torch.float32, device=DEVICE)          # (2,)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    best = {"val": -1.0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, eb, yb in DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0):
            xb, eb = xb.to(DEVICE), eb.to(DEVICE)                       # (B, C, T), (B, 2, T)
            opt.zero_grad(set_to_none=True)
            loss_fn(model.score_candidates(xb, eb), torch.as_tensor(yb, device=DEVICE)).backward()
            opt.step()
        v = _balanced(_logits(model, X_va, E_va), y_va)
        if v > best["val"]:
            best = {"val": v, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    return model.eval(), best["epoch"], best["val"]


def auroc(scores, positive):
    """Mann-Whitney AUROC of scores for a boolean positive set (ties averaged)."""
    scores, positive = np.asarray(scores, dtype=float), np.asarray(positive, dtype=bool)
    r = rankdata(scores)
    n1, n0 = positive.sum(), (~positive).sum()
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank_of(scores, idx):
    """1 = highest score."""
    scores = np.asarray(scores, dtype=float)
    return int((scores > scores[idx]).sum() + 1)


def zmean(vectors):
    """Average of per-subject z-scored score vectors."""
    return np.mean([(v - v.mean()) / (v.std() + 1e-12) for v in map(np.asarray, vectors)], axis=0)


EDGES = None   # directed edges of the fixed DTU graph, set by the first model


def run_one(cond, snr_db, seed, stage):
    """Generate data, train, explain on the test subjects, score against the truth."""
    global EDGES
    t0 = time.time()
    if cond == "null":
        spec = PlantedSpec(snr_db=None, decoy=DECOY, decoy_snr_db=DECOY_SNR_DB)
    else:
        spec = PlantedSpec(center=CONDITIONS[cond]["center"], band_hz=CONDITIONS[cond]["band_hz"],
                           snr_db=snr_db, decoy=DECOY, decoy_snr_db=DECOY_SNR_DB)
    trials, truth = make_planted_trials(spec, n_subjects=N_SUBJECTS, trials_per_story=TRIALS_PER_STORY,
                                        trial_s=TRIAL_S, sfreq=SFREQ, seed=seed)
    fold = next(iter(subject_group_story_out(trials, seed=FOLD_SEED)))
    assert_story_disjoint_fold_integrity(trials, fold)
    tr, va, te = [euclidean_align([trials[i] for i in idx], shrinkage=EA_SHRINKAGE)
                  for idx in (fold.train_idx, fold.val_idx, fold.test_idx)]
    ds_tr = WindowedEEGAudioDataset(tr, window_s=WINDOW_S, overlap_s=WINDOW_S - TRAIN_HOP_S, standardize_eeg=True)
    X_va, E_va, y_va = _stack(WindowedEEGAudioDataset(va, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=True))
    ds_te = WindowedEEGAudioDataset(te, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=True)
    X_te, E_te, y_te = _stack(ds_te)
    te_subj = np.array([w.subject_id for w in ds_te.window_indices])

    model, ep, val = train_sgat(ds_tr, X_va, E_va, y_va.numpy(), seed)
    logits_te = _logits(model, X_te, E_te)
    if EDGES is None:
        EDGES = sf.directed_edges(model)
    planted = cond != "null"
    gt_edges = truth.planted_edges(EDGES) if planted else None

    rec = {"condition": cond, "snr_db": snr_db, "seed": seed, "stage": stage, "best_epoch": ep, "val": val,
           "test_acc": float(np.mean(logits_te.argmax(1) == y_te.numpy())),
           "planted_idx": truth.planted_idx.tolist(), "decoy_idx": truth.decoy_idx,
           "band_hz": list(truth.band_hz), "lags_s": truth.lags_s, "subjects": []}
    for subj in fold.meta["test_subjects"]:
        st = [trials[i] for i in fold.test_idx if trials[i].subject_id == subj]
        W = inverse_sqrtm(euclidean_reference([t.eeg for t in st], EA_SHRINKAGE))        # (C, C)
        x, e, y = (t.to(DEVICE) for t in _stack(WindowedEEGAudioDataset(st, window_s=WINDOW_S, overlap_s=0.0,
                                                                         standardize_eeg=False)))
        scorer = sf.ElectrodeSpaceScorer(model, torch.as_tensor(W, dtype=torch.float32)).to(DEVICE)
        with torch.no_grad():
            ours = torch.cat([scorer(x[i:i + 256], e[i:i + 256]) for i in range(0, len(x), 256)]).cpu().numpy()
        agree = float(np.mean(ours.argmax(1) == logits_te[te_subj == subj].argmax(1)))
        assert agree >= 0.99, (cond, seed, subj, agree)
        s = {"subject": subj, "n_windows": len(y), "agree": agree,
             "acc": float(np.mean(ours.argmax(1) == y.cpu().numpy())),
             "channel": {"occlusion_zero": sf.occlusion_scores(scorer, x, e, y, mode="zero").tolist(),
                         "occlusion_permute": sf.occlusion_scores(scorer, x, e, y, mode="permute").tolist(),
                         "ig": sf.ig_channel_scores(scorer, x, e, y, steps=IG_STEPS).tolist(),
                         "attention": sf.attention_node_scores(scorer, x, e, y).tolist()},
             "band_attr": sf.band_feature_attribution(scorer, x, e, y, steps=BAND_STEPS).tolist(),
             "band_stop": [float(m) for _, _, m in sf.band_stop_drops(scorer, x, e, y, SFREQ)],
             "edge": {m: sf.edge_scores(scorer, x, e, y, EDGES, method=m).tolist() for m in EDGE_METHODS}}
        if planted:
            _, base = sf.evaluate(scorer, x, e, y)
            gt = [EDGES[i] for i in np.where(gt_edges)[0]]
            with sf.edges_removed(model, gt):
                gt_drop = base - sf.evaluate(scorer, x, e, y)[1]
            rng = np.random.default_rng(seed)
            others = np.where(~gt_edges)[0]
            rand = []
            for _ in range(N_EDGE_SETS):
                with sf.edges_removed(model, [EDGES[i] for i in rng.choice(others, len(gt), replace=False)]):
                    rand.append(base - sf.evaluate(scorer, x, e, y)[1])
            s["edge_intervention"] = {"gt_drop": float(gt_drop), "random_drops": [float(v) for v in rand]}
        rec["subjects"].append(s)
    rec["metrics"] = metrics(rec, truth, gt_edges, planted)
    rec["seconds"] = time.time() - t0
    return rec


def metrics(rec, truth, gt_edges, planted):
    subs = rec["subjects"]
    out = {}
    for m in CH_METHODS:
        sc = zmean([s["channel"][m] for s in subs])
        out[f"decoy_rank_{m}"] = rank_of(sc, truth.decoy_idx)
        if planted:
            out[f"auroc_{m}"] = auroc(sc, truth.planted_mask())
    ba = np.mean([s["band_attr"] for s in subs], axis=0)
    bs = np.mean([s["band_stop"] for s in subs], axis=0)
    out["band_attr"], out["band_stop"] = ba.tolist(), bs.tolist()
    if planted:
        b = list(DEFAULT_BANDS_HZ).index(tuple(truth.band_hz))
        out["band_attr_rank"], out["band_stop_rank"] = rank_of(ba, b), rank_of(bs, b)
        for m in EDGE_METHODS:
            out[f"edge_auroc_{m}"] = auroc(zmean([s["edge"][m] for s in subs]), gt_edges)
        out["edge_gt_drop"] = float(np.mean([s["edge_intervention"]["gt_drop"] for s in subs]))
        out["edge_random_drop"] = float(np.mean([np.mean(s["edge_intervention"]["random_drops"]) for s in subs]))
    return out


def save(results, main_snr=None):
    with open(OUT_DIR / "xai_level3.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "main_snr_db": main_snr,
                   "config": {"conditions": {k: {"center": v["center"], "band_hz": list(v["band_hz"])}
                                             for k, v in CONDITIONS.items()},
                              "decoy": DECOY, "decoy_snr_db": DECOY_SNR_DB, "snr_sweep_db": SNR_SWEEP_DB,
                              "sweep_seeds": SWEEP_SEEDS, "main_seeds": MAIN_SEEDS, "null_seeds": NULL_SEEDS,
                              "main_acc": MAIN_ACC, "n_subjects": N_SUBJECTS,
                              "trials_per_story": TRIALS_PER_STORY, "trial_s": TRIAL_S, "epochs": EPOCHS,
                              "ig_steps": IG_STEPS, "band_steps": BAND_STEPS, "n_edge_sets": N_EDGE_SETS},
                   "edges": [list(map(int, e)) for e in (EDGES or [])], "results": results}, f)


def show(r):
    m = r["metrics"]
    au = " ".join(f"{k[6:]}={m[k]:.2f}" for k in m if k.startswith("auroc_"))
    print(f"{r['stage']:5s} {r['condition']:9s} snr={r['snr_db']} seed={r['seed']}: acc={r['test_acc']:.3f} "
          f"(ep {r['best_epoch']}) {au} decoy={[m[f'decoy_rank_{c}'] for c in CH_METHODS]} "
          f"band_attr_rank={m.get('band_attr_rank')} band_stop_rank={m.get('band_stop_rank')} "
          f"({r['seconds']:.0f}s)", flush=True)

# %% [markdown]
# ## 4. Stage 1 (noise sweep), main SNR, stage 2, null

# %%
results = []
t_all = time.time()
for snr in SNR_SWEEP_DB:
    for cond in CONDITIONS:
        for seed in SWEEP_SEEDS:
            results.append(run_one(cond, snr, seed, "sweep"))
            show(results[-1])
            save(results)

med_acc = {snr: min(np.median([r["test_acc"] for r in results if r["snr_db"] == snr and r["condition"] == c])
                    for c in CONDITIONS) for snr in SNR_SWEEP_DB}
reached = [snr for snr in SNR_SWEEP_DB if med_acc[snr] >= MAIN_ACC]
MAIN_SNR = min(reached) if reached else max(SNR_SWEEP_DB)
print(f"median accuracy (worse condition) by SNR: {med_acc}; main SNR {MAIN_SNR} dB "
      f"({'valid' if reached else 'NO SNR REACHES 0.70: benchmark invalid'})")

for cond in CONDITIONS:
    for seed in MAIN_SEEDS:
        if seed not in SWEEP_SEEDS:
            results.append(run_one(cond, MAIN_SNR, seed, "main"))
            show(results[-1])
            save(results, MAIN_SNR)
for seed in NULL_SEEDS:
    results.append(run_one("null", None, seed, "null"))
    show(results[-1])
    save(results, MAIN_SNR)
print(f"all runs done in {(time.time() - t_all) / 60:.0f} min")

# %% [markdown]
# ## 5. Pre-registered evaluation

# %%
def one_sided(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(wilcoxon(d, alternative="greater").pvalue) if np.any(d) else 1.0


main = {c: [r for r in results if r["condition"] == c and r["snr_db"] == MAIN_SNR] for c in CONDITIONS}
print(f"main SNR {MAIN_SNR} dB; seeds per condition {[len(v) for v in main.values()]}")
valid = {}
for c, rs in main.items():
    M = [r["metrics"] for r in rs]
    v1 = np.median([r["test_acc"] for r in rs]) >= MAIN_ACC
    v2 = np.mean([m["band_stop_rank"] == 1 for m in M]) >= 0.9
    p3 = one_sided([m["edge_gt_drop"] for m in M], [m["edge_random_drop"] for m in M])
    valid[c] = {"V1": bool(v1), "V2": bool(v2), "V3": bool(p3 < 0.05)}
    print(f"{c}: median acc {np.median([r['test_acc'] for r in rs]):.3f} (V1 {v1}); band-stop ranks "
          f"{[m['band_stop_rank'] for m in M]} (V2 {v2}); edge removal planted {np.mean([m['edge_gt_drop'] for m in M]):.4f} "
          f"vs random {np.mean([m['edge_random_drop'] for m in M]):.4f}, p={p3:.4f} (V3 {p3 < 0.05})")

print("\nElectrodes (median AUROC per condition; pass if >= 0.9 in both) and decoy (never in top 6):")
for m in CH_METHODS:
    med = {c: np.median([r["metrics"][f"auroc_{m}"] for r in rs]) for c, rs in main.items()}
    worst_decoy = min(r["metrics"][f"decoy_rank_{m}"] for rs in main.values() for r in rs)
    print(f"  {m:18s} " + "  ".join(f"{c} {v:.3f}" for c, v in med.items())
          + f"  -> electrodes {'PASS' if min(med.values()) >= 0.9 else 'fail'}; best decoy rank {worst_decoy}"
          + f" -> decoy {'PASS' if worst_decoy > TOP10 else 'fail'}")

print("\nBands (planted band first in >= 90% of seeds, both conditions):")
share = {c: np.mean([r["metrics"]["band_attr_rank"] == 1 for r in rs]) for c, rs in main.items()}
print("  band attribution: " + "  ".join(f"{c} {v:.2f}" for c, v in share.items())
      + f" -> {'PASS' if min(share.values()) >= 0.9 else 'fail'}")

print("\nEdges (only if V3 holds in both conditions):")
if all(v["V3"] for v in valid.values()):
    for m in EDGE_METHODS:
        med = {c: np.median([r["metrics"][f"edge_auroc_{m}"] for r in rs]) for c, rs in main.items()}
        print(f"  {m:12s} " + "  ".join(f"{c} {v:.3f}" for c, v in med.items())
              + f" -> {'PASS' if min(med.values()) >= 0.9 else 'fail'}")
else:
    print("  edge ground truth not identifiable for this model: edge methods not scored")

print("\nSweep (descriptive): median accuracy and AUROC by SNR")
for c in CONDITIONS:
    for snr in SNR_SWEEP_DB:
        rs = [r for r in results if r["condition"] == c and r["snr_db"] == snr and r["seed"] in SWEEP_SEEDS]
        print(f"  {c:9s} {snr:+5.0f} dB acc {np.median([r['test_acc'] for r in rs]):.3f}  " + "  ".join(
            f"{m} {np.median([r['metrics'][f'auroc_{m}'] for r in rs]):.2f}" for m in CH_METHODS))
nul = [r for r in results if r["condition"] == "null"]
print(f"\nNull: acc {[round(r['test_acc'], 3) for r in nul]}; decoy ranks "
      + str({m: [r['metrics'][f'decoy_rank_{m}'] for r in nul] for m in CH_METHODS}))
