# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interpretability level 3b: planted ground truth at realistic accuracy (Kaggle)
#
# Level 3 (RESULTS_LOG 2026-11-09) planted a known attended-speech response in simulated EEG
# and scored each explanation method on recovering it. The model was at 100% test accuracy at
# every SNR of that sweep (-20 to -5 dB), so its verdicts are an upper bound. Real DTU accuracy
# is about 0.69 at 5 s. This run repeats the benchmark where the model is near 0.70.
#
# Unchanged from level 3: generator (src/aad_xai/data/planted_sources.py), conditions
# (delta 1-4 Hz around T7; theta 4-8 Hz around P8; 6 planted electrodes), decoy (Cz,
# attention-blind, 0 dB), round-9 StimulusGAT recipe, fold sgso_g0_k0, explanation harness,
# and every pass/fail threshold (src/aad_xai/xai/planted_eval.py).
# The decoy stays at 0 dB, so it is now 20-40 dB stronger than the plant: a harder test.
#
# Why this SNR range (decided before any model was trained at these SNRs): a matched-filter
# oracle that knows the planted electrodes, band and TRF (scripts/planted_oracle.py; an
# approximate upper bound) reaches 0.70 at about -40 dB (delta) and -43 dB (theta), and about
# 1.00 at -20 dB. A trained model should cross 0.70 above the oracle, and level 3 showed the
# model at 1.00 at -20 dB.
#
# Pre-registered rules (written before any run of this notebook):
# - Stage 1, sweep: SNR -40, -35, -32.5, -30, -27.5, -25, -22.5, -20 dB, seeds 0-2, per
#   condition. -20 dB repeats level 3 (gate: median accuracy >= 0.99 there in each condition).
# - Main SNR, per condition: the lowest sweep SNR whose median test accuracy is >= 0.70
#   (select_main_snr). Deviation from level 3, which used one SNR for both conditions; the
#   two conditions reach 0.70 at different SNRs.
# - Stage 2: seeds 0-9 at the main SNR (0-2 reused). Null (no plant): seeds 0-2.
# - Verdicts at the main SNR: the level-3 rules unchanged (planted_eval.verdicts): V1-V3,
#   electrode AUROC >= 0.9, decoy never in the top 6, band attribution first in >= 90% of
#   seeds, edges (only if V3 holds) AUROC >= 0.9.
# - Primary: verdicts on all seeds (comparable with level 3). Secondary: verdicts on learned
#   seeds only (class-balanced validation accuracy >= 0.60, planted_eval.is_learned). A method
#   that fails on all seeds but passes on learned seeds is a "model failure", not a
#   "method failure".
# - Regime label per condition (planted_eval.regime): "realistic" if the main-SNR median
#   accuracy is <= 0.85 and at least half the seeds lie in [0.60, 0.85]; "near ceiling" if the
#   median is above 0.85; otherwise "spread".
# - Descriptive: accuracy, AUROC and decoy rank along the sweep; verdict changes vs level 3.
# Analysis: scripts/analyse_xai_level3b.py on the outputs of parts A and B, run once.
#
# Parts (two kernels in parallel): A = delta_T7 + null, B = theta_P8.
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
from aad_xai.xai.planted_eval import select_main_snr

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
PART = "A"                     # A: delta_T7 + null; B: theta_P8 (two kernels in parallel)
ALL_CONDITIONS = {"delta_T7": {"center": "T7", "band_hz": (1.0, 4.0)},
                  "theta_P8": {"center": "P8", "band_hz": (4.0, 8.0)}}
CONDITIONS = {k: ALL_CONDITIONS[k] for k in {"A": ["delta_T7"], "B": ["theta_P8"]}[PART]}
DECOY, DECOY_SNR_DB = "Cz", 0.0
SNR_SWEEP_DB = [-40.0, -35.0, -32.5, -30.0, -27.5, -25.0, -22.5, -20.0]
SWEEP_SEEDS = [0, 1, 2]
MAIN_SEEDS = list(range(10))
NULL_SEEDS = [0, 1, 2] if PART == "A" else []
MAIN_ACC = 0.70
IG_STEPS = 32
BAND_STEPS = 16
N_EDGE_SETS = 20
CH_METHODS = ("occlusion_zero", "occlusion_permute", "ig", "attention")
EDGE_METHODS = ("attention", "attn_x_grad")
TOP10 = 6                      # top 10% of 64 electrodes
OUT_DIR = Path(f"/kaggle/working/results_xai_level3b_{PART}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"part {PART}; device {DEVICE}; conditions {list(CONDITIONS)}; sweep {SNR_SWEEP_DB} dB")

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
    with open(OUT_DIR / f"xai_level3b_{PART}.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "main_snr_db": main_snr,
                   "config": {"part": PART, "conditions": {k: {"center": v["center"], "band_hz": list(v["band_hz"])}
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
# ## 4. Stage 1 (sweep), main SNR per condition, stage 2, null

# %%
results = []
t_all = time.time()
for snr in SNR_SWEEP_DB:
    for cond in CONDITIONS:
        for seed in SWEEP_SEEDS:
            results.append(run_one(cond, snr, seed, "sweep"))
            show(results[-1])
            save(results)

MAIN_SNR = {}
for cond in CONDITIONS:
    acc = {snr: [r["test_acc"] for r in results if r["condition"] == cond and r["snr_db"] == snr]
           for snr in SNR_SWEEP_DB}
    MAIN_SNR[cond] = select_main_snr(acc, MAIN_ACC)
    print(f"{cond}: median accuracy by SNR " + ", ".join(f"{s:+.1f} {np.median(a):.3f}" for s, a in acc.items())
          + f" -> main SNR {MAIN_SNR[cond]}"
          + ("" if MAIN_SNR[cond] is not None else " (NO SNR REACHES 0.70: no pass/fail claim)"), flush=True)
save(results, MAIN_SNR)

for cond in CONDITIONS:
    if MAIN_SNR[cond] is None:
        continue
    for seed in MAIN_SEEDS:
        if seed not in SWEEP_SEEDS:
            results.append(run_one(cond, MAIN_SNR[cond], seed, "main"))
            show(results[-1])
            save(results, MAIN_SNR)
for seed in NULL_SEEDS:
    results.append(run_one("null", None, seed, "null"))
    show(results[-1])
    save(results, MAIN_SNR)
print(f"part {PART}: {len(results)} runs in {(time.time() - t_all) / 60:.0f} min")
