# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Architecture-only reference maps: explanations of randomised StimulusGAT models (Kaggle, GPU)
#
# Exploratory (section 8 of docs/preregistration_kul_external_test.md). Follow-up of the DTU -> KUL
# explanation transfer (scripts/analyse_xai_transfer.py): the DTU and KUL group maps of attention
# agree strongly, but attention fails the randomisation sanity check on KUL. How much of the
# cross-dataset agreement do the architecture, the electrode graph and the data statistics give
# on their own, without anything learned?
#
# For every fold, test subject and trained seed of the X1 runs (DTU: round-9 checkpoints, 18 folds x
# 5 seeds; KUL: KUL external-test checkpoints, 16 folds x 5 seeds), the trained model is replaced by
# a fully randomised copy (sgat_faithfulness.fully_randomised: every parameter re-drawn, Gaussian at
# the parameter's own scale, electrode graph kept; the final state of the R1 sanity cascade). Each
# (dataset, fold, seed) gets its own random draw, so the DTU and KUL random networks are independent,
# as the trained ones are. On exactly the X1 windows (raw test windows of the subject, electrode-space
# scorer with the subject's alignment matrix):
# - channel scores on all windows: occlusion, integrated gradients, attention received
#   (the inputs of the X1 group maps);
# - edge scores of the seed-42 random model on each cross-fit half (attention, attention x gradient),
#   averaged over the halves as for the trained edge maps;
# - the random model's accuracy (expected near 0.5).
# Analysis and decision rules: scripts/analyse_xai_random_reference.py (written before this run).
#
# DATASET "DTU": attach `dulanamanjitha/aad-xai-artifacts` and the outputs of
# `story-disjoint-v14-round9-fixa` / `-fixb`.
# DATASET "KUL": attach `dulanamanjitha/kul-dataset` and the outputs of `kul-external-sgat-a` / `-b`.
# GPU required.

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
# ## 2. Dataset and configuration (X1 settings)

# %%
DATASET = "DTU"
assert DATASET in ("DTU", "KUL")

import glob
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch

from aad_xai.config import PreprocessConfig
from aad_xai.data.alignment import euclidean_reference, inverse_sqrtm
from aad_xai.data.cv_splits import KUL_STORY_FOLDS, kul_story_part, subject_group_story_out
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.data.kul_dataset import KULeuvenDataset
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.models.sgat import DTU_MONTAGE_PATH, StimulusGAT
from aad_xai.xai import sgat_faithfulness as sf
assert hasattr(sf, "fully_randomised"), "repo predates the random-reference helper"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_S = 5.0
FOLD_SEED = 42
EA_SHRINKAGE = 1e-3
SEEDS = [42, 1, 2, 3, 4]
FAITH_SEED = 42
IG_STEPS = 32
SPEECH_KERNEL_SIZES = (33, 33)
SGAT_ENVELOPE_METHOD = "powerlaw_subbands"
ENVELOPE_BAND_HZ = (1.0, 8.0)
SGAT_EEG_BAND_HZ = (1.0, 31.0)
RAND_SEED_BASE = {"DTU": 10_000, "KUL": 20_000}[DATASET]
CKPT_PATTERN = {"DTU": "sgat_v14_fix_s{seed}_{fold}.pt", "KUL": "sgat_kul_s{seed}_{fold}.pt"}[DATASET]
N_CKPT = {"DTU": 90, "KUL": 80}[DATASET]
OUT_DIR = Path(f"/kaggle/working/results_xai_random_{DATASET}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pp = PreprocessConfig(bandpass_hz=SGAT_EEG_BAND_HZ, envelope_bandpass_hz=ENVELOPE_BAND_HZ)
SFREQ = pp.sfreq_out
MONTAGE = pd.read_csv(DTU_MONTAGE_PATH).sort_values("channel_index")["electrode_name"].tolist()

# %% [markdown]
# ## 3. Data and folds (identical to the X1 notebooks of each dataset)

# %%
_t0 = time.time()
if DATASET == "DTU":
    DTU_ROOT = next(p for p in ["/kaggle/input/aad-xai-artifacts/datasets/DTU",
                                "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU"]
                    if os.path.isdir(p))
    trials = list(DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=pp,
                             envelope_method=SGAT_ENVELOPE_METHOD).trials())
    folds = list(subject_group_story_out(trials, seed=FOLD_SEED))
    assert len(folds) == 18
else:
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

    def _unwrap(x):
        a = np.asarray(x)
        while a.dtype == object and a.size == 1:
            a = np.asarray(a.flat[0])
        return a

    _tr = scipy.io.loadmat(str(KUL_ROOT / "S1.mat"))["trials"]
    _t1 = _unwrap(_tr[0, 0] if _tr.ndim == 2 else _tr[0])
    _labels = [str(np.asarray(lab).flat[0]) for lab in _unwrap(_unwrap(_t1["FileHeader"])["Channels"])["Label"].flat]
    assert _labels == MONTAGE, f"KUL channel order differs from the montage: {_labels[:6]} vs {MONTAGE[:6]}"
    del _tr, _t1
    kul = KULeuvenDataset(KUL_ROOT, preprocess=pp, load_audio=True, envelope_method=SGAT_ENVELOPE_METHOD,
                          include_experiments=(1, 2), envelope_source="dry")
    trials = list(kul.trials())
    assert len(trials) == 128 and len({t.subject_id for t in trials}) == 16, len(trials)
    folds = list(subject_group_story_out(trials, content_group=kul_story_part, story_folds=KUL_STORY_FOLDS,
                                         n_subject_groups=8, n_val_subjects=2, seed=FOLD_SEED))
    assert len(folds) == 16
print(f"{DATASET}: {len(trials)} trials, {len(folds)} folds in {time.time() - _t0:.0f}s; device {DEVICE}")

CKPT = {p.name: p for p in Path("/kaggle/input").rglob(CKPT_PATTERN.format(seed="*", fold="sgso_*"))}
assert len(CKPT) == N_CKPT, f"{len(CKPT)} checkpoints; attach the training kernels' outputs"


def load_model(seed, fold_id):
    m = StimulusGAT(n_channels=64, sfreq=SFREQ, film_pooling="corr", speech_kernel_sizes=SPEECH_KERNEL_SIZES,
                    freq_length_normalize=True).to(DEVICE)
    m.load_state_dict(torch.load(CKPT[CKPT_PATTERN.format(seed=seed, fold=fold_id)], map_location=DEVICE))
    return m.eval()


def _stack(ds):
    xs, es, ys = zip(*[ds[i] for i in range(len(ds))])
    return (torch.stack(xs).to(DEVICE), torch.stack(es).to(DEVICE),
            torch.as_tensor(ys, dtype=torch.long, device=DEVICE))


# %% [markdown]
# ## 4. Explanations of the randomised models, per fold and test subject

# %%
EDGES = None
results = []
t_all = time.time()
for fi, fold in enumerate(folds):
    t0 = time.time()
    test_trials = [trials[i] for i in fold.test_idx]
    rand_seeds = {s: RAND_SEED_BASE + 10 * fi + k for k, s in enumerate(SEEDS)}
    rand_models = {s: sf.fully_randomised(load_model(s, fold.fold_id), seed=rand_seeds[s]) for s in SEEDS}
    edges = sf.directed_edges(rand_models[FAITH_SEED])
    EDGES = edges if EDGES is None else EDGES
    assert edges == EDGES, "electrode graph differs between folds"
    for subj in fold.meta["test_subjects"]:
        st = [t for t in test_trials if t.subject_id == subj]
        W = inverse_sqrtm(euclidean_reference([t.eeg for t in st], EA_SHRINKAGE))        # (C, C)
        Wt = torch.as_tensor(W, dtype=torch.float32)
        ds = WindowedEEGAudioDataset(st, window_s=WINDOW_S, overlap_s=0.0, standardize_eeg=False)
        x, e, y = _stack(ds)                                                              # raw windows
        trial_ids = np.array([w.trial_id for w in ds.window_indices])
        rec = {"fold_id": fold.fold_id, "subject": subj, "n_windows": len(y), "rand_seeds": rand_seeds,
               "rand_acc": {}, "rand_scores": {}}
        for s in SEEDS:
            scorer = sf.ElectrodeSpaceScorer(rand_models[s], Wt).to(DEVICE)
            rec["rand_acc"][s] = sf.evaluate(scorer, x, e, y)[0]
            rec["rand_scores"][s] = {"occlusion": sf.occlusion_scores(scorer, x, e, y).tolist(),
                                     "ig": sf.ig_channel_scores(scorer, x, e, y, steps=IG_STEPS).tolist(),
                                     "attention": sf.attention_node_scores(scorer, x, e, y).tolist()}
        scorer = sf.ElectrodeSpaceScorer(rand_models[FAITH_SEED], Wt).to(DEVICE)
        halves = []
        for mask in sf.cross_fit_halves(trial_ids):
            idx = torch.as_tensor(np.where(mask)[0], device=DEVICE)
            halves.append({m: sf.edge_scores(scorer, x[idx], e[idx], y[idx], edges, method=m)
                           for m in ("attention", "attn_x_grad")})
        rec["rand_edge_scores"] = {m: np.mean([h[m] for h in halves], axis=0).tolist()
                                   for m in ("attention", "attn_x_grad")}
        results.append(rec)
        print(f"[{fi + 1:>2}/{len(folds)}] {fold.fold_id} {subj}: n={len(y)} random-model acc "
              + ", ".join(f"{a:.3f}" for a in rec["rand_acc"].values()), flush=True)
    print(f"   fold done in {time.time() - t0:.0f}s (total {(time.time() - t_all) / 60:.0f} min)", flush=True)
    with open(OUT_DIR / f"xai_random_{DATASET}.json", "w") as f:
        json.dump({"commit": GIT_COMMIT, "dataset": DATASET,
                   "config": {"seeds": SEEDS, "faith_seed": FAITH_SEED, "ig_steps": IG_STEPS,
                              "rand_seed_base": RAND_SEED_BASE, "folds": [f_.fold_id for f_ in folds],
                              "montage": MONTAGE,
                              "randomisation": "sgat_faithfulness.fully_randomised (all parameters, "
                                               "Gaussian at each parameter's scale; graph kept)"},
                   "edges": [list(e_) for e_ in EDGES], "results": results}, f)

print(f"Done: {len(results)} subject-folds in {(time.time() - t_all) / 60:.0f} min")
