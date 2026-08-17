# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FAConformer -- SI/LOSO port, single-subject pilot (all 8 SI folds)
#
# Phase A of the new DHGCN/FAConformer track, superseding ST-GCN Phases
# 3-6 (ST-GCN Phases 1-2 stay closed/documented as-is -- see
# `stgcn/outputs/phase2_gcn_only/PHASE2_REPORT.md`).
#
# Ports FAConformer's real model
# (`external/FAConformer/models/FAConformer.py`, unmodified) into this
# project's subject-independent, leakage-safe DTU pipeline
# (`DTUDataset.createSICrossValidation` -- the same folds ST-GCN and
# AADNet/VLAAI use), replacing FAConformer's own native chronological
# 90/10 subject-dependent split.
#
# **CSP dropped as of this version.** A deep-network-free diagnostic
# (`kaggle_faconformer_csp_probe_diagnostic.py`) found that CSP fit on the
# pooled, cross-subject SI training set produces substantially weaker
# features than CSP fit within one subject (a trivial logistic-regression
# probe: 62.6% train / 55.9% test on SI vs. 77.6% train / 76.7% test on
# the single-subject case) -- a well-documented failure mode for naively
# pooling CSP covariances across subjects with different head geometries.
# CSP was never part of FAConformer's own paper architecture to begin
# with (Section III-C/D describes raw channels -> FFT band decomposition
# -> CNN-Transformer directly) -- only the repo's own undocumented
# preprocessing addition. This version feeds the windowed raw 64-channel
# EEG straight into the existing `filter_signal_by_fft` band-decomposition
# step, matching the paper's actual described pipeline.
#
# Exactly three things change versus FAConformer's own SD pipeline;
# everything else (model, band-filter preprocessing, `attend_mf` label,
# hyperparameters) is unchanged:
#
# 1. **Data source.** Instead of FAConformer's own `sliding_window`'s
#    internal chronological 90/10 split on ONE subject's OWN trials, each
#    SI fold's train pool is the OTHER 17 subjects' trials (leakage-
#    filtered by `createSICrossValidation`'s own shared-attended-stimulus
#    exclusion) and the test set is the held-out subject's specific
#    fold-of-trials. A validation slice is carved from the training pool
#    via `train_test_split(..., test_size=0.2, random_state=subject_id)`
#    -- the exact convention `kaggle_diag_stgcn_epoch_budget_test.py`'s
#    `build_partition` already established for this project's SI pipeline.
#    `createSICrossValidation`'s returned eeg lists are already the same
#    raw per-trial (64, T) channel-major arrays this project's FAConformer
#    adapter has always worked with -- no new format to handle, just a
#    different source list per split.
# 2. **Windowing.** FAConformer's own `sliding_window` does an internal
#    90/10 split *within* each trial (chronological). That's the wrong
#    split now -- the split boundary is at the subject/trial level via SI
#    cross-validation, not within a trial. `windows_for_trials` below
#    reuses FAConformer's own overlapping-window extraction per trial but
#    returns ALL windows for a given trial list, with no internal split --
#    called once each for the training pool, validation slice, and test
#    fold.
# 3. **Nothing else.** Confirmed via AskUserQuestion before building this:
#    FAConformer's own `train_model_FAConformer` already selects its
#    checkpoint by best VALIDATION LOSS via patience=10 early stopping --
#    functionally identical to the fix that mattered for ST-GCN Phase 2
#    (best-val-loss selection, not final-epoch or best-val-accuracy). No
#    change needed there. Optimizer/LR/weight-decay/lambda/max_epoch stay
#    exactly as published (Adam lr=5e-4, weight_decay=3e-4, lambda=1,
#    max_epoch=200) -- ST-GCN's own training-recipe findings are NOT
#    assumed to transfer to a different architecture; that's a separate
#    question for later if numbers look wrong, not pre-empted here.
#
# **Scope: single subject, all 8 SI folds, single seed.** Confirm the port
# runs end-to-end and produces a sane per-fold accuracy distribution
# before scaling to the full 18-subject sweep -- same pilot-before-sweep
# discipline as every prior step in this project. Each fold's training
# pool is far larger than FAConformer's own within-subject pool (~17
# other subjects' trials vs. one subject's own 60), so per-epoch cost is
# expected to be substantially higher than the earlier single-subject
# pilot -- this is flagged, not silently absorbed.
#
# **Kaggle setup requirements:** Internet enabled, GPU accelerator,
# `dulanamanjitha/aad-xai-artifacts` dataset attached. No Kaggle Secret needed.

# %% [markdown]
# ## 1. Clone repository (with submodules) + install dependencies

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
    print(f"Repository already cloned at {REPO_DIR}")

os.chdir(REPO_DIR)

subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True, cwd=REPO_DIR)

try:
    import torch as _torch_preinstalled
    with open("requirements.txt") as _f:
        _reqs_no_torch = [ln for ln in _f if ln.strip() and not ln.strip().lower().startswith("torch")]
    with open("/tmp/requirements_no_torch.txt", "w") as _f:
        _f.writelines(_reqs_no_torch)
    subprocess.run(["pip", "install", "-q", "-r", "/tmp/requirements_no_torch.txt"], check=True)
except ImportError:
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

subprocess.run(["pip", "install", "-q", "-e", "."], check=True)
subprocess.run(["pip", "install", "-q", "dotmap", "einops"], check=True)

for extra in ("src", "external/AADNet"):
    p = os.path.join(REPO_DIR, extra)
    if p not in sys.path:
        sys.path.insert(0, p)

print("Setup done.")

# %% [markdown]
# ## 2. GPU sanity check (with fast compatibility probe)

# %%
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    try:
        _probe = torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")
        torch.cuda.synchronize()
        print("GPU compatibility probe: OK")
    except RuntimeError as e:
        raise RuntimeError(
            f"GPU compatibility probe FAILED on {torch.cuda.get_device_name(0)}: {e}\n"
            "Known P100/sm_60 incompatibility -- re-push/re-run for a different GPU."
        ) from e

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert DEVICE.type == "cuda", "FAConformer's code hardcodes .cuda() throughout -- GPU required."

# %% [markdown]
# ## 3. Phase 1 (AADNet namespace) -- bulk-load all 18 subjects, build SI
#    folds for the ONE pilot subject

# %%
import yaml
import numpy as np
import random

PILOT_SUBJECT_ID = 0    # AADNet 0-indexed -> FAConformer's "S1"
SEED = 42
TIME_LEN = 2.0           # seconds -- unchanged from the validated pipeline
VAL_FRACTION = 0.2       # matches kaggle_diag_stgcn_epoch_budget_test.py's build_partition


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_global_seed(SEED)

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'."

BASE_CONFIG_PATH = os.path.join(REPO_DIR, "external", "AADNet", "config", "config_AADNet_SI_DTU_kaggle.yml")
with open(BASE_CONFIG_PATH, encoding="utf-8") as f:
    raw_config = yaml.safe_load(f)
raw_config["dataset"]["folder"] = os.path.join(DTU_ROOT, "eeg_new") + "/"
raw_config["dataset"]["stimuli_path"] = os.path.join(DTU_ROOT, "Audio")

from utils.config import Config
aadnet_config = Config.load_config(raw_config)

from aadnet.dataset import DTUDataset
from sklearn.model_selection import train_test_split

import time as time_module
_load_start = time_module.time()
DTUDataset.loadData(aadnet_config, subject=None)   # pre-populate the cache; createSICrossValidation reuses it
print(f"Bulk-loaded 18 subjects in {time_module.time() - _load_start:.1f}s")

crossSIData = DTUDataset.createSICrossValidation(PILOT_SUBJECT_ID, aadnet_config)
N_FOLDS = len(crossSIData)
print(f"Subject {PILOT_SUBJECT_ID}: {N_FOLDS} SI folds built")

# Capture into plain Python lists before the AADNet namespace cleanup --
# these are raw per-trial (64, T) channel-major arrays + audio + 0/1
# labels, same format this project's FAConformer adapter has always used.
si_folds_raw = []
for fold_idx in range(N_FOLDS):
    (tr_eeg, tr_aud, tr_label), (te_eeg, te_aud, te_label) = crossSIData[fold_idx]
    si_folds_raw.append({
        "tr_eeg": list(tr_eeg), "tr_label": list(tr_label),
        "te_eeg": list(te_eeg), "te_label": list(te_label),
    })
    print(f"  fold {fold_idx}: {len(tr_eeg)} train trials, {len(te_eeg)} test trials, "
          f"train labels {sorted(set(int(l) for l in tr_label))}, "
          f"test labels {sorted(set(int(l) for l in te_label))}")

# %% [markdown]
# ## 4. Clean up AADNet's sys.path/module cache before switching namespaces

# %%
aadnet_dir = os.path.join(REPO_DIR, "external", "AADNet")
if aadnet_dir in sys.path:
    sys.path.remove(aadnet_dir)

for mod_name in list(sys.modules):
    if mod_name in ("utils", "aadnet") or mod_name.startswith("utils.") or mod_name.startswith("aadnet."):
        del sys.modules[mod_name]

print("AADNet namespace cleared.")

# %% [markdown]
# ## 5. Phase 2 (FAConformer namespace) -- SI-aware data adapter

# %%
faconformer_dir = os.path.join(REPO_DIR, "external", "FAConformer")
sys.path.insert(0, faconformer_dir)
os.chdir(faconformer_dir)  # FAConformer's own relative imports assume this

import math
from dotmap import DotMap
from utils.functions import filter_signal_by_fft
from torch.utils.data import Dataset, DataLoader


class CustomDatasets(Dataset):
    def __init__(self, data, label):
        self.data = torch.Tensor(data)
        self.label = torch.tensor(label, dtype=torch.uint8)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class RawWindowDataset(Dataset):
    """Holds only the raw, windowed 64-channel EEG (one 'band' worth of
    data, ~3GB for a large SI fold) -- NOT the 8-band-filtered
    representation. Kept as a plain numpy array; band-filtering happens
    per-batch in make_band_filter_collate, not here. No CSP -- see this
    notebook's header for why it was dropped."""
    def __init__(self, windows, label):
        self.windows = windows
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.windows[index], self.label[index]


def make_band_filter_collate(points, window_length):
    """Computes the 8-band FFT filtering for one BATCH at a time, not the
    whole dataset. FAConformer's own get_DTU_data precomputes all 8 bands
    for the entire dataset upfront -- fine at their own within-subject
    scale (~54 trials), but this SI port's per-fold training pool is
    ~17-20x larger (all other subjects' trials), and holding 8 full-size
    copies of that (~24-30GB, confirmed by an OOM kernel death on the
    first pilot push) doesn't fit. Deferring filter_signal_by_fft to
    per-batch keeps peak memory bounded by batch_size, not dataset size,
    at the cost of repeating the FFT filtering every epoch instead of once."""
    def collate(batch):
        windows = np.stack([item[0] for item in batch], axis=0)
        labels = np.stack([item[1] for item in batch], axis=0)
        bands = [filter_signal_by_fft(windows, lo, hi, window_length) for lo, hi in points]
        stacked = np.stack(bands, axis=1)
        return torch.Tensor(stacked), torch.tensor(labels, dtype=torch.uint8)
    return collate


def windows_for_trials(eeg_data_list, event_data_list, window_size, overlap, eeg_channel):
    """All overlapping windows for a list of trials -- no internal split.
    Reuses FAConformer's own sliding_window's per-trial window-extraction
    logic verbatim; drops its chronological 90/10 split, since the SI
    port's split boundary is at the subject/trial level instead."""
    stride = int(window_size * (1 - overlap))
    all_windows, all_labels = [], []
    for trial_eeg, label in zip(eeg_data_list, event_data_list):
        for i in range(0, trial_eeg.shape[0] - window_size + 1, stride):
            all_windows.append(trial_eeg[i:i + window_size, :])
            all_labels.append(label)
    windows = np.stack(all_windows, axis=0).reshape(-1, window_size, eeg_channel)
    labels = np.array(all_labels).reshape(-1, 1)
    return windows, labels


def build_faconformer_si_loaders(train_eeg_list, train_label_list,
                                  valid_eeg_list, valid_label_list,
                                  test_eeg_list, test_label_list, time_len):
    """SI-aware version of build_faconformer_loaders (kaggle_faconformer_pilot.py) --
    the train/valid/test SPLIT is supplied externally (SI folds) instead of
    being derived internally from one subject's own chronological 90/10
    split. No CSP (dropped -- see this notebook's header): raw 64-channel
    windows feed directly into the 8-band FFT decomposition, matching the
    paper's own described pipeline (Section III-C/D) rather than the
    repo's undocumented CSP preprocessing addition."""
    args = DotMap()
    args.fs = 64
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.eeg_channel = 64
    args.n_components = 64   # no CSP -- model's in_planes matches the raw channel count directly

    train_data, train_label = windows_for_trials(train_eeg_list, train_label_list,
                                                  args.window_length, args.overlap, args.eeg_channel)
    valid_data, valid_label = windows_for_trials(valid_eeg_list, valid_label_list,
                                                  args.window_length, args.overlap, args.eeg_channel)
    test_data, test_label = windows_for_trials(test_eeg_list, test_label_list,
                                                args.window_length, args.overlap, args.eeg_channel)

    args.delta_low, args.delta_high = 1, 4
    args.theta_low, args.theta_high = 4, 8
    args.alpha1_low, args.alpha1_high = 8, 10
    args.alpha2_low, args.alpha2_high = 10, 13
    args.beta1_low, args.beta1_high = 13, 16
    args.beta2_low, args.beta2_high = 16, 20
    args.beta3_low, args.beta3_high = 20, 26
    args.gamma1_low, args.gamma1_high = 26, 32
    args.frequency_resolution = args.fs / args.window_length

    band_edges = [
        (args.delta_low, args.delta_high), (args.theta_low, args.theta_high),
        (args.alpha1_low, args.alpha1_high), (args.alpha2_low, args.alpha2_high),
        (args.beta1_low, args.beta1_high), (args.beta2_low, args.beta2_high),
        (args.beta3_low, args.beta3_high), (args.gamma1_low, args.gamma1_high),
    ]
    points = [(math.ceil(lo / args.frequency_resolution), math.ceil(hi / args.frequency_resolution) + 1)
              for lo, hi in band_edges]

    # Channel-major, matching filter_signal_by_fft's axis=1-is-time
    # convention -- unrelated to CSP, still needed with CSP dropped.
    train_data = train_data.transpose(0, 2, 1)
    valid_data = valid_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)

    # No CSP fit/transform here -- raw 64-channel windows feed directly
    # into the 8-band FFT decomposition below.

    # 8-band FFT filtering deferred to per-batch (see make_band_filter_collate) --
    # NOT precomputed for the whole dataset here, unlike FAConformer's own
    # get_DTU_data. See that function's docstring for why.
    collate_fn = make_band_filter_collate(points, args.window_length)

    train_loader = DataLoader(RawWindowDataset(train_data, train_label), batch_size=args.batch_size,
                               drop_last=True, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(RawWindowDataset(valid_data, valid_label), batch_size=args.batch_size,
                               drop_last=True, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(RawWindowDataset(test_data, test_label), batch_size=args.batch_size,
                              drop_last=True, pin_memory=True, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader, args


# %% [markdown]
# ## 6. Per-fold train/eval -- FAConformer's own hyperparameters and
#    patience=10 best-val-loss early stopping, unchanged

# %%
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from models.FAConformer import FAConformer
import pandas as pd
from pathlib import Path
import json

OUT_DIR = Path("/kaggle/working/faconformer_si_pilot")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / f"DTU_FAConformer_SI_subject{PILOT_SUBJECT_ID}_seed{SEED}_results.csv"

MAX_EPOCH = 200
LAMDA = 1.0
PATIENCE = 10   # FAConformer's own published value -- confirmed unchanged via AskUserQuestion

criterion = nn.CrossEntropyLoss().to(DEVICE)


def make_model_and_optimizer(in_planes, window_length):
    model = FAConformer(
        data_name="DTU", in_planes=in_planes, out_planes_branch=64, out_planes_total=32,
        kernel_size=63, radix=1, patch_size=32, time_points=window_length,
        num_classes=2, depth_branch=2, num_heads_branch=2,
        depth_total=2, num_heads_total=2, num_bands=8, dim_feedforward=16,
    ).to(DEVICE)
    optimizer = Adam(params=model.parameters(), lr=0.0005, weight_decay=3e-4)
    return model, optimizer


def train_epoch(model, optimizer, train_loader):
    model.train()
    loss_sum, acc_sum, n = 0.0, 0, 0
    for data, label in train_loader:
        label = label.squeeze(-1).to(DEVICE).long()
        data = data.to(DEVICE)
        preds, *aux = model(data)
        loss = criterion(preds, label)
        loss_aux = sum(criterion(p, label) for p in aux) / len(aux)
        loss_all = loss + LAMDA * loss_aux
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        bs = label.size(0)
        loss_sum += loss_all.item() * bs
        acc_sum += (preds.argmax(1) == label).sum().item()
        n += bs
    return loss_sum / n, acc_sum / n


def evaluate(model, loader):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0, 0
    with torch.no_grad():
        for data, label in loader:
            label = label.squeeze(-1).to(DEVICE).long()
            data = data.to(DEVICE)
            preds, *aux = model(data)
            loss = criterion(preds, label)
            loss_aux = sum(criterion(p, label) for p in aux) / len(aux)
            loss_all = loss + LAMDA * loss_aux
            bs = label.size(0)
            loss_sum += loss_all.item() * bs
            acc_sum += (preds.argmax(1) == label).sum().item()
            n += bs
    return loss_sum / n, acc_sum / n


def train_one_fold(fold_idx, train_loader, valid_loader, test_loader, data_args):
    model, optimizer = make_model_and_optimizer(data_args.n_components, data_args.window_length)

    if fold_idx == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"FAConformer parameter count: {n_params:,}")

    best_valid = float("inf")
    best_state = None
    best_epoch = 0
    waiting = 0
    n_epochs_run = 0
    min_train_loss = float("inf")

    t_start = time_module.time()
    for epoch in tqdm(range(1, MAX_EPOCH + 1), desc=f"fold{fold_idx}", leave=False):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader)
        valid_loss, valid_acc = evaluate(model, valid_loader)
        n_epochs_run = epoch
        min_train_loss = min(min_train_loss, train_loss)

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            waiting = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
            if waiting > PATIENCE:
                break

    train_wall_seconds = time_module.time() - t_start
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader)

    print(f"Fold {fold_idx} | best_epoch={best_epoch}/{n_epochs_run} | "
          f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} min_train_loss={min_train_loss:.4f} | "
          f"{train_wall_seconds:.1f}s")

    return {
        "fold": fold_idx, "subject_id": PILOT_SUBJECT_ID, "seed": SEED,
        "best_epoch": best_epoch, "n_epochs_run": n_epochs_run,
        "test_loss": test_loss, "test_acc": test_acc, "min_train_loss": min_train_loss,
        "train_wall_seconds": train_wall_seconds,
    }


# %% [markdown]
# ## 7. Run all 8 SI folds for the pilot subject, incremental CSV writes

# %%
results = []
pilot_start = time_module.time()

for fold_idx, fold_raw in enumerate(si_folds_raw):
    tr_eeg_list = fold_raw["tr_eeg"]
    tr_label_list = fold_raw["tr_label"]
    te_eeg_list = fold_raw["te_eeg"]
    te_label_list = fold_raw["te_label"]

    # Carve validation from the training pool -- same convention as
    # kaggle_diag_stgcn_epoch_budget_test.py's build_partition.
    tr_eeg2, va_eeg, tr_label2, va_label = train_test_split(
        tr_eeg_list, tr_label_list, test_size=VAL_FRACTION, random_state=PILOT_SUBJECT_ID
    )

    train_eeg_tmajor = [trial.T.astype(np.float64) for trial in tr_eeg2]
    valid_eeg_tmajor = [trial.T.astype(np.float64) for trial in va_eeg]
    test_eeg_tmajor = [trial.T.astype(np.float64) for trial in te_eeg_list]
    train_label_int = [int(l) for l in tr_label2]
    valid_label_int = [int(l) for l in va_label]
    test_label_int = [int(l) for l in te_label_list]

    train_loader, valid_loader, test_loader, data_args = build_faconformer_si_loaders(
        train_eeg_tmajor, train_label_int, valid_eeg_tmajor, valid_label_int,
        test_eeg_tmajor, test_label_int, TIME_LEN
    )
    print(f"[fold {fold_idx}] train batches={len(train_loader)} valid batches={len(valid_loader)} "
          f"test batches={len(test_loader)}")

    row = train_one_fold(fold_idx, train_loader, valid_loader, test_loader, data_args)
    results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)   # rewritten after every fold -- crash-safe partial output
    print(f"[{fold_idx + 1}/{N_FOLDS}] flushed to {RESULTS_CSV}")

pilot_wall_seconds = time_module.time() - pilot_start

# %% [markdown]
# ## 8. Pilot summary

# %%
results_df = pd.DataFrame(results)
print(f"\n=== SI PORT PILOT SUMMARY (subject {PILOT_SUBJECT_ID}, seed {SEED}, {N_FOLDS} folds) ===")
print(results_df[["fold", "best_epoch", "n_epochs_run", "test_loss", "test_acc", "min_train_loss",
                   "train_wall_seconds"]].to_string(index=False))
print(f"\nMean test_acc: {results_df['test_acc'].mean():.4f}  Std: {results_df['test_acc'].std():.4f}")
print(f"Total pilot wall-clock: {pilot_wall_seconds:.1f}s ({pilot_wall_seconds/60:.1f} min)")

with open(OUT_DIR / f"si_pilot_subject{PILOT_SUBJECT_ID}_seed{SEED}_summary.json", "w") as f:
    json.dump({
        "subject_id": PILOT_SUBJECT_ID, "seed": SEED, "time_len": TIME_LEN, "n_folds": N_FOLDS,
        "mean_test_acc": results_df["test_acc"].mean(), "std_test_acc": results_df["test_acc"].std(),
        "total_pilot_seconds": pilot_wall_seconds,
    }, f, indent=2)

print(f"Written to {OUT_DIR}")
