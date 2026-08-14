# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FAConformer -- full DTU sweep, one seed x all 18 subjects per kernel
#
# Scaled-up version of `kaggle_faconformer_pilot.py` (which validated, on
# subject S1/seed 42, that FAConformer trains sensibly on our AADNet-sourced
# raw DTU data once CSP is regularized with Ledoit-Wolf shrinkage). This
# notebook runs FAConformer's own full evaluation protocol for **one seed**
# across **all 18 DTU subjects**; the plan is to push this same script 5
# times with `SEED` set to 41, 42, 43, 44, 45 (FAConformer's own
# `seed_list = list(range(41,46))` from `main.py::main()`), as 5 separate
# Kaggle kernels -- see the project's plan file for the full rationale
# (reload-overhead elimination, per-kernel GPU-time budget, crash safety).
#
# Two structural differences versus the pilot, both required for a correct
# multi-subject sweep (not just "loop the pilot 18 times"):
# 1. **Bulk data load.** `DTUDataset.loadData(config, subject=<int>)`
#    discards the class-level cache every call (`dataset.py` line ~452) --
#    looping that per subject would redo ~192s of MNE preprocessing 18
#    times. Calling `loadData(config, subject=None)` instead loads all 18
#    subjects ONCE into a persistent cache; we capture that into a plain
#    Python list before the AADNet namespace cleanup.
# 2. **Fresh model per subject, seed set once per kernel (not per subject).**
#    FAConformer's own `run_experiment(seed, ...)` calls `set_global_seed`
#    once, before its subject loop, so each subject's model-init and
#    data-shuffle draws from the seed's continuing RNG stream rather than
#    an identical reset. The pilot (single subject) harmlessly re-seeded
#    right before model creation; this sweep must NOT do that, or all 18
#    subjects' models would start from identical initial weights.
#
# Every subject's result is flushed to the output CSV immediately after
# that subject finishes, so a Kaggle timeout/crash partway through still
# leaves the completed subjects' results downloadable.
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

# external/FAConformer is a git submodule -- a plain clone leaves it empty.
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
# FAConformer-specific deps not already in requirements.txt:
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
# ## 3. Phase 1 (AADNet namespace) -- bulk-load ALL 18 subjects once

# %%
import yaml
import numpy as np

SEED = 41               # <-- the one line changed across the 5 kernel pushes (41, 42, 43, 44, 45)
TIME_LEN = 2.0           # seconds -- FAConformer's own DTU default
N_SUBJECTS = 18


def set_global_seed(seed):
    """Verbatim from FAConformer's main.py::set_global_seed."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Called ONCE per kernel (= once per seed), before any data loading or
# subject loop -- matches FAConformer's own run_experiment() exactly.
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

import time as time_module
_load_start = time_module.time()
DTUDataset.loadData(aadnet_config, subject=None)   # subject=None -> loads all 18, caches persistently
all_subjects_raw = [list(s_data) for s_data in DTUDataset.all_data]
print(f"Bulk-loaded {len(all_subjects_raw)} subjects in {time_module.time() - _load_start:.1f}s")
assert len(all_subjects_raw) == N_SUBJECTS, f"Expected {N_SUBJECTS} subjects, got {len(all_subjects_raw)}"
for s in range(N_SUBJECTS):
    print(f"  Subject {s}: {len(all_subjects_raw[s])} trials, "
          f"labels present: {sorted(set(int(t['label']) for t in all_subjects_raw[s]))}")

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
# ## 5. Phase 2 (FAConformer namespace) -- data adapter (verbatim from the pilot)

# %%
faconformer_dir = os.path.join(REPO_DIR, "external", "FAConformer")
sys.path.insert(0, faconformer_dir)
os.chdir(faconformer_dir)  # FAConformer's own relative imports assume this

import math
from dotmap import DotMap
from mne.decoding import CSP
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


def sliding_window(eeg_datas, labels, window_size, overlap, eeg_channel):
    stride = int(window_size * (1 - overlap))
    train_eeg, test_eeg, train_label, test_label = [], [], [], []
    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows, new_label = [], []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            windows.append(eeg[i:i + window_size, :])
            new_label.append(label)
        train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
        test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
        train_label.append(np.array(new_label)[:int(len(windows) * 0.9)])
        test_label.append(np.array(new_label)[int(len(windows) * 0.9):])
    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)
    return train_eeg, test_eeg, train_label, test_label


def build_faconformer_loaders(eeg_data_list, event_data_list, time_len):
    """Adapted from FAConformer's utils/data_loader.py::get_DTU_data,
    FAConformer branch only -- identical logic from the point where their
    own file-reading closure would have returned (eeg_data, event_data),
    just fed our own arrays instead. Verbatim from the pilot."""
    args = DotMap()
    args.fs = 64
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.eeg_channel = 64          # raw EEG channel count -- used only by sliding_window's reshape
    # CSP output components. AADNet's DTUDataset self-average-references each
    # subject's 64 selected channels against the mean of that SAME 64-channel
    # set (dataset.py: ordinary_channels == selected_chs for DTU), which forces
    # sum_channels(eeg[t]) == 0 at every timepoint and caps the TRUE rank at 63
    # for every subject, deterministically. Requesting 64 components let MNE's
    # internal rank estimator decide per-subject whether to silently reduce to
    # 63 -- which it did inconsistently (S1-S10 got 64, S11 got 63), crashing
    # FAConformer's in_planes=64-hardcoded first conv layer on the mismatch.
    # Fixing n_components at 63 for every subject matches the true, structural
    # rank ceiling so CSP never needs to auto-reduce again.
    args.csp_comp = 63

    event_data = np.array(event_data_list)      # already 0/1, no "-1" step

    train_data, test_data, train_label, test_label = sliding_window(
        eeg_data_list, event_data, args.window_length, args.overlap, args.eeg_channel
    )

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

    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)

    # See kaggle_faconformer_pilot.py for why reg='ledoit_wolf' is needed
    # (AADNet's self-average-referencing caps the true channel rank at 63,
    # which crashes FAConformer's own un-regularized reg=None CSP call).
    csp = CSP(n_components=args.csp_comp, reg='ledoit_wolf', log=None, cov_est='concat',
              transform_into='csp_space', norm_trace=True)
    train_label = np.squeeze(train_label)
    train_data = csp.fit_transform(train_data, train_label)
    test_data = csp.transform(test_data)
    train_label = train_label.reshape(-1, 1)

    train_bands = [filter_signal_by_fft(train_data, lo, hi, args.window_length) for lo, hi in points]
    test_bands = [filter_signal_by_fft(test_data, lo, hi, args.window_length) for lo, hi in points]

    args.n_test = len(test_label)
    args.n_train = len(train_label) - args.n_test

    # Global np.random.shuffle -- relies on set_global_seed() already having
    # been called once at the top of this kernel, and continuing to advance
    # naturally across subjects (not reset per subject).
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)

    train_label = train_label[indices]
    train_bands = [b[indices] for b in train_bands]

    valid_label = train_label[args.n_train:]
    train_label = train_label[:args.n_train]
    valid_bands = [b[args.n_train:] for b in train_bands]
    train_bands = [b[:args.n_train] for b in train_bands]

    train_data = np.stack(train_bands, axis=1)
    valid_data = np.stack(valid_bands, axis=1)
    test_data = np.stack(test_bands, axis=1)

    train_loader = DataLoader(CustomDatasets(train_data, train_label), batch_size=args.batch_size,
                               drop_last=True, pin_memory=True)
    valid_loader = DataLoader(CustomDatasets(valid_data, valid_label), batch_size=args.batch_size,
                               drop_last=True, pin_memory=True)
    test_loader = DataLoader(CustomDatasets(test_data, test_label), batch_size=args.batch_size,
                              drop_last=True, pin_memory=True)
    return train_loader, valid_loader, test_loader, args


# %% [markdown]
# ## 6. Per-subject train/eval (fresh model each subject, verbatim training
#    logic from the pilot's train_epoch/evaluate/early-stopping)

# %%
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from models.FAConformer import FAConformer
import pandas as pd
from pathlib import Path
import json

OUT_DIR = Path("/kaggle/working/faconformer_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / f"DTU_FAConformer_{TIME_LEN:g}s_seed{SEED}_results.csv"

MAX_EPOCH = 200
LAMDA = 1.0
PATIENCE = 10


def make_model_and_optimizer(in_planes, window_length):
    model = FAConformer(
        data_name="DTU", in_planes=in_planes, out_planes_branch=64, out_planes_total=32,
        kernel_size=63, radix=1, patch_size=32, time_points=window_length,
        num_classes=2, depth_branch=2, num_heads_branch=2,
        depth_total=2, num_heads_total=2, num_bands=8, dim_feedforward=16,
    ).to(DEVICE)
    optimizer = Adam(params=model.parameters(), lr=0.0005, weight_decay=3e-4)
    return model, optimizer


criterion = nn.CrossEntropyLoss().to(DEVICE)


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


def train_one_subject(subject_idx, eeg_data_list, event_data_list):
    train_loader, valid_loader, test_loader, data_args = build_faconformer_loaders(
        eeg_data_list, event_data_list, TIME_LEN
    )
    model, optimizer = make_model_and_optimizer(data_args.csp_comp, data_args.window_length)

    if subject_idx == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"FAConformer parameter count: {n_params:,}")

    best_valid = float("inf")
    best_state = None
    best_epoch = 0
    waiting = 0
    n_epochs_run = 0

    t_start = time_module.time()
    for epoch in tqdm(range(1, MAX_EPOCH + 1), desc=f"S{subject_idx + 1}", leave=False):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader)
        valid_loss, valid_acc = evaluate(model, valid_loader)
        n_epochs_run = epoch

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

    print(f"Subject {subject_idx} (S{subject_idx + 1}) | best_epoch={best_epoch}/{n_epochs_run} | "
          f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | {train_wall_seconds:.1f}s")

    return {
        "subject": subject_idx, "seed": SEED, "best_epoch": best_epoch,
        "n_epochs_run": n_epochs_run, "test_loss": test_loss, "test_acc": test_acc,
        "train_wall_seconds": train_wall_seconds,
    }


# %% [markdown]
# ## 7. Run the sweep: this seed, all 18 subjects, incremental CSV writes

# %%
results = []
sweep_start = time_module.time()

for subject_idx in range(N_SUBJECTS):
    s_data = all_subjects_raw[subject_idx]
    eeg_data_list = [trial["eeg"].T.astype(np.float64) for trial in s_data]
    event_data_list = [int(trial["label"]) for trial in s_data]

    row = train_one_subject(subject_idx, eeg_data_list, event_data_list)
    results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)   # rewritten after every subject -- crash-safe partial output
    print(f"[{subject_idx + 1}/{N_SUBJECTS}] flushed to {RESULTS_CSV}")

sweep_wall_seconds = time_module.time() - sweep_start

# %% [markdown]
# ## 8. Per-seed summary

# %%
results_df = pd.DataFrame(results)
print(f"\n=== SEED {SEED} SUMMARY ({N_SUBJECTS} subjects) ===")
print(results_df[["subject", "best_epoch", "test_loss", "test_acc", "train_wall_seconds"]].to_string(index=False))
print(f"\nMean test_acc: {results_df['test_acc'].mean():.4f}  Std: {results_df['test_acc'].std():.4f}")
print(f"Total sweep wall-clock: {sweep_wall_seconds:.1f}s")

with open(OUT_DIR / f"faconformer_sweep_seed{SEED}_summary.json", "w") as f:
    json.dump({
        "seed": SEED, "time_len": TIME_LEN, "n_subjects": N_SUBJECTS,
        "mean_test_acc": results_df["test_acc"].mean(), "std_test_acc": results_df["test_acc"].std(),
        "total_sweep_seconds": sweep_wall_seconds,
    }, f, indent=2)

print(f"Written to {OUT_DIR}")
