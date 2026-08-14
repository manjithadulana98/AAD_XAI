# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FAConformer -- single-subject, single-seed pilot on our own DTU data
#
# FAConformer (external/FAConformer, added as a git submodule) is a
# benchmark codebase for auditory attention decoding. As committed, its
# `main.py` cannot run: 9 of 13 imported baseline models are missing from
# the repo, and its data loader expects a `<subject>_data_preproc.mat`
# format we don't have (we have AADNet's raw format instead).
#
# This notebook is a minimal, single-subject/single-seed pilot that:
# 1. Loads ONE subject's raw per-trial EEG via AADNet's own `DTUDataset`
#    (already validated throughout the ST-GCN work) -- NOT touching
#    FAConformer's `get_data_from_mat` file reader at all.
# 2. Adapts the resulting arrays (transpose channel-major -> time-major;
#    our label is already 0/1, so FAConformer's own `-1` step is skipped)
#    into exactly the shape their own `get_DTU_data` (FAConformer branch)
#    produces from that point onward -- sliding-window, CSP, per-band FFT
#    filtering, DataLoader construction are all reused verbatim from their
#    code, just fed our data instead of their file format.
# 3. Trains FAConformer itself (not the missing baselines) with their own
#    hyperparameters (Adam lr=5e-4, weight_decay=3e-4, band-wise auxiliary
#    loss lamda=1.0, max_epoch=200, patience=10), single subject, single
#    seed -- a "does it run and converge sensibly on our data" check, not
#    a full reproduction attempt (their exact original preprocessing
#    recipe for `_data_preproc.mat` isn't available to us).
#
# **Namespace note:** external/AADNet and external/FAConformer both have
# their own `utils` package. This notebook loads AADNet's data first, then
# explicitly clears the AADNet-related sys.path entries and sys.modules
# cache before switching to FAConformer's namespace, to avoid a collision.
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
# ## 3. Phase 1 (AADNet namespace) -- load ONE subject's raw per-trial data

# %%
import yaml
import numpy as np

PILOT_SUBJECT_ID = 0   # AADNet 0-indexed -> FAConformer's "S1"
SEED = 42
TIME_LEN = 2.0          # seconds -- FAConformer's own DTU default


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


# FAConformer's own run_experiment() calls set_global_seed() once, before any
# data loading -- the train/valid shuffle inside get_DTU_data relies on that
# already-seeded global numpy RNG stream. Replicated here for the same reason.
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

s_data = DTUDataset.loadData(aadnet_config, subject=PILOT_SUBJECT_ID)
print(f"Subject {PILOT_SUBJECT_ID}: {len(s_data)} trials loaded")
print(f"First trial eeg shape (channel-major, as AADNet stores it): {s_data[0]['eeg'].shape}")
print(f"Label values present: {sorted(set(int(t['label']) for t in s_data))}")

# Transpose channel-major (64, T) -> time-major (T, 64), matching what
# FAConformer's own get_data_from_mat produces. Our label is already 0/1
# (AADNet's attend_mf - 1); FAConformer's own pipeline does `- 1` on a raw
# 1/2 value, so we feed 0/1 directly and skip that step below.
eeg_data_list = [trial["eeg"].T.astype(np.float64) for trial in s_data]
event_data_list = [int(trial["label"]) for trial in s_data]
print(f"Adapted: {len(eeg_data_list)} trials, each shape {eeg_data_list[0].shape} (time, channels)")

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
# ## 5. Phase 2 (FAConformer namespace) -- adapt into their pipeline shape

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
    just fed our own arrays instead."""
    args = DotMap()
    args.fs = 64
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.eeg_channel = 64          # raw EEG channel count -- used only by sliding_window's reshape
    # CSP output components. AADNet's DTUDataset self-average-references each
    # subject's 64 selected channels against the mean of that SAME 64-channel
    # set, capping the TRUE rank at 63 for every subject, deterministically
    # (see the CSP reg='ledoit_wolf' note below). Requesting 64 components
    # lets MNE's internal rank estimator decide per-subject whether to
    # silently reduce to 63 -- confirmed (in the full 18-subject sweep) to be
    # inconsistent across subjects, crashing FAConformer's in_planes=64-
    # hardcoded first conv layer whenever a given subject's estimate lands on
    # 63. Fixing n_components at 63 unconditionally matches the true,
    # structural rank ceiling so CSP never needs to auto-reduce.
    args.csp_comp = 63

    # Their own get_DTU_data forces all trials into one uniform ndarray via
    # np.array(eeg_data)+vstack+reshape -- but that's only needed there to
    # slice off non-EEG columns (data.eeg has extra channels beyond 64).
    # Our AADNet-loaded trials are already exactly 64 channels, and
    # sliding_window() below processes each trial independently by its own
    # length, so we skip that forced-uniform step and pass trials through
    # as-is (per-trial length is allowed to vary by a few samples).
    lengths = [t.shape[0] for t in eeg_data_list]
    print(f"Per-trial lengths: min={min(lengths)} max={max(lengths)} (n_trials={len(lengths)})")
    event_data = np.array(event_data_list)      # already 0/1, no "-1" step (see note above)

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

    # FAConformer's own default is reg=None, but AADNet's DTUDataset re-references
    # each subject's 64 selected channels against the mean of that SAME 64-channel
    # set (dataset.py: ordinary_channels == selected_chs for DTU) -- a self-average
    # reference that forces sum_channels(eeg[t]) == 0 at every timepoint, capping
    # the true channel rank at 63. CSP's un-regularized GED needs a full-rank (64)
    # covariance and fails on this deterministically (LinAlgError: leading minor of
    # order 64 of B is not positive definite) on every subject, not just this one.
    # Ledoit-Wolf shrinkage regularizes the covariance so CSP can still fit.
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

    # Global np.random.shuffle (not a local RandomState) -- matches the
    # original exactly, and relies on set_global_seed() already having been
    # called once before data loading, same as their run_experiment().
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


train_loader, valid_loader, test_loader, data_args = build_faconformer_loaders(
    eeg_data_list, event_data_list, TIME_LEN
)
print(f"train batches={len(train_loader)}  valid batches={len(valid_loader)}  test batches={len(test_loader)}")

# %% [markdown]
# ## 6. Build FAConformer model + training loop (adapted from main.py's
#    train_model_FAConformer -- identical logic, just not importing main.py
#    itself since that fails on the 9 missing baseline model imports)

# %%
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from models.FAConformer import FAConformer

torch.manual_seed(SEED)
np.random.seed(SEED)

model_args = DotMap()
model_args.channel = data_args.csp_comp  # 63 -- must match build_faconformer_loaders' CSP output, not raw eeg_channel
model_args.output_size_branch = 64
model_args.output_size_total = 32
model_args.patch_size = 32
model_args.num_heads_branch = 2
model_args.depth_branch = 2
model_args.num_heads_total = 2
model_args.depth_total = 2
model_args.num_bands = 8
model_args.dim_feedforward = 16
model_args.time = data_args.window_length
model_args.lamda = 1.0
model_args.max_epoch = 200
model_args.hidden = 0.6
model_args.model = "FAConformer"

model = FAConformer(
    data_name="DTU", in_planes=model_args.channel,
    out_planes_branch=model_args.output_size_branch, out_planes_total=model_args.output_size_total,
    kernel_size=63, radix=1, patch_size=model_args.patch_size, time_points=model_args.time,
    num_classes=2, depth_branch=model_args.depth_branch, num_heads_branch=model_args.num_heads_branch,
    depth_total=model_args.depth_total, num_heads_total=model_args.num_heads_total,
    num_bands=model_args.num_bands, dim_feedforward=model_args.dim_feedforward,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"FAConformer parameter count: {n_params:,}")

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = Adam(params=model.parameters(), lr=0.0005, weight_decay=3e-4)


def train_epoch():
    model.train()
    loss_sum, acc_sum, n = 0.0, 0, 0
    for data, label in train_loader:
        label = label.squeeze(-1).to(DEVICE).long()
        data = data.to(DEVICE)
        preds, *aux = model(data)
        loss = criterion(preds, label)
        loss_aux = sum(criterion(p, label) for p in aux) / len(aux)
        loss_all = loss + model_args.lamda * loss_aux
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        bs = label.size(0)
        loss_sum += loss_all.item() * bs
        acc_sum += (preds.argmax(1) == label).sum().item()
        n += bs
    return loss_sum / n, acc_sum / n


def evaluate(loader):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0, 0
    with torch.no_grad():
        for data, label in loader:
            label = label.squeeze(-1).to(DEVICE).long()
            data = data.to(DEVICE)
            preds, *aux = model(data)
            loss = criterion(preds, label)
            loss_aux = sum(criterion(p, label) for p in aux) / len(aux)
            loss_all = loss + model_args.lamda * loss_aux
            bs = label.size(0)
            loss_sum += loss_all.item() * bs
            acc_sum += (preds.argmax(1) == label).sum().item()
            n += bs
    return loss_sum / n, acc_sum / n


# %% [markdown]
# ## 7. Train (their own patience=10 early stopping on validation loss)

# %%
import time as time_module

best_valid = float("inf")
best_state = None
best_epoch = 0
waiting = 0
history = []

t_start = time_module.time()
for epoch in tqdm(range(1, model_args.max_epoch + 1), desc="Training"):
    train_loss, train_acc = train_epoch()
    valid_loss, valid_acc = evaluate(valid_loader)
    history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                     "valid_loss": valid_loss, "valid_acc": valid_acc})
    print(f"Epoch {epoch:3d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
          f"Valid Loss {valid_loss:.4f} Acc {valid_acc:.4f}")

    if valid_loss < best_valid:
        best_valid = valid_loss
        best_epoch = epoch
        waiting = 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        waiting += 1
        if waiting > 10:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

t_total = time_module.time() - t_start
model.load_state_dict(best_state)
test_loss, test_acc = evaluate(test_loader)

print(f"\n=== PILOT RESULT (subject {PILOT_SUBJECT_ID}, seed {SEED}) ===")
print(f"Best epoch: {best_epoch} / {len(history)} run")
print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")
print(f"Total wall-clock: {t_total:.1f}s")

# %% [markdown]
# ## 8. Write output

# %%
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/kaggle/working/faconformer_pilot")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(history).to_csv(OUT_DIR / "pilot_history.csv", index=False)

import json
with open(OUT_DIR / "pilot_summary.json", "w") as f:
    json.dump({
        "subject_id": PILOT_SUBJECT_ID, "seed": SEED, "time_len": TIME_LEN,
        "n_params": n_params, "best_epoch": best_epoch, "n_epochs_run": len(history),
        "test_loss": test_loss, "test_acc": test_acc, "total_seconds": t_total,
    }, f, indent=2)

print(f"Written to {OUT_DIR}")
