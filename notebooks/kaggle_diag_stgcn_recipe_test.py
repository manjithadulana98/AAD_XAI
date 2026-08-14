# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ST-GCN Phase 2 diagnostic -- DHGCN's training-recipe ingredients, bundled,
# on the EXISTING ST-GCN (meanvar) model
#
# Closing gate on "does training recipe explain the Phase 2 gap," not a new
# round of single-variable isolation. Two more consequential interventions
# already returned near-null results on this question: the full NSR
# procedure swap, and the epoch-budget-alone extension
# (`kaggle_diag_stgcn_epoch_budget_test.py` -- recovery on 3/5 valid folds
# was exactly zero, 2/5 modest; see that notebook and the project memory
# entry for the fold-(3,0) early-stopping-artifact lesson this test
# incorporates via a longer patience). Given that pattern, the prior that
# DHGCN's three remaining recipe ingredients individually explain the gap is
# low enough that bundling them into one test is the right call -- if this
# moves nothing, isolating which of the three "did it" isn't worth the
# extra cost; if it moves a lot, THEN isolate.
#
# Bundled changes vs. the validated meanvar baseline, everything else
# (architecture, adjacency, data pipeline, SI split, 6 folds, seed=42)
# IDENTICAL:
# - Optimizer: AdamW(lr=0.003) instead of Adam(lr=1e-3) -- DHGCN's own
#   `main.py::initiate()` values.
# - Scheduler: CosineAnnealingWarmRestarts(T_0=20, T_mult=2, eta_min=3e-4)
#   -- confirmed from `train_model_DHGCN` to step once per EPOCH (integer,
#   not per-batch fractional stepping).
# - Loss: LabelSmoothing(smoothing=0.1) instead of plain CrossEntropyLoss --
#   copied verbatim from `main.py`'s own `LabelSmoothing` class.
# - Patience: 20 (not DHGCN's own 50, and not the epoch-budget-test's 12 --
#   raised specifically because that test's fold (3,0) showed patience=12
#   can cut a run short before a real, later improvement; 20 is toward the
#   generous end while still being far cheaper than DHGCN's native 50).
# - Max epochs: 100 (raised from 80 so patience=20 has genuine room without
#   an artificial ceiling, given folds (6,0)/(15,0) in the epoch-budget-test
#   already moved their best epoch out to 48/62).
#
# Partition-hash assertion (from `kaggle_diag_stgcn_seed_variance.py`)
# reused here for the same reason: guarantee the SI-fold + val-split
# partition is bit-identical to every prior diagnostic on these 6 folds,
# so any accuracy change is attributable to the recipe change alone.
#
# **Kaggle setup requirements:** Internet enabled, GPU accelerator,
# `dulanamanjitha/aad-xai-artifacts` dataset attached. No Kaggle Secret needed.

# %% [markdown]
# ## 1. Clone repository + install dependencies

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

for extra in ("src", "external/AADNet", "stgcn"):
    p = os.path.join(REPO_DIR, extra)
    if p not in sys.path:
        sys.path.insert(0, p)

print("Setup done.")

# %% [markdown]
# ## 2. GPU sanity check (with fast compatibility probe)

# %%
import torch
import torch.nn as nn

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

# %% [markdown]
# ## 3. Configuration -- meanvar pooling, DHGCN's recipe, longer patience

# %%
from pathlib import Path
import yaml
import json
import time
import hashlib

MAX_EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 32
MAX_TRAIN_WINDOWS_PER_EPOCH = 2000
N_KERNELS = 5
FC_HIDDEN = 8
VAL_FRACTION = 0.2
SEED = 42
POOLING = "meanvar"

ADAMW_LR = 0.003
COSINE_T0 = 20
COSINE_TMULT = 2
COSINE_ETA_MIN = ADAMW_LR / 10
LABEL_SMOOTHING = 0.1

DIAG_FOLDS = [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0)]

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
raw_config["dataset"]["training_window"] = 1
raw_config["dataset"]["duplicate"] = False

from utils.config import Config
aadnet_config = Config.load_config(raw_config)

OUT_DIR = Path("/kaggle/working/stgcn_diag_recipe_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np

# %% [markdown]
# ## 4. Fixed adjacency -- IDENTICAL to every prior ST-GCN run

# %%
from adjacency import load_montage, build_adjacency_distance

montage = load_montage(os.path.join(REPO_DIR, "config", "aadnet_dtu_channel_montage.csv"))
ADJACENCY = build_adjacency_distance(montage, k=6)

# %% [markdown]
# ## 5. Model -- IDENTICAL STGCNVariant("meanvar"), no architecture change

# %%
from model import GraphConvKW


class STGCNVariant(nn.Module):
    def __init__(self, adjacency, n_kernels, fc_hidden, n_channels, T, pooling, dropout=0.3):
        super().__init__()
        self.graph_conv = GraphConvKW(adjacency, n_kernels=n_kernels)
        self.pooling = pooling
        flat_dim = n_kernels * n_channels * (2 if pooling == "meanvar" else 1)
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, 2)

    def forward(self, x):
        B = x.size(0)
        f = self.graph_conv(x)
        C, N, T = f.shape[1], f.shape[2], f.shape[3]
        f = f.reshape(B, C * N, T)
        if self.pooling == "mean":
            pooled = f.mean(dim=-1)
        else:
            mean = f.mean(dim=-1)
            var = f.var(dim=-1, unbiased=False)
            pooled = torch.cat([mean, var], dim=-1)
        h = self.act(self.bn1(self.fc1(pooled)))
        h = self.dropout(h)
        return self.fc2(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(pooling, T, seed):
    torch.manual_seed(seed)
    m = STGCNVariant(ADJACENCY, n_kernels=N_KERNELS, fc_hidden=FC_HIDDEN,
                      n_channels=64, T=T, pooling=pooling, dropout=0.3).to(DEVICE)
    return m


# %% [markdown]
# ## 6. DHGCN's LabelSmoothing loss -- verbatim from FAConformer's main.py

# %%
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# %% [markdown]
# ## 7. Partition construction + hash assertion (from seed_variance.py --
#    guarantees this test's 6 folds are bit-identical to every prior
#    diagnostic on them)

# %%
from sklearn.model_selection import train_test_split
from aadnet.dataset import DTUDataset


def hash_partition(tr_label, va_label, te_label):
    payload = repr((
        [int(x) for x in tr_label], [int(x) for x in va_label], [int(x) for x in te_label]
    )).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_partition(subject_id, fold):
    crossSIData = DTUDataset.createSICrossValidation(subject_id, aadnet_config)
    tr_split, te_split = crossSIData[fold]
    tr_eeg, tr_aud, tr_label = tr_split
    te_eeg, te_aud, te_label = te_split
    tr_eeg2, va_eeg, tr_aud2, va_aud, tr_label2, va_label = train_test_split(
        tr_eeg, tr_aud, tr_label, test_size=VAL_FRACTION, random_state=subject_id
    )
    h = hash_partition(tr_label2, va_label, te_label)
    return (tr_eeg2, tr_aud2, tr_label2), (va_eeg, va_aud, va_label), (te_eeg, te_aud, te_label), h


# Reference hashes established by kaggle_diag_stgcn_seed_variance.py's own
# first-run-per-fold construction -- re-verified here, not re-derived, so
# any mismatch fails loudly rather than silently drifting.
REFERENCE_PARTITION_HASHES = {
    (0, 0): "02289e5dcacea68cf09571ea66b132fd51d1898c0d9ab438bc1aac286cdda6df",
    (3, 0): "ba9b3e0fb3744456fbd3a3a47ba0d177f44a2623046d5da334f2d474a5afe360",
    (6, 0): "39e934f01c306d9774071a87c11d6e186e3620730d09c4b97e1550d0cb6fa07f",
    (9, 0): "632d423d99b99290bf9adfc5f66899613d5443141536dc244be97f9c165d3726",
    (12, 0): "57a8c22edcc08510d8c46de6b7796928ec42bacce99ad04ef38ae740bdabd303",
    (15, 0): "811aeded7d1f58a1d380024c524b76dbf3eb2e32a267864d21edd4a502ced8d7",
}

# %% [markdown]
# ## 8. Fold-training loop -- AdamW + CosineAnnealingWarmRestarts +
#    LabelSmoothing, patience=20 on validation loss, full per-epoch history
#    retained for direct comparison against the epoch-budget-test's own CSV.

# %%
def make_train_loader(ds, batch_size):
    sampler = torch.utils.data.RandomSampler(ds, replacement=True,
                                              num_samples=min(MAX_TRAIN_WINDOWS_PER_EPOCH, len(ds) * 5))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)


def run_fold(subject_id, fold, tr_parts, va_parts, te_parts, ref_hash):
    tr_eeg2, tr_aud2, tr_label2 = tr_parts
    va_eeg, va_aud, va_label = va_parts
    te_eeg, te_aud, te_label = te_parts

    this_hash = hash_partition(tr_label2, va_label, te_label)
    assert this_hash == ref_hash, (
        f"PARTITION DRIFT DETECTED for subject {subject_id} fold {fold}: "
        f"hash {this_hash} != reference {ref_hash}"
    )

    train_ds = DTUDataset(aadnet_config, tr_eeg2, tr_aud2, tr_label2)
    valid_ds = DTUDataset(aadnet_config, va_eeg, va_aud, va_label)
    test_ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)
    T = train_ds[0][0].shape[-1]

    model = build_model(POOLING, T, SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=ADAMW_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=COSINE_T0, T_mult=COSINE_TMULT, eta_min=COSINE_ETA_MIN
    )
    loss_fn = LabelSmoothing(smoothing=LABEL_SMOOTHING)

    torch.manual_seed(SEED)
    train_loader = make_train_loader(train_ds, BATCH_SIZE)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    epoch_rows = []
    best_val_loss = None
    best_idx = -1
    waiting = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for eeg, _audio, y in train_loader:
            eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
            opt.zero_grad()
            logits = model(eeg)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * y.size(0)
            tr_n += y.size(0)
        scheduler.step()  # once per epoch -- confirmed against DHGCN's own train_model_DHGCN

        model.eval()
        with torch.no_grad():
            va_loss_sum, va_n = 0.0, 0
            for eeg, _audio, y in valid_loader:
                eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
                logits = model(eeg)
                loss = loss_fn(logits, y)
                va_loss_sum += loss.item() * y.size(0)
                va_n += y.size(0)

            te_correct, te_n = 0, 0
            for eeg, _audio, y in test_loader:
                eeg, y = eeg.to(DEVICE).float(), y.to(DEVICE).long()
                logits = model(eeg)
                te_correct += (logits.argmax(1) == y).sum().item()
                te_n += y.size(0)

        val_loss = va_loss_sum / max(va_n, 1)
        epoch_rows.append({
            "epoch": epoch,
            "train_loss": tr_loss_sum / max(tr_n, 1),
            "valid_loss": val_loss,
            "test_acc": te_correct / max(te_n, 1),
            "lr": opt.param_groups[0]["lr"],
        })

        if best_val_loss is None or val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_idx = epoch
            waiting = 0
        else:
            waiting += 1
        if waiting > PATIENCE:
            break

    n_epochs_run = len(epoch_rows)
    return {
        "subject_id": subject_id, "fold": fold,
        "n_epochs_run": n_epochs_run,
        "stopped_early": n_epochs_run < MAX_EPOCHS,
        "best_val_epoch": epoch_rows[best_idx]["epoch"],
        "best_val_loss": epoch_rows[best_idx]["valid_loss"],
        "test_at_best_val": epoch_rows[best_idx]["test_acc"],
        "min_train_loss": min(r["train_loss"] for r in epoch_rows),
        "final_train_loss": epoch_rows[-1]["train_loss"],
    }, epoch_rows


# %% [markdown]
# ## 9. Run all 6 folds

# %%
t_start = time.time()
summary_rows = []
all_epoch_rows = []
for subject_id, fold in DIAG_FOLDS:
    t0 = time.time()
    tr_parts, va_parts, te_parts, this_hash = build_partition(subject_id, fold)
    ref_hash = REFERENCE_PARTITION_HASHES[(subject_id, fold)]
    assert this_hash == ref_hash, (
        f"PARTITION DRIFT at construction for subject {subject_id} fold {fold}: "
        f"hash {this_hash} != reference {ref_hash}"
    )
    result, epoch_rows = run_fold(subject_id, fold, tr_parts, va_parts, te_parts, ref_hash)
    summary_rows.append(result)
    for r in epoch_rows:
        r2 = dict(r)
        r2["subject_id"] = subject_id
        r2["fold"] = fold
        all_epoch_rows.append(r2)
    print(f"[{time.time()-t_start:6.0f}s] subject {subject_id} fold {fold} ({time.time()-t0:.0f}s): "
          f"ran {result['n_epochs_run']} epochs (stopped_early={result['stopped_early']}), "
          f"best_val_epoch={result['best_val_epoch']}, min_train_loss={result['min_train_loss']:.4f}, "
          f"test@best_val={result['test_at_best_val']:.3f}")

t_total = time.time() - t_start
print(f"\nTotal wall-clock: {t_total:.1f}s ({t_total/60:.1f} min)")

# %% [markdown]
# ## 10. Write output

# %%
import pandas as pd

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "recipe_test_summary.csv", index=False)

epoch_df = pd.DataFrame(all_epoch_rows)
epoch_df.to_csv(OUT_DIR / "recipe_test_per_epoch.csv", index=False)

print(summary_df.to_string(index=False))
