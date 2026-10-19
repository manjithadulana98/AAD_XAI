# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # StimulusGAT -- Gradient & Attention Instrumentation on Real DTU Data (Kaggle)
#
# Two hypotheses tested and falsified in a row:
# - FiLM-init fix (v3): still stuck at chance on full 18-fold LOSO (0.4952 +/- 0.0261)
# - Per-window EEG standardization: train accuracy STILL pinned at chance
#   (best 0.5120, train_loss ~0.693 throughout) on the same 6,000-window
#   real training population that produces this exact pattern every time.
#
# Both were "guess a fix, check if accuracy improves" -- informative by
# elimination, but neither told us WHY the model can't learn. This notebook
# switches strategy: instrument a real training run directly, so we can see
# the actual mechanism instead of inferring it from accuracy curves.
#
# Logged every epoch:
# 1. **Per-submodule gradient norm** (temporal_encoder, frequency_encoder,
#    node_fusion, speech_encoder, gat, readout) -- reveals vanishing/exploding
#    gradients and which submodule(s), if any, are getting no usable signal.
# 2. **Attention weight entropy** vs. the reference entropy of a uniform
#    distribution over each node's actual neighbor count -- reveals whether
#    attention is collapsing to uniform (contributing nothing
#    discriminative) or to a degenerate near-one-hot pattern.
# 3. **Per-submodule output activation stats** (mean, std) -- reveals
#    saturation or numerical blow-up in the forward pass itself.
#
# **Kaggle setup requirements**
# - Enable Internet, attach `dulanamanjitha/aad-xai-artifacts`
# - GPU recommended

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
print(f"aad_xai importable from: {aad_xai.__file__}")

# %% [markdown]
# ## 2. Resolve the DTU dataset path, GPU check

# %%
from pathlib import Path
import torch

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, (
    "DTU dataset not found. Attach 'dulanamanjitha/aad-xai-artifacts'. "
    "Tried: " + ", ".join(DTU_KAGGLE_ROOT_CANDIDATES)
)
print(f"DTU dataset: {DTU_ROOT}")
print(f"CUDA available: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 3. Load data (same 14-subject pool as the train-vs-test-gap diagnostic,
# for direct comparability), with EEG standardization enabled

# %%
from dataclasses import asdict

from aad_xai.config import RunConfig, PreprocessConfig, TrainConfig, SplitConfig, WindowConfig
from aad_xai.data.base import BaseDataset
from aad_xai.data.dtu_dataset import DTUDataset
from aad_xai.train import _prepare_data

N_SUBJECTS = 14
WINDOW_S = 5.0
SEED = 42
EPOCHS = 20  # enough to see the instability pattern observed around epochs 19-28 previously
LR = 1e-2
BATCH_SIZE = 64

_all_subject_ids = sorted(p.stem for p in (Path(DTU_ROOT) / "eeg_new").glob("S*.mat"))
diag_subjects = _all_subject_ids[:N_SUBJECTS]
print(f"Using {len(diag_subjects)} of {len(_all_subject_ids)} subjects: {diag_subjects}")


class _SubjectSubset(BaseDataset):
    def __init__(self, inner, subjects):
        self.inner = inner
        self.subjects = set(subjects)
    def trials(self):
        return (t for t in self.inner.trials() if t.subject_id in self.subjects)


cfg = RunConfig(
    preprocess=PreprocessConfig(),
    window=WindowConfig(),
    split=SplitConfig(seed=SEED),
    train=TrainConfig(model="sgat", epochs=EPOCHS, num_seeds=1, device=str(DEVICE), lr=LR),
    dataset="dtu",
    dataset_root=DTU_ROOT,
    output_dir="/kaggle/working/results_sgat_instrumentation",
)

_full_dtu = DTUDataset(root=DTU_ROOT, load_audio=True, preprocess=cfg.preprocess)
dataset = _SubjectSubset(_full_dtu, diag_subjects)
trials, split, ds_train, ds_val, ds_test = _prepare_data(dataset, cfg, WINDOW_S)
print(f"Windows: train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")
print(f"Split -- test subject: {split.test}")

# %% [markdown]
# ## 4. Instrumented training loop
#
# Hooks register on each top-level submodule to capture output activation
# stats on every forward pass; gradient norms are read directly from
# `.parameters()` after `.backward()`, before `optimizer.step()`.

# %%
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from aad_xai.models.sgat import StimulusGAT
from aad_xai.utils.seed import seed_everything

seed_everything(SEED)
n_channels = ds_train[0][0].shape[0]
model = StimulusGAT(n_channels=n_channels, sfreq=cfg.preprocess.sfreq_out).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg.train.weight_decay)
loss_fn = nn.CrossEntropyLoss()

SUBMODULE_NAMES = [name for name, _ in model.named_children()]
print(f"Top-level submodules: {SUBMODULE_NAMES}")

# --- forward-activation-stats hooks ---
activation_stats = {}

def _make_hook(name):
    def hook(module, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(t):
            activation_stats[name] = (t.detach().mean().item(), t.detach().std().item())
    return hook

for name, child in model.named_children():
    child.register_forward_hook(_make_hook(name))

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Fixed batch for attention-entropy tracking (same batch every epoch, for a clean trend).
_fixed_eeg, _fixed_env, _fixed_labels = next(iter(dl_val))
_fixed_eeg, _fixed_env = _fixed_eeg.to(DEVICE), _fixed_env.to(DEVICE)

adj_mask = model.gat.adj_mask  # (N, N) bool
neighbor_counts = adj_mask.float().sum(dim=-1)  # (N,) -- includes self-loop
max_entropy_per_node = torch.log(neighbor_counts.clamp(min=1.0))  # uniform-over-neighbors entropy
reference_max_entropy = max_entropy_per_node.mean().item()
print(f"Reference max attention entropy (uniform over each node's own neighbor count): {reference_max_entropy:.4f}")


def _eval_acc(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, envb, yb in loader:
            xb, envb = xb.to(DEVICE), envb.to(DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
            logits = model(xb, envb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / max(total, 1)


def _attention_entropy():
    model.eval()
    with torch.no_grad():
        _, attn = model(_fixed_eeg, _fixed_env, return_attention_weights=True)
        attn_a = attn["attn_a"]  # (B, heads, N, N)
        eps = 1e-12
        row_entropy = -(attn_a * (attn_a + eps).log()).sum(dim=-1)  # (B, heads, N)
        return row_entropy.mean().item()


history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    grad_norm_sums = {name: 0.0 for name in SUBMODULE_NAMES}
    grad_norm_counts = 0
    train_correct, train_total, train_loss_sum = 0, 0, 0.0
    first_batch_activation_stats = None

    for batch_idx, (xb, envb, yb) in enumerate(dl_train):
        xb, envb = xb.to(DEVICE), envb.to(DEVICE)
        yb = torch.as_tensor(yb, dtype=torch.long, device=DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(xb, envb)
        if batch_idx == 0:
            # Snapshot real training-time activations before any later eval
            # forward pass (which also fires these hooks) overwrites them.
            first_batch_activation_stats = dict(activation_stats)
        loss = loss_fn(logits, yb)
        loss.backward()

        for name, child in model.named_children():
            total_sq = 0.0
            for p in child.parameters():
                if p.grad is not None:
                    total_sq += p.grad.detach().pow(2).sum().item()
            grad_norm_sums[name] += total_sq ** 0.5
        grad_norm_counts += 1

        opt.step()
        train_loss_sum += loss.item() * xb.size(0)
        train_correct += (logits.argmax(1) == yb).sum().item()
        train_total += xb.size(0)

    avg_grad_norms = {name: grad_norm_sums[name] / max(grad_norm_counts, 1) for name in SUBMODULE_NAMES}
    train_acc = train_correct / max(train_total, 1)
    train_loss = train_loss_sum / max(train_total, 1)
    val_acc = _eval_acc(dl_val)
    test_acc = _eval_acc(dl_test)
    attn_entropy = _attention_entropy()

    history.append({
        "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
        "val_acc": val_acc, "test_acc": test_acc,
        "grad_norms": avg_grad_norms, "attn_entropy": attn_entropy,
        "activation_stats": first_batch_activation_stats,
    })

    grad_str = "  ".join(f"{n}={avg_grad_norms[n]:.2e}" for n in SUBMODULE_NAMES)
    print(f"epoch {epoch:>3}: loss={train_loss:.4f} train_acc={train_acc:.4f} "
          f"val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
          f"attn_entropy={attn_entropy:.4f}/{reference_max_entropy:.4f}")
    print(f"    grad_norms: {grad_str}")
    act_str = "  ".join(f"{n}=(mean={m:.3f},std={s:.3f})" for n, (m, s) in first_batch_activation_stats.items())
    print(f"    activations (1st train batch): {act_str}")

# %% [markdown]
# ## 5. Summary

# %%
import json

print(f"\n{'epoch':>5} {'train_acc':>10} {'test_acc':>9} {'attn_H':>8}  " +
      "  ".join(f"{n:>14}" for n in SUBMODULE_NAMES))
for h in history:
    gn = "  ".join(f"{h['grad_norms'][n]:>14.2e}" for n in SUBMODULE_NAMES)
    print(f"{h['epoch']:>5} {h['train_acc']:>10.4f} {h['test_acc']:>9.4f} {h['attn_entropy']:>8.4f}  {gn}")

Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
with open(Path(cfg.output_dir) / "instrumentation_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"\nFull history written to {cfg.output_dir}/instrumentation_history.json")

# %% [markdown]
# ## 6. Decision
#
# Read this alongside the printed per-epoch table above -- the narrative
# below is a rough guide, not a substitute for looking at the actual numbers.

# %%
import numpy as np

final_grad_norms = history[-1]["grad_norms"]
min_grad_module = min(final_grad_norms, key=final_grad_norms.get)
max_grad_module = max(final_grad_norms, key=final_grad_norms.get)
final_entropy = history[-1]["attn_entropy"]

print("Decision:")
print(f"  Smallest final gradient norm: {min_grad_module} ({final_grad_norms[min_grad_module]:.2e})")
print(f"  Largest final gradient norm:  {max_grad_module} ({final_grad_norms[max_grad_module]:.2e})")
print(f"  Attention entropy: {final_entropy:.4f} vs. reference max (uniform) {reference_max_entropy:.4f} "
      f"({100*final_entropy/reference_max_entropy:.0f}% of max)")

if final_entropy / reference_max_entropy > 0.95:
    print("  -> Attention is essentially UNIFORM over neighbors -- the GAT mechanism is not")
    print("     discriminating between neighbors at all; speech conditioning via FiLM may not be")
    print("     differentiating node features enough to produce non-uniform attention scores.")
if min(final_grad_norms.values()) < 1e-6:
    print(f"  -> {min_grad_module} has a near-zero gradient norm -- likely vanishing gradient")
    print("     specifically in this submodule, blocking learning upstream/downstream of it.")
if max(final_grad_norms.values()) / (min(final_grad_norms.values()) + 1e-12) > 1e4:
    print("  -> Gradient norms span >4 orders of magnitude across submodules -- a severe")
    print("     imbalance that could prevent joint optimization (one part trains, others stall).")
print("\n  Inspect the per-epoch table above for trends (not just final values) -- e.g. does any")
print("  submodule's gradient norm spike then collapse, consistent with the loss instability")
print("  (0.70-0.81 excursions) seen in the pre-fix train-vs-test-gap run?")
