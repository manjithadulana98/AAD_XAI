# %% [markdown]
# # AADNet Focused XAI — Kaggle Notebook
#
# Runs a VLAAI-style XAI suite on the 18 subject-specific (SS) AADNet models
# trained on the DTU dataset. Model checkpoints are pulled from
# `gs://addnet_results/models/` via a service-account JSON stored as a
# Kaggle Secret named `GCP_SA_JSON`.
#
# **Sections executed:**
# - A. Stream / block ablation (EEG-branch vs audio-branch vs fusion)
# - B. Channel occlusion (64 channels, bootstrap CI + sign-flip perm)
# - C. Channel permutation (across-window shuffle)
# - D. Integrated Gradients (per-channel attribution)
# - E. ROI aggregation (via `config/dtu_channel_montage.csv`)
# - F. Frequency-band occlusion per ROI (delta / theta / alpha / beta)
# - G. Subject × channel heatmap + pairwise Spearman ρ
# - H. Subject-level Wilcoxon + BH-FDR + split-half reliability
#
# **Kaggle setup requirements**
# - Enable Internet in notebook settings
# - Enable GPU accelerator (T4 x2 recommended)
# - Attach dataset `dulanamanjitha/aad-xai-artifacts` (holds DTU EEG + Audio)
# - Add a Kaggle Secret `GCP_SA_JSON` containing a GCP service-account JSON
#   with `roles/storage.objectViewer` on `gs://addnet_results`
#
# Outputs land in `/kaggle/working/xai_results_aadnet/`.

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
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run(["pip", "install", "-q", "-e", "."], check=True)
subprocess.run(["pip", "install", "-q", "google-cloud-storage"], check=True)

# Make src/ and external/AADNet/ importable
for extra in ("src", "external/AADNet"):
    p = os.path.join(REPO_DIR, extra)
    if p not in sys.path:
        sys.path.insert(0, p)

print("Setup done.")

# %% [markdown]
# ## 2. GPU sanity check

# %%
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Device count    : {torch.cuda.device_count()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 3. Configuration

# %%
from pathlib import Path

# Runtime knobs
FOLD_STRATEGY = "per_fold_holdout"   # only supported value in v1
MAX_WINDOWS_PER_SUBJECT = 450        # cap XAI windows per subject
N_BOOT = 500                         # bootstrap iterations
N_PERM = 5000                        # sign-flip permutations
N_IG_WINDOWS = 30                    # windows used for Integrated Gradients
IG_STEPS = 32                        # IG interpolation steps
FDR_ALPHA = 0.05
SPLIT_HALF_ITER = 1000
RANDOM_SEED = 42

# Paths
DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]

DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, (
    "DTU dataset not found. Attach the 'dulanamanjitha/aad-xai-artifacts' dataset "
    "to this notebook. Tried: " + ", ".join(DTU_KAGGLE_ROOT_CANDIDATES)
)
print(f"DTU dataset  : {DTU_ROOT}")

os.environ["DATASET"] = DTU_ROOT

GCP_PROJECT = "MscVM"                              # GCP project that owns the bucket
GCS_BUCKET = "addnet_results"
GCS_MODEL_PREFIX = "models/"                       # inside bucket
LOCAL_CKPT_DIR = Path(REPO_DIR) / "external" / "AADNet" / "output" / "AADNet_DTUDataset"
LOCAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("/kaggle/working/xai_results_aadnet")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir   : {OUT_DIR}")

import numpy as np
import random
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# %% [markdown]
# ## 4. Authenticate to GCS
#
# The notebook uses whichever credential source is available. If you added a
# Kaggle Secret named `GCP_SA_JSON` (a service-account JSON key), it's used
# automatically. Otherwise Application Default Credentials are used
# (e.g. from `gcloud auth application-default login` on your workstation).

# %%
import json
import tempfile

try:
    from kaggle_secrets import UserSecretsClient  # type: ignore
    _sa_raw = UserSecretsClient().get_secret("GCP_SA_JSON")
    _sa_path = os.path.join(tempfile.gettempdir(), "gcp_sa.json")
    with open(_sa_path, "w") as f:
        f.write(_sa_raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _sa_path
    sa_info = json.loads(_sa_raw)
    print(f"Kaggle Secret GCP_SA_JSON found. Auth as: {sa_info.get('client_email','?')}")
except Exception as _e:
    print(f"Kaggle Secret GCP_SA_JSON not used ({type(_e).__name__}); falling back to Application Default Credentials.")

# %% [markdown]
# ## 5. List the bucket
#
# Confirm the checkpoint layout before downloading. Expected filename pattern
# (from `external/AADNet/cross_validate_ss.py`):
#     `AADNet_SS_T_10_s_{subject}_fold_{fold}.pth`   for subject in 0..17, fold in 0..7

# %%
from google.cloud import storage  # type: ignore

client = storage.Client(project=GCP_PROJECT)
bucket = client.bucket(GCS_BUCKET)

all_blobs = list(client.list_blobs(GCS_BUCKET, prefix=GCS_MODEL_PREFIX))
print(f"Total blobs under gs://{GCS_BUCKET}/{GCS_MODEL_PREFIX}: {len(all_blobs)}")

ss_blobs = [b for b in all_blobs if "AADNet_SS_T_" in b.name and b.name.endswith(".pth")]
print(f"SS checkpoint blobs: {len(ss_blobs)}")
for b in ss_blobs[:5] + (["..."] if len(ss_blobs) > 10 else []) + ss_blobs[-3:]:
    if isinstance(b, str):
        print(f"  {b}")
    else:
        print(f"  {b.name}   ({b.size/1e6:.2f} MB)")

# %% [markdown]
# ## 6. Download SS checkpoints

# %%
import re

CKPT_RE = re.compile(r"AADNet_SS_T_(?P<T>\d+)_s_(?P<s>\d+)_fold_(?P<fold>\d+)\.pth$")

def parse_ckpt_name(name):
    m = CKPT_RE.search(name)
    if not m:
        return None
    return {"T": int(m["T"]), "subject": int(m["s"]), "fold": int(m["fold"])}

ckpt_index = {}   # (subject, fold) -> local path
for blob in ss_blobs:
    meta = parse_ckpt_name(blob.name)
    if meta is None:
        continue
    local_path = LOCAL_CKPT_DIR / os.path.basename(blob.name)
    if not local_path.exists():
        blob.download_to_filename(str(local_path))
    ckpt_index[(meta["subject"], meta["fold"])] = local_path

print(f"Downloaded / cached {len(ckpt_index)} SS checkpoints")
subjects_found = sorted({s for (s, _) in ckpt_index})
folds_per_subject = {s: sorted({f for (ss, f) in ckpt_index if ss == s}) for s in subjects_found}
print(f"Subjects: {subjects_found}")
print(f"Folds per subject (example subject {subjects_found[0]}): {folds_per_subject[subjects_found[0]]}")

# %% [markdown]
# ## 7. Load AADNet config and build model factory

# %%
from utils.config import Config  # from external/AADNet/utils
from aad_xai.models.aadnet_external import ExternalAADNet

CONFIG_PATH = os.path.join(REPO_DIR, "external", "AADNet", "config", "config_AADNet_SS_DTU.yml")
aadnet_config = Config.load_config(CONFIG_PATH)

CHANNELS   = list(aadnet_config.get(("dataset", "channels")))
N_CHANNELS = len(CHANNELS)
SR         = int(aadnet_config.get(("dataset", "sr")))
TRAIN_WIN  = int(aadnet_config.get(("dataset", "training_window")))
WINDOW_SAMPLES = TRAIN_WIN * SR

print(f"AADNet DTU config — channels={N_CHANNELS}, sr={SR} Hz, window={TRAIN_WIN}s ({WINDOW_SAMPLES} samples)")

def build_aadnet(state_dict_path):
    m = ExternalAADNet(n_channels=N_CHANNELS, window_samples=WINDOW_SAMPLES, sfreq=SR)
    state = torch.load(str(state_dict_path), map_location=DEVICE)
    m.model.load_state_dict(state)
    m.eval().to(DEVICE)
    return m

# Smoke test: load one checkpoint and forward a dummy tensor
_smoke_key = sorted(ckpt_index)[0]
_smoke_model = build_aadnet(ckpt_index[_smoke_key])
with torch.no_grad():
    _eeg = torch.zeros(2, N_CHANNELS, WINDOW_SAMPLES, device=DEVICE)
    _env = torch.zeros(2, 2, WINDOW_SAMPLES, device=DEVICE)
    _logits = _smoke_model(_eeg, _env)
print(f"Smoke test — loaded {_smoke_key}, logits shape {tuple(_logits.shape)}")
del _smoke_model, _eeg, _env, _logits
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %% [markdown]
# ## 8. Data pipeline — per-subject held-out windowed batches
#
# For each subject:
#   1. Get its 8 SS folds via `DTUDataset.createSSCrossValidation`
#   2. For each fold, build a `DTUDataset` on the held-out (test) trials only
#   3. Iterate windows via DataLoader and stack up to `MAX_WINDOWS_PER_SUBJECT`

# %%
from torch.utils.data import DataLoader
from aadnet.dataset import DTUDataset

def collect_subject_windows(subject_id, max_windows, seed=RANDOM_SEED):
    """Return dict with tensors per subject:
       eeg  (N, 64, WINDOW_SAMPLES)
       env  (N, 2,  WINDOW_SAMPLES)
       y    (N,)
       fold_ids (N,)   which fold each window came from (matches ckpt)
    """
    # Force DTUDataset to reload for this subject
    DTUDataset.all_data = None
    crossSSData = DTUDataset.createSSCrossValidation(subject=subject_id, config=aadnet_config)
    n_folds = len(crossSSData)
    eegs_all, envs_all, ys_all, fids_all = [], [], [], []

    # Round-robin across folds so we don't over-sample from any one fold
    per_fold_cap = max_windows // n_folds + 1
    rng = np.random.RandomState(seed + subject_id)
    for fold_idx, (_, te) in enumerate(crossSSData):
        te_eeg, te_aud, te_label = te
        if len(te_eeg) == 0:
            continue
        ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)
        # window step is `step` seconds; DTUDataset handles it internally
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        buf_e, buf_a, buf_y = [], [], []
        for eeg, env, y in loader:
            buf_e.append(eeg.float()); buf_a.append(env.float()); buf_y.append(y.long())
            if sum(x.shape[0] for x in buf_e) >= per_fold_cap:
                break
        if not buf_e:
            continue
        buf_e = torch.cat(buf_e, 0)[:per_fold_cap]
        buf_a = torch.cat(buf_a, 0)[:per_fold_cap]
        buf_y = torch.cat(buf_y, 0)[:per_fold_cap]
        eegs_all.append(buf_e); envs_all.append(buf_a); ys_all.append(buf_y)
        fids_all.append(torch.full((buf_e.shape[0],), fold_idx, dtype=torch.long))

    if not eegs_all:
        return None
    eeg = torch.cat(eegs_all, 0)
    env = torch.cat(envs_all, 0)
    y   = torch.cat(ys_all, 0)
    fids= torch.cat(fids_all, 0)

    # Trim to max_windows via random sample
    if eeg.shape[0] > max_windows:
        idx = torch.from_numpy(rng.choice(eeg.shape[0], max_windows, replace=False))
        eeg, env, y, fids = eeg[idx], env[idx], y[idx], fids[idx]

    return {"eeg": eeg, "env": env, "y": y, "fold_ids": fids}

# Smoke test
_sw = collect_subject_windows(subject_id=0, max_windows=64)
print("Subject 0 sample — eeg", tuple(_sw["eeg"].shape), "env", tuple(_sw["env"].shape),
      "y unique", torch.unique(_sw["y"]).tolist(), "folds", torch.unique(_sw["fold_ids"]).tolist())
del _sw

# %% [markdown]
# ## 9. XAI helper functions
#
# Ported from `scripts/run_focused_xai.py` and adapted for AADNet's
# `(B, C, T)` layout and dual (eeg, env) forward signature.

# %%
import pandas as pd
from collections import OrderedDict
from scipy.signal import butter, filtfilt
from scipy.stats import wilcoxon, spearmanr


def bootstrap_ci(values, n_boot=N_BOOT, ci=0.95, seed=RANDOM_SEED):
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, len(values), size=len(values))
        means[b] = values[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(values.mean()), float(lo), float(hi)

def sign_flip_p_value(values, n_perm=N_PERM, seed=RANDOM_SEED):
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    obs = abs(values.mean())
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = abs((values * rng.choice([-1, 1], size=len(values))).mean())
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))

def fdr_correction(p_values, alpha=FDR_ALPHA):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        rank = i + 1
        bh_val = sorted_p[i] * n / rank
        adjusted[sorted_idx[i]] = min(bh_val, adjusted[sorted_idx[i + 1]])
    adjusted = np.clip(adjusted, 0.0, 1.0)
    return adjusted, adjusted < alpha

def load_montage(csv_path):
    df = pd.read_csv(csv_path)
    ch_by_idx = {int(r.channel_index): r.electrode_name for r in df.itertuples()}
    roi_by_idx = {int(r.channel_index): r.roi for r in df.itertuples()}
    rois = OrderedDict()
    for ch_idx, roi in sorted(roi_by_idx.items()):
        rois.setdefault(roi, []).append(ch_idx)
    return ch_by_idx, roi_by_idx, rois

MONTAGE_PATH = os.path.join(REPO_DIR, "config", "dtu_channel_montage.csv")
CH_NAME, CH_ROI, ROIS = load_montage(MONTAGE_PATH)
print(f"Loaded montage — {len(CH_NAME)} channels, {len(ROIS)} ROIs: {list(ROIS)}")

BANDS = OrderedDict([
    ("delta", (0.5, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 13.0)),
    ("beta",  (13.0, 30.0)),
])

# %% [markdown]
# ### Batch-mode P(attended) scoring
#
# For AADNet we want `softmax(logits, -1)[:, y_attended]` — the probability
# the model assigns to the true attended stream. Occlusion / permutation
# subtract this from a baseline to get ΔP.

# %%
def batched_forward(model, eeg, env, batch_size=128):
    """Forward pass in mini-batches; returns logits tensor on CPU."""
    outs = []
    n = eeg.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            e = eeg[i:i+batch_size].to(DEVICE, non_blocking=True)
            a = env[i:i+batch_size].to(DEVICE, non_blocking=True)
            outs.append(model(e, a).detach().cpu())
    return torch.cat(outs, 0)

def p_attended(model, eeg, env, y, batch_size=128):
    logits = batched_forward(model, eeg, env, batch_size=batch_size)
    probs = torch.softmax(logits, dim=-1)
    return probs.gather(1, y.unsqueeze(1)).squeeze(1).numpy()

# %% [markdown]
# ## 10. Per-subject XAI runner (Sections A/B/C/D/F on one subject)
#
# For each subject we iterate over folds, use each fold's checkpoint on
# **its own** held-out windows, compute per-window ΔP, and concatenate the
# per-window deltas.

# %%
def bandpass_channel_block(eeg_ct, low, high, fs=SR, order=4):
    """Bandpass filter each (channel, time) row.
    eeg_ct: numpy array (C, T).
    """
    ny = 0.5 * fs
    lo = max(1e-3, low / ny)
    hi = min(0.999, high / ny)
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, eeg_ct, axis=-1)

def subject_perturbations(subject_id, samples):
    """Compute per-subject occlusion/permutation/IG/freq deltas by iterating folds.

    Returns dict with keys:
        base_p         (N,)    baseline P(attended)
        base_acc       float
        occ_dp         (N, 64) ΔP per channel per window
        perm_dp        (N, 64)
        ig_attr        (N_ig, 64)   IG attribution per channel (mean over time abs)
        freq_dp        dict[roi] -> dict[band] -> (N_r,) list of per-window ΔP
    """
    eeg   = samples["eeg"]      # CPU tensors
    env   = samples["env"]
    y     = samples["y"]
    fids  = samples["fold_ids"].numpy()
    N     = eeg.shape[0]

    base_p   = np.zeros(N)
    occ_dp   = np.zeros((N, N_CHANNELS))
    perm_dp  = np.zeros((N, N_CHANNELS))
    ig_attr_rows = []
    ig_taken = 0
    freq_dp  = {roi: {band: [] for band in BANDS} for roi in ROIS}

    rng = np.random.RandomState(RANDOM_SEED + subject_id)

    # Precompute permutation index (same across channels for this subject)
    perm_idx = torch.from_numpy(rng.permutation(N)).long()

    unique_folds = sorted(set(fids.tolist()))
    for fold in unique_folds:
        mask = fids == fold
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        ckpt_path = ckpt_index.get((subject_id, fold))
        if ckpt_path is None:
            print(f"  subj {subject_id} fold {fold}: no checkpoint, skipping")
            continue
        model = build_aadnet(ckpt_path)

        eeg_f = eeg[idx]
        env_f = env[idx]
        y_f   = y[idx]

        # --- Baseline ---
        base = p_attended(model, eeg_f, env_f, y_f)
        base_p[idx] = base

        # --- Occlusion (zero each channel) ---
        for ch in range(N_CHANNELS):
            eeg_m = eeg_f.clone()
            eeg_m[:, ch, :] = 0.0
            m_p = p_attended(model, eeg_m, env_f, y_f)
            occ_dp[idx, ch] = base - m_p

        # --- Permutation (shuffle each channel across all windows for this subject) ---
        # We use the subject-level permutation index restricted to this fold's indices
        fold_perm_local = torch.from_numpy(rng.permutation(len(idx))).long()
        for ch in range(N_CHANNELS):
            eeg_m = eeg_f.clone()
            eeg_m[:, ch, :] = eeg_f[fold_perm_local][:, ch, :]
            m_p = p_attended(model, eeg_m, env_f, y_f)
            perm_dp[idx, ch] = base - m_p

        # --- Integrated Gradients (limited number of windows per subject) ---
        remaining_ig = max(0, N_IG_WINDOWS - ig_taken)
        if remaining_ig > 0:
            take = min(remaining_ig, eeg_f.shape[0])
            sel = np.arange(take)
            eeg_ig = eeg_f[sel].to(DEVICE)
            env_ig = env_f[sel].to(DEVICE)
            y_ig   = y_f[sel].to(DEVICE)
            baseline_zero = torch.zeros_like(eeg_ig)
            attrs = torch.zeros_like(eeg_ig)
            for step in range(IG_STEPS):
                alpha = (step + 1) / IG_STEPS
                x_i = baseline_zero + alpha * (eeg_ig - baseline_zero)
                x_i.requires_grad_(True)
                logits = model(x_i, env_ig)
                p = torch.softmax(logits, dim=-1).gather(1, y_ig.unsqueeze(1)).squeeze(1)
                grads = torch.autograd.grad(p.sum(), x_i, retain_graph=False, create_graph=False)[0]
                attrs = attrs + grads.detach()
            attrs = attrs / IG_STEPS * (eeg_ig - baseline_zero)
            # Reduce (N, C, T) -> (N, C) via mean(abs)
            ig_per_win = attrs.abs().mean(dim=-1).detach().cpu().numpy()
            ig_attr_rows.append(ig_per_win)
            ig_taken += take
            del eeg_ig, env_ig, y_ig, baseline_zero, attrs

        # --- Frequency-band occlusion per ROI (subtract-band mode) ---
        eeg_np = eeg_f.numpy()  # (B, C, T)
        for roi_name, roi_channels in ROIS.items():
            for band_name, (lo, hi) in BANDS.items():
                eeg_m = eeg_np.copy()
                for ch in roi_channels:
                    band = bandpass_channel_block(eeg_np[:, ch, :], lo, hi, fs=SR)
                    eeg_m[:, ch, :] = eeg_np[:, ch, :] - band
                eeg_m_t = torch.from_numpy(eeg_m.astype(np.float32))
                m_p = p_attended(model, eeg_m_t, env_f, y_f)
                freq_dp[roi_name][band_name].extend((base - m_p).tolist())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ig_attr = np.concatenate(ig_attr_rows, axis=0) if ig_attr_rows else np.zeros((0, N_CHANNELS))
    return {
        "base_p":  base_p,
        "base_acc": float((base_p > 0.5).mean()),
        "occ_dp":  occ_dp,
        "perm_dp": perm_dp,
        "ig_attr": ig_attr,
        "freq_dp": {r: {b: np.array(v) for b, v in bands.items()} for r, bands in freq_dp.items()},
    }

# %% [markdown]
# ## 11. Section A — Stream / block ablation

# %%
def zero_output_hook(module, inp, out):
    return torch.zeros_like(out)

def stream_ablation_for_fold(model, eeg, env, y, target_module_name):
    """Register a forward hook that zeros the target module's output, then score."""
    target = dict(model.model.named_modules())[target_module_name]
    handle = target.register_forward_hook(zero_output_hook)
    try:
        return p_attended(model, eeg, env, y)
    finally:
        handle.remove()

def section_a_stream_ablation(samples_by_subject):
    """For each (subject, fold), compare baseline vs each block zeroed."""
    modules_to_ablate = ["inception_1_eeg", "inception_1_aud", "fc1"]
    rows = []
    for subject_id, samples in samples_by_subject.items():
        fids = samples["fold_ids"].numpy()
        for fold in sorted(set(fids.tolist())):
            ckpt_path = ckpt_index.get((subject_id, fold))
            if ckpt_path is None: continue
            m = build_aadnet(ckpt_path)
            mask = fids == fold
            idx = np.where(mask)[0]
            eeg_f, env_f, y_f = samples["eeg"][idx], samples["env"][idx], samples["y"][idx]
            base = p_attended(m, eeg_f, env_f, y_f)
            for mod_name in modules_to_ablate:
                ab = stream_ablation_for_fold(m, eeg_f, env_f, y_f, mod_name)
                rows.append({
                    "subject": subject_id,
                    "fold": fold,
                    "module": mod_name,
                    "base_p": float(base.mean()),
                    "ablated_p": float(ab.mean()),
                    "delta_p": float((base - ab).mean()),
                    "base_acc": float((base > 0.5).mean()),
                    "ablated_acc": float((ab > 0.5).mean()),
                })
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)

# %% [markdown]
# ## 12. Run the pipeline across all 18 subjects

# %%
import time

t_start = time.time()
subject_ids = sorted({s for (s, _) in ckpt_index})
print(f"Running XAI on {len(subject_ids)} subjects: {subject_ids}")

# 12a. Collect windows per subject
samples_by_subject = {}
for sid in subject_ids:
    print(f"[{time.time()-t_start:6.0f}s] loading windows for subject {sid} ...")
    samples_by_subject[sid] = collect_subject_windows(sid, MAX_WINDOWS_PER_SUBJECT)
print(f"Loaded windows for {len(samples_by_subject)} subjects "
      f"({sum(s['eeg'].shape[0] for s in samples_by_subject.values())} total)")

# 12b. Section A — stream ablation
print(f"\n[{time.time()-t_start:6.0f}s] SECTION A — stream ablation")
df_stream = section_a_stream_ablation(samples_by_subject)
df_stream.to_csv(OUT_DIR / "stream_ablation.csv", index=False)
print(df_stream.groupby("module")[["delta_p", "ablated_acc"]].mean())

# 12c. Sections B/C/D/F per subject
per_subject_results = {}
for sid in subject_ids:
    print(f"\n[{time.time()-t_start:6.0f}s] SECTIONS B–F for subject {sid}")
    per_subject_results[sid] = subject_perturbations(sid, samples_by_subject[sid])
    r = per_subject_results[sid]
    print(f"   base_acc={r['base_acc']:.3f}  "
          f"top-3 occ ch idx={list(np.argsort(-r['occ_dp'].mean(0))[:3])}  "
          f"top-3 perm ch idx={list(np.argsort(-r['perm_dp'].mean(0))[:3])}")

print(f"\n[{time.time()-t_start:6.0f}s] Sections A/B/C/D/F complete.")

# %% [markdown]
# ## 13. Section E — ROI aggregation

# %%
def channel_to_roi_matrix():
    mat = np.zeros((len(ROIS), N_CHANNELS))
    roi_names = list(ROIS)
    for ri, roi in enumerate(roi_names):
        for ch in ROIS[roi]:
            mat[ri, ch] = 1.0 / len(ROIS[roi])
    return roi_names, mat  # mat @ channel_mean gives ROI-mean

roi_names, roi_mat = channel_to_roi_matrix()

def aggregate_channel_matrix(per_subject_matrices):
    """per_subject_matrices: dict[sid] -> (N_windows, 64)  -> flat concatenated"""
    stacked = np.concatenate(list(per_subject_matrices.values()), axis=0)
    return stacked  # (total_windows, 64)

# Channel-level per-subject means for downstream
occ_subj_ch  = np.stack([per_subject_results[s]["occ_dp"].mean(0)  for s in subject_ids])   # (18, 64)
perm_subj_ch = np.stack([per_subject_results[s]["perm_dp"].mean(0) for s in subject_ids])   # (18, 64)
occ_roi      = occ_subj_ch  @ roi_mat.T   # (18, n_roi) — subject means then group
perm_roi     = perm_subj_ch @ roi_mat.T
print("ROI channel counts:", {r: len(v) for r, v in ROIS.items()})

# %% [markdown]
# ## 14. Section G — Subject × channel heatmap + pairwise ρ

# %%
subject_pairwise_rho = np.zeros((len(subject_ids), len(subject_ids)))
for i in range(len(subject_ids)):
    for j in range(len(subject_ids)):
        rho, _ = spearmanr(occ_subj_ch[i], occ_subj_ch[j])
        subject_pairwise_rho[i, j] = 0.0 if np.isnan(rho) else rho

mean_pairwise_rho = float(subject_pairwise_rho[np.triu_indices_from(subject_pairwise_rho, k=1)].mean())
print(f"Mean pairwise Spearman ρ (subject-level occlusion vectors): {mean_pairwise_rho:+.3f}")

# %% [markdown]
# ## 15. Section H — Wilcoxon + BH-FDR + split-half reliability

# %%
def wilcoxon_fdr_across_subjects(subj_by_ch, alpha=FDR_ALPHA):
    """subj_by_ch: (n_subjects, n_channels)  -> per-channel Wilcoxon vs 0."""
    p_values = np.ones(subj_by_ch.shape[1])
    for ch in range(subj_by_ch.shape[1]):
        vals = subj_by_ch[:, ch]
        try:
            if np.any(vals != 0):
                _, p = wilcoxon(vals)
                p_values[ch] = p
        except ValueError:
            p_values[ch] = 1.0
    adj, sig = fdr_correction(p_values, alpha=alpha)
    return p_values, adj, sig

occ_p, occ_adj, occ_sig = wilcoxon_fdr_across_subjects(occ_subj_ch)
perm_p, perm_adj, perm_sig = wilcoxon_fdr_across_subjects(perm_subj_ch)

def split_half_reliability(subj_by_ch, n_iter=SPLIT_HALF_ITER, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    n = subj_by_ch.shape[0]
    rhos = np.zeros(n_iter)
    for i in range(n_iter):
        perm = rng.permutation(n)
        h1 = perm[:n // 2]
        h2 = perm[n // 2:]
        m1 = subj_by_ch[h1].mean(0)
        m2 = subj_by_ch[h2].mean(0)
        r, _ = spearmanr(m1, m2)
        rhos[i] = 0.0 if np.isnan(r) else r
    return {
        "median_rho": float(np.median(rhos)),
        "ci_lo": float(np.percentile(rhos, 2.5)),
        "ci_hi": float(np.percentile(rhos, 97.5)),
        "n_iter": n_iter,
    }

split_half = split_half_reliability(occ_subj_ch)
print("Occlusion split-half reliability:", split_half)
print(f"Occlusion channels FDR-sig: {int(occ_sig.sum())}/{N_CHANNELS}")
print(f"Permutation channels FDR-sig: {int(perm_sig.sum())}/{N_CHANNELS}")

# %% [markdown]
# ## 16. Persist results to `/kaggle/working/xai_results_aadnet/`

# %%
# ---- Save NPY matrices ----
np.save(OUT_DIR / "occ_subj_ch.npy", occ_subj_ch)
np.save(OUT_DIR / "perm_subj_ch.npy", perm_subj_ch)
np.save(OUT_DIR / "occ_roi_subj.npy", occ_roi)
np.save(OUT_DIR / "perm_roi_subj.npy", perm_roi)
np.save(OUT_DIR / "subject_pairwise_rho.npy", subject_pairwise_rho)

# ---- Save per-subject dumps ----
for sid, r in per_subject_results.items():
    np.savez(
        OUT_DIR / f"subject_{sid:02d}.npz",
        base_p=r["base_p"],
        occ_dp=r["occ_dp"],
        perm_dp=r["perm_dp"],
        ig_attr=r["ig_attr"],
        base_acc=r["base_acc"],
    )

# ---- Channel-level summary CSV ----
occ_mean_ch = occ_subj_ch.mean(0)
perm_mean_ch = perm_subj_ch.mean(0)
ch_summary = pd.DataFrame({
    "channel_idx": np.arange(N_CHANNELS),
    "electrode": [CH_NAME[i] for i in range(N_CHANNELS)],
    "roi": [CH_ROI[i] for i in range(N_CHANNELS)],
    "occ_mean_dp": occ_mean_ch,
    "occ_wilcoxon_p": occ_p,
    "occ_wilcoxon_p_adj": occ_adj,
    "occ_fdr_sig": occ_sig,
    "perm_mean_dp": perm_mean_ch,
    "perm_wilcoxon_p": perm_p,
    "perm_wilcoxon_p_adj": perm_adj,
    "perm_fdr_sig": perm_sig,
})
ch_summary.to_csv(OUT_DIR / "channel_importance.csv", index=False)

# ---- ROI summary CSV ----
roi_summary_rows = []
for ri, roi in enumerate(roi_names):
    occ_vals  = occ_roi[:, ri]
    perm_vals = perm_roi[:, ri]
    m_occ,  lo_occ,  hi_occ  = bootstrap_ci(occ_vals)
    m_perm, lo_perm, hi_perm = bootstrap_ci(perm_vals)
    roi_summary_rows.append({
        "roi": roi,
        "n_channels": len(ROIS[roi]),
        "occ_mean_dp": m_occ,
        "occ_ci_lo": lo_occ, "occ_ci_hi": hi_occ,
        "perm_mean_dp": m_perm,
        "perm_ci_lo": lo_perm, "perm_ci_hi": hi_perm,
    })
pd.DataFrame(roi_summary_rows).to_csv(OUT_DIR / "roi_importance.csv", index=False)

# ---- Frequency-band × ROI CSV ----
freq_rows = []
for sid in subject_ids:
    for roi, bands in per_subject_results[sid]["freq_dp"].items():
        for band, vals in bands.items():
            if len(vals) == 0: continue
            freq_rows.append({
                "subject": sid, "roi": roi, "band": band,
                "mean_dp": float(vals.mean()),
                "n_windows": int(len(vals)),
            })
pd.DataFrame(freq_rows).to_csv(OUT_DIR / "frequency_by_roi_subject.csv", index=False)

# ---- Run config + high-level summary JSON ----
run_config = {
    "fold_strategy": FOLD_STRATEGY,
    "max_windows_per_subject": MAX_WINDOWS_PER_SUBJECT,
    "n_boot": N_BOOT,
    "n_perm": N_PERM,
    "n_ig_windows": N_IG_WINDOWS,
    "ig_steps": IG_STEPS,
    "fdr_alpha": FDR_ALPHA,
    "seed": RANDOM_SEED,
    "n_subjects": len(subject_ids),
    "channels": N_CHANNELS,
    "window_seconds": TRAIN_WIN,
    "sample_rate_hz": SR,
    "gcs_bucket": GCS_BUCKET,
    "gcs_prefix": GCS_MODEL_PREFIX,
    "n_checkpoints_used": len(ckpt_index),
    "mean_pairwise_rho": mean_pairwise_rho,
    "occ_fdr_sig_channels": int(occ_sig.sum()),
    "perm_fdr_sig_channels": int(perm_sig.sum()),
    "occ_split_half": split_half,
    "mean_base_acc": float(np.mean([per_subject_results[s]["base_acc"] for s in subject_ids])),
}
with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

# ---- Human-readable summary ----
top15_occ  = np.argsort(-occ_mean_ch)[:15]
top15_perm = np.argsort(-perm_mean_ch)[:15]
report_lines = [
    "AADNet Focused XAI — Kaggle run summary",
    f"Subjects: {len(subject_ids)}    Channels: {N_CHANNELS}    Window: {TRAIN_WIN}s @ {SR}Hz",
    f"Mean baseline P(attended) accuracy: {run_config['mean_base_acc']:.3f}",
    f"Mean inter-subject Spearman ρ (occlusion): {mean_pairwise_rho:+.3f}",
    f"Occlusion split-half ρ: median {split_half['median_rho']:.3f}  "
    f"[{split_half['ci_lo']:.3f}, {split_half['ci_hi']:.3f}]",
    f"Occlusion FDR-sig channels: {int(occ_sig.sum())}/{N_CHANNELS}",
    f"Permutation FDR-sig channels: {int(perm_sig.sum())}/{N_CHANNELS}",
    "",
    "Top-15 channels by occlusion ΔP (mean across subjects):",
]
for rank, ch in enumerate(top15_occ, 1):
    report_lines.append(
        f"  {rank:2d}. ch{ch:02d} {CH_NAME[ch]:6s} {CH_ROI[ch]:20s} "
        f"occ={occ_mean_ch[ch]:+.5f}  perm={perm_mean_ch[ch]:+.5f}  "
        f"FDR_occ={'*' if occ_sig[ch] else '.'} FDR_perm={'*' if perm_sig[ch] else '.'}"
    )
report_lines.append("")
report_lines.append("Stream ablation (mean across subject/fold):")
for mod, sub in df_stream.groupby("module")[["delta_p", "base_acc", "ablated_acc"]].mean().iterrows():
    report_lines.append(f"  zero '{mod}': ΔP={sub['delta_p']:+.4f}  base_acc={sub['base_acc']:.3f}  ablated_acc={sub['ablated_acc']:.3f}")
(OUT_DIR / "AADNET_XAI_REPORT.txt").write_text("\n".join(report_lines))
print("\n".join(report_lines))

# %% [markdown]
# ## 17. Verification vs published SS accuracies
#
# Compares this notebook's baseline P(attended) accuracy per subject to the
# accuracies stored in `results_gce/results/SS_AADNet_DTU_final_SS_acc.npy`
# (rows = windows [1,2,5,10,20,40]s, cols = subjects).

# %%
_ss_acc_path = os.path.join(REPO_DIR, "results_gce", "results", "SS_AADNet_DTU_final_SS_acc.npy")
if os.path.exists(_ss_acc_path):
    published = np.load(_ss_acc_path)
    published = published.squeeze(0) if published.ndim == 3 else published  # (n_windows, n_subjects)
    # window row for 10 s → index of 10 in [1,2,5,10,20,40]
    windows_paper = [1, 2, 5, 10, 20, 40]
    row_10 = windows_paper.index(10)
    pub_10 = published[row_10, :]
    ours = np.array([per_subject_results[s]["base_acc"] for s in subject_ids])
    delta = ours - pub_10[:len(subject_ids)]
    print("Per-subject baseline acc vs published SS AADNet 10-s accuracy:")
    print(pd.DataFrame({
        "subject": subject_ids,
        "notebook_acc": ours,
        "published_acc_10s": pub_10[:len(subject_ids)],
        "delta": delta,
    }).to_string(index=False))
    print(f"\nMean |delta| = {np.mean(np.abs(delta)):.4f}   (should be small; large gap => misload)")
else:
    print(f"Reference accuracy file not found: {_ss_acc_path}")
