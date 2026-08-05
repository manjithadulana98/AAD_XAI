# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AADNet — finer-resolution delta-band rerun
#
# **Why this notebook exists.** Both `frequency_by_roi_subject.csv` (this
# repo) and the standalone frequency-analysis deck flag the same caveat on
# every delta-band (0.5-4 Hz) result: the pipeline's own output schema
# attaches "Delta-band interpretation is limited by short analysis window.
# Treat cautiously." to every delta row, in both models. This notebook tests
# that caveat directly rather than just repeating it: delta is split into
# three narrower sub-bands --
#
#   delta_1: 0.5-1.5 Hz   delta_2: 1.5-2.5 Hz   delta_3: 2.5-4.0 Hz
#
# -- while theta/alpha/beta stay exactly as they were (they were never
# flagged with this caveat). If the delta effect seen in the broad-band
# rerun is concentrated in one narrow sub-band rather than spread evenly,
# that's informative; if it's roughly flat across all three, that's
# consistent with (not proof of) the broad-band result being a reasonable
# summary despite the resolution caveat.
#
# **Identical machinery to `kaggle_rerun_aadnet_freq_by_roi.py`** (see that
# notebook's markdown for the full rationale) -- reuses cached
# occlusion/permutation/IG results, only redoes the ROI x band loop, recomputes
# baseline fresh rather than trusting cached per-window order. The only change
# is `BANDS_CONFIG` below and the output path (this writes to its own scoped
# directory, not over the existing 4-band `frequency_by_roi_subject.csv`).
#
# **Kaggle setup requirements:** Internet enabled, GPU accelerator,
# `dulanamanjitha/aad-xai-artifacts` dataset attached, Kaggle Secret
# `GCP_SA_JSON`.
#
# Output: `/kaggle/working/xai_results_aadnet_freq_finer_delta/frequency_by_roi_subject_finer_delta.csv`
# (same schema as the original -- `subject,roi,band,mean_dp,n_windows` -- with
# `band` now one of delta_1/delta_2/delta_3/theta/alpha/beta).

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
    print(f"Pre-installed torch {_torch_preinstalled.__version__} found "
          f"(CUDA available: {_torch_preinstalled.cuda.is_available()}) -- "
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
subprocess.run(["pip", "install", "-q", "google-cloud-storage"], check=True)

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 3. Configuration

# %%
from pathlib import Path

MAX_WINDOWS_PER_SUBJECT = 450   # must match the original run so window counts/CIs are comparable
RANDOM_SEED = 42

BANDS_CONFIG = [
    ("delta_1", 0.5, 1.5), ("delta_2", 1.5, 2.5), ("delta_3", 2.5, 4.0),
    ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0),
]

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, (
    "DTU dataset not found. Attach the 'dulanamanjitha/aad-xai-artifacts' dataset. "
    "Tried: " + ", ".join(DTU_KAGGLE_ROOT_CANDIDATES)
)
os.environ["DATASET"] = DTU_ROOT
print(f"DTU dataset  : {DTU_ROOT}")

GCP_PROJECT = "MscVM"
GCS_BUCKET = "addnet_results"
GCS_MODEL_PREFIX = "models/"
LOCAL_CKPT_DIR = Path(REPO_DIR) / "external" / "AADNet" / "output" / "AADNet_DTUDataset"
LOCAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("/kaggle/working/xai_results_aadnet_freq_finer_delta")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir   : {OUT_DIR}  (scoped rerun -- NOT xai_results_aadnet/)")

import numpy as np
import random
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# %% [markdown]
# ## 4. Authenticate to GCS + list the bucket

# %%
import json
import tempfile
import time

from google.cloud import storage  # type: ignore

GCS_CONNECT_MAX_ATTEMPTS = 5
GCS_CONNECT_RETRY_DELAY_S = 20


def _authenticate_gcs():
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        _sa_raw = UserSecretsClient().get_secret("GCP_SA_JSON")
        _sa_path = os.path.join(tempfile.gettempdir(), "gcp_sa.json")
        with open(_sa_path, "w") as f:
            f.write(_sa_raw)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _sa_path
        sa_info = json.loads(_sa_raw)
        print(f"Kaggle Secret GCP_SA_JSON found. Auth as: {sa_info.get('client_email', '?')}")
    except Exception as _e:
        print(f"Kaggle Secret GCP_SA_JSON not used ({type(_e).__name__}); falling back to Application Default Credentials.")


client = None
all_blobs = None
_last_exc = None
for _attempt in range(1, GCS_CONNECT_MAX_ATTEMPTS + 1):
    try:
        _authenticate_gcs()
        client = storage.Client(project=GCP_PROJECT)
        all_blobs = list(client.list_blobs(GCS_BUCKET, prefix=GCS_MODEL_PREFIX))
        print(f"GCS connected on attempt {_attempt}/{GCS_CONNECT_MAX_ATTEMPTS}.")
        break
    except Exception as _e:
        _last_exc = _e
        print(f"  GCS connect attempt {_attempt}/{GCS_CONNECT_MAX_ATTEMPTS} failed: {type(_e).__name__}: {_e}")
        if _attempt < GCS_CONNECT_MAX_ATTEMPTS:
            time.sleep(GCS_CONNECT_RETRY_DELAY_S)

if all_blobs is None:
    raise RuntimeError(
        f"Could not connect to gs://{GCS_BUCKET}/{GCS_MODEL_PREFIX} after "
        f"{GCS_CONNECT_MAX_ATTEMPTS} attempts. Last error: {_last_exc!r}."
    ) from _last_exc

print(f"Total blobs under gs://{GCS_BUCKET}/{GCS_MODEL_PREFIX}: {len(all_blobs)}")
ss_blobs = [b for b in all_blobs if "AADNet_SS_T_" in b.name and b.name.endswith(".pth")]
print(f"SS checkpoint blobs: {len(ss_blobs)}")

# %% [markdown]
# ## 5. Download SS checkpoints (skips any already cached on local disk)

# %%
import re

CKPT_RE = re.compile(r"AADNet_SS_T_(?P<T>\d+)_s_(?P<s>\d+)_fold_(?P<fold>\d+)\.pth$")


def parse_ckpt_name(name):
    m = CKPT_RE.search(name)
    return None if not m else {"T": int(m["T"]), "subject": int(m["s"]), "fold": int(m["fold"])}


ckpt_index = {}
for blob in ss_blobs:
    meta = parse_ckpt_name(blob.name)
    if meta is None:
        continue
    local_path = LOCAL_CKPT_DIR / os.path.basename(blob.name)
    if not local_path.exists():
        blob.download_to_filename(str(local_path))
    ckpt_index[(meta["subject"], meta["fold"])] = local_path

print(f"Downloaded / cached {len(ckpt_index)} SS checkpoints")
subject_ids = sorted({s for (s, _) in ckpt_index})
print(f"Subjects: {subject_ids}")

# %% [markdown]
# ## 6. Config, corrected montage, and model factory

# %%
from utils.config import Config  # from external/AADNet/utils
from aad_xai.models.aadnet_external import ExternalAADNet

CONFIG_PATH = os.path.join(REPO_DIR, "external", "AADNet", "config", "config_AADNet_SS_DTU.yml")
aadnet_config = Config.load_config(CONFIG_PATH)

CHANNELS = list(aadnet_config.get(("dataset", "channels")))
N_CHANNELS = len(CHANNELS)
SR = int(aadnet_config.get(("dataset", "sr")))
TRAIN_WIN = int(aadnet_config.get(("dataset", "training_window")))
WINDOW_SAMPLES = TRAIN_WIN * SR
print(f"AADNet DTU config — channels={N_CHANNELS}, sr={SR} Hz, window={TRAIN_WIN}s ({WINDOW_SAMPLES} samples)")


def load_montage(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    ch_by_idx = {int(r.channel_index): r.electrode_name for r in df.itertuples()}
    roi_by_idx = {int(r.channel_index): r.roi for r in df.itertuples()}
    rois = {}
    for ch_idx, roi in sorted(roi_by_idx.items()):
        rois.setdefault(roi, []).append(ch_idx)
    return ch_by_idx, roi_by_idx, rois


MONTAGE_PATH = os.path.join(REPO_DIR, "config", "aadnet_dtu_channel_montage.csv")
CH_NAME, CH_ROI, ROIS = load_montage(MONTAGE_PATH)
BANDS = {name: (lo, hi) for name, lo, hi in BANDS_CONFIG}
print(f"Loaded corrected montage — {len(CH_NAME)} channels, {len(ROIS)} ROIs: {list(ROIS)}")
print(f"Bands ({len(BANDS)}): {BANDS}")
assert "Mastoid" not in ROIS, (
    "Mastoid ROI present -- this should not happen with aadnet_dtu_channel_montage.csv; "
    "stop, this indicates the montage file or its loading changed unexpectedly."
)
assert len(ROIS) == 8, f"Expected 8 ROIs for AADNet's real channel set, got {len(ROIS)}: {list(ROIS)}"
assert len(BANDS) == 6, f"Expected 6 bands (3 delta sub-bands + theta/alpha/beta), got {len(BANDS)}"


def build_aadnet(state_dict_path):
    m = ExternalAADNet(n_channels=N_CHANNELS, window_samples=WINDOW_SAMPLES, sfreq=SR)
    state = torch.load(str(state_dict_path), map_location=DEVICE)
    m.model.load_state_dict(state)
    m.eval().to(DEVICE)
    return m

# %% [markdown]
# ## 7. Data pipeline — per-subject held-out windowed batches

# %%
from torch.utils.data import DataLoader
from aadnet.dataset import DTUDataset


def collect_subject_windows(subject_id, max_windows, seed=RANDOM_SEED):
    DTUDataset.all_data = None
    crossSSData = DTUDataset.createSSCrossValidation(subject=subject_id, config=aadnet_config)
    n_folds = len(crossSSData)
    eegs_all, envs_all, ys_all, fids_all = [], [], [], []
    per_fold_cap = max_windows // n_folds + 1
    rng = np.random.RandomState(seed + subject_id)
    for fold_idx, (_, te) in enumerate(crossSSData):
        te_eeg, te_aud, te_label = te
        if len(te_eeg) == 0:
            continue
        ds = DTUDataset(aadnet_config, te_eeg, te_aud, te_label)
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
    y = torch.cat(ys_all, 0)
    fids = torch.cat(fids_all, 0)
    if eeg.shape[0] > max_windows:
        idx = torch.from_numpy(rng.choice(eeg.shape[0], max_windows, replace=False))
        eeg, env, y, fids = eeg[idx], env[idx], y[idx], fids[idx]
    return {"eeg": eeg, "env": env, "y": y, "fold_ids": fids}

# %% [markdown]
# ## 8. Frequency-only per-subject computation (6 bands: 3 delta sub-bands + theta/alpha/beta)

# %%
from scipy.signal import butter, filtfilt


def bandpass_channel_block(eeg_ct, low, high, fs=SR, order=4):
    ny = 0.5 * fs
    lo = max(1e-3, low / ny)
    hi = min(0.999, high / ny)
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, eeg_ct, axis=-1)


def batched_forward(model, eeg, env, batch_size=128):
    outs = []
    with torch.no_grad():
        for i in range(0, eeg.shape[0], batch_size):
            e = eeg[i:i + batch_size].to(DEVICE, non_blocking=True)
            a = env[i:i + batch_size].to(DEVICE, non_blocking=True)
            outs.append(model(e, a).detach().cpu())
    return torch.cat(outs, 0)


def p_attended(model, eeg, env, y, batch_size=128):
    logits = batched_forward(model, eeg, env, batch_size=batch_size)
    probs = torch.softmax(logits, dim=-1)
    return probs.gather(1, y.unsqueeze(1)).squeeze(1).numpy()


def subject_frequency_only(subject_id, samples):
    eeg = samples["eeg"]
    env = samples["env"]
    y = samples["y"]
    fids = samples["fold_ids"].numpy()

    freq_dp = {roi: {band: [] for band in BANDS} for roi in ROIS}
    base_p_all = []

    unique_folds = sorted(set(fids.tolist()))
    for fold in unique_folds:
        idx = np.where(fids == fold)[0]
        if idx.size == 0:
            continue
        ckpt_path = ckpt_index.get((subject_id, fold))
        if ckpt_path is None:
            print(f"  subj {subject_id} fold {fold}: no checkpoint, skipping")
            continue
        model = build_aadnet(ckpt_path)

        eeg_f, env_f, y_f = eeg[idx], env[idx], y[idx]

        base = p_attended(model, eeg_f, env_f, y_f)
        base_p_all.append(base)

        eeg_np = eeg_f.numpy()
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

    base_p_all = np.concatenate(base_p_all) if base_p_all else np.zeros(0)
    return {
        "freq_dp": {r: {b: np.array(v) for b, v in bands.items()} for r, bands in freq_dp.items()},
        "base_acc": float((base_p_all > 0.5).mean()) if base_p_all.size else float("nan"),
    }

# %% [markdown]
# ## 9. Run across all subjects, with per-subject timing

# %%
t_start = time.time()
per_subject_timing = {}
per_subject_results = {}

for sid in subject_ids:
    t_subj_start = time.time()
    print(f"[{t_subj_start - t_start:6.0f}s] subject {sid}: loading windows ...")
    samples = collect_subject_windows(sid, MAX_WINDOWS_PER_SUBJECT)
    if samples is None:
        print(f"  subject {sid}: no windows returned, skipping")
        continue
    print(f"[{time.time() - t_start:6.0f}s] subject {sid}: running frequency-only ROI x band loop "
          f"({len(ROIS)} ROIs x {len(BANDS)} bands, {tuple(samples['eeg'].shape)} eeg windows) ...")
    per_subject_results[sid] = subject_frequency_only(sid, samples)
    t_subj_elapsed = time.time() - t_subj_start
    per_subject_timing[sid] = t_subj_elapsed
    print(f"[{time.time() - t_start:6.0f}s] subject {sid} done in {t_subj_elapsed:.1f}s "
          f"(base_acc={per_subject_results[sid]['base_acc']:.3f})")

t_total = time.time() - t_start
print(f"\nTotal wall-clock time for {len(per_subject_results)} subjects: {t_total:.1f}s "
      f"({t_total / 60:.1f} min)")
print(f"Mean per-subject time: {np.mean(list(per_subject_timing.values())):.1f}s")

# %% [markdown]
# ## 10. Write `frequency_by_roi_subject_finer_delta.csv`

# %%
import pandas as pd

freq_rows = []
for sid in per_subject_results:
    for roi, bands in per_subject_results[sid]["freq_dp"].items():
        for band, vals in bands.items():
            if len(vals) == 0:
                continue
            freq_rows.append({
                "subject": sid, "roi": roi, "band": band,
                "mean_dp": float(vals.mean()),
                "n_windows": int(len(vals)),
            })
freq_df = pd.DataFrame(freq_rows)
freq_df.to_csv(OUT_DIR / "frequency_by_roi_subject_finer_delta.csv", index=False)

roi_values = sorted(freq_df["roi"].unique().tolist())
band_values = sorted(freq_df["band"].unique().tolist())
print(f"ROIs present ({len(roi_values)}): {roi_values}")
print(f"Bands present ({len(band_values)}): {band_values}")
assert "Mastoid" not in roi_values, "Mastoid still present -- stop and investigate."
assert len(roi_values) == 8, f"Expected 8 ROIs, got {len(roi_values)}."
assert len(band_values) == 6, f"Expected 6 bands, got {len(band_values)}."
print("PASS -- 8 ROIs, 6 bands (delta_1/delta_2/delta_3/theta/alpha/beta), no Mastoid.")

with open(OUT_DIR / "rerun_timing.json", "w") as f:
    json.dump({
        "per_subject_seconds": per_subject_timing,
        "total_seconds": t_total,
        "total_minutes": t_total / 60,
        "n_subjects": len(per_subject_results),
        "n_rois": len(ROIS),
        "n_bands": len(BANDS),
        "bands_config": BANDS_CONFIG,
        "montage_file": "config/aadnet_dtu_channel_montage.csv",
        "purpose": "finer delta-band resolution test (0.5-1.5 / 1.5-2.5 / 2.5-4 Hz), theta/alpha/beta unchanged",
    }, f, indent=2)

print(f"\nWritten to {OUT_DIR}/frequency_by_roi_subject_finer_delta.csv")
print(f"Timing/config record: {OUT_DIR}/rerun_timing.json")
