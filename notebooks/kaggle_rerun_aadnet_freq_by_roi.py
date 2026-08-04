# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AADNet — targeted rerun of `frequency_by_roi_subject.csv` ONLY
#
# **Why this notebook exists.** `kaggle_output_AADnet/xai_results_aadnet/` was
# generated while AADNet's channels were mislabeled against the wrong montage
# (`config/dtu_channel_montage.csv` instead of AADNet's real
# `config/aadnet_dtu_channel_montage.csv` -- see the "NOTE (bug fix)" comment
# in `kaggle_run_xai_aadnet.py`). Per-channel files (`channel_importance.csv`,
# `candidate_channels.csv`, `high_confidence_channels.csv`,
# `hierarchical_channel_stats.csv`) were simple mislabels and have already been
# corrected in place via `scripts/relabel_aadnet_channels.py`. `roi_importance.csv`
# and `subject_level_roi_stats.csv` needed full recomputation from the cached
# per-subject-per-channel arrays (`occ_subj_ch.npy` / `perm_subj_ch.npy`) --
# also already done.
#
# `frequency_by_roi_subject.csv` is different: the frequency-band ablation
# itself iterates `ROIS.items()` **at computation time** (see
# `subject_perturbations` in `kaggle_run_xai_aadnet.py`, "Frequency-band
# occlusion per ROI" block) and bandpass-filters+zeros each ROI's channels
# before a fresh model forward pass. The wrong `ROIS` dict was used when this
# ran originally, so the *numbers* are wrong, not just the labels -- there is
# no cached array to relabel or recompute from. It needs a real rerun.
#
# **What this notebook does NOT do:** it does not repeat occlusion (Section B,
# 64 channels/fold), permutation (Section C, 64 channels/fold), or Integrated
# Gradients (Section D, `IG_STEPS` gradient steps/fold) -- those are already
# correct in `channel_importance.csv` / `occ_subj_ch.npy` / `perm_subj_ch.npy`
# and redoing them here would be wasted compute *and* a second unnecessary
# place for a silent bug to creep in. Per subject/fold this notebook runs only:
#   1. one baseline forward pass (recomputed fresh -- see note below), and
#   2. `len(ROIS) x len(BANDS)` = 8 x 4 = 32 forward passes for the
#      frequency-band loop,
# versus the original's 1 + 64 + 64 + (IG_STEPS grad steps) + 32 per fold.
#
# **Note on `base_p`:** the original run's `subject_XX.npz` files do cache a
# per-window `base_p`, computed on the exact same windows this notebook will
# reload via `collect_subject_windows`. Reusing it was considered, but it
# would rely on `collect_subject_windows` reproducing byte-identical
# window *order* to the original run purely from the RNG seed + code being
# unchanged -- an assumption that, if silently wrong, would misalign
# baselines to windows without any error being raised. Recomputing the
# baseline fresh (immediately after loading each fold's windows, same as the
# original code's own ordering) costs one extra cheap forward pass per fold
# and removes that assumption entirely, so that's what this notebook does.
#
# **Dependencies confirmed NOT cached anywhere** (verified against
# `kaggle_output_AADnet/xai_results_aadnet/`, which has aggregate arrays and
# per-subject `base_p`/`occ_dp`/`perm_dp`/`ig_attr` dumps, but no raw
# per-window EEG/envelope/label tensors):
#   - Raw per-fold EEG/envelope/label windows -- must reload via
#     `collect_subject_windows`, which requires the DTU dataset attached.
#   - The model checkpoint for each (subject, fold) as a *loaded* model
#     object -- the `.pth` files themselves may already be cached locally
#     (`external/AADNet/output/AADNet_DTUDataset/`), but still need GCS access
#     to confirm/fill any gaps, and still need a live GPU forward pass.
# Everything else this section needs (`ROIS`, `BANDS`, `SR`, `CH_NAME`,
# `CH_ROI`, the corrected montage) is cheap to rebuild from config and is
# rebuilt below, not reused from any cache.
#
# **Kaggle setup requirements** (same as `kaggle_run_xai_aadnet.py`):
# Internet enabled, GPU accelerator, `dulanamanjitha/aad-xai-artifacts`
# dataset attached, Kaggle Secret `GCP_SA_JSON`.
#
# Output: `/kaggle/working/xai_results_aadnet_freq_rerun/frequency_by_roi_subject.csv`
# (schema-identical to the original -- `subject,roi,band,mean_dp,n_windows` --
# a drop-in replacement for the stale file, plus a timing/config JSON for the
# record). It deliberately does NOT write into `xai_results_aadnet/` directly,
# so the already-correct files there can't be touched by a bug in this script.

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

BANDS_CONFIG = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0)]

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

OUT_DIR = Path("/kaggle/working/xai_results_aadnet_freq_rerun")
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
#
# Identical retry logic to `kaggle_run_xai_aadnet.py` -- Kaggle's secrets/GCP
# proxy has been observed to fail during the first ~minute of a fresh kernel.

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
#
# Uses `config/aadnet_dtu_channel_montage.csv` -- the corrected montage,
# already the default in the fixed `kaggle_run_xai_aadnet.py` -- so `ROIS`
# here has 8 entries (no `Mastoid`), matching the already-regenerated
# `roi_importance.csv`.

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
assert "Mastoid" not in ROIS, (
    "Mastoid ROI present -- this should not happen with aadnet_dtu_channel_montage.csv; "
    "stop, this indicates the montage file or its loading changed unexpectedly."
)
assert len(ROIS) == 8, f"Expected 8 ROIs for AADNet's real channel set, got {len(ROIS)}: {list(ROIS)}"


def build_aadnet(state_dict_path):
    m = ExternalAADNet(n_channels=N_CHANNELS, window_samples=WINDOW_SAMPLES, sfreq=SR)
    state = torch.load(str(state_dict_path), map_location=DEVICE)
    m.model.load_state_dict(state)
    m.eval().to(DEVICE)
    return m

# %% [markdown]
# ## 7. Data pipeline — per-subject held-out windowed batches
#
# Unchanged from `kaggle_run_xai_aadnet.py` -- raw windows are not cached
# anywhere, so this must run again regardless of what else is reused.

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
# ## 8. Frequency-only per-subject computation
#
# Deliberately omits the occlusion/permutation/IG loops from
# `subject_perturbations` in `kaggle_run_xai_aadnet.py` -- everything below
# this cell is the only part of that function this rerun actually needs.

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
    """Baseline + frequency-band-per-ROI ΔP only. No occlusion/permutation/IG.

    Returns dict[roi] -> dict[band] -> list[float] (per-window ΔP), plus
    base_acc for a cheap sanity cross-check against channel_importance.csv's
    already-correct baseline accuracy for this subject.
    """
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

        # Baseline recomputed fresh (see markdown note above on why we don't
        # reuse subject_XX.npz's cached base_p here).
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
# ## 10. Write `frequency_by_roi_subject.csv` (schema-identical to the original)

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
freq_df.to_csv(OUT_DIR / "frequency_by_roi_subject.csv", index=False)

# Verification: ROI set must be exactly the corrected 8, never Mastoid.
roi_values = sorted(freq_df["roi"].unique().tolist())
print(f"ROIs present in regenerated file ({len(roi_values)}): {roi_values}")
assert "Mastoid" not in roi_values, "Mastoid still present -- the fix did not take effect, stop and investigate."
assert len(roi_values) == 8, f"Expected 8 ROIs, got {len(roi_values)}."
print("PASS -- 8 ROIs, no Mastoid.")

with open(OUT_DIR / "rerun_timing.json", "w") as f:
    json.dump({
        "per_subject_seconds": per_subject_timing,
        "total_seconds": t_total,
        "total_minutes": t_total / 60,
        "n_subjects": len(per_subject_results),
        "n_rois": len(ROIS),
        "n_bands": len(BANDS),
        "montage_file": "config/aadnet_dtu_channel_montage.csv",
        "skipped_sections": ["B (occlusion, 64ch/fold)", "C (permutation, 64ch/fold)", "D (integrated gradients)"],
        "reused_from_original_run": [
            "channel_importance.csv", "candidate_channels.csv", "high_confidence_channels.csv",
            "hierarchical_channel_stats.csv", "subject_level_roi_stats.csv", "roi_importance.csv",
            "occ_subj_ch.npy", "perm_subj_ch.npy", "stream_ablation.csv",
        ],
    }, f, indent=2)

print(f"\nWritten to {OUT_DIR}/frequency_by_roi_subject.csv")
print(f"Timing/config record: {OUT_DIR}/rerun_timing.json")
print(
    "\nNext step (manual, not done by this notebook): back up the stale "
    "kaggle_output_AADnet/xai_results_aadnet/frequency_by_roi_subject.csv "
    "(e.g. rename to *_STALE_WRONG_MONTAGE.csv) and replace it with this file."
)
