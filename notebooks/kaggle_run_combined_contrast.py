# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Combined-Group Contrast — Kaggle Notebook
#
# Runs `scripts/run_combined_region_frequency_contrast.py` for VLAAI (GPU) and
# an AADNet-specific equivalent (inline, since AADNet's per-subject-per-fold
# checkpoint loading isn't factored into an importable module) for both
# models. Neither reruns the full XAI pipeline — both only occlude/permute a
# *combined group* of channels (or remove a frequency band from all 64
# channels at once) in one forward pass, which is fast even across 18
# subjects.
#
# **What this answers, that the existing per-channel/per-ROI/per-band tables
# don't:** does jointly removing the core-ROI group (or the important
# frequency band) hurt the model's P(attended) *more* than jointly removing
# an equally-sized "other" group — a real synergy/redundancy test, not an
# average of independently-measured single-channel effects.
#
# **Kaggle setup requirements**
# - Enable Internet + GPU
# - Attach dataset `dulanamanjitha/aad-xai-artifacts` (DTU EEG + audio, for
#   AADNet's `external/AADNet` DTUDataset)
# - VLAAI's own model (`models/vlaai.h5`) and DTU windows
#   (`data/vlaai_dtu_npz`) are committed to the repo directly -- no separate
#   dataset needed for that half.

# %% [markdown]
# ## 1. Clone repository and install dependencies

# %%
import os
import subprocess
import sys

REPO_DIR = "/kaggle/working/AAD_XAI"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "https://github.com/manjithadulana98/AAD_XAI.git", REPO_DIR], check=True)
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
# ## 2. GPU check

# %%
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## 3. VLAAI — combined-group contrast (reuses the tested standalone script)

# %%
OUT_VLAAI = "/kaggle/working/combined_contrast_vlaai"
cmd = [sys.executable, "scripts/run_combined_region_frequency_contrast.py",
       "--device", DEVICE, "--output-dir", OUT_VLAAI]
print("Command:", " ".join(cmd))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="")
proc.wait()
print(f"Exit code: {proc.returncode}")

# %% [markdown]
# ## 4. AADNet — authenticate to GCS + download SS checkpoints
#
# Same retry-with-backoff logic as `kaggle_run_xai_aadnet.py` (Kaggle's
# secrets/GCP-link proxy has been observed to fail for the first ~minute of a
# fresh session).

# %%
import json
import re
import tempfile
import time

from google.cloud import storage

GCP_PROJECT = "MscVM"
GCS_BUCKET = "addnet_results"
GCS_MODEL_PREFIX = "models/"
from pathlib import Path
LOCAL_CKPT_DIR = Path(REPO_DIR) / "external" / "AADNet" / "output" / "AADNet_DTUDataset"
LOCAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)

DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]
DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, "DTU dataset not found -- attach dulanamanjitha/aad-xai-artifacts."
os.environ["DATASET"] = DTU_ROOT
print(f"DTU dataset: {DTU_ROOT}")

RANDOM_SEED = 42
GCS_CONNECT_MAX_ATTEMPTS = 5
GCS_CONNECT_RETRY_DELAY_S = 20


def _authenticate_gcs():
    try:
        from kaggle_secrets import UserSecretsClient
        _sa_raw = UserSecretsClient().get_secret("GCP_SA_JSON")
        _sa_path = os.path.join(tempfile.gettempdir(), "gcp_sa.json")
        with open(_sa_path, "w") as f:
            f.write(_sa_raw)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _sa_path
        print(f"Kaggle Secret GCP_SA_JSON found.")
    except Exception as _e:
        print(f"Kaggle Secret GCP_SA_JSON not used ({type(_e).__name__}); falling back to ADC.")


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
    raise RuntimeError(f"Could not connect to GCS after {GCS_CONNECT_MAX_ATTEMPTS} attempts: {_last_exc!r}")

ss_blobs = [b for b in all_blobs if "AADNet_SS_T_" in b.name and b.name.endswith(".pth")]
CKPT_RE = re.compile(r"AADNet_SS_T_(?P<T>\d+)_s_(?P<s>\d+)_fold_(?P<fold>\d+)\.pth$")
ckpt_index = {}
for blob in ss_blobs:
    m = CKPT_RE.search(blob.name)
    if not m:
        continue
    local_path = LOCAL_CKPT_DIR / os.path.basename(blob.name)
    if not local_path.exists():
        blob.download_to_filename(str(local_path))
    ckpt_index[(int(m["s"]), int(m["fold"]))] = local_path
subject_ids = sorted({s for (s, _) in ckpt_index})
print(f"Downloaded/cached {len(ckpt_index)} SS checkpoints, subjects: {subject_ids}")

# %% [markdown]
# ## 5. AADNet — config, montage, model factory

# %%
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon
from utils.config import Config
from aad_xai.models.aadnet_external import ExternalAADNet
from aadnet.dataset import DTUDataset
from torch.utils.data import DataLoader

CONFIG_PATH = os.path.join(REPO_DIR, "external", "AADNet", "config", "config_AADNet_SS_DTU.yml")
aadnet_config = Config.load_config(CONFIG_PATH)
CHANNELS = list(aadnet_config.get(("dataset", "channels")))
N_CHANNELS = len(CHANNELS)
SR = int(aadnet_config.get(("dataset", "sr")))
TRAIN_WIN = int(aadnet_config.get(("dataset", "training_window")))
WINDOW_SAMPLES = TRAIN_WIN * SR
print(f"AADNet config -- channels={N_CHANNELS}, sr={SR} Hz, window={TRAIN_WIN}s ({WINDOW_SAMPLES} samples)")

MONTAGE_PATH = os.path.join(REPO_DIR, "config", "aadnet_dtu_channel_montage.csv")


def load_montage(csv_path):
    df = pd.read_csv(csv_path)
    roi_by_idx = {int(r.channel_index): r.roi for r in df.itertuples()}
    rois = OrderedDict()
    for ch_idx, roi in sorted(roi_by_idx.items()):
        rois.setdefault(roi, []).append(ch_idx)
    return roi_by_idx, rois


CH_ROI, ROIS = load_montage(MONTAGE_PATH)
CORE_ROIS = ["Fronto-Central", "Central", "Temporal", "Centro-Parietal"]
CORE_CHS = sorted({ch for roi in CORE_ROIS for ch in ROIS.get(roi, [])})
OTHER_CHS = sorted(set(range(N_CHANNELS)) - set(CORE_CHS))
print(f"Core ROI group: {len(CORE_CHS)} channels; other: {len(OTHER_CHS)} channels")

BANDS = OrderedDict([("delta", (0.5, 4.0)), ("theta", (4.0, 8.0)),
                     ("alpha", (8.0, 13.0)), ("beta", (13.0, 30.0))])
IMPORTANT_BAND = "theta"
OTHER_BAND = "alpha"
MAX_WINDOWS_PER_SUBJECT = 450


def build_aadnet(state_dict_path):
    m = ExternalAADNet(n_channels=N_CHANNELS, window_samples=WINDOW_SAMPLES, sfreq=SR)
    state = torch.load(str(state_dict_path), map_location=DEVICE)
    m.model.load_state_dict(state)
    m.eval().to(DEVICE)
    return m


def batched_forward(model, eeg, env, batch_size=128):
    outs = []
    with torch.no_grad():
        for i in range(0, eeg.shape[0], batch_size):
            outs.append(model(eeg[i:i + batch_size].to(DEVICE), env[i:i + batch_size].to(DEVICE)).cpu())
    return torch.cat(outs, 0)


def p_attended(model, eeg, env, y, batch_size=128):
    logits = batched_forward(model, eeg, env, batch_size=batch_size)
    probs = torch.softmax(logits, dim=-1)
    return probs.gather(1, y.unsqueeze(1)).squeeze(1).numpy()


def band_content(eeg_np, band_name):
    """eeg_np: (N, C, T). Vectorized bandpass along the time axis (axis=2)."""
    lo, hi = BANDS[band_name]
    nyq = SR / 2.0
    sos = butter(4, [max(lo / nyq, 0.01), min(hi / nyq, 0.99)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, eeg_np, axis=2, padtype="even", padlen=64)


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
    eeg = torch.cat(eegs_all, 0); env = torch.cat(envs_all, 0)
    y = torch.cat(ys_all, 0); fids = torch.cat(fids_all, 0)
    if eeg.shape[0] > max_windows:
        idx = torch.from_numpy(rng.choice(eeg.shape[0], max_windows, replace=False))
        eeg, env, y, fids = eeg[idx], env[idx], y[idx], fids[idx]
    return {"eeg": eeg, "env": env, "y": y, "fold_ids": fids}


print("Helpers ready.")

# %% [markdown]
# ## 6. AADNet — per-subject, per-fold combined-group contrast
#
# For each subject, iterate its folds (own held-out windows, own checkpoint)
# exactly like the main pipeline's `subject_perturbations`, but instead of
# 64 single-channel loops, jointly manipulate the core-ROI group, the
# other-ROI group, and the whole-brain theta/alpha bands.

# %%
subj_condition_means = {k: [] for k in
                         ["core_roi_occ", "core_roi_perm", "other_roi_occ", "other_roi_perm",
                          "band_theta", "band_alpha"]}
subjects_done = []

for sid in subject_ids:
    samples = collect_subject_windows(sid, MAX_WINDOWS_PER_SUBJECT)
    if samples is None:
        print(f"  subj {sid}: no windows, skipping")
        continue
    eeg, env, y, fids = samples["eeg"], samples["env"], samples["y"], samples["fold_ids"].numpy()
    N = eeg.shape[0]
    rng = np.random.RandomState(RANDOM_SEED + sid)
    accum = {k: np.zeros(N) for k in subj_condition_means}

    for fold in sorted(set(fids.tolist())):
        mask = fids == fold
        idx = np.where(mask)[0]
        ckpt_path = ckpt_index.get((sid, fold))
        if ckpt_path is None or idx.size == 0:
            continue
        model = build_aadnet(ckpt_path)
        eeg_f, env_f, y_f = eeg[idx], env[idx], y[idx]
        base = p_attended(model, eeg_f, env_f, y_f)

        for name, chs in [("core_roi", CORE_CHS), ("other_roi", OTHER_CHS)]:
            eeg_occ = eeg_f.clone()
            eeg_occ[:, chs, :] = 0.0
            accum[f"{name}_occ"][idx] = base - p_attended(model, eeg_occ, env_f, y_f)

            eeg_perm = eeg_f.clone()
            for ch in chs:
                perm = torch.from_numpy(rng.permutation(len(idx))).long()
                eeg_perm[:, ch, :] = eeg_f.index_select(0, perm)[:, ch, :]
            accum[f"{name}_perm"][idx] = base - p_attended(model, eeg_perm, env_f, y_f)

        eeg_np = eeg_f.numpy()
        for band in (IMPORTANT_BAND, OTHER_BAND):
            bc = band_content(eeg_np, band)
            eeg_band = eeg_f.clone() - torch.from_numpy(bc.astype(np.float32))
            accum[f"band_{band}"][idx] = base - p_attended(model, eeg_band, env_f, y_f)

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    for k in subj_condition_means:
        subj_condition_means[k].append(float(accum[k].mean()))
    subjects_done.append(sid)
    print(f"  subj {sid} done ({N} windows across {len(set(fids.tolist()))} folds)")

print(f"\nSubjects completed: {len(subjects_done)}/{len(subject_ids)}")

# %% [markdown]
# ## 7. AADNet — subject-level statistics + paired contrasts

# %%
def bootstrap_ci(values, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    means = np.array([values[rng.randint(0, len(values), len(values))].mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(x):
    x = np.asarray(x)
    if np.allclose(x, 0):
        return 1.0
    _, p = wilcoxon(x)
    return float(p)


def cohens_d(x):
    x = np.asarray(x)
    std = np.std(x, ddof=1)
    return float(np.mean(x) / std) if std > 1e-12 else 0.0


aadnet_results = {}
for key, vals in subj_condition_means.items():
    sv = np.array(vals)
    mean, lo, hi = bootstrap_ci(sv)
    aadnet_results[key] = {"subject_values": sv.tolist(), "mean_dp": mean, "ci_lo": lo, "ci_hi": hi,
                           "cohens_d": cohens_d(sv), "wilcox_p": wilcoxon_p(sv)}
    print(f"  {key:14s} mean dP={mean:+.6f} [{lo:+.6f},{hi:+.6f}] d={aadnet_results[key]['cohens_d']:+.3f} p={aadnet_results[key]['wilcox_p']:.4f}")

aadnet_contrasts = {}
for metric in ("occ", "perm"):
    a = np.array(aadnet_results[f"core_roi_{metric}"]["subject_values"])
    b = np.array(aadnet_results[f"other_roi_{metric}"]["subject_values"])
    diff = a - b
    mean, lo, hi = bootstrap_ci(diff)
    aadnet_contrasts[f"core_vs_other_{metric}"] = {"mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                                                    "cohens_d": cohens_d(diff), "wilcox_p": wilcoxon_p(diff)}
a = np.array(aadnet_results[f"band_{IMPORTANT_BAND}"]["subject_values"])
b = np.array(aadnet_results[f"band_{OTHER_BAND}"]["subject_values"])
diff = a - b
mean, lo, hi = bootstrap_ci(diff)
aadnet_contrasts[f"{IMPORTANT_BAND}_vs_{OTHER_BAND}"] = {"mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                                                          "cohens_d": cohens_d(diff), "wilcox_p": wilcoxon_p(diff)}

print("\nAADNet paired contrasts (important minus other, per subject):")
for key, c in aadnet_contrasts.items():
    print(f"  {key:22s} diff={c['mean_diff']:+.6f} [{c['ci_lo']:+.6f},{c['ci_hi']:+.6f}] d={c['cohens_d']:+.3f} p={c['wilcox_p']:.4f}")

out = {
    "n_subjects": len(subjects_done), "subjects": subjects_done,
    "core_roi_channels": CORE_CHS, "other_roi_channels": OTHER_CHS,
    "important_band": IMPORTANT_BAND, "other_band": OTHER_BAND,
    "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "subject_values"} for k, v in aadnet_results.items()},
    "contrasts": aadnet_contrasts,
}
OUT_AADNET_DIR = Path("/kaggle/working/combined_contrast_aadnet")
OUT_AADNET_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_AADNET_DIR / "combined_contrast_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {OUT_AADNET_DIR / 'combined_contrast_results.json'}")

# %% [markdown]
# ## 8. Display VLAAI's results for comparison

# %%
vlaai_json_path = os.path.join(OUT_VLAAI, "combined_contrast_results.json")
if os.path.isfile(vlaai_json_path):
    with open(vlaai_json_path) as f:
        print(json.dumps(json.load(f), indent=2))
else:
    print("VLAAI combined_contrast_results.json not found -- check Section 3 output for errors.")
