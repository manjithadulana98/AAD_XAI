# %% [markdown]
# # AADNet Training — Google Compute Engine
#
# Trains AADNet on the DTU dataset using a GCE GPU VM.
# Dataset is read from `gs://aad_data/datasets/DTU` and cached locally.
#
# **Quick start (SSH into your VM):**
# ```bash
# git clone https://github.com/manjithadulana98/AAD_XAI.git ~/AAD_XAI
# cd ~/AAD_XAI
# bash scripts/setup_gcp_vm.sh        # one-time setup
# screen -S train                      # optional: keep alive after SSH drop
# bash scripts/train_gcp.sh            # runs SI then SS training end-to-end
# ```
#
# This notebook mirrors that shell script step-by-step so you can run
# individual cells interactively (e.g. from a Vertex AI Workbench instance).
#
# **Requirements:**
# - VM with NVIDIA GPU (T4 or L4 recommended)
# - Service account with `roles/storage.objectViewer` on `aad_data` bucket
# - `setup_gcp_vm.sh` already executed (Python env + CUDA torch installed)

# %% [markdown]
# ## 1. Environment & paths

# %%
import os
import subprocess
import sys

# Adjust if you cloned the repo elsewhere.
REPO_DIR = os.path.expanduser("~/AAD_XAI")
LOCAL_DATASET_DIR = os.path.join(REPO_DIR, "aad_data", "datasets", "DTU")
GCS_BUCKET_URI = "gs://aad_data/datasets/DTU"
ARTIFACTS_BUCKET = "gs://aad_data/artifacts/aad_xai"

os.chdir(REPO_DIR)
print(f"Repo dir    : {REPO_DIR}")
print(f"Dataset dir : {LOCAL_DATASET_DIR}")
print(f"GCS bucket  : {GCS_BUCKET_URI}")

# %% [markdown]
# ## 2. Verify GPU

# %%
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU count       : {torch.cuda.device_count()}")
else:
    raise RuntimeError("No CUDA GPU detected — check driver or VM accelerator config.")

# %% [markdown]
# ## 3. Sync DTU dataset from GCS bucket
#
# Uses `gsutil -m rsync` so subsequent runs only transfer changed files.

# %%
os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

print(f"Syncing {GCS_BUCKET_URI}  →  {LOCAL_DATASET_DIR} ...")
result = subprocess.run(
    ["gsutil", "-m", "rsync", "-r", GCS_BUCKET_URI, LOCAL_DATASET_DIR],
    check=True,
    text=True,
)

eeg_dir   = os.path.join(LOCAL_DATASET_DIR, "eeg_new")
audio_dir = os.path.join(LOCAL_DATASET_DIR, "Audio")
assert os.path.isdir(eeg_dir),   f"Missing: {eeg_dir}"
assert os.path.isdir(audio_dir), f"Missing: {audio_dir}"

eeg_files = [f for f in os.listdir(eeg_dir) if f.endswith(".mat")]
print(f"EEG files : {len(eeg_files)} subjects")
print(f"Audio dir : {audio_dir}")

# %% [markdown]
# ## 4. Build GCE configs
#
# The committed YAML files use `${DATASET}` as a placeholder.
# We write resolved copies to a temp dir and never modify the originals.

# %%
import tempfile
import shutil
import yaml

DATASET = os.path.realpath(LOCAL_DATASET_DIR)
print(f"DATASET = {DATASET}")

aadnet_dir = os.path.join(REPO_DIR, "external", "AADNet")
tmpdir = tempfile.mkdtemp(prefix="aadnet_cfg_")

def resolve_config(src_name: str, dst_name: str, overrides: dict | None = None) -> str:
    """Expand ${DATASET} in a config template and apply optional key overrides."""
    src = os.path.join(aadnet_dir, "config", src_name)
    with open(src) as f:
        text = f.read().replace("${DATASET}", DATASET)
    cfg = yaml.safe_load(text)
    if overrides:
        for dotkey, value in overrides.items():
            keys = dotkey.split(".")
            node = cfg
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = value
    dst = os.path.join(tmpdir, dst_name)
    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return dst

NUM_WORKERS = 6  # tune to your VM's vCPU count

si_cfg  = resolve_config("config_AADNet_SI_DTU.yml", "si.yml",
                         {"learning.running.num_workers": NUM_WORKERS})
ss_cfg  = resolve_config("config_AADNet_SS_DTU.yml", "ss.yml",
                         {"learning.running.num_workers": NUM_WORKERS})
print(f"SI config : {si_cfg}")
print(f"SS config : {ss_cfg}")

# %% [markdown]
# ## 5. SI (LOSO) Training
#
# Leave-one-subject-out cross-validation across all 18 subjects, 8 folds, 20 epochs.
# Expected runtime: **~2–3 hours** on an L4 or T4.

# %%
os.chdir(aadnet_dir)
os.makedirs("output", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 70)
print("STARTING SI (LOSO) TRAINING")
print("=" * 70)

proc = subprocess.Popen(
    [sys.executable, "cross_validate_loso.py", "-c", si_cfg, "-j", "LOSO_AADNet_DTU_GCE"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
print("=" * 70)
print(f"SI training exit code: {proc.returncode}")
assert proc.returncode == 0, "SI training failed — check output above."

# %% [markdown]
# ## 6. SS (Subject-Specific) Training — Part 1 (Subjects 0–8)
#
# Fine-tunes SI pretrained weights per subject. Subjects 0–8.
# Expected runtime: **~4–5 hours** on L4/T4.

# %%
ss_cfg_p1 = resolve_config(
    "config_AADNet_SS_DTU.yml", "ss_part1.yml",
    {
        "learning.running.num_workers": NUM_WORKERS,
        "dataset.from_sbj": 0,
        "dataset.to_sbj": 9,
    }
)

print("=" * 70)
print("STARTING SS TRAINING — PART 1 (Subjects 0–8)")
print("=" * 70)

proc = subprocess.Popen(
    [sys.executable, "cross_validate_ss.py", "-c", ss_cfg_p1, "-j", "SS_AADNet_DTU_GCE_part1"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
print("=" * 70)
print(f"SS part 1 exit code: {proc.returncode}")

# %% [markdown]
# ## 7. SS (Subject-Specific) Training — Part 2 (Subjects 9–17)
#
# Continue SS fine-tuning for subjects 9–17.
# Expected runtime: **~4–5 hours** on L4/T4.

# %%
ss_cfg_p2 = resolve_config(
    "config_AADNet_SS_DTU.yml", "ss_part2.yml",
    {
        "learning.running.num_workers": NUM_WORKERS,
        "dataset.from_sbj": 9,
        "dataset.to_sbj": 18,
    }
)

print("=" * 70)
print("STARTING SS TRAINING — PART 2 (Subjects 9–17)")
print("=" * 70)

proc = subprocess.Popen(
    [sys.executable, "cross_validate_ss.py", "-c", ss_cfg_p2, "-j", "SS_AADNet_DTU_GCE_part2"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
print("=" * 70)
print(f"SS part 2 exit code: {proc.returncode}")

# %% [markdown]
# ## 8. Verify outputs

# %%
os.chdir(aadnet_dir)

for dirname in ("output/AADNet_DTUDataset", "results"):
    print(f"\n=== {dirname} ===")
    if os.path.isdir(dirname):
        for item in sorted(os.listdir(dirname)):
            full = os.path.join(dirname, item)
            if os.path.isdir(full):
                print(f"  {item}/ ({len(os.listdir(full))} files)")
            else:
                print(f"  {item} ({os.path.getsize(full):,} bytes)")
    else:
        print(f"  Not found: {dirname}")

# %% [markdown]
# ## 9. Upload artifacts to GCS

# %%
import datetime

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
upload_base = f"{ARTIFACTS_BUCKET}/{run_id}"

os.chdir(REPO_DIR)
for src, tag in [
    ("external/AADNet/output",  "aadnet_output"),
    ("external/AADNet/results", "aadnet_results"),
]:
    if os.path.isdir(src):
        dest = f"{upload_base}/{tag}"
        print(f"Uploading {src}  →  {dest}")
        subprocess.run(["gsutil", "-m", "rsync", "-r", src, dest], check=True)

print(f"\nArtifacts at: {upload_base}")

# Cleanup temp configs
shutil.rmtree(tmpdir, ignore_errors=True)

# %% [markdown]
# ## Summary
#
# | Step | Description | Est. Time (L4/T4) |
# |------|-------------|-------------------|
# | SI (LOSO) | 18 subjects × 8 folds × 20 epochs | ~2–3 hrs |
# | SS Part 1 | Subjects 0–8 × 8 folds × 20 epochs | ~4–5 hrs |
# | SS Part 2 | Subjects 9–17 × 8 folds × 20 epochs | ~4–5 hrs |
# | **Total** | | **~10–13 hrs** |
#
# For an unattended full run use the shell script:
# ```bash
# screen -S train bash scripts/train_gcp.sh
# ```
