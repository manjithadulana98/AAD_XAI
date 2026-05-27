# %% [markdown]
# # AADNet Training — Kaggle Notebook
#
# Trains AADNet on DTU dataset using Kaggle GPU.
#
# **Training plan (fits within Kaggle's 12-hr session limit):**
# - **Session 1:** SI (LOSO) training (~2-3 hrs) + SS subjects 0-8 (~4-5 hrs)
# - **Session 2:** SS subjects 9-17 (~4-5 hrs)
#
# **Requirements:**
# - Enable **Internet** in notebook settings
# - Enable **GPU T4 x2** accelerator
# - Attach dataset: `dulanamanjitha/aad-xai-artifacts`

# %% [markdown]
# ## 1. Clone repository and install dependencies

# %%
import os
REPO_DIR = "/kaggle/working/AAD_XAI"

if not os.path.exists(REPO_DIR):
    os.system("git clone https://github.com/manjithadulana98/AAD_XAI.git " + REPO_DIR)
else:
    print(f"Repository already cloned at {REPO_DIR}")

os.chdir(REPO_DIR)
os.system("pip install -q -r requirements.txt")
os.system("pip install -q -e .")

# %% [markdown]
# ## 2. Verify GPU and dataset

# %%
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

# Verify DTU dataset accessible via Kaggle input
DTU_PATH = "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts"
eeg_dir = os.path.join(DTU_PATH, "eeg_new")
audio_dir = os.path.join(DTU_PATH, "Audio")

assert os.path.isdir(eeg_dir), f"Missing: {eeg_dir}"
assert os.path.isdir(audio_dir), f"Missing: {audio_dir}"

eeg_files = [f for f in os.listdir(eeg_dir) if f.endswith(".mat")]
print(f"\nEEG files: {len(eeg_files)} subjects in {eeg_dir}")
print(f"Audio dir: {audio_dir}")

# %% [markdown]
# ## 3. SI (LOSO) Training
#
# Leave-one-subject-out cross-validation. All 18 subjects, 8 folds, 20 epochs.
# Expected runtime: ~2-3 hours on T4.

# %%
os.chdir(os.path.join(REPO_DIR, "external", "AADNet"))

import subprocess, sys

print("=" * 70)
print("STARTING SI (LOSO) TRAINING")
print("=" * 70)

cmd_si = [
    sys.executable, "cross_validate_loso.py",
    "-c", "config/config_AADNet_SI_DTU_kaggle.yml",
    "-j", "LOSO_AADNet_DTU",
]

process = subprocess.Popen(
    cmd_si,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in process.stdout:
    print(line, end="")
process.wait()
print("=" * 70)
print(f"SI training exit code: {process.returncode}")

# %% [markdown]
# ## 4. SS Training — Part 1 (Subjects 0-8)
#
# Subject-specific fine-tuning using SI pretrained weights.
# Trains subjects 0 through 8. Expected runtime: ~4-5 hours on T4.
#
# **Before running:** Verify SI training completed successfully (check output above).

# %%
import yaml

# Create a temporary config for SS part 1 (subjects 0-8)
config_ss_path = "config/config_AADNet_SS_DTU_kaggle.yml"
config_ss_part1_path = "config/config_AADNet_SS_DTU_kaggle_part1.yml"

with open(config_ss_path, "r") as f:
    ss_config = yaml.safe_load(f)

ss_config["dataset"]["from_sbj"] = 0
ss_config["dataset"]["to_sbj"] = 9

with open(config_ss_part1_path, "w") as f:
    yaml.dump(ss_config, f, default_flow_style=False)

print(f"Created SS part 1 config: from_sbj=0, to_sbj=9")
print(f"Saved to: {config_ss_part1_path}")

# %%
print("=" * 70)
print("STARTING SS TRAINING — PART 1 (Subjects 0-8)")
print("=" * 70)

cmd_ss1 = [
    sys.executable, "cross_validate_ss.py",
    "-c", config_ss_part1_path,
    "-j", "SS_AADNet_DTU_part1",
]

process = subprocess.Popen(
    cmd_ss1,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in process.stdout:
    print(line, end="")
process.wait()
print("=" * 70)
print(f"SS part 1 exit code: {process.returncode}")

# %% [markdown]
# ## 5. SS Training — Part 2 (Subjects 9-17)
#
# **If running in the same session** and time allows (~4-5 hrs remaining),
# run this cell. Otherwise, start a **new Kaggle session** and:
# 1. Re-run cells 1-2 (clone + verify)
# 2. Copy SI outputs from your previous session's output artifact
# 3. Run this cell

# %%
# Create config for SS part 2 (subjects 9-17)
config_ss_part2_path = "config/config_AADNet_SS_DTU_kaggle_part2.yml"

with open(config_ss_path, "r") as f:
    ss_config = yaml.safe_load(f)

ss_config["dataset"]["from_sbj"] = 9
ss_config["dataset"]["to_sbj"] = 18

with open(config_ss_part2_path, "w") as f:
    yaml.dump(ss_config, f, default_flow_style=False)

print(f"Created SS part 2 config: from_sbj=9, to_sbj=18")

# %%
print("=" * 70)
print("STARTING SS TRAINING — PART 2 (Subjects 9-17)")
print("=" * 70)

cmd_ss2 = [
    sys.executable, "cross_validate_ss.py",
    "-c", config_ss_part2_path,
    "-j", "SS_AADNet_DTU_part2",
]

process = subprocess.Popen(
    cmd_ss2,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in process.stdout:
    print(line, end="")
process.wait()
print("=" * 70)
print(f"SS part 2 exit code: {process.returncode}")

# %% [markdown]
# ## 6. Verify outputs

# %%
os.chdir(os.path.join(REPO_DIR, "external", "AADNet"))

output_dir = "output/AADNet_DTUDataset"
results_dir = "results"

print("=== Output directory ===")
if os.path.isdir(output_dir):
    for item in sorted(os.listdir(output_dir)):
        full = os.path.join(output_dir, item)
        if os.path.isdir(full):
            count = len(os.listdir(full))
            print(f"  {item}/ ({count} files)")
        else:
            print(f"  {item} ({os.path.getsize(full):,} bytes)")
else:
    print(f"  Not found: {output_dir}")

print("\n=== Results directory ===")
if os.path.isdir(results_dir):
    for item in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, item)
        print(f"  {item} ({os.path.getsize(full):,} bytes)")
else:
    print(f"  Not found: {results_dir}")

# %% [markdown]
# ## 7. Copy outputs for download
#
# Copy training results to `/kaggle/working/` so they appear as notebook output artifacts.

# %%
import shutil

kaggle_output = "/kaggle/working/aadnet_results"
os.makedirs(kaggle_output, exist_ok=True)

# Copy output models
src_output = os.path.join(REPO_DIR, "external", "AADNet", "output")
if os.path.isdir(src_output):
    shutil.copytree(src_output, os.path.join(kaggle_output, "output"), dirs_exist_ok=True)
    print(f"Copied output/ to {kaggle_output}/output/")

# Copy results
src_results = os.path.join(REPO_DIR, "external", "AADNet", "results")
if os.path.isdir(src_results):
    shutil.copytree(src_results, os.path.join(kaggle_output, "results"), dirs_exist_ok=True)
    print(f"Copied results/ to {kaggle_output}/results/")

print(f"\nAll artifacts saved to: {kaggle_output}")
print("Use 'Save Version' to download as output artifact.")

# %% [markdown]
# ## Summary
#
# | Step | Description | Est. Time |
# |------|-------------|-----------|
# | SI (LOSO) | 18 subjects × 8 folds × 20 epochs | ~2-3 hrs |
# | SS Part 1 | Subjects 0-8 × 8 folds × 20 epochs | ~4-5 hrs |
# | SS Part 2 | Subjects 9-17 × 8 folds × 20 epochs | ~4-5 hrs |
# | **Total** | | **~10-13 hrs (across 2 sessions)** |
