# %% [markdown]
# # VLAAI XAI Analysis — Kaggle Notebook
# 
# Runs the full focused XAI pipeline (`run_focused_xai.py`) on Kaggle GPU.
# 
# **Requirements:**
# - Enable **Internet** in notebook settings (for git clone + pip install)
# - Enable **GPU T4 x2** accelerator
# - Dataset: `dulanamanjitha/aad-xai-artifacts` (DTU EEG data) — attached but NOT used directly by this notebook
#   (VLAAI uses its own preprocessed .npz format included in the git repo)

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
# ## 2. Verify GPU and paths

# %%
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

# Verify data files exist
DATA_DIR = os.path.join(REPO_DIR, "data", "vlaai_dtu_npz")
H5_PATH = os.path.join(REPO_DIR, "models", "vlaai.h5")
MONTAGE = os.path.join(REPO_DIR, "config", "dtu_channel_montage.csv")
OUTPUT = "/kaggle/working/xai_results"

assert os.path.isdir(DATA_DIR), f"Missing data dir: {DATA_DIR}"
assert os.path.isfile(H5_PATH), f"Missing model: {H5_PATH}"
assert os.path.isfile(MONTAGE), f"Missing montage: {MONTAGE}"

npz_count = len([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])
print(f"\nData dir: {DATA_DIR} ({npz_count} .npz files)")
print(f"Model: {H5_PATH} ({os.path.getsize(H5_PATH) / 1e6:.1f} MB)")
print(f"Montage: {MONTAGE}")
print(f"Output: {OUTPUT}")

# %% [markdown]
# ## 3. Run full XAI analysis
# 
# This runs all sections (A through F) with:
# - **All windows** (`--max-samples -1`)
# - **2000 bootstrap** iterations for tight CIs
# - **5000 sign-flip permutations** for robust p-values
# - **100 IG windows**, 50 interpolation steps
# - **GPU acceleration** (`--device cuda`)

# %%
import subprocess, sys

cmd = [
    sys.executable, "scripts/run_focused_xai.py",
    "--data-dir", DATA_DIR,
    "--h5-path", H5_PATH,
    "--montage-file", MONTAGE,
    "--output-dir", OUTPUT,
    "--max-samples", "-1",
    "--n-boot", "2000",
    "--n-perm", "5000",
    "--ig-samples", "100",
    "--ig-steps", "50",
    "--device", "cuda",
]

print("Command:", " ".join(cmd))
print("=" * 70)

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in process.stdout:
    print(line, end="")
process.wait()
print("=" * 70)
print(f"Exit code: {process.returncode}")

# %% [markdown]
# ## 4. Display results

# %%
report_path = os.path.join(OUTPUT, "FOCUSED_XAI_REPORT.txt")
if os.path.isfile(report_path):
    with open(report_path, "r") as f:
        print(f.read())
else:
    print("Report not found — check for errors above.")

# %%
# List all output files
print("Output files:")
if os.path.isdir(OUTPUT):
    for f in sorted(os.listdir(OUTPUT)):
        size = os.path.getsize(os.path.join(OUTPUT, f))
        print(f"  {f:50s}  {size:>10,} bytes")

# %% [markdown]
# ## 5. Display plots (if generated)

# %%
from IPython.display import Image, display
import glob

plot_files = sorted(glob.glob(os.path.join(OUTPUT, "*.png")))
for pf in plot_files:
    print(f"\n--- {os.path.basename(pf)} ---")
    display(Image(filename=pf))

# %% [markdown]
# ## 6. Download results
# 
# All results are saved to `/kaggle/working/xai_results/`.
# Use the Kaggle "Save Version" button to download them as an output artifact.
