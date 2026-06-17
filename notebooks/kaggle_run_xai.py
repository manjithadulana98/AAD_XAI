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
# This runs all sections (A through G) with:
# - **All windows** (`--max-samples -1`)
# - **2000 bootstrap** iterations for tight CIs
# - **5000 sign-flip permutations** for robust p-values
# - **100 IG windows**, 50 interpolation steps
# - **GPU acceleration** (`--device cuda`)
# - **Subject Specificity Analysis** (Section G): per-channel cross-subject variability
#   metrics (agreement fraction, std, coefficient of variation) correlated against
#   occlusion ΔP, permutation ΔP, IG importance, and combined score

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
# ## 5. Subject Specificity Analysis
#
# Section G of the pipeline computes three cross-subject variability metrics per channel
# and correlates each with the four XAI importance features:
#
# | Specificity metric | Meaning |
# |---|---|
# | **agree_frac** | Fraction of subjects whose importance sign matches the group mean |
# | **subj_std** | Standard deviation of per-subject importance scores |
# | **CV** | Coefficient of variation (`std / |mean|`) — scale-free spread |
#
# Positive `r(agree_frac, importance)` → high-importance channels are consistently
# important across subjects (low specificity).
# Positive `r(subj_std, importance)` or `r(CV, importance)` → high-importance channels
# vary more across subjects (high specificity).

# %%
import json, pandas as pd
from IPython.display import display as ipy_display

spec_json = os.path.join(OUTPUT, "subject_specificity.json")
spec_csv  = os.path.join(OUTPUT, "subject_specificity.csv")

if os.path.isfile(spec_json):
    with open(spec_json, encoding="utf-8") as f:
        spec = json.load(f)

    print("=== Inter-subject profile similarity ===")
    print(f"  N subjects                  : {spec.get('n_subjects', 'N/A')}")
    print(f"  Mean r across subject pairs : {spec.get('intersubj_r_mean', float('nan')):+.4f}")
    print(f"  Std  r across subject pairs : {spec.get('intersubj_r_std',  float('nan')):.4f}")

    print("\n=== Per-subject profile similarity to group mean (Spearman ρ) ===")
    print(f"  Mean ρ = {spec.get('per_subject_spearman_mean', float('nan')):+.4f} "
          f"± {spec.get('per_subject_spearman_std', float('nan')):.4f}")
    ps_rho = spec.get("per_subject_spearman_rho", {})
    if ps_rho:
        rho_rows = [{"Subject": s, "Spearman ρ vs group": v} for s, v in sorted(ps_rho.items())]
        df_rho = pd.DataFrame(rho_rows).set_index("Subject")
        df_rho["Spearman ρ vs group"] = df_rho["Spearman ρ vs group"].map(
            lambda x: f"{x:+.4f}" if pd.notna(x) else "NaN")
        ipy_display(df_rho)

    print("\n=== Pearson r — Specificity vs Importance Features ===")
    corr = spec.get("feature_vs_specificity_correlations", {})
    rows = []
    for key, r_val in corr.items():
        parts = key.split("_vs_", 1)
        specificity = parts[0] if len(parts) == 2 else key
        feature     = parts[1] if len(parts) == 2 else ""
        rows.append({"Specificity metric": specificity, "XAI feature": feature,
                     "r": r_val if r_val is not None else float("nan")})
    df_corr = pd.DataFrame(rows).set_index(["Specificity metric", "XAI feature"])
    df_corr["r"] = df_corr["r"].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "NaN")
    ipy_display(df_corr)
else:
    print("subject_specificity.json not found — run the pipeline first.")

# %%
if os.path.isfile(spec_csv):
    df_spec = pd.read_csv(spec_csv)
    print(f"subject_specificity.csv  ({len(df_spec)} channels)")
    # Sort by absolute occlusion score descending for quick inspection
    if "occ_score" in df_spec.columns:
        df_spec = df_spec.reindex(df_spec["occ_score"].abs().sort_values(ascending=False).index)
    ipy_display(df_spec.head(20).reset_index(drop=True))
else:
    print("subject_specificity.csv not found — run the pipeline first.")

# %% [markdown]
# ## 6. Display plots (if generated)

# %%
from IPython.display import Image, display
import glob

# Subject-specificity and comparison plots shown first for quick review
spec_plots = [
    "subject_vs_group_heatmap.png",
    "subject_profile_similarity.png",
    "channel_disagreement.png",
    "subject_specificity_correlation.png",
    "subject_specificity_correlation_summary.png",
]
for name in spec_plots:
    pf = os.path.join(OUTPUT, name)
    if os.path.isfile(pf):
        print(f"\n--- {name} ---")
        display(Image(filename=pf))

# All remaining plots
plot_files = sorted(glob.glob(os.path.join(OUTPUT, "*.png")))
for pf in plot_files:
    name = os.path.basename(pf)
    if name not in spec_plots:
        print(f"\n--- {name} ---")
        display(Image(filename=pf))

# %% [markdown]
# ## 7. Download results
# 
# All results are saved to `/kaggle/working/xai_results/`.
# Use the Kaggle "Save Version" button to download them as an output artifact.
