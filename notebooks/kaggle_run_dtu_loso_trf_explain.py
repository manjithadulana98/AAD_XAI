# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DTU LOSO TRF Explainability -- Full 18-Subject Sweep (Kaggle Notebook)
#
# Refits the DTU LOSO TRF baseline (`kaggle_run_dtu_loso_trf_full.py`'s
# validated pipeline, mean_acc=0.5344) with per-fold weight-saving enabled,
# then runs the new `aad_xai.xai.trf_explain` module end to end: Haufe
# activation patterns, window- and subject-level channel importance
# (occlusion + permutation + BH-FDR), ROI-level stats, a lag-block
# cascading-randomization sanity check, and deletion/insertion faithfulness
# curves -- full statistical parity with the AADNet/VLAAI XAI pipelines.
#
# **Kaggle setup requirements**
# - Enable Internet in notebook settings (for git clone + pip install)
# - Attach dataset `dulanamanjitha/aad-xai-artifacts` (holds DTU EEG + Audio)
# - No GPU needed -- TRF is CPU-only ridge regression throughout
#
# Outputs land in `/kaggle/working/results_dtu_loso_trf_explain/`.

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
    print(f"Repository already cloned at {REPO_DIR}")

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
print("Setup done.")

# %% [markdown]
# ## 2. Resolve the DTU dataset path

# %%
DTU_KAGGLE_ROOT_CANDIDATES = [
    "/kaggle/input/aad-xai-artifacts/datasets/DTU",
    "/kaggle/input/datasets/dulanamanjitha/aad-xai-artifacts/datasets/DTU",
]

DTU_ROOT = next((p for p in DTU_KAGGLE_ROOT_CANDIDATES if os.path.isdir(p)), None)
assert DTU_ROOT is not None, (
    "DTU dataset not found. Attach the 'dulanamanjitha/aad-xai-artifacts' dataset "
    "to this notebook. Tried: " + ", ".join(DTU_KAGGLE_ROOT_CANDIDATES)
)
print(f"DTU dataset: {DTU_ROOT}")

FIT_OUTPUT_DIR = "/kaggle/working/results_dtu_loso_trf_explain_fit"
EXPLAIN_OUTPUT_DIR = "/kaggle/working/results_dtu_loso_trf_explain"

# %% [markdown]
# ## 3. Refit the full 18-subject LOSO sweep, with per-fold weight-saving
#
# Same pipeline/window/seed as the validated accuracy-only sweep
# (`dulanamanjitha/dtu-loso-trf-full`) -- the only difference is that
# `loso_runner.py`'s `_collect` hook now also persists each fold's fitted
# TRF weights (`trf_coef`/`trf_lags`/`trf_x_mean`/`trf_x_std`) to
# `trf_weights/{subject}.npz`, which `trf_explain.py` needs. Re-fitting from
# scratch is cheap (~15 min observed for the accuracy-only sweep) so there's
# no value in trying to reuse the earlier run's outputs.

# %%
cmd = [
    sys.executable, "-m", "aad_xai.evaluation.loso_runner",
    "--data-dir", DTU_ROOT,
    "--output-dir", FIT_OUTPUT_DIR,
    "--window-s", "5",
    "--seed", "42",
]
print("Command:", " ".join(cmd))
print("=" * 70)
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="")
process.wait()
print("=" * 70)
print(f"Exit code: {process.returncode}")
assert process.returncode == 0, "LOSO refit failed -- see log above."

# %%
from pathlib import Path

run_dirs = sorted(Path(FIT_OUTPUT_DIR).glob("DTU_TRF_LOSO_*"))
assert run_dirs, f"No run directory found under {FIT_OUTPUT_DIR}"
run_dir = run_dirs[-1]
weights_dir = run_dir / "trf_weights"
print(f"Run directory: {run_dir}")
print(f"Weights saved: {sorted(p.name for p in weights_dir.glob('*.npz'))}")
assert len(list(weights_dir.glob("*.npz"))) == 18, "Expected one .npz per subject."

# %% [markdown]
# ## 4. Run the explainability pipeline
#
# Haufe patterns, window/subject-level channel importance, ROI-level stats,
# the lag-cascade sanity check, and deletion/insertion faithfulness curves.
#
# Cost note: a local benchmark (64 channels, TRFDecisionWrapper's per-window
# Python loop) measured ~9ms/forward-pass-window, and every DTU subject has
# ~600 test windows at a 5s decision window -- at full window counts and
# `trf_explain`'s own AADNet/VLAAI-parity defaults (k_step=4, 17 K-values,
# n_random_perms=20 -> 714 passes/fold), faithfulness alone would be
# ~20 CPU-hours across 18 folds. The flags below cap windows per fold at
# 200 (matching `run_xai_trf_comparison.py`'s own `--max-samples` default --
# not a new compromise, the same cap that pipeline already applies) and
# coarsen the faithfulness sweep to 9 K-values x 5 random perms. Estimated
# total from the same benchmark: ~70 min window-level importance + ~60 min
# faithfulness + ~15 min refit + <5 min Haufe/sanity/ROI =~ 2.5h, with
# comfortable margin under Kaggle's 12h cap even if this CPU runs slower.

# %%
EXPLAIN_CMD = [
    sys.executable, "-m", "aad_xai.xai.trf_explain",
    "--data-dir", DTU_ROOT,
    "--weights-dir", str(weights_dir),
    "--output-dir", EXPLAIN_OUTPUT_DIR,
    "--window-s", "5",
    "--seed", "42",
    "--max-windows-per-subject", "200",
    "--faithfulness-k-step", "8",
    "--faithfulness-random-perms", "5",
]
print("Command:", " ".join(EXPLAIN_CMD))
print("=" * 70)
process = subprocess.Popen(EXPLAIN_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="")
process.wait()
print("=" * 70)
print(f"Exit code: {process.returncode}")
assert process.returncode == 0, "trf_explain failed -- see log above."

# %% [markdown]
# ## 5. Display results

# %%
import json

out_dir = Path(EXPLAIN_OUTPUT_DIR)


def _load(name):
    p = out_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


window_table = _load("window_level_channel_importance.json")
subject_stats = _load("subject_level_channel_stats.json")
roi_stats = _load("roi_level_stats.json")
sanity = _load("sanity_check_lag_cascade.json")
faithfulness = _load("faithfulness_summary.json")

if window_table:
    top10 = sorted(window_table, key=lambda c: abs(c["combined_score"]), reverse=True)[:10]
    print("Top-10 channels by pooled window-level |combined_score|:")
    for c in top10:
        print(f"  ch{c['channel']:2d}  occ={c['occ_score']:+.5f}  perm={c['perm_score']:+.5f}  "
              f"type={c['contribution_type']:12s}  robust={c['robust_significant']}  "
              f"score={c['combined_score']:+.3f}")

if subject_stats:
    print(f"\nSubject-level tiers (n_subjects={subject_stats['n_subjects']}):")
    print(f"  Tier-1 (high-confidence): {subject_stats['n_tier1_high_confidence']} channels")
    print(f"  Tier-2 (candidate):       {subject_stats['n_tier2_candidate']} channels")
    for r in subject_stats["tier1_channels"]:
        print(f"    ch{r['channel']:2d}  tier={r['tier']}  occ_mean={r['occ_subj_mean']:+.5f}  "
              f"haufe_mean={r['haufe_subj_mean']:+.5f}")

if roi_stats:
    print("\nROI-level stats:")
    for r in roi_stats:
        print(f"  [{r['method']}] {r['roi']:16s}  mean_dP={r['mean_dP']:+.5f}  "
              f"d={r['cohens_d']:+.3f}  fdr_sig={r['fdr_sig']}")

if sanity:
    print("\nSanity check (lag-block cascading randomization), mean rho vs original by step:")
    n_lags = len(next(iter(sanity.values()))["cascade_steps"])
    for step_i in range(n_lags):
        rhos = [s["cascade_steps"][step_i]["rho_vs_original"] for s in sanity.values()]
        print(f"  after randomizing {step_i + 1}/{n_lags} lags: mean rho = {sum(rhos)/len(rhos):+.3f}")

if faithfulness:
    print("\nFaithfulness AUC summary:")
    print(json.dumps(faithfulness["auc_summary"], indent=2))

# %% [markdown]
# ## 6. Next steps
#
# These results are the "TRF LOSO explainability" subsection for the
# findings-log artifact -- same reporting shape as the existing AADNet/
# VLAAI sections (window/subject-level importance, Tier-1/2, faithfulness
# AUC), plus the Haufe-pattern and lag-cascade results unique to TRF's
# linear structure.
