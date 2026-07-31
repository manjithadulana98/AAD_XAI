"""Relabel already-downloaded AADNet XAI results with the correct channel montage.

Background (see config/aadnet_dtu_channel_montage.csv and the fix in
notebooks/kaggle_run_xai_aadnet.py): AADNet's XAI notebook used to label its
64 channels via config/dtu_channel_montage.csv, a generic BioSemi-64 template
that does NOT match AADNet's actual channel order ("Fuglsang-64", hardcoded
at external/AADNet/aadnet/dataset.py:382). 58 of 64 names matched at
different index positions; 6 didn't match at all. This means every already
-completed AADNet run's channel names/ROIs are wrong, even though the
underlying per-channel-INDEX numbers are unaffected.

This script corrects an already-downloaded results directory WITHOUT needing
a fresh ~40-minute Kaggle re-run, in two different ways depending on the file:

1. Per-channel-indexed CSVs (channel_importance.csv, and -- if present from
   the Phase 5 backfill -- candidate_channels.csv, high_confidence_channels.csv,
   hierarchical_channel_stats.csv): a simple column relabel. The channel-index
   column is the ground truth; only the electrode-name/ROI columns attached
   to it were wrong.

2. ROI-level CSVs (roi_importance.csv, and subject_level_roi_stats.csv if
   present): NOT a simple relabel. These were computed by AVERAGING channels
   into ROI groups using the wrong channel-to-ROI mapping, so the aggregated
   numbers themselves are wrong, not just their labels. This script
   RECOMPUTES them from the cached per-subject-per-channel arrays
   (occ_subj_ch.npy / perm_subj_ch.npy, which every AADNet run already saves)
   using the corrected ROI grouping -- the same bootstrap-CI formula the
   original pipeline uses.

3. frequency_by_roi_subject.csv CANNOT be corrected by this script: the
   underlying frequency-band ablation itself was computed on the wrong set
   of channels per ROI at run time (not just mislabeled afterward), and that
   requires re-running the ablation against real model checkpoints. This
   script leaves it untouched and prints an explicit warning rather than
   silently ignoring it.

4. stream_ablation.csv has no channel names (only architecture-module names
   like "fc1") -- unaffected by this bug, left untouched.

Usage:
    python scripts/relabel_aadnet_channels.py --results-dir kaggle_output_AADnet/xai_results_aadnet
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORRECT_MONTAGE = ROOT / "config" / "aadnet_dtu_channel_montage.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Relabel already-downloaded AADNet XAI results with the correct channel montage.")
    p.add_argument("--results-dir", type=str, required=True,
                    help="Path to an AADNet xai_results_aadnet directory (e.g. kaggle_output_AADnet/xai_results_aadnet).")
    p.add_argument("--montage-file", type=str, default=str(CORRECT_MONTAGE),
                    help="Corrected montage CSV (default: config/aadnet_dtu_channel_montage.csv).")
    p.add_argument("--backup", action="store_true", default=True,
                    help="Copy each file to <name>.pre_relabel_backup.csv before overwriting (default: on).")
    p.add_argument("--no-backup", dest="backup", action="store_false")
    p.add_argument("--n-boot", type=int, default=500, help="Bootstrap iterations for ROI recomputation.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bootstrap_ci(values, n_boot=500, ci=0.95, seed=42):
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    means = np.array([values[rng.randint(0, len(values), len(values))].mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(values.mean()), float(lo), float(hi)


def load_correct_montage(path):
    df = pd.read_csv(path)
    ch_name = dict(zip(df.channel_index, df.electrode_name))
    ch_roi = dict(zip(df.channel_index, df.roi))
    rois = {}
    for idx, roi in sorted(ch_roi.items()):
        rois.setdefault(roi, []).append(idx)
    return ch_name, ch_roi, rois


def backup_file(path: Path, do_backup: bool):
    if do_backup:
        bak = path.with_suffix(".pre_relabel_backup.csv")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"    backed up original to {bak.name}")


def relabel_channel_csv(path: Path, ch_name: dict, ch_roi: dict, do_backup: bool):
    """Simple relabel: channel-index column is ground truth; overwrite the
    electrode-name/roi columns using the corrected mapping."""
    df = pd.read_csv(path)
    idx_col = "channel_idx" if "channel_idx" in df.columns else ("channel" if "channel" in df.columns else None)
    name_col = "electrode" if "electrode" in df.columns else ("electrode_name" if "electrode_name" in df.columns else None)
    if idx_col is None or name_col is None or "roi" not in df.columns:
        print(f"  SKIPPED {path.name}: expected a channel-index column plus electrode-name/roi columns, found {list(df.columns)}")
        return False
    backup_file(path, do_backup)
    df[name_col] = df[idx_col].map(ch_name)
    df["roi"] = df[idx_col].map(ch_roi)
    if df[name_col].isna().any() or df["roi"].isna().any():
        missing = df[df[name_col].isna() | df["roi"].isna()][idx_col].tolist()
        raise ValueError(f"{path.name}: channel indices {missing} not found in corrected montage -- refusing to write a partially-relabeled file.")
    df.to_csv(path, index=False)
    print(f"  RELABELED {path.name} ({len(df)} rows, columns '{name_col}'/'roi' corrected)")
    return True


def recompute_roi_csv(path: Path, results_dir: Path, rois: dict, n_boot: int, seed: int, do_backup: bool):
    """ROI-level files were aggregated with the WRONG channel groupings at
    computation time -- relabeling the roi column alone would not fix the
    underlying numbers. Recompute from the cached per-subject-per-channel
    arrays using the corrected ROI grouping."""
    occ_path = results_dir / "occ_subj_ch.npy"
    perm_path = results_dir / "perm_subj_ch.npy"
    if not occ_path.exists() or not perm_path.exists():
        print(f"  SKIPPED {path.name}: needs occ_subj_ch.npy + perm_subj_ch.npy (not found in {results_dir}) to recompute "
              "-- a simple column relabel is NOT sufficient here since the original aggregation used the wrong channels per ROI.")
        return False
    backup_file(path, do_backup)
    occ_subj_ch = np.load(occ_path)
    perm_subj_ch = np.load(perm_path)
    rows = []
    for roi_name, chs in rois.items():
        occ_vals = occ_subj_ch[:, chs].mean(axis=1)
        perm_vals = perm_subj_ch[:, chs].mean(axis=1)
        om, olo, ohi = bootstrap_ci(occ_vals, n_boot, seed=seed)
        pm, plo, phi = bootstrap_ci(perm_vals, n_boot, seed=seed)
        rows.append({
            "roi": roi_name, "n_channels": len(chs),
            "occ_mean_dp": om, "occ_ci_lo": olo, "occ_ci_hi": ohi,
            "perm_mean_dp": pm, "perm_ci_lo": plo, "perm_ci_hi": phi,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  RECOMPUTED {path.name} ({len(rows)} ROIs, from occ_subj_ch.npy/perm_subj_ch.npy with corrected grouping)")
    return True


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    ch_name, ch_roi, rois = load_correct_montage(args.montage_file)
    print(f"Loaded corrected montage: {len(ch_name)} channels, {len(rois)} ROIs")
    print(f"Relabeling results in: {results_dir}\n")

    fixed, skipped = [], []

    # 1. Per-channel-indexed CSVs -- simple relabel
    for name in ["channel_importance.csv", "candidate_channels.csv",
                 "high_confidence_channels.csv", "hierarchical_channel_stats.csv"]:
        path = results_dir / name
        if not path.exists():
            continue
        # candidate_channels.csv / high_confidence_channels.csv / hierarchical_channel_stats.csv
        # use "channel_index"/"channel_name" rather than "channel_idx"/"electrode" -- handle both.
        df_check = pd.read_csv(path, nrows=0)
        if "channel_name" in df_check.columns and "channel_index" in df_check.columns:
            df = pd.read_csv(path)
            backup_file(path, args.backup)
            df["channel_name"] = df["channel_index"].map(ch_name)
            df["roi"] = df["channel_index"].map(ch_roi)
            df.to_csv(path, index=False)
            print(f"  RELABELED {name} ({len(df)} rows, columns 'channel_name'/'roi' corrected)")
            fixed.append(name)
        elif relabel_channel_csv(path, ch_name, ch_roi, args.backup):
            fixed.append(name)
        else:
            skipped.append(name)

    # 2. ROI-level CSVs -- full recomputation from cached per-subject arrays
    for name in ["roi_importance.csv", "subject_level_roi_stats.csv"]:
        path = results_dir / name
        if not path.exists():
            continue
        if recompute_roi_csv(path, results_dir, rois, args.n_boot, args.seed, args.backup):
            fixed.append(name)
        else:
            skipped.append(name)

    # 3. Cannot be fixed retroactively
    freq_path = results_dir / "frequency_by_roi_subject.csv"
    if freq_path.exists():
        print(f"\n  NOT FIXED: {freq_path.name} -- the frequency-band ablation itself was computed "
              "on the wrong set of channels per ROI at run time (not just mislabeled afterward). "
              "This requires re-running Section F against real model checkpoints on Kaggle; "
              "left untouched.")
        skipped.append(freq_path.name + " (requires fresh Kaggle run)")

    # 4. Unaffected
    stream_path = results_dir / "stream_ablation.csv"
    if stream_path.exists():
        print(f"\n  UNAFFECTED: {stream_path.name} -- has no channel names (only architecture-module "
              "names like 'fc1'), left untouched.")

    print(f"\nDone. Fixed: {fixed}")
    if skipped:
        print(f"Not fixed (see messages above): {skipped}")


if __name__ == "__main__":
    main()
