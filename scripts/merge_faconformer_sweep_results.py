"""Merge FAConformer's per-seed DTU sweep result CSVs into one summary.

Reads the per-seed `DTU_FAConformer_{time_len}s_seed{seed}_results.csv` files
produced by `notebooks/kaggle_faconformer_sweep.py` (one CSV per Kaggle
kernel push, columns: subject, seed, best_epoch, n_epochs_run, test_loss,
test_acc, train_wall_seconds), concatenates them, and computes:
  - per-subject mean/std test_acc across whichever seeds are present
  - overall mean/std test_acc across all (subject, seed) rows

Local, dependency-light -- run after downloading each seed kernel's output
CSV from Kaggle. Works with however many of the 5 seed CSVs currently
exist, so it can be run for a partial-progress check before all 5 finish.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", type=str, default=None,
                    help="Directory to glob for DTU_FAConformer_*s_seed*_results.csv files.")
    p.add_argument("--csv-paths", type=str, nargs="+", default=None,
                    help="Explicit list of per-seed result CSV paths (overrides --csv-dir).")
    p.add_argument("--output-dir", type=str, default="xai_results_faconformer_sweep_merged")
    return p.parse_args()


def find_csv_paths(args):
    if args.csv_paths:
        return [Path(p) for p in args.csv_paths]
    if args.csv_dir:
        return sorted(Path(args.csv_dir).glob("DTU_FAConformer_*s_seed*_results.csv"))
    raise ValueError("Provide either --csv-dir or --csv-paths.")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = find_csv_paths(args)
    if not csv_paths:
        raise FileNotFoundError("No per-seed result CSVs found.")

    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        print(f"Loaded {p}: {len(df)} subjects, seed(s) present: {sorted(df['seed'].unique())}")
        frames.append(df)
    all_results = pd.concat(frames, ignore_index=True)

    seeds_present = sorted(all_results["seed"].unique())
    n_rows = len(all_results)
    print("=" * 70)
    print("FACONFORMER DTU SWEEP -- MERGED RESULTS")
    print("=" * 70)
    print(f"Seeds present: {seeds_present}  ({len(seeds_present)}/5)")
    print(f"Total (subject, seed) rows: {n_rows}")

    per_subject = (
        all_results.groupby("subject")["test_acc"]
        .agg(mean_test_acc="mean", std_test_acc="std", n_seeds="count")
        .reset_index()
        .sort_values("subject")
    )
    print("\nPer-subject test accuracy across seeds:")
    print(per_subject.to_string(index=False))

    overall_mean = all_results["test_acc"].mean()
    overall_std = all_results["test_acc"].std()
    print(f"\nOverall test_acc: mean={overall_mean:.4f}  std={overall_std:.4f}  (n={n_rows})")

    per_subject_path = out_dir / "faconformer_sweep_per_subject.csv"
    per_subject.to_csv(per_subject_path, index=False)

    all_results_path = out_dir / "faconformer_sweep_all_rows.csv"
    all_results.to_csv(all_results_path, index=False)

    summary_path = out_dir / "faconformer_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "seeds_present": seeds_present,
            "n_seeds_present": len(seeds_present),
            "n_rows": n_rows,
            "overall_mean_test_acc": overall_mean,
            "overall_std_test_acc": overall_std,
        }, f, indent=2)

    print(f"\nSaved {per_subject_path}")
    print(f"Saved {all_results_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
