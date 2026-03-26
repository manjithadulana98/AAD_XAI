"""Regenerate subject-specific 5-fold TRF results across window lengths.

This runs `cv=within_subject_5fold` with `model=trf` on KULeuven and reports
accuracy vs. decision window length.

Notes
-----
- This is *subject-specific* CV (train/val/test come from the same subject).
- Overlap defaults to a common AAD setting: step=max(0.5s, L/5) => overlap=L-step.

Example
-------
  c:/.../.venv/Scripts/python.exe scripts/regenerate_within_subject_5fold_trf.py \
    --data-dir data/KULeuven --windows 1,2,5,10 --output results_trf_within5fold
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aad_xai.config import PreprocessConfig
from aad_xai.data.kul_dataset import KULeuvenDataset
from aad_xai.run_experiments import run_experiment


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _write_markdown(rows: list[dict], md_path: Path, *, settings: dict) -> None:
    md: list[str] = []
    md.append("# TRF — Within-subject 5-fold Accuracy vs Window")
    md.append("")
    md.append("Protocol: `cv=within_subject_5fold`, model=`trf`, KULeuven.")
    md.append("Default stepping (literature-style): step=max(0.5s, L/5) so overlap=L-step (unless --overlap is provided).")
    md.append("")
    md.append("Settings:")
    md.append(f"- lags: tmin={settings['trf_tmin']:.3f}s, tmax={settings['trf_tmax']:.3f}s")
    md.append(f"- max_train_seconds: {settings['max_train_seconds']:.0f}s")
    md.append(f"- tune_alpha: {settings['trf_tune_alpha']} (metric={settings['trf_alpha_metric']})")
    md.append(f"- alpha_grid: {settings['trf_alphas']}")
    md.append("")
    md.append("| Decision window (s) | Train window (s) | Overlap (s) | Folds | Acc (mean +/- std) | Time (s) |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {int(round(r['window_s']))} | {int(round(r['train_window_s']))} | {r['overlap_s']:.2f} | {r['n_folds']} | {r['mean_accuracy']:.4f} +/- {r['std_accuracy']:.4f} | {r['time_s']:.0f} |"
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/KULeuven")
    ap.add_argument("--output", type=str, default="results_trf_within_subject_5fold")
    ap.add_argument("--windows", type=str, default="1,2,5,10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--trf-tune-alpha", action="store_true", help="Tune TRF ridge alpha on the fold validation split.")
    ap.add_argument(
        "--trf-alpha-metric",
        type=str,
        default="aad_acc",
        choices=["corr", "aad_acc"],
        help="Metric for choosing alpha when --trf-tune-alpha is set.",
    )
    ap.add_argument(
        "--trf-alphas",
        type=str,
        default=None,
        help="Optional comma-separated alpha grid (e.g., '1e-2,1e-1,1,10,100').",
    )
    ap.add_argument("--trf-tmin", type=float, default=0.0)
    ap.add_argument("--trf-tmax", type=float, default=0.5)
    ap.add_argument(
        "--overlap",
        type=float,
        default=float("nan"),
        help="If NaN (default), uses paper-style step=max(0.5s, L/5) => overlap=L-step.",
    )
    ap.add_argument(
        "--max-train-seconds",
        type=float,
        default=1200.0,
        help="Cap TRF training seconds per fold (training segment sampling).",
    )
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = _parse_csv_floats(args.windows)

    print("=" * 72)
    print("Regenerate within-subject 5-fold TRF results")
    print("=" * 72)

    print("Loading KUL data once (audio=True)...")
    t0 = time.time()
    ds = KULeuvenDataset(root=args.data_dir, preprocess=PreprocessConfig(), load_audio=True)
    trials = list(ds.trials())
    print(f"Loaded {len(trials)} trials in {time.time() - t0:.1f}s")

    device = torch.device("cpu")

    settings = {
        "trf_tmin": float(args.trf_tmin),
        "trf_tmax": float(args.trf_tmax),
        "max_train_seconds": float(args.max_train_seconds),
        "trf_tune_alpha": bool(args.trf_tune_alpha),
        "trf_alpha_metric": str(args.trf_alpha_metric),
        "trf_alphas": str(args.trf_alphas) if args.trf_alphas is not None else "logspace(-7..7,15)",
    }

    # If a previous run exists in this output directory, merge into it so
    # running a subset of windows (e.g. only 40/60s) doesn't wipe prior rows.
    summary_json_path = out_dir / "within_subject_5fold_trf_summary.json"
    existing_rows: list[dict] = []
    if summary_json_path.exists():
        try:
            existing_rows = json.loads(summary_json_path.read_text(encoding="utf-8"))
            if not isinstance(existing_rows, list):
                existing_rows = []
        except Exception:
            existing_rows = []

    by_window: dict[float, dict] = {}
    for r in existing_rows:
        try:
            by_window[float(r.get("window_s"))] = r
        except Exception:
            continue

    rows: list[dict] = list(existing_rows)
    for w in windows:
        decision_w = float(w)
        train_w = decision_w

        if np.isnan(float(args.overlap)):
            step_s = max(0.5, decision_w / 5.0)
            overlap_s = max(0.0, decision_w - step_s)
        else:
            overlap_s = float(args.overlap)

        print(f"\n--- within_subject_5fold x TRF x window={decision_w:.0f}s (overlap={overlap_s:.2f}s) ---")
        start = time.time()
        results = run_experiment(
            trials=trials,
            cv_name="within_subject_5fold",
            model_name="trf",
            window_s=decision_w,
            train_window_s=train_w,
            epochs=1,
            patience=1,
            device=device,
            seed=args.seed,
            lr=1e-3,
            batch_size=64,
            overlap_s=overlap_s,
            weight_decay=1e-4,
            max_folds=args.max_folds,
            output_dir=out_dir,
            trf_tune_alpha=bool(args.trf_tune_alpha),
            trf_alpha_metric=str(args.trf_alpha_metric),
            trf_alphas=args.trf_alphas,
            trf_max_train_seconds=float(args.max_train_seconds),
            trf_tmin_s=float(args.trf_tmin),
            trf_tmax_s=float(args.trf_tmax),
        )

        accs = [r["test_accuracy"] for r in results]
        row = {
            "cv": "within_subject_5fold",
            "model": "trf",
            "window_s": decision_w,
            "train_window_s": train_w,
            "overlap_s": overlap_s,
            "n_folds": len(results),
            "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
            "std_accuracy": float(np.std(accs)) if accs else 0.0,
            "time_s": float(time.time() - start),
        }
        rows.append(row)
        by_window[float(decision_w)] = row
        rows = list(by_window.values())
        print(
            f"=> TRF within_subject_5fold {decision_w:.0f}s: {row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f} ({row['n_folds']} folds, {row['time_s']:.0f}s)"
        )

        # Incremental save so partial progress is visible even if the run is interrupted.
        rows.sort(key=lambda r: r["window_s"])
        summary_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        _write_markdown(rows, out_dir / "within_subject_5fold_trf_summary.md", settings=settings)
        print("(summary updated)")

    rows.sort(key=lambda r: r["window_s"])
    summary_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_markdown(rows, out_dir / "within_subject_5fold_trf_summary.md", settings=settings)

    print("\nSaved:")
    print(f"  - {out_dir / 'within_subject_5fold_trf_summary.json'}")
    print(f"  - {out_dir / 'within_subject_5fold_trf_summary.md'}")


if __name__ == "__main__":
    main()
