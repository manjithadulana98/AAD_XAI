"""Regenerate paper-style LOSO accuracy-vs-window results.

This script is designed to produce a *paper-like* curve over decision window
lengths for the KULeuven dataset under LOSO (subject-independent) evaluation.

Key idea (common in AAD papers):
- Train/evaluate models on a base input window (e.g., 10s)
- For longer decision windows (e.g., 20s, 40s, 60s), aggregate base-window
  predictions into larger decision windows.

Notes:
- This workspace currently uses a CPU-only PyTorch build. Deep models across
  all LOSO folds and many windows will take a long time on CPU.
- TRF is fast enough to run full LOSO across multiple windows.

Examples:
  # TRF curve (full LOSO)
  python scripts/regenerate_paper_results_loso.py --models trf --output results_paper_loso_trf

  # AADNet-ext curve (WARNING: slow on CPU; consider --max-folds)
  python scripts/regenerate_paper_results_loso.py --models aadnet_ext --max-folds 2 --epochs 5 --output results_paper_loso_aadnetext

  # Include AADNet paper reference curve (from external/AADNet/results)
  python scripts/regenerate_paper_results_loso.py --models trf --compare-paper --output results_paper_loso_trf
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


_PAPER_WINDOWS = [1, 2, 5, 10, 20, 40]


def _parse_csv_list(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _load_aadnet_paper_curve() -> dict[str, dict[int, float]]:
    """Load reference (mean over subjects) curves from external/AADNet/results.

    Returns mapping:
      curve_name -> {window_seconds: mean_accuracy}

    Curves included:
      - SI_AADNet, SI_CCA, SI_LSQ, SI_NSR
      - SS_AADNet, SS_CCA, SS_LSQ, SS_NSR
    """

    base = Path("external/AADNet/results")
    windows = _PAPER_WINDOWS
    curves: dict[str, str] = {
        "SI_AADNet": "LOSO_AADNet_Das_final_SI_acc.npy",
        "SI_CCA": "LOSO_CCA_Das_final_SI_acc.npy",
        "SI_LSQ": "LOSO_LSQ_Das_final_SI_acc.npy",
        "SI_NSR": "LOSO_NSR_Das_final_SI_acc.npy",
        "SS_AADNet": "SS_AADNet_Das_final_SS_acc.npy",
        "SS_CCA": "SS_CCA_Das_final_SS_acc.npy",
        "SS_LSQ": "SS_LSQ_Das_final_SS_acc.npy",
        "SS_NSR": "SS_NSR_Das_final_SS_acc.npy",
    }

    out: dict[str, dict[int, float]] = {}
    for name, fn in curves.items():
        arr = np.load(base / fn)
        arr = arr.squeeze(0)  # (n_windows, n_subjects)
        mean = arr.mean(axis=1)
        out[name] = {int(w): float(m) for w, m in zip(windows, mean)}
    return out


def _write_markdown(rows: list[dict], md_path: Path, paper: dict[str, dict[int, float]] | None) -> None:
    md: list[str] = []
    md.append("# Paper-style LOSO Accuracy vs Window")
    md.append("")
    md.append("This run uses `cv=loso` on KULeuven. For decision windows > base train-window, deep models aggregate base-window predictions into larger decisions; TRF uses direct correlation scoring on the decision window.")
    md.append("")

    md.append("| Model | Decision window (s) | Train window (s) | Folds | Acc (mean +/- std) |")
    md.append("|---|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['model']} | {int(r['window_s'])} | {int(r['train_window_s'])} | {r['n_folds']} | {r['mean_accuracy']:.4f} +/- {r['std_accuracy']:.4f} |"
        )

    if paper:
        md.append("")
        md.append("## AADNet reference (external/AADNet) — SI mean curve")
        md.append("These are the mean accuracies shipped with the upstream AADNet repo (Das2019/KUL-style), for LOSO (subject-independent).")
        md.append("")
        md.append("| Window (s) | SI_LSQ | SI_CCA | SI_NSR | SI_AADNet |")
        md.append("|---:|---:|---:|---:|---:|")
        for w in _PAPER_WINDOWS:
            md.append(
                f"| {w} | {paper['SI_LSQ'][w]:.3f} | {paper['SI_CCA'][w]:.3f} | {paper['SI_NSR'][w]:.3f} | {paper['SI_AADNet'][w]:.3f} |"
            )

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/KULeuven")
    ap.add_argument("--output", type=str, default="results_paper_loso")
    ap.add_argument("--models", type=str, default="trf", help="Comma-separated: trf,aadnet_ext,cnn,stgcn")
    ap.add_argument("--windows", type=str, default=",".join(str(w) for w in _PAPER_WINDOWS))
    ap.add_argument("--base-train-window", type=float, default=10.0, help="Base input window used when aggregating long decision windows.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # If NaN, we auto-select paper-style overlap per window (step=max(0.5s, window/5)).
    ap.add_argument("--overlap", type=float, default=float("nan"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--compare-paper", action="store_true")
    ap.add_argument("--trf-tune-alpha", action="store_true", help="Tune TRF ridge alpha on the fold validation split (closer to baseline practice).")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = _parse_csv_list(args.models)
    windows = [float(x) for x in _parse_csv_list(args.windows)]

    # CPU-only torch in this environment.
    device = torch.device("cpu")

    print("=" * 72)
    print("Regenerate paper-style LOSO results")
    print("=" * 72)

    need_audio = any(m in {"trf", "aadnet_ext"} for m in models)
    print(f"Loading KUL data once (audio={need_audio})...")
    ds = KULeuvenDataset(
        root=args.data_dir,
        preprocess=PreprocessConfig(),
        load_audio=need_audio,
    )
    t0 = time.time()
    trials = list(ds.trials())
    print(f"Loaded {len(trials)} trials in {time.time() - t0:.1f}s")

    rows: list[dict] = []
    for model in models:
        for w in windows:
            decision_w = float(w)
            train_w = decision_w if decision_w <= float(args.base_train_window) else float(args.base_train_window)

            # Paper-style windowing: step=max(0.5s, L/5)
            if np.isnan(float(args.overlap)):
                # Overlap controls *base/train-window* segmentation in this codebase.
                step_s = max(0.5, train_w / 5.0)
                overlap_s = max(0.0, train_w - step_s)
            else:
                overlap_s = float(args.overlap)

            print(f"\n--- loso x {model.upper()} x decision={decision_w:.0f}s (train={train_w:.0f}s) ---")
            start = time.time()
            results = run_experiment(
                trials=trials,
                cv_name="loso",
                model_name=model,
                window_s=decision_w,
                train_window_s=train_w,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                seed=args.seed,
                lr=args.lr,
                batch_size=args.batch_size,
                overlap_s=overlap_s,
                weight_decay=args.weight_decay,
                max_folds=args.max_folds,
                output_dir=out_dir,
                trf_tune_alpha=bool(args.trf_tune_alpha),
            )
            accs = [r["test_accuracy"] for r in results]
            row = {
                "cv": "loso",
                "model": model,
                "window_s": decision_w,
                "train_window_s": train_w,
                "n_folds": len(results),
                "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
                "std_accuracy": float(np.std(accs)) if accs else 0.0,
                "time_s": float(time.time() - start),
            }
            rows.append(row)
            print(
                f"=> {model.upper()} decision {decision_w:.0f}s: {row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f} ({row['n_folds']} folds, {row['time_s']:.0f}s)"
            )

    rows.sort(key=lambda r: (r["model"], r["window_s"]))
    (out_dir / "paper_loso_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    paper = _load_aadnet_paper_curve() if args.compare_paper else None
    _write_markdown(rows, out_dir / "paper_loso_summary.md", paper)

    print("\nSaved:")
    print(f"  - {out_dir / 'paper_loso_summary.json'}")
    print(f"  - {out_dir / 'paper_loso_summary.md'}")


if __name__ == "__main__":
    main()
