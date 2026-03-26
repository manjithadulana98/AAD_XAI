"""Run strict subject-independent story-disjoint pilot matrix.

Pilot settings:
- CV: strict_si_story
- Models: trf, aadnet_ext, stgcn
- Windows: 1, 5, 10, 60 s
- max_folds=1 (fast comparable pilot)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from aad_xai.config import PreprocessConfig
from aad_xai.data.kul_dataset import KULeuvenDataset
from aad_xai.run_experiments import run_experiment


def main() -> None:
    output_dir = Path("results_strict_matrix_pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = [1.0, 5.0, 10.0, 60.0]
    models = ["trf", "aadnet_ext", "stgcn"]

    print("=" * 72)
    print("Strict SI + Story-disjoint Pilot Matrix")
    print("=" * 72)

    print("Loading KUL data once...")
    ds = KULeuvenDataset(
        root="data/KULeuven",
        preprocess=PreprocessConfig(),
        load_audio=True,
    )
    t0 = time.time()
    trials = list(ds.trials())
    print(f"Loaded {len(trials)} trials in {time.time() - t0:.1f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows: list[dict] = []
    for model in models:
        for window_s in windows:
            print(f"\n--- strict_si_story x {model.upper()} x {window_s:.0f}s ---")
            start = time.time()
            train_window_s = 10.0 if int(round(float(window_s))) == 60 else float(window_s)
            results = run_experiment(
                trials=trials,
                cv_name="strict_si_story",
                model_name=model,
                window_s=window_s,
                train_window_s=train_window_s,
                epochs=5,
                patience=2,
                device=device,
                seed=42,
                lr=1e-3,
                batch_size=64,
                overlap_s=0.0,
                weight_decay=1e-4,
                max_folds=1,
                output_dir=output_dir,
            )
            accs = [r["test_accuracy"] for r in results]
            mean_acc = float(np.mean(accs)) if accs else 0.0
            std_acc = float(np.std(accs)) if accs else 0.0
            elapsed = time.time() - start
            row = {
                "cv": "strict_si_story",
                "model": model,
                "window_s": window_s,
                "n_folds": len(results),
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "time_s": elapsed,
            }
            rows.append(row)
            print(
                f"=> {model.upper()} @ {window_s:.0f}s: "
                f"{mean_acc:.4f} +/- {std_acc:.4f} "
                f"({len(results)} fold, {elapsed:.0f}s)"
            )

    summary_path = output_dir / "pilot_matrix_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Also create a compact markdown table
    md_path = output_dir / "pilot_matrix_summary.md"
    lines = [
        "# Strict SI Story-Disjoint Pilot Matrix",
        "",
        "| Model | Window (s) | Accuracy (mean +/- std) | Folds | Time (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['window_s']:.0f} | "
            f"{r['mean_accuracy']:.4f} +/- {r['std_accuracy']:.4f} | "
            f"{r['n_folds']} | {r['time_s']:.0f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSaved:")
    print(f"  - {summary_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
