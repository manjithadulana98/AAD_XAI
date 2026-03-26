"""Run strict SI + story-disjoint matrix with multiple folds and summarize results.

Default profile:
- CV: strict_si_story
- Models: trf, aadnet_ext, stgcn
- Windows: 1, 5, 10, 60 s
- max_folds: 2 (faster than full, better than 1-fold pilot)
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


def _summarize_results(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(out_dir.glob("strict_si_story_*_w*s.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        accs = [float(r["test_accuracy"]) for r in j.get("per_fold", [])]
        row = {
            "file": p.name,
            "cv": j.get("cv_strategy"),
            "model": j.get("model"),
            "window_s": float(j.get("window_s")),
            "n_folds": int(j.get("n_folds", 0)),
            "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
            "std_accuracy": float(np.std(accs)) if accs else 0.0,
        }
        rows.append(row)
    return rows


def _write_markdown(rows: list[dict], path: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["model"], r["window_s"]))
    lines = [
        "# Strict SI Story-Disjoint Multi-fold Matrix",
        "",
        "| Model | Window (s) | Folds | Accuracy (mean +/- std) |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['window_s']:.0f} | {r['n_folds']} | "
            f"{r['mean_accuracy']:.4f} +/- {r['std_accuracy']:.4f} |"
        )

    # Best per model
    lines.append("")
    lines.append("## Best per model")
    best: dict[str, dict] = {}
    for r in rows:
        m = r["model"]
        if m not in best or r["mean_accuracy"] > best[m]["mean_accuracy"]:
            best[m] = r
    for m, r in sorted(best.items()):
        lines.append(
            f"- {m}: {r['mean_accuracy']:.4f} at {r['window_s']:.0f}s ({r['n_folds']} folds)"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="results_strict_matrix_multifold")
    ap.add_argument("--max-folds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = [1.0, 5.0, 10.0, 60.0]
    models = ["trf", "aadnet_ext", "stgcn"]

    print("=" * 72)
    print("Strict SI + Story-disjoint Multi-fold Matrix")
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
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                seed=42,
                lr=args.lr,
                batch_size=args.batch_size,
                overlap_s=0.0,
                weight_decay=1e-4,
                max_folds=args.max_folds,
                output_dir=out_dir,
            )
            accs = [r["test_accuracy"] for r in results]
            mean_acc = float(np.mean(accs)) if accs else 0.0
            std_acc = float(np.std(accs)) if accs else 0.0
            elapsed = time.time() - start
            print(
                f"=> {model.upper()} @ {window_s:.0f}s: "
                f"{mean_acc:.4f} +/- {std_acc:.4f} "
                f"({len(results)} folds, {elapsed:.0f}s)"
            )

    rows = _summarize_results(out_dir)
    json_path = out_dir / "multifold_summary.json"
    md_path = out_dir / "multifold_summary.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_markdown(rows, md_path)

    print("\nSaved:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
