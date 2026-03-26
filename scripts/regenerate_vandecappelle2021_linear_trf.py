"""Reproduce Vandecappelle et al. (eLife 2021) linear stimulus reconstruction baseline.

This script runs the TRF/linear stimulus reconstruction model on the KU Leuven
AAD dataset with settings aligned to the eLife 2021 baseline:

- Subject-specific CV with leave-one-story+speaker-out (only 2 valid folds per subject)
- Linear-model preprocessing: EEG bandpass 1–9 Hz and downsample to 20 Hz
- Exclude experiment-3 short repeat trials (use only exp 1 & 2)
- Speech envelope: powerlaw subbands (gammatone filterbank + |.|**0.6 + sum)
- TRF lag window: 0–250 ms
- Windowing: 50% overlap (overlap = window/2)

Example
-------
  c:/.../.venv/Scripts/python.exe scripts/regenerate_vandecappelle2021_linear_trf.py \
    --data-dir data/KULeuven --windows 5,10,20,40,60 --output results_trf_vandecappelle2021
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
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _write_markdown(rows: list[dict], md_path: Path, *, settings: dict) -> None:
    md: list[str] = []
    md.append("# TRF — Vandecappelle et al. (eLife 2021) — Reproduction")
    md.append("")
    md.append("Protocol:")
    md.append("- cv: `within_subject_story_speaker_out` (2 folds per subject; only stories with unique speakers)")
    md.append("- envelope: `powerlaw_subbands` (gammatone + powerlaw 0.6 + sum)")
    md.append("- EEG preprocessing: bandpass 1–9 Hz, sfreq_out=20 Hz, average reference")
    md.append("- include_experiments: 1,2 (exclude experiment-3 repeats)")
    md.append("- TRF lags: 0–250 ms")
    md.append("- window overlap: 50% (overlap = window/2)")
    md.append("")
    md.append("Settings:")
    md.append(f"- bandpass_hz: {settings['bandpass_hz']}")
    md.append(f"- sfreq_out: {settings['sfreq_out']} Hz")
    md.append(f"- trf_tmin: {settings['trf_tmin']:.3f}s")
    md.append(f"- trf_tmax: {settings['trf_tmax']:.3f}s")
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
    ap.add_argument("--output", type=str, default="results_trf_vandecappelle2021_linear")
    ap.add_argument("--windows", type=str, default="5,10,20,40,60")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--subjects", type=str, default=None, help="Optional comma-separated subject IDs to include.")

    # TRF knobs
    ap.add_argument("--trf-tune-alpha", action="store_true", help="Tune TRF ridge alpha.")
    ap.add_argument(
        "--trf-alpha-metric",
        type=str,
        default="corr",
        choices=["corr", "aad_acc"],
        help="Alpha selection metric. 'corr' uses GroupKFold RidgeCV + Pearson scoring.",
    )
    ap.add_argument("--trf-alphas", type=str, default=None)
    ap.add_argument("--trf-tmin", type=float, default=0.0)
    ap.add_argument("--trf-tmax", type=float, default=0.25)
    ap.add_argument("--max-train-seconds", type=float, default=1200.0)

    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = _parse_csv_floats(args.windows)

    preprocess = PreprocessConfig(sfreq_out=20, bandpass_hz=(1.0, 9.0), reref="average")

    print("=" * 72)
    print("Regenerate Vandecappelle (2021) linear TRF baseline")
    print("=" * 72)

    print("Loading KUL data once (audio=True)...")
    t0 = time.time()
    ds = KULeuvenDataset(
        root=args.data_dir,
        preprocess=preprocess,
        load_audio=True,
        envelope_method="powerlaw_subbands",
        include_experiments=(1, 2),
    )
    trials = list(ds.trials())

    if args.subjects:
        keep = {s.strip() for s in args.subjects.split(",") if s.strip()}
        trials = [t for t in trials if str(t.subject_id) in keep]

    print(f"Loaded {len(trials)} trials in {time.time() - t0:.1f}s")
    print(f"Subjects: {sorted(set(t.subject_id for t in trials))}")

    expected_folds = len(set(t.subject_id for t in trials)) * 2

    device = torch.device("cpu")

    settings = {
        "sfreq_out": preprocess.sfreq_out,
        "bandpass_hz": preprocess.bandpass_hz,
        "trf_tmin": float(args.trf_tmin),
        "trf_tmax": float(args.trf_tmax),
        "max_train_seconds": float(args.max_train_seconds),
        "trf_tune_alpha": bool(args.trf_tune_alpha),
        "trf_alpha_metric": str(args.trf_alpha_metric),
        "trf_alphas": str(args.trf_alphas) if args.trf_alphas is not None else "logspace(-7..7,15)",
    }

    summary_json_path = out_dir / "vandecappelle2021_linear_trf_summary.json"
    by_window: dict[float, dict] = {}
    if summary_json_path.exists():
        try:
            existing = json.loads(summary_json_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for r in existing:
                    by_window[float(r.get("window_s"))] = r
        except Exception:
            pass

    for w in windows:
        decision_w = float(w)
        train_w = decision_w
        overlap_s = decision_w / 2.0

        existing = by_window.get(decision_w)
        if existing and int(existing.get("n_folds", 0)) == int(expected_folds):
            print(f"\n(skip) window={decision_w:.0f}s already complete ({expected_folds} folds)")
            continue

        print(f"\n--- story+speaker-out x TRF x window={decision_w:.0f}s (overlap={overlap_s:.2f}s) ---")
        start = time.time()
        results = run_experiment(
            trials=trials,
            cv_name="within_subject_story_speaker_out",
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
            write_artifacts=False,
            trf_tune_alpha=bool(args.trf_tune_alpha),
            trf_alpha_metric=str(args.trf_alpha_metric),
            trf_alphas=args.trf_alphas,
            trf_max_train_seconds=float(args.max_train_seconds),
            trf_tmin_s=float(args.trf_tmin),
            trf_tmax_s=float(args.trf_tmax),
        )

        accs = [r["test_accuracy"] for r in results]
        row = {
            "cv": "within_subject_story_speaker_out",
            "model": "trf",
            "window_s": decision_w,
            "train_window_s": train_w,
            "overlap_s": overlap_s,
            "n_folds": len(results),
            "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
            "std_accuracy": float(np.std(accs)) if accs else 0.0,
            "time_s": float(time.time() - start),
        }
        by_window[decision_w] = row

        rows = list(by_window.values())
        rows.sort(key=lambda r: r["window_s"])
        summary_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        _write_markdown(rows, out_dir / "vandecappelle2021_linear_trf_summary.md", settings=settings)

        print(
            f"=> {decision_w:.0f}s: {row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f} "
            f"({row['n_folds']} folds, {row['time_s']:.0f}s)"
        )
        print("(summary updated)")

    print("\nSaved:")
    print(f"  - {out_dir / 'vandecappelle2021_linear_trf_summary.json'}")
    print(f"  - {out_dir / 'vandecappelle2021_linear_trf_summary.md'}")


if __name__ == "__main__":
    main()
