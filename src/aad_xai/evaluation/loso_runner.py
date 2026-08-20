"""DTU + TRF baseline reproduction under true Leave-One-Subject-Out (Phase 1,
Work Packages 1 & 3, TRF only).

Split into a pure, filesystem-free core (`run_dtu_loso_trf_on_trials`, easy to
unit-test against synthetic fixtures) and a thin CLI-facing loader
(`run_dtu_loso_trf`) that constructs `DTUDataset`. Reuses the existing,
battle-tested `run_experiments.run_experiment()` orchestrator (fold loop,
resume support, artifact writing) via its `on_fold_result` hook rather than
duplicating or extracting its TRF fold-training logic.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import PreprocessConfig
from ..data.base import Trial
from ..data.cv_splits import CV_STRATEGIES, CVFold, assert_loso_fold_integrity
from ..data.dtu_dataset import DTUDataset
from ..data.windowing import assert_no_cross_split_overlap, make_windows
from ..run_experiments import run_experiment
from ..utils.logging import experiment_id, log_run_metadata, save_json
from ..utils.seed import seed_everything
from .baseline_table import write_baseline_performance_csv
from .schema import PredictionRecord, aggregate_subject_records, build_prediction_records, write_prediction_csv, write_subject_csv

ALLOWED_WINDOWS_S: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)


def _check_fold_windowing(trials: list[Trial], fold: CVFold, window_s: float, overlap_s: float) -> None:
    """Defense-in-depth: build per-split WindowIndex lists at `window_s` and
    assert no cross-split overlap. LOSO structurally guarantees this (splits
    are subject-disjoint), so this is cheap insurance, not expected to fire.
    """
    def _windows_for(idx: list[int]):
        out = []
        for i in idx:
            t = trials[i]
            out.extend(
                make_windows(
                    n_times=t.eeg.shape[-1], sfreq=t.sfreq, window_s=window_s,
                    overlap_s=overlap_s, label=t.label,
                    subject_id=t.subject_id, trial_id=t.trial_id,
                )
            )
        return out

    assert_no_cross_split_overlap(
        _windows_for(fold.train_idx), _windows_for(fold.val_idx), _windows_for(fold.test_idx),
    )


def run_dtu_loso_trf_on_trials(
    trials: list[Trial],
    *,
    output_dir: str | Path,
    window_s: float = 5.0,
    seed: int = 42,
    overlap_s: float = 0.0,
    trf_tmin_s: float = 0.0,
    trf_tmax_s: float = 0.5,
    trf_tune_alpha: bool = False,
    trf_alpha_metric: str = "corr",
    trf_alphas: Optional[str] = None,
    trf_max_train_seconds: float = 1200.0,
    max_folds: Optional[int] = None,
) -> dict:
    """Pure core: no filesystem/dataset dependency; caller loads `trials`
    (from `DTUDataset` in production, from a hand-built fixture in tests).
    """
    if window_s not in ALLOWED_WINDOWS_S:
        raise ValueError(f"window_s must be one of {ALLOWED_WINDOWS_S}, got {window_s}")

    exp_id = experiment_id("dtu", "trf", "loso", window_s, seed)
    run_dir = Path(output_dir) / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)

    # Upfront integrity pre-pass -- fail fast, before spending any compute.
    folds = list(CV_STRATEGIES["loso"](trials, seed=seed))
    if max_folds is not None:
        folds = folds[:max_folds]
    for fold in folds:
        assert_loso_fold_integrity(trials, fold)
        _check_fold_windowing(trials, fold, window_s, overlap_s)

    # `run_experiment` is resumable (skips folds already present in its JSON
    # summary with a matching config fingerprint) -- but a skipped fold never
    # invokes `on_fold_result`, which would silently leave window_records
    # incomplete (or entirely empty, on a resumed fully-complete run). The
    # standardized outputs below must always be complete for the folds this
    # call runs, so force a fresh recompute every time.
    window_tag = str(window_s).replace(".", "p")
    stale_json = run_dir / f"loso_trf_w{window_tag}s.json"
    if stale_json.exists():
        stale_json.unlink()

    window_records: list[PredictionRecord] = []
    weights_dir = run_dir / "trf_weights"

    def _collect(_fold: CVFold, res: dict) -> None:
        window_rows = res.get("window_rows") or []
        window_records.extend(
            build_prediction_records(
                window_rows, dataset="dtu", model="trf", window_seconds=window_s, seed=seed,
            )
        )

        # Persist the fitted TRF's weights for downstream Haufe-transform
        # explainability (trf_explain.py). Every fitted TRFDecoder in this
        # repo is otherwise fit-and-discarded in-process -- this is the
        # first place any TRF weights survive past a single function call.
        # Named by the fold's TEST subject (meta["test_subject"], not
        # val_idx -- LOSO's val_idx is a *different* subject held out only
        # for early-stopping/model-selection, not the subject this fold's
        # accuracy is actually reported against).
        coef = res.get("trf_coef")
        if coef is not None:
            weights_dir.mkdir(parents=True, exist_ok=True)
            held_out_subject = _fold.meta["test_subject"]
            np.savez(
                weights_dir / f"{held_out_subject}.npz",
                coef=coef,
                lags=res["trf_lags"],
                x_mean=res["trf_x_mean"],
                x_std=res["trf_x_std"],
            )

    fold_results = run_experiment(
        trials=trials,
        cv_name="loso",
        model_name="trf",
        window_s=window_s,
        train_window_s=window_s,
        epochs=1,
        patience=1,
        device=torch.device("cpu"),
        seed=seed,
        overlap_s=overlap_s,
        max_folds=max_folds,
        output_dir=run_dir,
        write_artifacts=True,
        trf_tune_alpha=trf_tune_alpha,
        trf_alpha_metric=trf_alpha_metric,
        trf_alphas=trf_alphas,
        trf_max_train_seconds=trf_max_train_seconds,
        trf_tmin_s=trf_tmin_s,
        trf_tmax_s=trf_tmax_s,
        on_fold_result=_collect,
    )

    n_subjects = len({t.subject_id for t in trials})
    if max_folds is None and len(fold_results) != n_subjects:
        raise AssertionError(
            f"LOSO should run exactly one fold per subject: got {len(fold_results)} "
            f"fold results for {n_subjects} subjects."
        )

    subject_records = aggregate_subject_records(window_records)

    write_prediction_csv(run_dir / "window_predictions.csv", window_records)
    write_subject_csv(run_dir / "subject_predictions.csv", subject_records)
    baseline_summary = write_baseline_performance_csv(
        run_dir / "baseline_performance.csv", subject_records, seed=seed,
    )

    log_run_metadata(
        run_dir,
        experiment_id=exp_id,
        dataset="dtu",
        model="trf",
        cv_strategy="loso",
        window_s=window_s,
        overlap_s=overlap_s,
        seed=seed,
        n_subjects=n_subjects,
        n_folds=len(fold_results),
    )

    return {
        "experiment_id": exp_id,
        "run_dir": str(run_dir),
        "n_folds": len(fold_results),
        "n_windows": len(window_records),
        "baseline_summary": baseline_summary,
    }


def run_dtu_loso_trf(
    *,
    data_dir: str,
    output_dir: str,
    window_s: float = 5.0,
    seed: int = 42,
    subjects: Optional[list[str]] = None,
    preprocess: Optional[PreprocessConfig] = None,
    **kwargs,
) -> dict:
    """CLI-facing: constructs `DTUDataset(root=data_dir, load_audio=True,
    preprocess=preprocess or PreprocessConfig())`, loads trials, optionally
    filters `--subjects`, delegates to the pure core.
    """
    preprocess = preprocess or PreprocessConfig()
    ds = DTUDataset(root=data_dir, load_audio=True, preprocess=preprocess)
    trials = list(ds.trials())
    if subjects:
        keep = set(subjects)
        trials = [t for t in trials if t.subject_id in keep]
    return run_dtu_loso_trf_on_trials(trials, output_dir=output_dir, window_s=window_s, seed=seed, **kwargs)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="DTU TRF baseline reproduction under true LOSO.")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default="results_dtu_loso_trf")
    ap.add_argument(
        "--windows", type=str, default=None,
        help="Comma-separated window lengths (subset of 1,2,5,10). Overrides --window-s.",
    )
    ap.add_argument("--window-s", type=float, default=5.0, choices=list(ALLOWED_WINDOWS_S))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs, e.g. S1,S2.")
    ap.add_argument("--trf-tune-alpha", action="store_true")
    ap.add_argument("--trf-alpha-metric", type=str, default="corr", choices=["corr", "aad_acc"])
    ap.add_argument("--trf-alphas", type=str, default=None)
    ap.add_argument("--trf-max-train-seconds", type=float, default=1200.0)
    ap.add_argument("--trf-tmin", type=float, default=0.0)
    ap.add_argument("--trf-tmax", type=float, default=0.5)
    ap.add_argument("--sfreq-out", type=int, default=64)
    ap.add_argument("--bandpass", type=str, default="1,8")
    args = ap.parse_args(argv)

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    low_hz, high_hz = (float(x) for x in args.bandpass.split(","))
    preprocess = PreprocessConfig(sfreq_out=int(args.sfreq_out), bandpass_hz=(low_hz, high_hz))

    windows = (
        [float(w) for w in args.windows.split(",") if w.strip()]
        if args.windows else [float(args.window_s)]
    )

    summaries = []
    for w in windows:
        print(f"=== DTU LOSO TRF: window={w}s ===")
        t0 = time.time()
        summary = run_dtu_loso_trf(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            window_s=w,
            seed=args.seed,
            subjects=subjects,
            preprocess=preprocess,
            overlap_s=args.overlap,
            max_folds=args.max_folds,
            trf_tune_alpha=args.trf_tune_alpha,
            trf_alpha_metric=args.trf_alpha_metric,
            trf_alphas=args.trf_alphas,
            trf_max_train_seconds=args.trf_max_train_seconds,
            trf_tmin_s=args.trf_tmin,
            trf_tmax_s=args.trf_tmax,
        )
        summary["time_s"] = time.time() - t0
        summaries.append(summary)
        print(
            f"  -> {summary['experiment_id']}: "
            f"mean_acc={summary['baseline_summary']['mean_accuracy']:.4f} "
            f"({summary['time_s']:.0f}s)"
        )

    if len(summaries) > 1:
        save_json(Path(args.output_dir) / "window_sweep_summary.json", summaries)


if __name__ == "__main__":
    main()
