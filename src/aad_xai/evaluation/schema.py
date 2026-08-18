"""Standardized prediction-record schema (Phase 1, Work Package 2).

Every model/dataset combination should eventually emit records in this
shape. For now, only the TRF row format (window rows carrying `corr_a`/
`corr_b`, added additively in `run_experiments.py::_train_trf_fold`) is
supported by `build_prediction_records`.
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REQUIRED_COLUMNS = [
    "dataset", "model", "subject_id", "trial_id", "window_id",
    "window_seconds", "seed", "target", "prediction", "probability", "correct",
]
TRF_EXTRA_COLUMNS = ["attended_correlation", "unattended_correlation", "correlation_margin"]


@dataclass(frozen=True)
class PredictionRecord:
    dataset: str
    model: str
    subject_id: str
    trial_id: str
    window_id: int
    window_seconds: float
    seed: int
    target: int
    prediction: int
    probability: float
    correct: int
    attended_correlation: Optional[float] = None
    unattended_correlation: Optional[float] = None
    correlation_margin: Optional[float] = None


@dataclass(frozen=True)
class SubjectRecord:
    dataset: str
    model: str
    subject_id: str
    window_seconds: float
    seed: int
    n_windows: int
    n_correct: int
    accuracy: float
    mean_attended_correlation: Optional[float] = None
    mean_unattended_correlation: Optional[float] = None
    mean_correlation_margin: Optional[float] = None


def _softmax_p1(corr_a: float, corr_b: float) -> float:
    """P(class=1) via a 2-way softmax over the raw correlations -- a
    continuous, evidence-scaled analog of the legacy hard 0/1 `p_class1`
    flag (`1.0 if corr_b >= corr_a else 0.0`), numerically stable."""
    m = max(corr_a, corr_b)
    ea, eb = math.exp(corr_a - m), math.exp(corr_b - m)
    return eb / (ea + eb)


def _row_to_record(
    row: dict, *, dataset: str, model: str, window_seconds: float, seed: int, window_id: int,
) -> PredictionRecord:
    target = int(row["y_true"])
    prediction = int(row["y_pred"])
    corr_a = float(row["corr_a"])
    corr_b = float(row["corr_b"])

    attended_correlation = corr_a if target == 0 else corr_b
    unattended_correlation = corr_b if target == 0 else corr_a
    correlation_margin = attended_correlation - unattended_correlation

    return PredictionRecord(
        dataset=str(dataset),
        model=str(model),
        subject_id=str(row["subject_id"]),
        trial_id=str(row["trial_id"]),
        window_id=int(window_id),
        window_seconds=float(window_seconds),
        seed=int(seed),
        target=target,
        prediction=prediction,
        probability=float(_softmax_p1(corr_a, corr_b)),
        # Independent of the margin sign -- at an exact tie (corr_a == corr_b)
        # the decision rule assigns prediction=1, which is correct whenever
        # target==1 despite a zero margin.
        correct=int(target == prediction),
        attended_correlation=float(attended_correlation),
        unattended_correlation=float(unattended_correlation),
        correlation_margin=float(correlation_margin),
    )


def build_prediction_records(
    window_rows: Sequence[dict], *, dataset: str, model: str, window_seconds: float, seed: int,
) -> list[PredictionRecord]:
    """Build standardized records from TRF-shaped window rows (each with
    subject_id, trial_id, start, y_true, y_pred, corr_a, corr_b).

    `window_id` is assigned as a per-(subject_id, trial_id) running counter,
    ordered by `start` -- correct even if `window_rows` interleaves rows from
    different trials.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in window_rows:
        key = (str(row["subject_id"]), str(row["trial_id"]))
        groups.setdefault(key, []).append(row)

    records: list[PredictionRecord] = []
    for rows in groups.values():
        rows_sorted = sorted(rows, key=lambda r: int(r["start"]))
        for window_id, row in enumerate(rows_sorted):
            records.append(
                _row_to_record(
                    row, dataset=dataset, model=model,
                    window_seconds=window_seconds, seed=seed, window_id=window_id,
                )
            )
    return records


def aggregate_subject_records(records: Sequence[PredictionRecord]) -> list[SubjectRecord]:
    by_subject: dict[str, list[PredictionRecord]] = {}
    for r in records:
        by_subject.setdefault(r.subject_id, []).append(r)

    subject_records: list[SubjectRecord] = []
    for subject_id, rows in by_subject.items():
        n_windows = len(rows)
        n_correct = sum(r.correct for r in rows)
        accuracy = float(n_correct) / n_windows if n_windows else 0.0

        att = [r.attended_correlation for r in rows if r.attended_correlation is not None]
        unatt = [r.unattended_correlation for r in rows if r.unattended_correlation is not None]
        margin = [r.correlation_margin for r in rows if r.correlation_margin is not None]

        subject_records.append(
            SubjectRecord(
                dataset=rows[0].dataset,
                model=rows[0].model,
                subject_id=subject_id,
                window_seconds=rows[0].window_seconds,
                seed=rows[0].seed,
                n_windows=n_windows,
                n_correct=n_correct,
                accuracy=accuracy,
                mean_attended_correlation=float(np.mean(att)) if att else None,
                mean_unattended_correlation=float(np.mean(unatt)) if unatt else None,
                mean_correlation_margin=float(np.mean(margin)) if margin else None,
            )
        )

    subject_records.sort(key=lambda r: r.subject_id)
    return subject_records


def write_prediction_csv(path: str | Path, records: Sequence[PredictionRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = REQUIRED_COLUMNS + TRF_EXTRA_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def write_subject_csv(path: str | Path, records: Sequence[SubjectRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "model", "subject_id", "window_seconds", "seed",
        "n_windows", "n_correct", "accuracy",
        "mean_attended_correlation", "mean_unattended_correlation", "mean_correlation_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
