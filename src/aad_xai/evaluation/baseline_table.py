"""Baseline performance table generation (Phase 1, Work Package 3)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from ..utils.logging import save_json
from ..xai.composite_stability import bootstrap_ci
from .schema import SubjectRecord


def write_baseline_performance_csv(
    path: str | Path,
    subject_records: Sequence[SubjectRecord],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Write one row per subject to *path*, plus a companion
    ``<stem>_summary.json`` with mean/std/95% CI (bootstrap over subjects,
    via `xai.composite_stability.bootstrap_ci`). Returns the summary dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["subject_id", "n_windows", "n_correct", "accuracy"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in subject_records:
            writer.writerow(
                {
                    "subject_id": r.subject_id,
                    "n_windows": r.n_windows,
                    "n_correct": r.n_correct,
                    "accuracy": r.accuracy,
                }
            )

    accuracies = [r.accuracy for r in subject_records]
    if accuracies:
        mean_acc, ci_low, ci_high = bootstrap_ci(accuracies, n_boot=n_boot, seed=seed)
        std_acc = float(np.std(accuracies))
    else:
        mean_acc, ci_low, ci_high, std_acc = 0.0, 0.0, 0.0, 0.0

    summary = {
        "n_subjects": len(subject_records),
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }
    save_json(path.with_name(path.stem + "_summary.json"), summary)
    return summary
