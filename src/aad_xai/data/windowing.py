from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence

from .splits import Split


@dataclass(frozen=True)
class WindowIndex:
    start: int
    stop: int
    label: int
    subject_id: str
    trial_id: str


def make_windows(
    n_times: int,
    sfreq: float,
    window_s: float,
    overlap_s: float,
    label: int,
    subject_id: str,
    trial_id: str,
) -> list[WindowIndex]:
    """Create windows for a *single* continuous segment (e.g., one trial).

    IMPORTANT: only call this *within* a single split partition.
    Do NOT create overlapping windows that cross train/val/test boundaries.
    """
    win = int(round(window_s * sfreq))
    hop = int(round((window_s - overlap_s) * sfreq))
    hop = max(1, hop)

    if win > n_times:
        return []  # trial shorter than window — skip

    idx: list[WindowIndex] = []
    for start in range(0, max(0, n_times - win + 1), hop):
        stop = start + win
        if stop > n_times:
            break
        idx.append(WindowIndex(start=start, stop=stop, label=int(label), subject_id=subject_id, trial_id=trial_id))
    return idx


def assert_no_cross_split_overlap(
    train_windows: Sequence[WindowIndex],
    val_windows: Sequence[WindowIndex],
    test_windows: Sequence[WindowIndex],
) -> None:
    """Verify no (subject, trial, start, stop) tuple appears in more than one split.

    This is a defence-in-depth check: if splits are subject-independent, windows
    from different splits cannot share a trial.  But we check anyway.
    """
    def _key(w: WindowIndex) -> tuple:
        return (w.subject_id, w.trial_id, w.start, w.stop)

    sets = [
        {_key(w) for w in train_windows},
        {_key(w) for w in val_windows},
        {_key(w) for w in test_windows},
    ]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            assert len(overlap) == 0, (
                f"Cross-split window overlap detected between split {i} and {j}: "
                f"{len(overlap)} shared windows."
            )

