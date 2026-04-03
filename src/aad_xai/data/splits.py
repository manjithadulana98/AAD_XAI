from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, List
import numpy as np

@dataclass(frozen=True)
class Record:
    subject_id: str
    trial_id: str
    # Use a *trial/story/session id* here to prevent shared stimulus segments across splits.
    # If your dataset has a better unit (story/session), use that.
    group_id: str

@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]
    test: list[str]

def subject_independent_split(subject_ids: Sequence[str], train_frac: float, val_frac: float, seed: int) -> Split:
    """Split by subject to avoid subject overlap across train/val/test."""
    rng = np.random.default_rng(seed)
    subjects = list(subject_ids)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = min(max(n_train, 1), n-2) if n >= 3 else max(n_train, 1)
    n_val = min(max(n_val, 1), n-n_train-1) if n >= 3 else max(n_val, 0)

    train = subjects[:n_train]
    val = subjects[n_train:n_train+n_val]
    test = subjects[n_train+n_val:]
    return Split(train=train, val=val, test=test)

def assert_no_leakage(records: Sequence[Record], split: Split) -> None:
    """Basic leakage checks.

    You should extend this once you know the dataset structure.
    """
    s_train, s_val, s_test = set(split.train), set(split.val), set(split.test)
    assert s_train.isdisjoint(s_val) and s_train.isdisjoint(s_test) and s_val.isdisjoint(s_test), "Subject overlap across splits!"

    # group_id (trial/story/session) should not overlap across splits to avoid shared stimulus segments.
    groups_by_subject: Dict[str, set[str]] = {}
    for r in records:
        groups_by_subject.setdefault(r.subject_id, set()).add(r.group_id)

    def groups_for(subjects: set[str]) -> set[str]:
        out: set[str] = set()
        for s in subjects:
            out |= groups_by_subject.get(s, set())
        return out

    g_train = groups_for(s_train)
    g_val = groups_for(s_val)
    g_test = groups_for(s_test)

    assert g_train.isdisjoint(g_val) and g_train.isdisjoint(g_test) and g_val.isdisjoint(g_test), (
        "Shared stimulus groups across splits! Split by trial/story/session."
    )
