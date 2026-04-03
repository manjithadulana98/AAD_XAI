"""Cross-validation strategies for the KULeuven AAD dataset.

Four evaluation protocols, each progressively harder:

1. **within_subject_trial_cv** -- Per-subject K-fold over trials (K=4 story groups).
   Keeps all repetitions of the same story content in the same fold.

1b. **within_subject_5fold** -- Per-subject 5-fold CV over trials (stratified by label).
    This is a standard subject-specific CV setting; it does *not* enforce story/content-disjointness
    (because KUL has only 4 coarse story-content groups).

2. **leave_one_story_out** -- Hold out one story-part (content group 1-4);
   train on the remaining parts.  Tests generalisation to unseen speech.

3. **cross_condition** -- Train on one condition (dry/hrtf), test on the other.
   Tests robustness to spatial processing changes.

4. **leave_one_subject_out** -- LOSO: train on N-1 subjects, test on 1.
   Tests cross-subject generalisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from .base import Trial


# ======================================================================== #
#  Fold dataclass
# ======================================================================== #

@dataclass(frozen=True)
class CVFold:
    """One fold of a cross-validation split."""
    fold_id: str
    train_idx: list[int]   # indices into the trial list
    val_idx: list[int]
    test_idx: list[int]
    meta: dict              # arbitrary metadata about the fold


# ======================================================================== #
#  Story-content mapping for KULeuven
# ======================================================================== #

def _content_group(trial: Trial) -> int:
    """Map a KUL trial to its underlying story-content group (1-4).

    - Experiments 1 & 2: parts 1-4 map directly to content groups 1-4.
    - Experiment 3: 12 short repetition segments, 3 per original part.
      parts 1-3 -> CG 1, parts 4-6 -> CG 2, 7-9 -> CG 3, 10-12 -> CG 4.

    The group_id format from the KUL loader is usually:
        ``{subject}_exp{E}_p{P}_rep{R}_{condition}``
    """
    # Parse group_id to extract experiment and part.
    # Backward-compatible with legacy format containing subject prefix.
    parts = trial.group_id.split("_")
    exp_idx = next(i for i, p in enumerate(parts) if p.startswith("exp"))
    part_idx = next(i for i, p in enumerate(parts) if p.startswith("p"))
    exp = int(parts[exp_idx].replace("exp", ""))
    part = int(parts[part_idx].replace("p", ""))

    if exp in (1, 2):
        return part            # already 1-4
    else:  # exp 3
        return math.ceil(part / 3)  # 1-3->1, 4-6->2, 7-9->3, 10-12->4


def _condition(trial: Trial) -> str:
    """Extract the condition (dry / hrtf) from the group_id."""
    return trial.group_id.rsplit("_", 1)[-1]


# ======================================================================== #
#  Option 1: Within-subject trial-wise K-fold CV
# ======================================================================== #

def within_subject_trial_cv(
    trials: list[Trial],
    val_fraction: float = 0.25,
    seed: int = 42,
) -> Iterator[CVFold]:
    """Per-subject, leave-one-story-group-out 4-fold CV.

    For each subject, the 20 trials fall into 4 story-content groups (5
    trials each).  Each fold holds out one group as test, uses another
    as validation, and trains on the rest.

    Keeps all repetitions (exp 1/2/3) of the same story content together.

    Yields
    ------
    CVFold
        One fold per (subject, held-out story group) combination.
        Total folds = n_subjects * 4.
    """
    rng = np.random.default_rng(seed)
    subject_ids = sorted({t.subject_id for t in trials})

    for subj in subject_ids:
        subj_trials = [(i, t) for i, t in enumerate(trials) if t.subject_id == subj]
        # Group indices by content group
        cg_map: dict[int, list[int]] = {}
        for idx, t in subj_trials:
            cg = _content_group(t)
            cg_map.setdefault(cg, []).append(idx)

        groups = sorted(cg_map.keys())  # [1, 2, 3, 4]

        for test_g in groups:
            # Pick a validation group (next in cycle)
            remaining = [g for g in groups if g != test_g]
            val_g = remaining[rng.integers(len(remaining))]
            train_gs = [g for g in groups if g != test_g and g != val_g]

            test_idx = cg_map[test_g]
            val_idx = cg_map[val_g]
            train_idx = [i for g in train_gs for i in cg_map[g]]

            yield CVFold(
                fold_id=f"{subj}_cgTest{test_g}",
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                meta={
                    "strategy": "within_subject_trial_cv",
                    "subject": subj,
                    "test_content_group": test_g,
                    "val_content_group": val_g,
                },
            )


# ======================================================================== #
#  Option 1b: Within-subject 5-fold CV (trial-wise)
# ======================================================================== #

def within_subject_5fold(
    trials: list[Trial],
    seed: int = 42,
) -> Iterator[CVFold]:
    """Per-subject 5-fold CV over trials (stratified by label).

    For each subject, split that subject's trials into 5 folds while roughly
    preserving label balance in each fold.

    Notes
    -----
    - This is *not* story-disjoint. KUL's coarse story grouping is 4-way, so a
      strict story-group CV cannot be 5-fold. Use `within_subject` or
      `leave_story_out` / `strict_si_story` if you need story-disjointness.
    - Validation is taken as the next fold in a cycle: val=(k+1) mod 5.
    """
    rng = np.random.default_rng(seed)
    subject_ids = sorted({t.subject_id for t in trials})

    for subj in subject_ids:
        subj_idxs = [i for i, t in enumerate(trials) if t.subject_id == subj]
        if len(subj_idxs) < 5:
            continue

        # Stratify by label so each fold has a similar class balance.
        idx0 = [i for i in subj_idxs if int(trials[i].label) == 0]
        idx1 = [i for i in subj_idxs if int(trials[i].label) == 1]
        rng.shuffle(idx0)
        rng.shuffle(idx1)

        folds: list[list[int]] = [[] for _ in range(5)]
        for j, i in enumerate(idx0):
            folds[j % 5].append(i)
        for j, i in enumerate(idx1):
            folds[j % 5].append(i)

        # Shuffle within each fold for nicer ordering / determinism.
        for f in folds:
            rng.shuffle(f)

        for k in range(5):
            test_idx = list(folds[k])
            val_idx = list(folds[(k + 1) % 5])
            train_idx = [i for kk in range(5) if kk not in {k, (k + 1) % 5} for i in folds[kk]]

            if not train_idx or not val_idx or not test_idx:
                continue

            yield CVFold(
                fold_id=f"{subj}_k5_{k+1}",
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                meta={
                    "strategy": "within_subject_5fold",
                    "subject": subj,
                    "k": k + 1,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                },
            )


# ======================================================================== #
#  Option 2: Leave-one-story / leave-one-part out
# ======================================================================== #

def leave_one_story_out(
    trials: list[Trial],
    seed: int = 42,
) -> Iterator[CVFold]:
    """Hold out one story-content group (across ALL subjects).

    4 folds total: content groups 1-4.  Each fold tests on all trials
    whose story content matches the held-out group.

    Yields
    ------
    CVFold
        One fold per content group.
    """
    rng = np.random.default_rng(seed)
    cg_map: dict[int, list[int]] = {}
    for i, t in enumerate(trials):
        cg = _content_group(t)
        cg_map.setdefault(cg, []).append(i)

    groups = sorted(cg_map.keys())

    for test_g in groups:
        remaining = [g for g in groups if g != test_g]
        val_g = remaining[rng.integers(len(remaining))]
        train_gs = [g for g in groups if g != test_g and g != val_g]

        yield CVFold(
            fold_id=f"storyOut_cg{test_g}",
            train_idx=[i for g in train_gs for i in cg_map[g]],
            val_idx=cg_map[val_g],
            test_idx=cg_map[test_g],
            meta={
                "strategy": "leave_one_story_out",
                "test_content_group": test_g,
                "val_content_group": val_g,
            },
        )


# ======================================================================== #
#  Option 2b: Within-subject leave-one-story+speaker-out (eLife-style)
# ======================================================================== #

def within_subject_leave_one_story_speaker_out(
    trials: list[Trial],
    seed: int = 42,
) -> Iterator[CVFold]:
    """Subject-specific leave-one-story+speaker-out, matching Vandecappelle et al. (2021).

    The KUL dataset contains 4 story-content groups. Stories 3 and 4 share the
    same narrator (speaker). A strict leave-one-story+speaker-out evaluation can
    therefore only use the two folds where the held-out story has a unique
    narrator (content groups 1 and 2). This matches the eLife paper's statement
    that only two folds remain when enforcing both story and speaker separation.

    For each subject, yields 2 folds:
      - test: content group 1 (speaker 1)
      - test: content group 2 (speaker 2)

    Validation is chosen from the remaining story groups; training uses the rest.
    """
    rng = np.random.default_rng(seed)

    # Fixed speaker mapping for the 4 stories in the KUL dataset.
    # Story 3 & 4 share the same speaker.
    speaker_by_group = {1: 1, 2: 2, 3: 3, 4: 3}

    subject_ids = sorted({t.subject_id for t in trials})
    for subj in subject_ids:
        subj_trials = [(i, t) for i, t in enumerate(trials) if t.subject_id == subj]
        cg_map: dict[int, list[int]] = {}
        for idx, t in subj_trials:
            cg = _content_group(t)
            cg_map.setdefault(cg, []).append(idx)

        groups = sorted(cg_map.keys())
        if not groups:
            continue

        # Only content groups with unique speakers are allowed as test sets.
        allowed_test_groups = [g for g in (1, 2) if g in cg_map]
        for test_g in allowed_test_groups:
            test_speaker = speaker_by_group.get(test_g)
            remaining = [g for g in groups if g != test_g]
            if not remaining:
                continue

            # Choose a validation story group from the remaining ones.
            # Validation speaker can overlap with training; both are part of the training procedure.
            val_g = int(remaining[rng.integers(len(remaining))])
            train_gs = [g for g in remaining if g != val_g]

            test_idx = list(cg_map.get(test_g, []))
            val_idx = list(cg_map.get(val_g, []))
            train_idx = [i for g in train_gs for i in cg_map.get(g, [])]

            # Ensure train/test have no speaker overlap.
            train_speakers = {speaker_by_group.get(g) for g in train_gs}
            if test_speaker in train_speakers:
                continue

            if not train_idx or not val_idx or not test_idx:
                continue

            yield CVFold(
                fold_id=f"{subj}_storySpeakerOut_testCG{test_g}",
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                meta={
                    "strategy": "within_subject_leave_one_story_speaker_out",
                    "subject": subj,
                    "test_content_group": test_g,
                    "val_content_group": val_g,
                    "test_speaker": test_speaker,
                },
            )


# ======================================================================== #
#  Option 3: Cross-condition (dry <-> HRTF)
# ======================================================================== #

def cross_condition(
    trials: list[Trial],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Iterator[CVFold]:
    """Train on one condition, test on the other.

    Two folds: train=dry/test=hrtf, and train=hrtf/test=dry.
    Validation is carved from the training condition.

    Yields
    ------
    CVFold
        Two folds total.
    """
    rng = np.random.default_rng(seed)

    dry_idx = [i for i, t in enumerate(trials) if _condition(t) == "dry"]
    hrtf_idx = [i for i, t in enumerate(trials) if _condition(t) == "hrtf"]

    for train_cond, train_pool, test_idx in [
        ("dry", dry_idx, hrtf_idx),
        ("hrtf", hrtf_idx, dry_idx),
    ]:
        pool = np.array(train_pool)
        rng.shuffle(pool)
        n_val = max(1, int(len(pool) * val_fraction))
        val = pool[:n_val].tolist()
        train = pool[n_val:].tolist()

        yield CVFold(
            fold_id=f"cond_train{train_cond}",
            train_idx=train,
            val_idx=val,
            test_idx=test_idx,
            meta={
                "strategy": "cross_condition",
                "train_condition": train_cond,
                "test_condition": "hrtf" if train_cond == "dry" else "dry",
            },
        )


# ======================================================================== #
#  Option 4: Leave-one-subject-out (LOSO)
# ======================================================================== #

def leave_one_subject_out(
    trials: list[Trial],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Iterator[CVFold]:
    """Standard LOSO: train on N-1 subjects, test on 1.

    Validation is one randomly-chosen subject from the training set.

    Yields
    ------
    CVFold
        One fold per subject (N folds total).
    """
    rng = np.random.default_rng(seed)
    subj_idx: dict[str, list[int]] = {}
    for i, t in enumerate(trials):
        subj_idx.setdefault(t.subject_id, []).append(i)

    subjects = sorted(subj_idx.keys())

    for test_subj in subjects:
        rest = [s for s in subjects if s != test_subj]
        val_subj = rest[rng.integers(len(rest))]
        train_subjs = [s for s in rest if s != val_subj]

        yield CVFold(
            fold_id=f"loso_{test_subj}",
            train_idx=[i for s in train_subjs for i in subj_idx[s]],
            val_idx=subj_idx[val_subj],
            test_idx=subj_idx[test_subj],
            meta={
                "strategy": "leave_one_subject_out",
                "test_subject": test_subj,
                "val_subject": val_subj,
            },
        )


# ======================================================================== #
#  Option 5: Strict subject-independent + story-disjoint
# ======================================================================== #

def strict_subject_independent_story_disjoint(
    trials: list[Trial],
    seed: int = 42,
) -> Iterator[CVFold]:
    """Strict protocol: unseen subject and unseen story content.

    For each test subject and content group g:
      - test:  trials from test subject with group g
      - val:   one random non-test subject with one random group h != g
      - train: all remaining subjects with groups not in {g, h}

    This prevents train/test overlap in both subject identity and story content.
    """
    rng = np.random.default_rng(seed)
    subjects = sorted({t.subject_id for t in trials})
    groups = sorted({_content_group(t) for t in trials})

    by_subj_group: dict[tuple[str, int], list[int]] = {}
    for i, t in enumerate(trials):
        key = (t.subject_id, _content_group(t))
        by_subj_group.setdefault(key, []).append(i)

    for test_subj in subjects:
        other_subjects = [s for s in subjects if s != test_subj]
        for test_g in groups:
            val_subj = other_subjects[rng.integers(len(other_subjects))]
            val_groups = [g for g in groups if g != test_g]
            val_g = val_groups[rng.integers(len(val_groups))]

            test_idx = by_subj_group.get((test_subj, test_g), [])
            val_idx = by_subj_group.get((val_subj, val_g), [])

            train_idx: list[int] = []
            for subj in subjects:
                if subj in {test_subj, val_subj}:
                    continue
                for g in groups:
                    if g in {test_g, val_g}:
                        continue
                    train_idx.extend(by_subj_group.get((subj, g), []))

            if not train_idx or not val_idx or not test_idx:
                continue

            yield CVFold(
                fold_id=f"strict_si_{test_subj}_cg{test_g}",
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                meta={
                    "strategy": "strict_subject_independent_story_disjoint",
                    "test_subject": test_subj,
                    "test_content_group": test_g,
                    "val_subject": val_subj,
                    "val_content_group": val_g,
                },
            )


# ======================================================================== #
#  Registry
# ======================================================================== #

CV_STRATEGIES = {
    "within_subject": within_subject_trial_cv,
    "within_subject_5fold": within_subject_5fold,
    "within_subject_story_speaker_out": within_subject_leave_one_story_speaker_out,
    "leave_story_out": leave_one_story_out,
    "cross_condition": cross_condition,
    "loso": leave_one_subject_out,
    "strict_si_story": strict_subject_independent_story_disjoint,
}
