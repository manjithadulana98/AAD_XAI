"""Explore KUL trial metadata across all subjects."""
import scipy.io as sio
import numpy as np
from pathlib import Path
from collections import Counter

root = Path("data/KULeuven")
mat_files = sorted(root.glob("S*.mat"))
print(f"Subjects found: {len(mat_files)}")

all_trials = []
for mf in mat_files:
    mat = sio.loadmat(str(mf), squeeze_me=False)
    trials = mat["trials"]
    for i in range(trials.shape[1]):
        t = trials[0, i]
        subj = str(t["subject"][0, 0].flat[0])
        exp = int(t["experiment"][0, 0].flat[0])
        part = int(t["part"][0, 0].flat[0])
        rep = int(t["repetition"][0, 0].flat[0])
        cond = str(t["condition"][0, 0].flat[0])
        ear = str(t["attended_ear"][0, 0].flat[0])
        at = int(t["attended_track"][0, 0].flat[0])
        stim = t["stimuli"][0, 0]
        s0 = str(stim[0, 0].flat[0])
        s1 = str(stim[1, 0].flat[0])
        eeg_shape = t["RawData"][0, 0]["EegData"][0, 0].shape
        all_trials.append(dict(
            subj=subj, exp=exp, part=part, rep=rep,
            cond=cond, ear=ear, at=at,
            stim_left=s0, stim_right=s1, eeg_shape=eeg_shape
        ))

print(f"Total trials: {len(all_trials)}")
subj_set = sorted(set(t["subj"] for t in all_trials))
print(f"Subjects: {subj_set}")
print(f"Experiments: {Counter(t['exp'] for t in all_trials)}")
print(f"Parts: {Counter(t['part'] for t in all_trials)}")
print(f"Conditions: {Counter(t['cond'] for t in all_trials)}")
print(f"Repetitions: {Counter(t['rep'] for t in all_trials)}")
print(f"Attended ear: {Counter(t['ear'] for t in all_trials)}")

# Unique (exp, part, rep, cond) combos
combos = set()
for t in all_trials:
    combos.add((t["exp"], t["part"], t["rep"], t["cond"]))
print(f"\nUnique (exp, part, rep, cond) combos ({len(combos)}):")
for c in sorted(combos):
    cnt = sum(1 for t in all_trials if (t["exp"], t["part"], t["rep"], t["cond"]) == c)
    print(f"  exp={c[0]} part={c[1]} rep={c[2]} cond={c[3]:4s} -> {cnt} trials")

# Per-subject trial count
print("\nTrials per subject:")
for s in subj_set:
    cnt = sum(1 for t in all_trials if t["subj"] == s)
    print(f"  {s}: {cnt}")

# Check: for one subject, what's the trial breakdown
print("\nS1 trial detail:")
for t in all_trials:
    if t["subj"] == "S1":
        print(f"  exp={t['exp']} part={t['part']} rep={t['rep']} cond={t['cond']:4s} ear={t['ear']} track={t['at']} eeg={t['eeg_shape']}")
