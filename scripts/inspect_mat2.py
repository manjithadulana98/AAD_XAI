"""Deeper inspection of trial fields in KULeuven MAT."""
import sys
import numpy as np
import scipy.io as sio

mat_path = sys.argv[1] if len(sys.argv) > 1 else "data/KULeuven/S1.mat"
mat = sio.loadmat(mat_path, squeeze_me=False)
trials = mat["trials"]

print(f"Total trials: {trials.shape[1]}")

for ti in range(min(3, trials.shape[1])):
    t = trials[0, ti]
    print(f"\n{'='*60}")
    print(f"Trial {ti}")
    print(f"{'='*60}")

    # Scalar fields
    for fname in ["attended_ear", "condition", "experiment", "part",
                   "attended_track", "repetition", "subject", "TrialID"]:
        val = t[fname][0, 0]
        # Try to extract value
        if val.dtype.names:
            print(f"  {fname}: structured dtype={val.dtype.names}")
        elif val.size == 1:
            print(f"  {fname}: {val.flat[0]}")
        elif val.dtype.kind in ('U', 'S', 'O'):
            print(f"  {fname}: {val}")
        else:
            print(f"  {fname}: shape={val.shape}, dtype={val.dtype}")

    # stimuli
    stim = t["stimuli"][0, 0]
    print(f"  stimuli: shape={stim.shape}, dtype={stim.dtype}")
    if stim.dtype == object:
        for i in range(stim.size):
            print(f"    stimuli[{i}]: {stim.flat[i]}")

    # RawData
    rd = t["RawData"][0, 0]
    print(f"  RawData: dtype fields = {rd.dtype.names}")
    for fname in rd.dtype.names:
        val = rd[fname][0, 0] if rd[fname].ndim >= 2 else rd[fname]
        sh = getattr(val, "shape", "N/A")
        dt = getattr(val, "dtype", "N/A")
        if hasattr(val, "size") and val.size <= 3:
            print(f"    RawData.{fname}: shape={sh}, dtype={dt}, value={val.flat[0] if val.size==1 else val}")
        else:
            print(f"    RawData.{fname}: shape={sh}, dtype={dt}")

    # FileHeader
    fh = t["FileHeader"][0, 0]
    print(f"  FileHeader: dtype fields = {fh.dtype.names}")
    for fname in fh.dtype.names:
        val = fh[fname][0, 0] if fh[fname].ndim >= 2 else fh[fname]
        sh = getattr(val, "shape", "N/A")
        if hasattr(val, "size") and val.size <= 5:
            print(f"    FileHeader.{fname}: value={val.flat[0] if val.size==1 else val}")
        else:
            print(f"    FileHeader.{fname}: shape={sh}")
