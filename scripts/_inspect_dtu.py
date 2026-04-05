"""Inspect a DTU EEG mat file (deep drill)."""
import sys
import scipy.io as sio
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "data/DTU/eeg_new/S1.mat"
mat = sio.loadmat(path, squeeze_me=False)
d0 = mat["data"][0, 0]

# fsample
fs = int(d0["fsample"][0, 0]["eeg"][0, 0])
print("fsample =", fs, "Hz")

# EEG
eeg = d0["eeg"][0, 0]
print("eeg shape:", eeg.shape, "  -> (samples, channels)")
print("  duration:", round(eeg.shape[0] / fs, 1), "s")

# dim -> channel names
dim = d0["dim"][0, 0]
print("dim fields:", dim.dtype.names)
chan = dim["chan"][0, 0] if "chan" in dim.dtype.names else None
if chan is not None:
    print("  channels:", getattr(chan, "shape", chan))
    if hasattr(chan, "flat"):
        print("  chan[0..3]:", [str(chan.flat[i]) for i in range(min(4, chan.size))])

# events
ev_raw = d0["event"][0, 0]["eeg"]
print("\nevent.eeg raw shape:", ev_raw.shape)
# The event array may be stored as nested object arrays
ev_samples_raw = ev_raw.flat[0]["sample"].flatten() if ev_raw.dtype.names else ev_raw
ev = ev_raw.flat[0] if ev_raw.shape == (1,1) else ev_raw
print("event inner fields:", ev.dtype.names if hasattr(ev,"dtype") and ev.dtype.names else type(ev))
samples_col = ev["sample"].flatten()
values_col  = ev["value"].flatten()
print("n events:", len(samples_col))
print("sample range: min=%d max=%d" % (min(int(s.flat[0]) for s in samples_col), max(int(s.flat[0]) for s in samples_col)))
uniq_vals = sorted(set(int(v.flat[0]) for v in values_col))
print("unique trigger values:", uniq_vals[:20], "..." if len(uniq_vals)>20 else "")
print("First 10 (sample, value):")
for i in range(min(10, len(samples_col))):
    s = int(samples_col[i].flat[0])
    v = int(values_col[i].flat[0])
    print("  sample=%d  value=%d" % (s, v))

# expinfo triggers
ei = mat["expinfo"][:, 0]
triggers = [int(ei[i]["trigger"][0,0]) for i in range(len(ei))]
print("\nexpinfo triggers (first 10):", triggers[:10])
print("expinfo triggers unique:", sorted(set(triggers)))
print("n_speakers first 10:", [int(ei[i]["n_speakers"][0,0]) for i in range(10)])
print("attend_mf first 10:", [int(ei[i]["attend_mf"][0,0]) for i in range(10)])





