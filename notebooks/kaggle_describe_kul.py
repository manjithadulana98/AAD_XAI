# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # KU Leuven AAD dataset: structure check before pre-registration (CPU)
#
# Input: the output of the kernel `dulanamanjitha/kul-aad-dataset-fetch` (Zenodo 4004271
# v2.0, every file MD5-verified there), attached as a kernel source.
#
# Prints the design facts the KUL protocol depends on, for every subject and trial:
# experiment, `part` field, repetition, condition, attended ear and track, the left / right
# stimulus files, the EEG array shape and duration, and the channel labels. Checks:
# 1. the EEG is (samples x 64) at 128 Hz for every trial;
# 2. the channel order (labels in the file) against our BioSemi-64 montage;
# 3. what `part` means for experiment-3 trials (index 1-12 vs the rep_partN file played),
#    because `aad_xai.data.cv_splits._content_group` assumes ceil(part / 3);
# 4. whether the two experiment-1/2 presentations of a part use the same stimulus files;
# 5. the EEG duration against the dry stimulus duration.
#
# No decoding, no model, no accuracy: KUL stays untouched until the protocol is frozen.

# %%
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io
import scipy.io.wavfile

hits = glob.glob("/kaggle/input/**/KULeuven/S1.mat", recursive=True)
assert len(hits) == 1, f"attach kernel output dulanamanjitha/kul-aad-dataset-fetch: {hits}"
ROOT = Path(hits[0]).parent
manifest = json.load(open(ROOT / "manifest.json"))
print("KUL root:", ROOT, "| record", manifest["record"], "version", manifest["version"])

# Re-check the MD5 of every subject file against the manifest written at fetch time.
for name, info in sorted(manifest["files"].items()):
    p = ROOT / name
    if not p.exists():
        assert name == "stimuli.zip", f"missing {name}"   # unpacked into stimuli/ at fetch time
        continue
    h = hashlib.md5()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == info["md5"], f"{name}: md5 changed since the fetch"
print(f"md5 re-checked for {sum((ROOT / n).exists() for n in manifest['files'])} files")

# %%
def _s(x):
    """MATLAB char / cell leaf -> str."""
    a = np.asarray(x)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.flat[0])
    return str(a.flat[0]) if a.dtype.kind in "US" and a.size == 1 else str(a.squeeze().tolist())


def _num(x):
    a = np.asarray(x)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.flat[0])
    return float(a.flat[0])


MONTAGE = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7',
           'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz',
           'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2',
           'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10',
           'PO8', 'PO4', 'O2']   # config/aadnet_dtu_channel_montage.csv order (BioSemi-64)

wav_s = {}
for w in sorted((ROOT / "stimuli").glob("*_dry.wav")):
    sr, a = scipy.io.wavfile.read(w)
    wav_s[w.name] = a.shape[0] / sr

FILE_PART = re.compile(r"^(rep_)?part(\d)_track(\d)_(dry|hrtf)\.wav$")
rows = []
for mat in sorted(ROOT.glob("S*.mat"), key=lambda p: int(p.stem[1:])):
    tr = scipy.io.loadmat(mat)["trials"]
    tr = tr[0] if tr.ndim == 2 else tr
    for k, t in enumerate(tr):
        t = t[0, 0] if t.dtype == object else t
        rd, fh = t["RawData"][0, 0], t["FileHeader"][0, 0]
        eeg = np.asarray(rd["EegData"][0, 0]) if rd["EegData"].dtype == object else np.asarray(rd["EegData"])
        if mat.stem == "S1" and k == 0:
            print("S1 trial 1: EegData", eeg.shape, eeg.dtype)
            for name, src in (("RawData.Channels", rd["Channels"]), ("FileHeader.Channels", fh["Channels"])):
                try:   # informational; layout not documented beyond "channel numbers (1 to 64)"
                    a = np.asarray(src)
                    while a.dtype == object and a.size == 1:
                        a = np.asarray(a.flat[0])
                    vals = [_s(v) for v in a.flat] if a.dtype == object else np.atleast_1d(a.squeeze()).tolist()
                    print(f"  {name}: dtype {a.dtype}, {len(vals)} entries: {vals[:70]}")
                    if len(vals) == 64 and all(isinstance(v, str) for v in vals):
                        print(f"    equals our BioSemi-64 montage order: {vals == MONTAGE}; "
                              f"position of Cz: {vals.index('Cz') + 1 if 'Cz' in vals else None}")
                except Exception as e:
                    print(f"  {name}: could not parse ({type(e).__name__}: {e})")
            for name, src in (("ElectrodeCap", fh["ElectrodeCap"]), ("HighPass", rd["HighPass"]),
                              ("FilterImplementation", rd["FilterImplementation"])):
                try:
                    print(f"  {name}: {_s(src)}")
                except Exception as e:
                    print(f"  {name}: could not parse ({type(e).__name__})")
        stim = t["stimuli"][0, 0]
        left, right = _s(stim[0]), _s(stim[1])
        m_l, m_r = FILE_PART.match(left), FILE_PART.match(right)
        assert m_l and m_r, (mat.stem, k, left, right)
        rows.append({
            "subject": mat.stem, "k": k, "trial_id": int(_num(t["TrialID"])),
            "exp": int(_num(t["experiment"])), "part": int(_num(t["part"])), "rep": int(_num(t["repetition"])),
            "cond": _s(t["condition"]), "ear": _s(t["attended_ear"]), "track": int(_num(t["attended_track"])),
            "left": left, "right": right, "file_part": int(m_l.group(2)),
            "left_track": int(m_l.group(3)), "right_track": int(m_r.group(3)),
            "n": eeg.shape[0], "c": eeg.shape[1], "fs": _num(fh["SampleRate"]),
            "dry_s": wav_s[re.sub(r"_(dry|hrtf)\.wav$", "_dry.wav", left)],
        })
    print(f"{mat.stem}: {sum(r['subject'] == mat.stem for r in rows)} trials", flush=True)

# %% [markdown]
# ## Checks

# %%
assert all(r["c"] == 64 and r["fs"] == 128.0 for r in rows), "EEG is not (samples x 64) at 128 Hz everywhere"
print(f"1. EEG (samples x 64) at 128 Hz in all {len(rows)} trials")

assert all(m_l == m_r for m_l, m_r in ((r["left"].split("_track")[0], r["right"].split("_track")[0]) for r in rows))
assert all({r["left_track"], r["right_track"]} == {1, 2} for r in rows)
att_side_track = [r["right_track"] if r["ear"] == "R" else r["left_track"] for r in rows]
print("   attended_track field == track of the file on the attended ear in",
      sum(a == r["track"] for a, r in zip(att_side_track, rows)), "of", len(rows), "trials")

print("\n3. experiment 3: (part field -> rep file part) over all subjects:")
print("  ", dict(sorted(Counter((r["part"], r["file_part"]) for r in rows if r["exp"] == 3).items())))
print("   experiments 1/2: part field == file part in",
      sum(r["part"] == r["file_part"] for r in rows if r["exp"] in (1, 2)), "of",
      sum(r["exp"] in (1, 2) for r in rows), "trials")
ceil_ok = sum(-(-r["part"] // 3) == r["file_part"] for r in rows if r["exp"] == 3)
print(f"   ceil(part/3) == file part (cv_splits._content_group assumption) in {ceil_ok} of "
      f"{sum(r['exp'] == 3 for r in rows)} experiment-3 trials")

print("\n4. per subject, experiments 1/2: part -> [(exp, cond, ear, attended track, left file track)]")
same_files = True
for s in sorted({r["subject"] for r in rows}, key=lambda s: int(s[1:])):
    d = defaultdict(list)
    for r in rows:
        if r["subject"] == s and r["exp"] in (1, 2):
            d[r["file_part"]].append((r["exp"], r["cond"], r["ear"], r["track"], r["left_track"]))
    for p, lst in d.items():
        same_files &= len(lst) == 2 and lst[0][1] == lst[1][1]
    print(f"   {s:>3}: " + "  ".join(f"p{p} {sorted(v)}" for p, v in sorted(d.items())))
print("   each part presented twice per subject in the same condition (exp 1 and 2):", same_files)
print("   attended track by experiment:", dict(sorted(Counter((r["exp"], r["track"]) for r in rows).items())))
print("   condition by part (exp 1/2):", dict(sorted(Counter((r["file_part"], r["cond"]) for r in rows
                                                            if r["exp"] in (1, 2)).items())))
print("   attended ear by experiment:", dict(sorted(Counter((r["exp"], r["ear"]) for r in rows).items())))

print("\n5. EEG duration minus dry stimulus duration (s), by experiment:")
for e in (1, 2, 3):
    d = np.array([r["n"] / r["fs"] - r["dry_s"] for r in rows if r["exp"] == e])
    print(f"   exp {e}: min {d.min():+.1f}  median {np.median(d):+.1f}  max {d.max():+.1f}")
mins = defaultdict(float)
for r in rows:
    mins[(r["subject"], r["exp"] in (1, 2))] += r["n"] / r["fs"] / 60
print("   EEG minutes per subject, exp 1+2:",
      f"{min(v for (s, k), v in mins.items() if k):.1f}-{max(v for (s, k), v in mins.items() if k):.1f};",
      "exp 3:", f"{min(v for (s, k), v in mins.items() if not k):.1f}-{max(v for (s, k), v in mins.items() if not k):.1f}")
print("\nper-part dry durations (s):", {k: round(v, 1) for k, v in wav_s.items() if not k.startswith("rep_")})
