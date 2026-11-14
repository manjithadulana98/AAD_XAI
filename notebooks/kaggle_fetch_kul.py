# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fetch the KU Leuven AAD dataset from Zenodo into Kaggle (CPU)
#
# Source: "Auditory Attention Detection Dataset KULeuven", version 2.0, Zenodo record
# 4004271 (doi 10.5281/zenodo.4004271), licence CC-BY-NC-SA 4.0 (research use; keep this
# kernel private, do not redistribute). 16 subjects (S1-S16.mat) and stimuli.zip, 5.15 GB.
#
# What this notebook does: downloads every file of the record straight from Zenodo,
# checks each file's MD5 against the checksum Zenodo publishes, unpacks stimuli.zip into
# stimuli/, and prints the dataset's structure (trials, sampling rate, channels, attended
# ear and track, stimulus pairs, conditions). The output folder KULeuven/ (S*.mat and
# stimuli/, the layout aad_xai.data.kul_dataset.KULeuvenDataset reads) is then attached to
# the decoding notebooks as a kernel source.
#
# What it does NOT do: no decoding and no accuracy of any kind. KUL is our held-out
# external test set for StimulusGAT: the model configuration, split and analysis are to be
# frozen and pre-registered before any notebook decodes attention on it.
#
# **Kaggle setup requirements**
# - Enable Internet
# - CPU is enough

# %%
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

RECORD = "4004271"
API = f"https://zenodo.org/api/records/{RECORD}"
ROOT = Path("/kaggle/working/KULeuven")
CHUNK = 8 * 1024 * 1024


def fetch_file(url: str, dst: Path, md5_expected: str, size_expected: int, tries: int = 4) -> float:
    """Stream ``url`` to ``dst``, checking size and MD5; retry on mismatch or network error.
    Returns seconds taken. Raises RuntimeError if every try fails."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, tries + 1):
        t0, h, n = time.time(), hashlib.md5(), 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aad-research-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                while True:
                    block = r.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
                    h.update(block)
                    n += len(block)
        except Exception as e:  # network error: retry
            print(f"  {dst.name}: attempt {attempt} failed ({type(e).__name__}: {e})")
            time.sleep(10 * attempt)
            continue
        if n == size_expected and h.hexdigest() == md5_expected:
            return time.time() - t0
        print(f"  {dst.name}: attempt {attempt} mismatch (size {n} vs {size_expected}, "
              f"md5 {h.hexdigest()} vs {md5_expected})")
    raise RuntimeError(f"could not fetch {url} with a matching checksum after {tries} tries")

# %% [markdown]
# ## 1. Download and verify every file of the record

# %%
with urllib.request.urlopen(urllib.request.Request(API, headers={"User-Agent": "aad-research-fetch/1.0"}),
                            timeout=60) as r:
    record = json.load(r)
meta = record["metadata"]
print(meta["title"], "| version", meta.get("version"), "| doi", record.get("doi"))
assert meta.get("version") == "2.0", "record version changed; re-check before use"

manifest = {"record": RECORD, "doi": record.get("doi"), "version": meta.get("version"), "files": {}}
t_all = time.time()
for f in sorted(record["files"], key=lambda f: f["key"]):
    algo, md5 = f["checksum"].split(":", 1)
    assert algo == "md5", f["checksum"]
    secs = fetch_file(f["links"]["self"], ROOT / f["key"], md5, f["size"])
    manifest["files"][f["key"]] = {"size": f["size"], "md5": md5}
    print(f"{f['key']:22s} {f['size'] / 1e6:8.1f} MB  md5 ok  ({secs:.0f}s)")
print(f"all {len(manifest['files'])} files verified in {time.time() - t_all:.0f}s")

# %% [markdown]
# ## 2. Unpack the stimuli into KULeuven/stimuli/

# %%
zpath = ROOT / "stimuli.zip"
tmp = ROOT / "_unzip"
with zipfile.ZipFile(zpath) as z:
    bad = z.testzip()
    assert bad is None, f"corrupt member {bad}"
    z.extractall(tmp)
wavs = sorted(tmp.rglob("*.wav"))
assert wavs, "no wav files in stimuli.zip"
stim_dir = ROOT / "stimuli"
stim_dir.mkdir(exist_ok=True)
for w in wavs:
    assert not (stim_dir / w.name).exists(), f"duplicate wav name {w.name}"
    shutil.move(str(w), stim_dir / w.name)
shutil.rmtree(tmp)
zpath.unlink()  # verified and unpacked; the md5 stays in the manifest
manifest["stimuli"] = sorted(w.name for w in stim_dir.glob("*.wav"))
print(f"{len(manifest['stimuli'])} wav files in stimuli/")

# %% [markdown]
# ## 3. Describe the data (structure only, no decoding)

# %%
import numpy as np
import scipy.io
import scipy.io.wavfile


def _s(x):
    """MATLAB char / cell leaf -> str."""
    a = np.asarray(x)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.flat[0])
    return str(a.flat[0]) if a.dtype.kind in "US" and a.size == 1 else str(a.squeeze().tolist())


for w in manifest["stimuli"]:
    sr, a = scipy.io.wavfile.read(stim_dir / w)
    print(f"  {w:32s} {sr} Hz  {a.shape[0] / sr:7.1f} s  channels {1 if a.ndim == 1 else a.shape[1]}")

subjects = {}
for mat in sorted(ROOT.glob("S*.mat"), key=lambda p: int(p.stem[1:])):
    m = scipy.io.loadmat(mat)
    tr = m["trials"]
    tr = tr[0] if tr.ndim == 2 else tr
    if mat.stem == "S1":
        t0 = tr[0][0, 0] if tr[0].dtype == object else tr[0]
        print("S1 trial fields:", t0.dtype.names)
        rd = t0["RawData"][0, 0]
        print("  RawData fields:", rd.dtype.names, "| EegData shape", rd["EegData"].shape)
        fh = t0["FileHeader"][0, 0]
        print("  FileHeader fields:", fh.dtype.names)
    rows, fs_set, ch_set, minutes = [], set(), set(), 0.0
    for t in tr:
        t = t[0, 0] if t.dtype == object else t
        fs = float(np.asarray(t["FileHeader"][0, 0]["SampleRate"][0, 0]).flat[0])
        eeg = t["RawData"][0, 0]["EegData"]
        fs_set.add(fs); ch_set.add(eeg.shape[1]); minutes += eeg.shape[0] / fs / 60
        stim = t["stimuli"][0, 0] if "stimuli" in t.dtype.names else None
        rows.append({
            "ear": _s(t["attended_ear"]) if "attended_ear" in t.dtype.names else None,
            "track": _s(t["attended_track"]) if "attended_track" in t.dtype.names else None,
            "condition": _s(t["condition"]) if "condition" in t.dtype.names else None,
            "experiment": _s(t["experiment"]) if "experiment" in t.dtype.names else None,
            "stimuli": tuple(_s(stim[i]) for i in range(stim.size)) if stim is not None else None,
            "seconds": round(eeg.shape[0] / fs, 1),
        })
    subjects[mat.stem] = rows
    print(f"{mat.stem}: {len(rows)} trials, fs {sorted(fs_set)}, EEG columns {sorted(ch_set)}, "
          f"{minutes:.1f} min; attended ear {dict(Counter(r['ear'] for r in rows))}; "
          f"experiments {dict(Counter(r['experiment'] for r in rows))}")

print("\nS1 trials (ear, track, condition, experiment, stimuli, seconds):")
for r in subjects["S1"]:
    print("  ", r["ear"], r["track"], r["condition"], r["experiment"], r["stimuli"], r["seconds"])
pairs = Counter(r["stimuli"] for rows in subjects.values() for r in rows)
print(f"\n{len(pairs)} distinct stimulus pairs over all subjects; most common:")
for p, n in pairs.most_common(12):
    print(f"  {n:3d}  {p}")
manifest["subjects"] = {s: len(r) for s, r in subjects.items()}
with open(ROOT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)
print("\nwrote", ROOT / "manifest.json")
