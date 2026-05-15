import numpy as np
from pathlib import Path
from collections import defaultdict

data_dir = Path("external/vlaai/evaluation_datasets/DTU")
files = sorted(data_dir.glob("*.npz"))
print(f"Total files: {len(files)}")

subj_files = defaultdict(list)
for f in files:
    parts = f.stem.split("_")
    subj = parts[1]
    subj_files[subj].append(f)

for s, fs in sorted(subj_files.items()):
    print(f"{s}: {len(fs)} trials")

s1_files = subj_files["S1"]
d0 = np.load(str(s1_files[0]))
d1 = np.load(str(s1_files[1]))
env0 = d0["envelope"]
env1 = d1["envelope"]
print(f"\nS1 trial 0 envelope shape: {env0.shape}")
print(f"S1 trial 1 envelope shape: {env1.shape}")
print(f"Same envelope? {np.array_equal(env0, env1)}")
print(f"Max diff: {np.abs(env0 - env1).max():.6f}")
print(f"Corr: {np.corrcoef(env0[:, 0], env1[:, 0])[0,1]:.4f}")
