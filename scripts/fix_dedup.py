"""Fix duplicate folds in loso_cnn.json"""
import json, numpy as np
from pathlib import Path

p = Path("results/loso_cnn.json")
d = json.loads(p.read_text(encoding="utf-8"))

seen = set()
unique = []
for r in d["per_fold"]:
    if r["fold_id"] not in seen:
        unique.append(r)
        seen.add(r["fold_id"])

d["per_fold"] = unique
d["n_folds"] = len(unique)
accs = [r["test_accuracy"] for r in unique]
d["mean_accuracy"] = float(np.mean(accs))
d["std_accuracy"] = float(np.std(accs))
p.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")

print(f"Fixed: {d['n_folds']} folds, mean={d['mean_accuracy']:.4f} +/- {d['std_accuracy']:.4f}")
for r in unique:
    print(f"  {r['fold_id']}: {r['test_accuracy']:.4f}")
