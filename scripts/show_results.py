"""Generate final results table from all experiment JSON files."""
import json
from pathlib import Path

results_dir = Path("results")
strategies = ["within_subject", "leave_story_out", "cross_condition", "loso"]
models = ["trf", "cnn", "stgcn"]

strategy_labels = {
    "within_subject": "Within-Subject",
    "leave_story_out": "Leave-Story-Out",
    "cross_condition": "Cross-Condition",
    "loso": "LOSO",
}

data = {}
for s in strategies:
    for m in models:
        f = results_dir / f"{s}_{m}.json"
        if f.exists():
            j = json.loads(f.read_text(encoding="utf-8"))
            data[(s, m)] = (j["mean_accuracy"], j["std_accuracy"], j["n_folds"])

# Print table
print()
print("=" * 80)
print("  FULL RESULTS: 4 CV Strategies x 3 Models on KULeuven (16 Subjects)")
print("=" * 80)
print()
header = f"  {'CV Strategy':<20s}  {'TRF':<22s}  {'CNN (AADNet)':<22s}  {'ST-GCN':<22s}"
print(header)
print(f"  {'-'*20}  {'-'*22}  {'-'*22}  {'-'*22}")

for s in strategies:
    row = f"  {strategy_labels[s]:<20s}"
    for m in models:
        if (s, m) in data:
            mean, std, n = data[(s, m)]
            cell = f"{mean*100:.1f}% +/- {std*100:.1f}%"
            row += f"  {cell:<22s}"
        else:
            row += f"  {'N/A':<22s}"
    print(row)

print()
print("  Notes:")
print("  - Window size: 2.0s, Epochs: 15, Patience: 5, Seed: 42")
print("  - Within-Subject: per-subject 4-fold (story-group-based), 64 total folds")
print("  - Leave-Story-Out: hold out 1/4 content groups, 4 folds")
print("  - Cross-Condition: train dry->test hrtf & vice versa, 2 folds")
print("  - LOSO: leave-one-subject-out, 16 folds")
print("  - Chance level: 50%")
print()

# Best results
print("  Best per strategy:")
for s in strategies:
    best_m, best_acc = "", 0
    for m in models:
        if (s, m) in data and data[(s, m)][0] > best_acc:
            best_acc = data[(s, m)][0]
            best_m = m.upper()
    print(f"    {strategy_labels[s]:<20s}: {best_m} ({best_acc*100:.1f}%)")

print()
print("  Best overall: ", end="")
best = max(data.items(), key=lambda x: x[0][0] != "within_subject" and x[1][0])
s, m = best[0]
mean = best[1][0]
print(f"{strategy_labels[s]} x {m.upper()} = {mean*100:.1f}%")
