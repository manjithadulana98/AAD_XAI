"""Train just the missing LOSO fold (S9) for CNN and append to results."""
import json, time, sys
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn
from aad_xai.config import PreprocessConfig
from aad_xai.data.kul_dataset import KULeuvenDataset
from aad_xai.data.cv_splits import leave_one_subject_out
from aad_xai.data.torch_dataset import WindowedEEGDataset
from aad_xai.models.aadnet import AADNet
from aad_xai.utils.seed import seed_everything
from torch.utils.data import DataLoader
from pathlib import Path

print("Loading data...")
ds = KULeuvenDataset(root="data/KULeuven", preprocess=PreprocessConfig(), load_audio=False)
trials = list(ds.trials())
print(f"  {len(trials)} trials loaded")

# Find the S9 fold
for fold in leave_one_subject_out(trials, seed=42):
    if fold.fold_id == "loso_S9":
        break
else:
    print("ERROR: loso_S9 fold not found")
    sys.exit(1)

print(f"Training loso_S9: train={len(fold.train_idx)}, val={len(fold.val_idx)}, test={len(fold.test_idx)}")

train_t = [trials[i] for i in fold.train_idx]
val_t = [trials[i] for i in fold.val_idx]
test_t = [trials[i] for i in fold.test_idx]

ds_tr = WindowedEEGDataset(train_t, window_s=2.0, overlap_s=0.0)
ds_val = WindowedEEGDataset(val_t, window_s=2.0, overlap_s=0.0)
ds_te = WindowedEEGDataset(test_t, window_s=2.0, overlap_s=0.0)
print(f"  Windows: tr={len(ds_tr)}, val={len(ds_val)}, te={len(ds_te)}")

seed_everything(42)
device = torch.device("cpu")
model = AADNet(n_channels=64).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True, num_workers=0)
dl_val = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=0)
dl_te = DataLoader(ds_te, batch_size=64, shuffle=False, num_workers=0)

best_val_loss, patience_cnt, best_state = float("inf"), 0, None
t0 = time.time()

for ep in range(1, 16):
    model.train()
    for xb, yb in dl_tr:
        xb = xb.to(device)
        yb = torch.as_tensor(yb, dtype=torch.long, device=device)
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()

    model.eval()
    vloss, vcorr, vtot = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(device)
            yb = torch.as_tensor(yb, dtype=torch.long, device=device)
            logits = model(xb)
            vloss += loss_fn(logits, yb).item() * xb.size(0)
            vcorr += (logits.argmax(1) == yb).sum().item()
            vtot += xb.size(0)
    vloss /= max(vtot, 1)
    vacc = vcorr / max(vtot, 1)
    print(f"  ep {ep}: vloss={vloss:.4f} vacc={vacc:.4f} ({time.time()-t0:.0f}s)")

    if vloss < best_val_loss:
        best_val_loss = vloss
        patience_cnt = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_cnt += 1
        if patience_cnt >= 5:
            break

if best_state:
    model.load_state_dict(best_state)
    model.to(device)

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in dl_te:
        xb = xb.to(device)
        yb = torch.as_tensor(yb, dtype=torch.long, device=device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += xb.size(0)
test_acc = correct / max(total, 1)
print(f"\n  loso_S9 test_accuracy = {test_acc:.4f}")

# Append to existing results
jf = Path("results/loso_cnn.json")
data = json.loads(jf.read_text(encoding="utf-8"))
data["per_fold"].append({
    "test_accuracy": test_acc,
    "best_val_loss": best_val_loss,
    "epochs_run": ep,
    "fold_id": "loso_S9",
    "meta": {"strategy": "leave_one_subject_out", "test_subject": "S9"},
})
accs = [r["test_accuracy"] for r in data["per_fold"]]
data["n_folds"] = len(accs)
data["mean_accuracy"] = float(np.mean(accs))
data["std_accuracy"] = float(np.std(accs))
jf.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
print(f"  Saved {data['n_folds']} folds to {jf}")
print(f"  LOSO CNN: {data['mean_accuracy']:.4f} +/- {data['std_accuracy']:.4f}")
