"""Quick SHAP-only run with minimal samples."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import json
from aad_xai.data.vlaai_dataset import VLAAIDTUDataset
from aad_xai.models import VLAAIPyTorch, AADDecisionEEGOnly
from aad_xai.xai.shap_explainer import shap_kernel_attribution

# Load data
ds = VLAAIDTUDataset("external/vlaai/evaluation_datasets/DTU", subjects=["S1"], window_length=320, hop=64)

# Load model
model = VLAAIPyTorch.from_h5("external/vlaai/pretrained_models/vlaai.h5")
model.eval()
decision = AADDecisionEEGOnly(model)

# Prepare batches
eeg_batch = torch.stack([ds[i][0] for i in range(20)])
att_batch = torch.stack([ds[i][1] for i in range(20)])
unatt_batch = torch.stack([ds[i][2] for i in range(20)])

def predict_fn(batch_np):
    t = torch.from_numpy(batch_np).float()
    bs = t.shape[0]
    decision.set_envelopes(att_batch[0:1].expand(bs, -1, -1), unatt_batch[0:1].expand(bs, -1, -1))
    with torch.no_grad():
        out = decision(t)
    return torch.softmax(out, dim=-1).cpu().numpy()

print("Running KernelSHAP (1 sample, 50 perturbations)...")
sv = shap_kernel_attribution(predict_fn, eeg_batch[:1].numpy(), eeg_batch[:10].numpy(), n_time_bins=10, n_samples=50)
shap_vals = sv["shap_values"]
print(f"SHAP values shape: {shap_vals.shape}")

# Report top features
if shap_vals.ndim == 3:
    # (n_classes, B, n_features) or (B, n_features, n_classes)
    if shap_vals.shape[-1] == 2:
        sv_att = shap_vals[0, :, 1]  # attended class
    else:
        sv_att = shap_vals[1, 0]
elif shap_vals.ndim == 2:
    sv_att = shap_vals[0]
else:
    sv_att = shap_vals.ravel()

feat_names = sv["feature_names"]
top5 = np.argsort(np.abs(sv_att))[-5:][::-1]
print("Top-5 important features (attended class):")
for idx in top5:
    print(f"  {feat_names[idx]}: {sv_att[idx]:.6f}")

out_dir = Path("results_vlaai_xai")
out_dir.mkdir(exist_ok=True)
np.save(out_dir / "shap_values.npy", shap_vals)
with open(out_dir / "shap_feature_names.json", "w") as f:
    json.dump(feat_names, f, indent=2)
print("SHAP results saved!")
