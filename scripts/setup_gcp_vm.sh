#!/bin/bash
# One-time VM setup for AAD_XAI training on GCP (T4 / L4 GPU).
# Run once after creating the VM:
#   bash scripts/setup_gcp_vm.sh
#
# Assumes:
#   • Debian/Ubuntu base image (Deep Learning VM or plain Ubuntu 22.04)
#   • NVIDIA driver already installed (Deep Learning VMs ship with it)
#   • gcloud auth already configured (VM service account with Storage access)
set -euo pipefail

echo "=== Setting up GCP VM for AAD_XAI training ==="

# ── System packages ────────────────────────────────────────────────────────────
sudo apt-get update -q
# gettext-base provides envsubst (needed by train_gcp.sh to expand ${DATASET})
# screen / tmux let you keep the training job alive after SSH disconnects
sudo apt-get install -y python3-pip python3-venv git gettext-base screen tmux

# ── Python virtual environment ─────────────────────────────────────────────────
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip --quiet

# ── PyTorch with CUDA (must come BEFORE requirements.txt to avoid CPU wheel) ──
# cu121 wheels work on CUDA 12.x (T4, L4, A100 on GCP).
pip install --quiet \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ── Project dependencies ───────────────────────────────────────────────────────
# Install torch-incompatible packages separately so they don't downgrade torch.
# tensorflow is optional for this project; skip it to save install time.
grep -v '^tensorflow' requirements.txt > /tmp/requirements_no_tf.txt || true
pip install --quiet -r /tmp/requirements_no_tf.txt
pip install --quiet -e .

# ── Verify CUDA is visible to PyTorch ─────────────────────────────────────────
echo ""
echo "=== Verifying PyTorch + CUDA ==="
python - <<'EOF'
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version    : {torch.version.cuda}")
else:
    print("WARNING: CUDA not visible — check driver / CUDA toolkit install.")
EOF

# ── Quick dataset sync check ───────────────────────────────────────────────────
echo ""
echo "=== Listing DTU bucket objects (first 10) ==="
gsutil ls gs://aad_data/datasets/DTU/ | head -10 || \
    echo "WARNING: Could not list bucket. Check service account Storage permissions."

echo ""
echo "=== Setup complete ==="
echo "Next step: bash scripts/train_gcp.sh"
echo "Tip: run inside 'screen' or 'tmux' so SSH disconnects don't kill the job."
echo "  screen -S train bash scripts/train_gcp.sh"
