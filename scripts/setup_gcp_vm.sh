#!/bin/bash
# One-time setup for GCP VM with T4 GPU
# Usage: bash scripts/setup_gcp_vm.sh
set -e

echo "=== Setting up GCP VM for AAD_XAI training ==="

# ── System packages ──
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# ── Create and activate virtual environment ──
python3 -m venv .venv
source .venv/bin/activate

# ── Install PyTorch with CUDA support (T4 = CUDA 12.x compatible) ──
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── Install project dependencies ──
pip install -r requirements.txt
pip install -e .

# ── Copy DTU dataset from GCS bucket ──
echo ""
echo "=== Copying DTU dataset from GCS bucket ==="
mkdir -p aad_data/datasets/DTU
gsutil -m cp -r gs://aad_data/datasets/DTU/* aad_data/datasets/DTU/

# ── Verify setup ──
echo ""
echo "=== Verifying setup ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"

echo ""
echo "=== Setup complete. Run: bash scripts/train_gcp.sh ==="
