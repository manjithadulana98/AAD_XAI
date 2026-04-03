#!/bin/bash
# Training script for GCP VM with T4 GPU
# Usage: bash scripts/train_gcp.sh
set -e

# ── Dataset path on GCP (GCS bucket mounted or copied locally) ──
export DATASET=aad_data/datasets/DTU

# ── Verify GPU is available ──
echo "=== GPU Info ==="
nvidia-smi || { echo "ERROR: No GPU detected. Ensure T4 is attached."; exit 1; }
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# ── Install dependencies ──
echo "=== Installing dependencies ==="
pip install -e .
pip install -r requirements.txt

# ── 1. Main framework: Train with DTU dataset on GPU ──
echo ""
echo "=== Training aad_xai models (DTU, CUDA) ==="
python -m aad_xai.train \
    --config runs/config.json \
    --dataset dtu \
    --data-dir "$DATASET" \
    --device cuda \
    --output runs

# ── 2. AADNet: Leave-One-Subject-Out cross-validation ──
echo ""
echo "=== AADNet LOSO Cross-Validation ==="
cd external/AADNet
python cross_validate_loso.py \
    -c config/config_AADNet_SI_DTU.yml \
    -j LOSO_AADNet_DTU

# ── 3. AADNet: Subject-Specific cross-validation ──
echo ""
echo "=== AADNet SS Cross-Validation ==="
python cross_validate_ss.py \
    -c config/config_AADNet_SS_DTU.yml \
    -j SS_AADNet_DTU

cd ../..
echo ""
echo "=== Training complete ==="
