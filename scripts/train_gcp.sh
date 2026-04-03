#!/bin/bash
# Training script for GCP VM with T4 GPU
# Usage: bash scripts/train_gcp.sh
set -e

# ── Dataset path on GCP (GCS bucket mounted or copied locally) ──
export DATASET=aad_data/datasets/DTU
ARTIFACTS_BUCKET_URI=${ARTIFACTS_BUCKET_URI:-gs://aad_data/artifacts/aad_xai}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

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

# ── Upload artifacts and metrics to GCS bucket ──
echo ""
echo "=== Uploading artifacts to ${ARTIFACTS_BUCKET_URI}/${RUN_ID} ==="
if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r runs "${ARTIFACTS_BUCKET_URI}/${RUN_ID}/runs"
    gsutil -m rsync -r external/AADNet/output "${ARTIFACTS_BUCKET_URI}/${RUN_ID}/aadnet_output"
    gsutil -m rsync -r external/AADNet/results "${ARTIFACTS_BUCKET_URI}/${RUN_ID}/aadnet_results"
    echo "Verifying uploaded objects..."
    gsutil ls "${ARTIFACTS_BUCKET_URI}/${RUN_ID}/runs/**" | head -n 5 || true
    gsutil ls "${ARTIFACTS_BUCKET_URI}/${RUN_ID}/aadnet_output/**" | head -n 5 || true
    gsutil ls "${ARTIFACTS_BUCKET_URI}/${RUN_ID}/aadnet_results/**" | head -n 5 || true
    echo "Full artifact path: ${ARTIFACTS_BUCKET_URI}/${RUN_ID}"
    echo "Artifacts uploaded successfully."
else
    echo "WARNING: gsutil not found. Skipping artifact upload."
fi

echo ""
echo "=== Training complete ==="
