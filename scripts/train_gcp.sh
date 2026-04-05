#!/bin/bash
# Training script for GCP VM with 1x NVIDIA GPU (T4/L4)
# Usage: bash scripts/train_gcp.sh
set -euo pipefail

# ── Dataset settings ──
DATASET_BUCKET_URI=${DATASET_BUCKET_URI:-gs://aad_data/datasets/DTU}
LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-aad_data/datasets/DTU}
# DATASET is set to an absolute path after the directory is created/synced below.
ARTIFACTS_BUCKET_URI=${ARTIFACTS_BUCKET_URI:-gs://aad_data/artifacts/aad_xai}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_AAD_XAI_STAGE=${RUN_AAD_XAI_STAGE:-0}
AADNET_NUM_WORKERS=${AADNET_NUM_WORKERS:-6}

# Keep CPU thread usage bounded for g2-standard-8.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-8}

# ── Verify GPU is available ──
echo "=== GPU Info ==="
nvidia-smi || { echo "ERROR: No GPU detected. Ensure T4 is attached."; exit 1; }

# ── Ensure dataset is present locally (sync from bucket if needed) ──
echo ""
echo "=== Syncing DTU dataset from ${DATASET_BUCKET_URI} to ${LOCAL_DATASET_DIR} ==="
mkdir -p "${LOCAL_DATASET_DIR}"
if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "${DATASET_BUCKET_URI}" "${LOCAL_DATASET_DIR}"
else
    echo "WARNING: gsutil not found. Assuming dataset already exists at ${LOCAL_DATASET_DIR}."
fi
if [ ! -d "${LOCAL_DATASET_DIR}/eeg_new" ] || [ ! -d "${LOCAL_DATASET_DIR}/Audio" ]; then
    echo "ERROR: Dataset not found at ${LOCAL_DATASET_DIR}. Expected eeg_new/ and Audio/ folders."
    exit 1
fi
# Resolve DATASET to an absolute path so it stays valid after `cd external/AADNet`
export DATASET="$(realpath "${LOCAL_DATASET_DIR}")"
echo "DATASET resolved to: ${DATASET}"

# ── Install dependencies ──
echo "=== Installing dependencies ==="
python -m pip install -e .
python -m pip install -r requirements.txt

# ── Verify torch/cuda after dependencies are installed ──
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# ── Optional: main aad_xai DTU stage (disabled by default) ──
if [ "$RUN_AAD_XAI_STAGE" = "1" ]; then
    echo ""
    echo "=== Training aad_xai models (DTU, CUDA) ==="
    python -m aad_xai.train \
        --config runs/config.json \
        --dataset dtu \
        --data-dir "$DATASET" \
        --device cuda \
        --output runs
else
    echo ""
    echo "=== Skipping aad_xai DTU stage (RUN_AAD_XAI_STAGE=0) ==="
fi

# ── 1) AADNet: Leave-One-Subject-Out cross-validation ──
echo ""
echo "=== AADNet LOSO Cross-Validation ==="
cd external/AADNet

# Tune dataloader workers for this VM.
sed -i "s/^\(\s*num_workers:\s*\).*/\1${AADNET_NUM_WORKERS}/" config/config_AADNet_SI_DTU.yml
sed -i "s/^\(\s*num_workers:\s*\).*/\1${AADNET_NUM_WORKERS}/" config/config_AADNet_SS_DTU.yml

mkdir -p output results

python cross_validate_loso.py \
    -c config/config_AADNet_SI_DTU.yml \
    -j LOSO_AADNet_DTU_${RUN_ID}

# ── 2) AADNet: Subject-Specific cross-validation ──
echo ""
echo "=== AADNet SS Cross-Validation ==="
python cross_validate_ss.py \
    -c config/config_AADNet_SS_DTU.yml \
    -j SS_AADNet_DTU_${RUN_ID}

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
