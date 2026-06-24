#!/bin/bash
# Training script for GCP VM with 1x NVIDIA GPU (T4/L4)
# Usage: bash scripts/train_gcp.sh
#
# Environment variables (all have sensible defaults):
#   DATASET_BUCKET_URI   — GCS bucket path to DTU dataset
#   LOCAL_DATASET_DIR    — local directory to rsync the dataset into
#   ARTIFACTS_BUCKET_URI — GCS path to upload model outputs
#   RUN_ID               — unique tag for this run (auto: timestamp)
#   AADNET_NUM_WORKERS   — DataLoader workers (default: 6)
#   RUN_AAD_XAI_STAGE    — set to 1 to also run the aad_xai training stage
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_BUCKET_URI=${DATASET_BUCKET_URI:-gs://aad_data/datasets/DTU}
LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-aad_data/datasets/DTU}
ARTIFACTS_BUCKET_URI=${ARTIFACTS_BUCKET_URI:-gs://aad_data/artifacts/aad_xai}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_AAD_XAI_STAGE=${RUN_AAD_XAI_STAGE:-0}
AADNET_NUM_WORKERS=${AADNET_NUM_WORKERS:-6}

# Keep CPU thread usage bounded (tune for your VM's vCPU count).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-8}

# ── GPU check ──────────────────────────────────────────────────────────────────
echo "=== GPU Info ==="
nvidia-smi || { echo "ERROR: No GPU detected. Ensure a GPU is attached to this VM."; exit 1; }

# ── Sync DTU dataset from GCS ─────────────────────────────────────────────────
echo ""
echo "=== Syncing DTU dataset from ${DATASET_BUCKET_URI} to ${LOCAL_DATASET_DIR} ==="
mkdir -p "${LOCAL_DATASET_DIR}"
if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "${DATASET_BUCKET_URI}" "${LOCAL_DATASET_DIR}"
else
    echo "WARNING: gsutil not found. Assuming dataset already present at ${LOCAL_DATASET_DIR}."
fi

if [ ! -d "${LOCAL_DATASET_DIR}/eeg_new" ] || [ ! -d "${LOCAL_DATASET_DIR}/Audio" ]; then
    echo "ERROR: Dataset missing at ${LOCAL_DATASET_DIR}. Expected eeg_new/ and Audio/ subdirs."
    exit 1
fi

# Resolve to absolute path — stays valid after cd external/AADNet.
export DATASET
DATASET="$(realpath "${LOCAL_DATASET_DIR}")"
echo "DATASET resolved to: ${DATASET}"

# ── Install dependencies ───────────────────────────────────────────────────────
echo ""
echo "=== Installing dependencies ==="
# Upgrade build tools first so setuptools supports PEP 660 editable installs.
pip install -q --upgrade pip setuptools wheel
pip install -q -e .
pip install -q -r requirements.txt

# ── Torch/CUDA sanity check ────────────────────────────────────────────────────
python - <<'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

# ── Optional aad_xai DTU stage (disabled by default) ─────────────────────────
if [ "${RUN_AAD_XAI_STAGE}" = "1" ]; then
    echo ""
    echo "=== Training aad_xai models (DTU, CUDA) ==="
    python -m aad_xai.train \
        --config runs/config.json \
        --dataset dtu \
        --data-dir "${DATASET}" \
        --device cuda \
        --output runs
else
    echo ""
    echo "(Skipping aad_xai DTU stage — set RUN_AAD_XAI_STAGE=1 to enable)"
fi

# ── AADNet training ────────────────────────────────────────────────────────────
cd external/AADNet
mkdir -p output results

# Use envsubst to expand ${DATASET} in the config templates, then patch
# num_workers in the temporary copies — never modify the committed files.
TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

envsubst < config/config_AADNet_SI_DTU.yml > "${TMPDIR}/si.yml"
envsubst < config/config_AADNet_SS_DTU.yml > "${TMPDIR}/ss.yml"

sed -i "s/^\(\s*num_workers:\s*\).*/\1${AADNET_NUM_WORKERS}/" "${TMPDIR}/si.yml"
sed -i "s/^\(\s*num_workers:\s*\).*/\1${AADNET_NUM_WORKERS}/" "${TMPDIR}/ss.yml"

echo ""
echo "=== AADNet SI (LOSO) Cross-Validation — RUN_ID=${RUN_ID} ==="
python cross_validate_loso.py \
    -c "${TMPDIR}/si.yml" \
    -j "LOSO_AADNet_DTU_${RUN_ID}"

echo ""
echo "=== AADNet SS (Subject-Specific) Cross-Validation ==="
python cross_validate_ss.py \
    -c "${TMPDIR}/ss.yml" \
    -j "SS_AADNet_DTU_${RUN_ID}"

cd ../..

# ── Upload artifacts to GCS ────────────────────────────────────────────────────
echo ""
echo "=== Uploading artifacts to ${ARTIFACTS_BUCKET_URI}/${RUN_ID} ==="
if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r runs             "${ARTIFACTS_BUCKET_URI}/${RUN_ID}/runs"          || true
    gsutil -m rsync -r external/AADNet/output  "${ARTIFACTS_BUCKET_URI}/${RUN_ID}/aadnet_output"  || true
    gsutil -m rsync -r external/AADNet/results "${ARTIFACTS_BUCKET_URI}/${RUN_ID}/aadnet_results" || true
    echo "Artifact path: ${ARTIFACTS_BUCKET_URI}/${RUN_ID}"
else
    echo "WARNING: gsutil not found. Skipping artifact upload."
fi

echo ""
echo "=== Training complete (RUN_ID=${RUN_ID}) ==="
