#!/usr/bin/env bash
# ============================================================
# build_and_push.sh
#
# Builds the AutoMend training Docker image and pushes it to
# Artifact Registry.
#
# Usage (run from repo root):
#   bash gcp/build_and_push.sh
#
# Optional — build a specific tag:
#   bash gcp/build_and_push.sh v1.2
# ============================================================
set -euo pipefail

# ---- config ----
PROJECT_ID="automend"
REGION="us-central1"
REPO="automend-images"
IMAGE_NAME="automend-train"
TAG="${1:-latest}"

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

echo "=============================================="
echo "  Building: ${REGISTRY}"
echo "=============================================="

# ---- ensure we're in repo root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

# ---- authenticate Docker to Artifact Registry ----
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ---- build ----
docker build \
  --file gcp/Dockerfile \
  --tag "${REGISTRY}" \
  --platform linux/amd64 \
  .

echo ""
echo "Build complete. Pushing to Artifact Registry..."

# ---- push ----
docker push "${REGISTRY}"

echo ""
echo "=============================================="
echo "  Image pushed: ${REGISTRY}"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "  # Workflow 1 — hyperparameter sweep (fully automated via Cloud Run webhook):"
echo "  python gcp/jobs/submit_sweep_agent.py --trials 10"
echo "  # → auto-creates sweep, launches trials on GCP"
echo "  # → W&B fires webhook when done → Cloud Run fetches best config → submits training pipeline"
echo ""
echo "  # Workflow 2 — full training manually (after sweep, without webhook):"
echo "  # Step 1: fetch best config from W&B + upload to GCS (no rebuild needed for config changes):"
echo "  python model_2_training/scripts/fetch_best_config.py --sweep-id <entity/project/sweep_id>"
echo "  # Step 2: submit pipeline (script above prints this command with the correct sweep path):"
echo "  python gcp/pipelines/submit_training_pipeline.py \\"
echo "      --train-config /gcs/automend-model2/configs/train/best_sweep_config.yaml"
echo ""
echo "  # Workflow 3 — retrain on new data:"
echo "  python gcp/pipelines/submit_training_pipeline.py --retrain-only"
