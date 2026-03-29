#!/usr/bin/env bash
# ============================================================
# build_and_push.sh — Model 1 Training Image
#
# Builds the AutoMend Model 1 Docker image and pushes it to
# Artifact Registry.
#
# Usage (run from repo root):
#   bash model_1_training/gcp/build_and_push.sh
#   bash model_1_training/gcp/build_and_push.sh v1.0
# ============================================================
set -euo pipefail

PROJECT_ID="automend"
REGION="us-central1"
REPO="automend-images"
IMAGE_NAME="automend-train-model1"
TAG="${1:-latest}"

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

echo "=============================================="
echo "  Building Model 1: ${REGISTRY}"
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build \
  --file model_1_training/docker/Dockerfile \
  --tag "${REGISTRY}" \
  --platform linux/amd64 \
  .

echo ""
echo "Build complete. Pushing to Artifact Registry..."

docker push "${REGISTRY}"

echo ""
echo "=============================================="
echo "  Image pushed: ${REGISTRY}"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "  # Run locally with Docker:"
echo "  docker run --gpus all -v ./data:/workspace/data ${REGISTRY} \\"
echo "    -m model_1_training.scripts.run_train"
echo ""
echo "  # Submit Vertex AI pipeline:"
echo "  python model_1_training/gcp/pipelines/submit_training_pipeline.py"
echo ""
echo "  # Run sweep on GCP:"
echo "  python model_1_training/gcp/jobs/submit_sweep_agent.py --trials 20"
echo ""
