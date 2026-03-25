#!/usr/bin/env bash
# ============================================================
# upload_splits.sh
#
# One-time script: split data locally, then upload splits to GCS.
# After this, GCP training jobs read directly from GCS splits —
# no need to re-split on every run.
#
# Usage (from repo root):
#   python model_2_training/scripts/run_split.py \
#       --config model_2_training/configs/data/track_b_chatml.yaml
#   bash gcp/scripts/upload_splits.sh
# ============================================================
set -euo pipefail

GCS_BUCKET="gs://automend-model2"
LOCAL_SPLITS="model_2_training/data/splits"
LOCAL_PROCESSED="model_2_training/data/processed"

echo "Uploading processed data..."
gsutil -m cp "${LOCAL_PROCESSED}/track_B_combined.jsonl" "${GCS_BUCKET}/data/processed/"

echo "Uploading splits..."
gsutil -m cp "${LOCAL_SPLITS}/train.jsonl" "${GCS_BUCKET}/data/splits/"
gsutil -m cp "${LOCAL_SPLITS}/val.jsonl"   "${GCS_BUCKET}/data/splits/"
gsutil -m cp "${LOCAL_SPLITS}/test.jsonl"  "${GCS_BUCKET}/data/splits/"

echo ""
echo "Done. GCS contents:"
gsutil ls "${GCS_BUCKET}/data/splits/"
echo ""
echo "GCP training jobs can now skip the split step."
