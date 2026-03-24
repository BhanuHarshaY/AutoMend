"""
config.py

Central GCP configuration for AutoMend training pipeline.
Import this in submit_pipeline.py and any other GCP scripts.
"""

# ---- GCP project ----
PROJECT_ID   = "automend"
REGION       = "us-central1"

# ---- Artifact Registry ----
AR_REPO      = "automend-images"
IMAGE_NAME   = "automend-train"
IMAGE_TAG    = "latest"
IMAGE_URI    = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO}/{IMAGE_NAME}:{IMAGE_TAG}"

# ---- GCS bucket ----
GCS_BUCKET   = "automend-model2"
GCS_ROOT     = f"gs://{GCS_BUCKET}"

# GCS paths
GCS_DATA_DIR       = f"{GCS_ROOT}/data"           # raw + splits
GCS_CHECKPOINTS    = f"{GCS_ROOT}/outputs/checkpoints"
GCS_PIPELINE_ROOT  = f"{GCS_ROOT}/pipeline_runs"  # Vertex AI pipeline artefacts

# ---- GCS FUSE mount (inside container) ----
GCS_FUSE_MOUNT     = "/gcs/automend-model2"        # mounted at container startup

# ---- Service accounts ----
TRAINER_SA   = f"automend-trainer@{PROJECT_ID}.iam.gserviceaccount.com"

# ---- Secret Manager ----
WANDB_SECRET_NAME  = f"projects/{PROJECT_ID}/secrets/WANDB_API_KEY/versions/latest"

# ---- W&B ----
WANDB_PROJECT      = "automend-model2"
WANDB_ENTITY       = "mlops-team-northeastern-university"

# ---- Pipeline ----
PIPELINE_NAME      = "automend-training-pipeline"
PIPELINE_DISPLAY_NAME = "AutoMend QLoRA Training"

# ---- Vertex AI compute ----
# Training step — L4 GPU (or T4 if L4 quota not available)
TRAIN_MACHINE_TYPE = "n1-standard-8"
TRAIN_ACCELERATOR  = "NVIDIA_TESLA_T4"
TRAIN_ACCEL_COUNT  = 1

# Eval / test steps — CPU is fine (inference on saved adapter)
EVAL_MACHINE_TYPE  = "n1-standard-4"
