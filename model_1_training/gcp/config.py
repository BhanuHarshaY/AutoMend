"""
config.py

Central GCP configuration for AutoMend Model 1 training pipeline.
Mirrors gcp/config.py but with Model 1-specific resources.
"""

# ---- GCP project ----
PROJECT_ID   = "automend"
REGION       = "us-central1"

# ---- Artifact Registry (shared repo, separate image) ----
AR_REPO      = "automend-images"
IMAGE_NAME   = "automend-train-model1"
IMAGE_TAG    = "latest"
IMAGE_URI    = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO}/{IMAGE_NAME}:{IMAGE_TAG}"

# ---- GCS bucket (separate from Model 2) ----
GCS_BUCKET   = "automend-model1"
GCS_ROOT     = f"gs://{GCS_BUCKET}"

GCS_DATA_DIR       = f"{GCS_ROOT}/data"
GCS_CHECKPOINTS    = f"{GCS_ROOT}/outputs/checkpoints"
GCS_PIPELINE_ROOT  = f"{GCS_ROOT}/pipeline_runs"

# ---- GCS FUSE mount (inside container) ----
GCS_FUSE_MOUNT     = "/gcs/automend-model1"

# ---- Service accounts ----
TRAINER_SA   = f"automend-trainer@{PROJECT_ID}.iam.gserviceaccount.com"

# ---- Secret Manager ----
WANDB_SECRET_NAME  = f"projects/{PROJECT_ID}/secrets/WANDB_API_KEY/versions/latest"

# ---- W&B ----
WANDB_PROJECT      = "automend-model1"
WANDB_ENTITY       = "mlops-team-northeastern-university"

# ---- Pipeline ----
PIPELINE_NAME      = "automend-model1-pipeline"
PIPELINE_DISPLAY_NAME = "AutoMend Model 1 — RoBERTa Anomaly Detector"

# ---- Vertex AI compute ----
TRAIN_MACHINE_TYPE = "g2-standard-8"
TRAIN_ACCELERATOR  = "NVIDIA_L4"
TRAIN_ACCEL_COUNT  = 1

EVAL_MACHINE_TYPE  = "n1-standard-4"
