# ============================================================
# AutoMend Training Container
# Base: PyTorch 2.5.1, Python 3.11, CUDA 12.4, cuDNN 9
# ============================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# ---- system deps + gcsfuse (gcloud CLI NOT needed — we use Python SDKs) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    fuse \
    curl \
    gnupg \
    lsb-release \
    && echo "deb https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/gcsfuse.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install -y --no-install-recommends gcsfuse \
    && rm -rf /var/lib/apt/lists/*

# ---- working directory ----
WORKDIR /workspace

# ---- Python deps: project requirements ----
COPY model_2_training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Python deps: GCP-specific additions ----
COPY gcp/requirements-gcp.txt /tmp/requirements-gcp.txt
RUN pip install --no-cache-dir -r /tmp/requirements-gcp.txt

# ---- copy source code ----
COPY model_2_training/ /workspace/model_2_training/
COPY gcp/scripts/ /workspace/gcp/scripts/

# ---- GCS mount point (used by GCS FUSE at runtime) ----
RUN mkdir -p /gcs/automend-model2

# ---- environment defaults (no secrets here — loaded from Secret Manager at runtime) ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    WANDB_START_METHOD=thread \
    WANDB_PROJECT=automend-model2 \
    WANDB_ENTITY=mlops-team-northeastern-university \
    PYTHONPATH=/workspace

# Default entrypoint — override per Vertex AI step
ENTRYPOINT ["python"]
