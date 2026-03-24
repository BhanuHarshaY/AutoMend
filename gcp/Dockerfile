# ============================================================
# AutoMend Training Container
# Base: Official PyTorch image — PyTorch 2.3.0, Python 3.11, CUDA 12.1, cuDNN 8
# ============================================================
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# ---- system deps + gcsfuse + gcloud CLI (all from Google apt repo) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    fuse \
    curl \
    gnupg \
    lsb-release \
    apt-transport-https \
    ca-certificates \
    && echo "deb https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/gcsfuse.list \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update \
    && apt-get install -y --no-install-recommends gcsfuse google-cloud-cli \
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
