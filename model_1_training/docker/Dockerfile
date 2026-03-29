# ============================================================
# AutoMend Model 1 Training Container
# Base: PyTorch 2.5.1, Python 3.11, CUDA 12.4, cuDNN 9
# ============================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# ---- system deps + gcsfuse ----
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

WORKDIR /workspace

# ---- Python deps: project requirements ----
COPY model_1_training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Python deps: GCP-specific ----
COPY model_1_training/docker/requirements-gcp.txt /tmp/requirements-gcp.txt
RUN pip install --no-cache-dir -r /tmp/requirements-gcp.txt

# ---- copy source code ----
COPY model_1_training/ /workspace/model_1_training/

# ---- GCS mount point ----
RUN mkdir -p /gcs/automend-model1

# ---- environment defaults ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    WANDB_START_METHOD=thread \
    WANDB_PROJECT=automend-model1 \
    WANDB_ENTITY=mlops-team-northeastern-university \
    PYTHONPATH=/workspace

ENTRYPOINT ["python"]
