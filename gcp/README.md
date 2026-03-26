# AutoMend GCP Deployment

End-to-end MLOps pipeline for fine-tuning a causal LM with QLoRA on Google Cloud Platform using Vertex AI Pipelines, Vertex AI Custom Training Jobs, Compute Engine GPU VMs, and Cloud Run.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           LOCAL MACHINE                                  │
│                                                                          │
│  gcp/build_and_push.sh ──────► Artifact Registry (Docker image)         │
│                                                                          │
│  gcp/jobs/submit_sweep_agent.py                                          │
│    --backend vertex         ──► Vertex AI Custom Training Jobs (W1)      │
│    --backend compute-engine ──► Compute Engine GPU VMs       (W1 alt)    │
│                                                                          │
│  gcp/pipelines/submit_training_pipeline.py ──► Vertex AI Pipeline (W2/3)│
└──────────────────────────────────────────────────────────────────────────┘
                │                                   │
      ┌─────────┴──────────┐                        ▼
      ▼                    ▼           ┌──────────────────────────────────┐
┌──────────────┐  ┌──────────────────┐ │  Vertex AI Pipeline (W2/W3)      │
│ Vertex AI    │  │ Compute Engine   │ │  1. Split      (CPU)             │
│ Custom Jobs  │  │ GPU VMs          │ │  2. Train      (L4 GPU)          │
│ (managed)    │  │ (self-deleting)  │ │  3. Eval       (L4 GPU)          │
│              │  │                  │ │  4. Test       (L4 GPU)          │
└──────┬───────┘  └────────┬─────────┘ │  5. Benchmark  (L4 GPU)          │
       └────────┬───────────┘           │  6. Robustness (L4 GPU)          │
                ▼                       └──────────────────────────────────┘
   W&B Sweep (Bayesian optimizer)
   N × L4 GPU trials → best config
                │
                ▼
┌──────────────────────┐    ┌──────────────────────────────────────┐
│  Cloud Scheduler     │───►│  Cloud Run (automend-webhook)        │
│  fires after N hours │    │  /trigger endpoint:                  │
│  (auto-created by    │    │  1. Check sweep finished             │
│   submit_sweep.py)   │    │  2. Fetch best config from W&B       │
└──────────────────────┘    │  3. Upload config to GCS             │
                            │  4. Submit Vertex AI Pipeline        │
                            └──────────────────────────────────────┘
                                          │
                                          ▼
                            ┌──────────────────────────────────────┐
                            │  GCS Bucket: gs://automend-model2    │
                            │  ├── data/                           │
                            │  │   ├── track_B_combined.jsonl      │
                            │  │   ├── splits/                     │
                            │  │   └── benchmarks/                 │
                            │  │       └── gold_benchmark.jsonl    │
                            │  ├── configs/train/                  │
                            │  │   └── best_sweep_config.yaml      │
                            │  └── outputs/runs/<run_id>/          │
                            │      ├── checkpoints/best_model/     │
                            │      └── reports/                    │
                            └──────────────────────────────────────┘
```

### Backend comparison (Workflow 1)

| | Vertex AI (`--backend vertex`) | Compute Engine (`--backend compute-engine`) |
|---|---|---|
| **GPU** | L4 24 GB (`g2-standard-8`) | L4 24 GB (`g2-standard-8`) |
| **Setup** | Zero — fully managed | Needs CE L4 quota + `compute.instanceAdmin.v1` IAM |
| **GPU quota** | Vertex AI: `NVIDIA_L4_GPUS` | CE: `NVIDIA_L4_GPUS` in `us-central1` |
| **VM lifecycle** | Managed by Vertex AI | VM self-deletes after trial completes |
| **Log streaming** | Cloud Logging via `job.stream_logs()` | Serial port polling |
| **Cold start** | ~2–3 min | ~5–8 min (DL image boot + Docker pull) |
| **Best for** | Default, cost-effective | More control, custom startup |

---

## GCP Infrastructure

| Resource | Name / Value |
|---|---|
| Project ID | `automend` |
| Region | `us-central1` |
| GCS Bucket | `gs://automend-model2` |
| Artifact Registry | `us-central1-docker.pkg.dev/automend/automend-images` |
| Docker Image | `automend-train:latest` |
| Training Machine | `g2-standard-8` + NVIDIA L4 24 GB |
| Eval / Benchmark / Robustness | `g2-standard-8` + NVIDIA L4 24 GB (GPU inference) |
| Split Machine | CPU only (`n1-standard-4`) |
| Trainer SA | `automend-trainer@automend.iam.gserviceaccount.com` |
| Secret | `WANDB_API_KEY` (in Secret Manager) |
| Cloud Run Service | `automend-webhook` |
| W&B Project | `automend-model2` (entity: `mlops-team-northeastern-university`) |

---

## Repository Structure

```
gcp/
├── Dockerfile                        # Training container (PyTorch 2.5.1 + CUDA 12.4 + gcsfuse)
├── .dockerignore                     # Excludes .env, outputs/, data/splits/, wandb/
├── requirements-gcp.txt              # GCP Python deps for container + local tools
├── config.py                         # Central GCP config (project, region, SA, machine types, etc.)
├── build_and_push.sh                 # Build Docker image + push to Artifact Registry
│
├── jobs/
│   └── submit_sweep_agent.py         # Workflow 1: W&B sweep trials on Vertex AI OR Compute Engine
│
├── pipelines/
│   └── submit_training_pipeline.py   # Workflow 2/3/4: training pipeline (KFP v2)
│
└── cloud_run/
    ├── main.py                       # Flask webhook: /trigger + /health
    ├── Dockerfile                    # Lightweight python:3.11-slim image
    └── requirements.txt              # Flask + W&B + GCP SDK deps
```

---

## Prerequisites

### 1. Install Tools

**Windows (Git Bash recommended):**
1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install-sdk#windows)
2. [Docker Desktop](https://www.docker.com/products/docker-desktop/) (enable WSL2 backend)
3. Python 3.11+ (Conda recommended)

**macOS / Linux:**
```bash
brew install --cask google-cloud-sdk   # macOS
# or follow https://cloud.google.com/sdk/docs/install for Linux
```

### 2. Authenticate

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project automend
```

### 3. Install Python Dependencies

```bash
# Core — needed for all workflows
pip install google-cloud-aiplatform kfp wandb pyyaml python-dotenv loguru

# Compute Engine backend (Workflow 1 alt)
pip install google-cloud-compute google-cloud-scheduler

# GCS upload (fetch_best_config.py)
pip install google-cloud-storage

# Secret Manager
pip install google-cloud-secret-manager
```

---

## One-Time GCP Setup

These steps only need to be done once per project.

### 1. Enable APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  compute.googleapis.com \
  logging.googleapis.com
```

### 2. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create automend-images \
  --repository-format=docker \
  --location=us-central1
```

### 3. Create GCS Bucket

```bash
gsutil mb -l us-central1 gs://automend-model2
```

### 4. Create Service Account and Grant Roles

```bash
PROJECT_ID=automend
SA="automend-trainer@${PROJECT_ID}.iam.gserviceaccount.com"

# Create SA
gcloud iam service-accounts create automend-trainer \
  --display-name="AutoMend Trainer"

# Core roles (all workflows)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/run.invoker"

# Extra roles for --backend compute-engine
# (SA runs inside each VM, needs to self-delete and pull from Artifact Registry)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/compute.instanceAdmin.v1"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/artifactregistry.reader"
```

### 5. Store W&B API Key in Secret Manager

```bash
echo -n "YOUR_WANDB_API_KEY" | \
  gcloud secrets create WANDB_API_KEY \
    --data-file=- \
    --replication-policy=automatic
```

> The secret name must be exactly `WANDB_API_KEY` — the training container fetches it by this name at runtime via the Python Secret Manager SDK.

### 6. Request GPU Quota

Go to: **IAM & Admin → Quotas** → search `NVIDIA_L4` in `us-central1`:
- **Vertex AI quota**: `NVIDIA_L4_GPUS` under **Vertex AI API**
- **Compute Engine quota**: `NVIDIA_L4_GPUS` under **Compute Engine API**

Request at least 1 for training pipeline runs, or more for parallel sweep trials.

### 7. Upload Required Data Files to GCS

The pipeline reads these files from the GCS FUSE mount inside the container. Upload them once before running any pipeline:

```bash
# Training data
gsutil cp model_2_training/data/processed/track_B_combined.jsonl \
  gs://automend-model2/data/track_B_combined.jsonl

# Gold benchmark (used in steps 5 and 6)
gsutil cp model_2_training/data/benchmarks/gold_benchmark.jsonl \
  gs://automend-model2/data/benchmarks/gold_benchmark.jsonl
```

Verify:
```bash
gsutil ls gs://automend-model2/data/
# Should show:
# gs://automend-model2/data/track_B_combined.jsonl
# gs://automend-model2/data/benchmarks/gold_benchmark.jsonl
```

---

## Build & Push Docker Image

Run from the **repo root** (one-time, or after changing container code):

```bash
# Windows (Git Bash)
bash gcp/build_and_push.sh

# macOS / Linux
bash gcp/build_and_push.sh

# Optional: specify a tag
bash gcp/build_and_push.sh v1.2
```

This builds `gcp/Dockerfile` and pushes `automend-train:latest` to Artifact Registry.

**Image contents:** PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9, gcsfuse, all training/eval dependencies.

> **When to rebuild:** Only when you change Python files inside `model_2_training/` or `gcp/scripts/`. Scripts that run locally (like `submit_training_pipeline.py`) do not require a rebuild.

---

## Workflow 1 — Hyperparameter Sweep

Uses W&B Bayesian optimization to find the best hyperparameters. Runs N parallel trials.

### How it works

```
submit_sweep_agent.py
  1. Creates W&B sweep from configs/sweep/wandb_sweep.yaml
  2. Launches N parallel trials on the chosen backend
  3. Each trial: wandb agent --count 1 <sweep_id>
  4. (Optional) Creates Cloud Scheduler → Cloud Run /trigger
     → checks sweep finished → fetches best config → submits training pipeline
```

### Run — Vertex AI backend (default, recommended)

```bash
# Auto-create sweep + launch 10 trials
python gcp/jobs/submit_sweep_agent.py --trials 10

# Resume an existing sweep
python gcp/jobs/submit_sweep_agent.py \
  --sweep-id mlops-team-northeastern-university/automend-model2/<sweep_id> \
  --trials 5

# Dry-run (prints config, does not submit)
python gcp/jobs/submit_sweep_agent.py --trials 3 --dry-run
```

### Run — Compute Engine backend

```bash
# Auto-create sweep + launch 10 trials as self-deleting GPU VMs
python gcp/jobs/submit_sweep_agent.py \
  --backend compute-engine \
  --trials 10

# Different zone
python gcp/jobs/submit_sweep_agent.py \
  --backend compute-engine \
  --trials 10 \
  --zone us-central1-b

# Resume an existing sweep
python gcp/jobs/submit_sweep_agent.py \
  --backend compute-engine \
  --sweep-id mlops-team-northeastern-university/automend-model2/<sweep_id> \
  --trials 5
```

**Compute Engine VM lifecycle:**
1. VM created from `deeplearning-platform-release/common-cu128-ubuntu-2204-nvidia-570`
2. Startup script pulls Docker image → runs `wandb agent --count 1 <sweep_id>`
3. VM **self-deletes** when the trial finishes (success or failure)

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--backend` | `vertex` | `vertex` or `compute-engine` |
| `--trials` | 10 | Number of parallel sweep trials |
| `--sweep-id` | (auto-created) | Resume an existing W&B sweep |
| `--image-tag` | `latest` | Docker image tag |
| `--zone` | `us-central1-a` | CE zone (ignored for `--backend vertex`) |
| `--dry-run` | `False` | Print config only, do not submit |

**Monitor:**
- Vertex AI Jobs: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=automend
- Compute Engine VMs: https://console.cloud.google.com/compute/instances?project=automend
- W&B Sweep: https://wandb.ai/mlops-team-northeastern-university/automend-model2

---

## Workflow 2 — Full Training Pipeline (after sweep)

Run after the sweep completes, using the best hyperparameters found.

### Step 1: Fetch best config from W&B

```bash
python model_2_training/scripts/fetch_best_config.py \
  --sweep-id mlops-team-northeastern-university/automend-model2/<sweep_id>
```

This script:
1. Connects to W&B and finds the best-scoring trial (by `benchmark/tax_valid_rate`)
2. Merges its hyperparameters into the base training config
3. Saves the merged config locally to `configs/train/best_sweep_config.yaml`
4. Uploads it to `gs://automend-model2/configs/train/best_sweep_config.yaml`

The GCS upload means **no image rebuild is needed** when the config changes — the container reads it via the FUSE mount at runtime.

### Step 2: Submit full training pipeline

```bash
# Windows (Git Bash) — MSYS_NO_PATHCONV=1 prevents Git Bash from mangling /gcs/ paths
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --train-config /gcs/automend-model2/configs/train/best_sweep_config.yaml

# macOS / Linux
python gcp/pipelines/submit_training_pipeline.py \
  --train-config /gcs/automend-model2/configs/train/best_sweep_config.yaml
```

**Pipeline stages (in order):**

| Step | Name | Compute | Output |
|---|---|---|---|
| 1 | Split data | CPU | `gs://automend-model2/data/splits/` |
| 2 | QLoRA training | L4 GPU | `gs://automend-model2/outputs/runs/<run_id>/checkpoints/` |
| 3 | Evaluate — val set | L4 GPU | `outputs/runs/<run_id>/reports/val/` |
| 4 | Evaluate — test set | L4 GPU | `outputs/runs/<run_id>/reports/test/` |
| 5 | Benchmark — gold set | L4 GPU | `outputs/runs/<run_id>/reports/benchmark/` |
| 6 | Robustness evaluation | L4 GPU | `outputs/runs/<run_id>/reports/robustness/` |

Each run gets a timestamped `run_id` (`YYYYMMDD-HHMMSS`) so outputs never overwrite each other.

**Monitor:** https://console.cloud.google.com/vertex-ai/pipelines?project=automend

---

## Workflow 3 — Retrain on New Data

Use when you have updated data and want to retrain without running a sweep again.

```bash
# Windows
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py --retrain-only

# macOS / Linux
python gcp/pipelines/submit_training_pipeline.py --retrain-only
```

Runs only: **Split → Train → Eval → Test** (skips Benchmark and Robustness).

---

## Workflow 4 — Resume from Existing Checkpoint

Use when the pipeline failed partway through (e.g., benchmark step failed) and training already completed. Skips split/train/eval/test and runs only Benchmark + Robustness against an existing checkpoint on GCS.

### Find your checkpoint

```bash
gsutil ls gs://automend-model2/outputs/runs/
# Example output:
# gs://automend-model2/outputs/runs/20260325-170915/

gsutil ls gs://automend-model2/outputs/runs/20260325-170915/checkpoints/
# Should show best_model/ directory
```

### Submit resume pipeline

```bash
# Windows
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --resume-from-checkpoint /gcs/automend-model2/outputs/runs/20260325-170915/checkpoints/best_model

# macOS / Linux
python gcp/pipelines/submit_training_pipeline.py \
  --resume-from-checkpoint /gcs/automend-model2/outputs/runs/20260325-170915/checkpoints/best_model
```

> Replace `20260325-170915` with your actual run_id from the `gsutil ls` output.

---

## Pipeline Flags Reference

| Flag | Default | Description |
|---|---|---|
| `--train-config` | `qlora_sft.yaml` | Train config filename (baked in image) or full GCS FUSE path |
| `--retrain-only` | `False` | Workflow 3: run split→train→eval→test only |
| `--resume-from-checkpoint` | `""` | Workflow 4: path to existing checkpoint, run only benchmark+robustness |
| `--image-tag` | `latest` | Docker image tag |
| `--dry-run` | `False` | Compile pipeline YAML only, do not submit |
| `--output-yaml` | `gcp/pipelines/training_pipeline.yaml` | Where to write compiled pipeline YAML |

---

## GCS Data Layout

```
gs://automend-model2/
├── data/
│   ├── track_B_combined.jsonl          ← upload before first run
│   ├── splits/
│   │   ├── train.jsonl                 ← written by split step
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── benchmarks/
│       └── gold_benchmark.jsonl        ← upload before first run
├── configs/
│   └── train/
│       └── best_sweep_config.yaml      ← written by fetch_best_config.py
├── outputs/
│   └── runs/
│       └── <YYYYMMDD-HHMMSS>/
│           ├── checkpoints/
│           │   ├── best_model/         ← final model (load this for inference)
│           │   ├── checkpoint-XXXX/    ← intermediate checkpoints
│           │   └── run_config_snapshot.json
│           └── reports/
│               ├── val/
│               ├── test/
│               ├── benchmark/
│               └── robustness/
└── pipeline_runs/                      ← Vertex AI pipeline artifacts (auto-managed)
```

---

## Cloud Run Webhook Setup

Only needed if using Workflow 1 fully automated mode (sweep → auto-trigger training):

```bash
# Build and push Cloud Run image
gcloud builds submit gcp/cloud_run/ \
  --tag us-central1-docker.pkg.dev/automend/automend-images/automend-webhook:latest

# Deploy to Cloud Run
gcloud run deploy automend-webhook \
  --image us-central1-docker.pkg.dev/automend/automend-images/automend-webhook:latest \
  --platform managed \
  --region us-central1 \
  --no-allow-unauthenticated \
  --service-account automend-trainer@automend.iam.gserviceaccount.com \
  --memory 512Mi \
  --timeout 600

# Get the service URL
gcloud run services describe automend-webhook \
  --region us-central1 \
  --format "value(status.url)"
```

Use the URL as `--cloud-run-url` when running `submit_sweep_agent.py`.

**Endpoints:**
- `POST /trigger` — triggered by Cloud Scheduler; fetches best config + submits training pipeline
- `GET /health` — returns `{"status": "ok"}`

---

## Configuration Reference

### gcp/config.py

Edit this file to change project-wide settings:

```python
PROJECT_ID         = "automend"
REGION             = "us-central1"
GCS_BUCKET         = "automend-model2"
TRAIN_MACHINE_TYPE = "g2-standard-8"    # 1× L4 24 GB, 8 vCPU, 32 GB RAM
TRAIN_ACCELERATOR  = "NVIDIA_L4"
TRAIN_ACCEL_COUNT  = 1
WANDB_PROJECT      = "automend-model2"
WANDB_ENTITY       = "mlops-team-northeastern-university"

# Compute Engine backend
CE_ZONE            = "us-central1-a"    # L4 available in -a, -b, -c, -f
CE_MACHINE_TYPE    = "g2-standard-8"
```

### Model / Training Configs

| Config | Location | Purpose |
|---|---|---|
| Data config | `model_2_training/configs/data/track_b_chatml.yaml` | Full pipeline |
| Data config (sweep) | `model_2_training/configs/data/track_b_chatml_sweep.yaml` | Fast sweep trials (capped samples) |
| Model config | `model_2_training/configs/model/qwen_baseline.yaml` | Base model + LoRA settings |
| Train config | `model_2_training/configs/train/qlora_sft.yaml` | Default training hyperparameters |
| Best sweep config | `model_2_training/configs/train/best_sweep_config.yaml` | Generated by `fetch_best_config.py` |
| Sweep config | `model_2_training/configs/sweep/wandb_sweep.yaml` | W&B Bayesian sweep search space |

---

## Monitoring

| Resource | URL |
|---|---|
| Vertex AI Custom Jobs | https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=automend |
| Vertex AI Pipelines | https://console.cloud.google.com/vertex-ai/pipelines?project=automend |
| Compute Engine Instances | https://console.cloud.google.com/compute/instances?project=automend |
| Cloud Run Logs | https://console.cloud.google.com/run/detail/us-central1/automend-webhook/logs?project=automend |
| Cloud Scheduler | https://console.cloud.google.com/cloudscheduler?project=automend |
| W&B Dashboard | https://wandb.ai/mlops-team-northeastern-university/automend-model2 |
| GCS Bucket | https://console.cloud.google.com/storage/browser/automend-model2 |

**Check CE serial port logs manually:**
```bash
gcloud compute instances get-serial-port-output automend-sweep-t01-<timestamp> \
  --zone=us-central1-a \
  --project=automend \
  --port=1
```

---

## Cost Estimates

| Resource | Unit Cost | Typical Usage | Estimate |
|---|---|---|---|
| L4 GPU (`g2-standard-8`) Vertex AI | ~$0.70/hr | 10 trials × 30 min | ~$3.50/sweep |
| L4 GPU (`g2-standard-8`) Compute Engine | ~$0.70/hr | 10 trials × 30 min + boot | ~$4.00/sweep |
| Full training pipeline (all 6 steps) | ~$0.70/hr | ~3–5 hours | ~$2.10–$3.50 |
| Cloud Run | ~$0 | Very low traffic | <$0.01 |
| GCS storage | ~$0.02/GB/mo | ~10–50 GB | <$1/mo |

> Check [GCP Pricing](https://cloud.google.com/pricing) for current rates.

---

## Troubleshooting

### `WANDB_API_KEY not set` in training container

The container fetches the key at runtime from Secret Manager using the Python SDK. Ensure:
1. The secret exists: `gcloud secrets list --project=automend` — you should see `WANDB_API_KEY`
2. The trainer SA has `roles/secretmanager.secretAccessor`
3. `PROJECT_ID` env var is set in the container (it is — set in `submit_training_pipeline.py`)

To test the secret exists:
```bash
gcloud secrets versions access latest --secret=WANDB_API_KEY --project=automend
```

### Git Bash path mangling on Windows

Git Bash converts paths starting with `/` to Windows paths (e.g. `/gcs/automend-model2` → `C:/Program Files/Git/gcs/automend-model2`).

**Fix:** prefix the command with `MSYS_NO_PATHCONV=1`:
```bash
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --train-config /gcs/automend-model2/configs/train/best_sweep_config.yaml
```

### `FileNotFoundError` for training data or benchmark inside container

The container reads data via the GCS FUSE mount (`/gcs/automend-model2`). Check the files exist on GCS:
```bash
gsutil ls gs://automend-model2/data/track_B_combined.jsonl
gsutil ls gs://automend-model2/data/benchmarks/gold_benchmark.jsonl
```

If missing, upload them (see [Upload Required Data Files](#7-upload-required-data-files-to-gcs)).

### Benchmark or robustness step failed but training completed

Use Workflow 4 (`--resume-from-checkpoint`) to rerun only steps 5+6 without re-training:
```bash
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --resume-from-checkpoint /gcs/automend-model2/outputs/runs/<run_id>/checkpoints/best_model
```

Find your `run_id`:
```bash
gsutil ls gs://automend-model2/outputs/runs/
```

### Vertex AI pipeline stuck in PENDING

Usually a GPU quota issue. Check and request:

**Vertex AI quota:**
```bash
gcloud compute regions describe us-central1 --format="value(quotas)" | grep NVIDIA_L4
```
Go to **IAM & Admin → Quotas** → `NVIDIA_L4_GPUS` under **Vertex AI API** in `us-central1`.

### `Permission denied` on pipeline submission

```bash
gcloud projects add-iam-policy-binding automend \
  --member="serviceAccount:automend-trainer@automend.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### Docker build fails on Windows

Use Git Bash or WSL2 to run `bash gcp/build_and_push.sh`. PowerShell may have path issues.

Or build manually:
```bash
docker build \
  --file gcp/Dockerfile \
  --tag us-central1-docker.pkg.dev/automend/automend-images/automend-train:latest \
  --platform linux/amd64 \
  .
docker push us-central1-docker.pkg.dev/automend/automend-images/automend-train:latest
```

### `Billing account not associated`

```bash
gcloud billing accounts list
gcloud billing projects link automend --billing-account=XXXXXX-XXXXXX-XXXXXX
```

### CE instance never self-deletes

Verify the SA has `roles/compute.instanceAdmin.v1`:
```bash
gcloud projects get-iam-policy automend \
  --flatten="bindings[].members" \
  --filter="bindings.members:automend-trainer@automend.iam.gserviceaccount.com" \
  --format="table(bindings.role)"
```

Delete manually if needed:
```bash
gcloud compute instances delete automend-sweep-t01-<timestamp> \
  --zone=us-central1-a --project=automend
```

---

## Quick Reference — End to End

```bash
# ── One-time setup ──────────────────────────────────────────────────────────

# 1. GCP infra (run once)
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com \
  run.googleapis.com cloudscheduler.googleapis.com secretmanager.googleapis.com \
  storage.googleapis.com compute.googleapis.com
gsutil mb -l us-central1 gs://automend-model2
gcloud artifacts repositories create automend-images --repository-format=docker --location=us-central1
echo -n "YOUR_WANDB_KEY" | gcloud secrets create WANDB_API_KEY --data-file=- --replication-policy=automatic

# 2. Upload data to GCS (run once, or when data changes)
gsutil cp model_2_training/data/processed/track_B_combined.jsonl \
  gs://automend-model2/data/track_B_combined.jsonl
gsutil cp model_2_training/data/benchmarks/gold_benchmark.jsonl \
  gs://automend-model2/data/benchmarks/gold_benchmark.jsonl

# 3. Build + push Docker image (run once, or when container code changes)
bash gcp/build_and_push.sh

# ── Workflow 1: Hyperparameter sweep ────────────────────────────────────────

python gcp/jobs/submit_sweep_agent.py --trials 10

# ── Workflow 2: Full training after sweep ───────────────────────────────────

# Fetch best config from W&B (uploads to GCS automatically)
python model_2_training/scripts/fetch_best_config.py \
  --sweep-id mlops-team-northeastern-university/automend-model2/<sweep_id>

# Submit pipeline (Windows)
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --train-config /gcs/automend-model2/configs/train/best_sweep_config.yaml

# ── Workflow 3: Retrain on new data ─────────────────────────────────────────

MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py --retrain-only

# ── Workflow 4: Resume from checkpoint (steps 5+6 only) ─────────────────────

gsutil ls gs://automend-model2/outputs/runs/   # find run_id
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --resume-from-checkpoint /gcs/automend-model2/outputs/runs/<run_id>/checkpoints/best_model
```
