# AutoMend GCP Deployment

End-to-end MLOps pipeline for fine-tuning **Qwen2.5-1.5B-Instruct** with QLoRA on Google Cloud Platform using Vertex AI, Cloud Run, and Cloud Scheduler.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LOCAL MACHINE                                │
│                                                                     │
│  gcp/build_and_push.sh ──► Artifact Registry (Docker image)        │
│  gcp/jobs/submit_sweep_agent.py ──► Vertex AI Custom Training Jobs  │
│  gcp/pipelines/submit_training_pipeline.py ──► Vertex AI Pipeline  │
└─────────────────────────────────────────────────────────────────────┘
                │                        │
                ▼                        ▼
┌──────────────────────┐    ┌──────────────────────────────────────┐
│  W&B Sweep (W1)      │    │  Vertex AI Pipeline (W2/W3)          │
│  10 × T4 GPU trials  │    │  Split → Train → Eval → Test →       │
│  Bayesian optimizer  │    │  Benchmark → Robustness              │
│  picks best config   │    │  (T4 GPU for training step)          │
└──────────┬───────────┘    └──────────────────────────────────────┘
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
                            │  ├── data/splits/                    │
                            │  ├── configs/best_sweep_config.yaml  │
                            │  ├── outputs/checkpoints/best_model/ │
                            │  └── outputs/reports/                │
                            └──────────────────────────────────────┘
```

---

## GCP Infrastructure

| Resource | Name / Value |
|---|---|
| Project ID | `automend` |
| Region | `us-central1` |
| GCS Bucket | `gs://automend-model2` |
| Artifact Registry | `us-central1-docker.pkg.dev/automend/automend-images` |
| Docker Image | `automend-train:latest` |
| Training Machine | `n1-standard-8` + NVIDIA Tesla T4 |
| Eval/Test Machine | `n1-standard-4` (CPU) |
| Trainer SA | `automend-trainer@automend.iam.gserviceaccount.com` |
| Secret | `wandb-api-key` (in Secret Manager) |
| Cloud Run Service | `automend-webhook` |
| W&B Project | `automend-model2` (entity: `mlops-team-northeastern-university`) |

---

## Repository Structure

```
gcp/
├── Dockerfile                        # Training container (PyTorch + CUDA + gcsfuse + gcloud CLI)
├── .dockerignore                     # Excludes .env, outputs/, data/splits/, wandb/
├── requirements-gcp.txt              # GCP Python deps for container
├── config.py                         # Central GCP config (project, region, SA, etc.)
├── build_and_push.sh                 # Build Docker image + push to Artifact Registry
│
├── jobs/
│   └── submit_sweep_agent.py         # Workflow 1: launch W&B sweep trials on Vertex AI
│
├── pipelines/
│   └── submit_training_pipeline.py   # Workflow 2/3: full training pipeline (KFP)
│
└── cloud_run/
    ├── main.py                       # Flask webhook: /trigger + /health
    ├── Dockerfile                    # Lightweight python:3.11-slim image
    └── requirements.txt              # Flask + W&B + GCP SDK deps
```

---

## Prerequisites

### Windows

1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install-sdk#windows)
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (enable WSL2 backend)
3. Install Python 3.11+ (Conda recommended)
4. Authenticate:
   ```powershell
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project automend
   ```

### macOS

1. Install Google Cloud SDK:
   ```bash
   brew install --cask google-cloud-sdk
   ```
2. Install Docker Desktop
3. Authenticate:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project automend
   ```

### Python Dependencies (both platforms)

```bash
pip install google-cloud-aiplatform kfp wandb google-cloud-scheduler pyyaml
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
  storage.googleapis.com
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

### 4. Create Service Account

```bash
# Create SA
gcloud iam service-accounts create automend-trainer \
  --display-name="AutoMend Trainer"

# Grant roles
PROJECT_ID=automend
SA="automend-trainer@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/run.invoker"
```

### 5. Store W&B API Key in Secret Manager

```bash
echo -n "YOUR_WANDB_API_KEY" | \
  gcloud secrets create wandb-api-key \
    --data-file=- \
    --replication-policy=automatic
```

### 6. Request GPU Quota

Go to: **IAM & Admin → Quotas** → search for `NVIDIA_TESLA_T4` in `us-central1` → request increase to at least 10.

---

## Build & Push Docker Image

Run from the **repo root**:

```bash
# Windows (Git Bash or WSL)
bash gcp/build_and_push.sh

# macOS / Linux
bash gcp/build_and_push.sh

# Optional: specify a custom tag
bash gcp/build_and_push.sh v1.2
```

This builds `gcp/Dockerfile` tagged as `automend-train:latest` and pushes to Artifact Registry.

> The Docker image includes: PyTorch 2.3 + CUDA 12.1, gcsfuse (GCS FUSE mount), gcloud CLI, and all training dependencies.

---

## Workflow 1 — Hyperparameter Sweep (Fully Automated)

Uses W&B Bayesian optimization to find the best hyperparameters, then automatically triggers full training.

### How it works

```
submit_sweep_agent.py
  └── 1. Creates W&B sweep from wandb_sweep.yaml
  └── 2. Launches N Vertex AI Custom Training Jobs (parallel)
       Each job: wandb agent --count 1 <sweep_id>
  └── 3. Creates Cloud Scheduler job (fires after --delay-hours)
       Cloud Scheduler → Cloud Run /trigger
         → checks sweep finished
         → fetches best config from W&B
         → uploads to gs://automend-model2/configs/best_sweep_config.yaml
         → submits Vertex AI full training pipeline
```

### Run

```bash
# Windows
python gcp/jobs/submit_sweep_agent.py \
  --trials 10 \
  --cloud-run-url https://automend-webhook-XXXX-uc.a.run.app \
  --delay-hours 6

# macOS / Linux
python gcp/jobs/submit_sweep_agent.py \
  --trials 10 \
  --cloud-run-url https://automend-webhook-XXXX-uc.a.run.app \
  --delay-hours 6
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--trials` | 10 | Number of parallel sweep trials |
| `--sweep-id` | (auto-created) | Resume an existing sweep |
| `--image-tag` | latest | Docker image tag |
| `--cloud-run-url` | None | Cloud Run URL — enables auto scheduling |
| `--delay-hours` | 6.0 | Hours until Cloud Scheduler fires (must be > expected sweep duration) |
| `--dry-run` | False | Print config only, do not submit |

**Dry-run example:**
```bash
python gcp/jobs/submit_sweep_agent.py --trials 3 --dry-run
```

**Resume an existing sweep:**
```bash
python gcp/jobs/submit_sweep_agent.py \
  --sweep-id mlops-team-northeastern-university/automend-model2/abc123 \
  --trials 5
```

**Monitor:**
- Vertex AI jobs: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=automend
- W&B sweep: https://wandb.ai/mlops-team-northeastern-university/automend-model2

---

## Workflow 2 — Full Training Pipeline (Manual)

Run after the sweep completes, using the best hyperparameters found.

```bash
# Step 1: Fetch best config from W&B sweep
python model_2_training/scripts/fetch_best_config.py \
  --sweep-id mlops-team-northeastern-university/automend-model2/<sweep_id>

# Step 2: Submit full training pipeline to Vertex AI
python gcp/pipelines/submit_training_pipeline.py \
  --train-config best_sweep_config.yaml
```

**Pipeline stages (in order):**
1. **Split** — train/val/test split (90/5/5)
2. **Train** — QLoRA fine-tuning on T4 GPU
3. **Eval** — validation set evaluation
4. **Test** — test set evaluation
5. **Benchmark** — gold benchmark scoring
6. **Robustness** — robustness evaluation

**Monitor:** https://console.cloud.google.com/vertex-ai/pipelines?project=automend

---

## Workflow 3 — Retrain on New Data

Use when you have new data and want to retrain without running a sweep again.

```bash
python gcp/pipelines/submit_training_pipeline.py --retrain-only
```

**Pipeline stages:**
1. Split → 2. Train → 3. Eval → 4. Test

---

## Cloud Run Webhook Setup

Deploy the automation service (only needed if using Workflow 1 fully automated mode):

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

# Get the URL
gcloud run services describe automend-webhook \
  --region us-central1 \
  --format "value(status.url)"
```

Use the URL as `--cloud-run-url` when running `submit_sweep_agent.py`.

**Endpoints:**
- `POST /trigger` — triggered by Cloud Scheduler, fetches best config + submits pipeline
- `GET /health` — returns `{"status": "ok"}`

---

## Configuration Reference

### gcp/config.py

Central config for all GCP resources:

```python
PROJECT_ID         = "automend"
REGION             = "us-central1"
GCS_BUCKET         = "automend-model2"
TRAIN_MACHINE_TYPE = "n1-standard-8"
TRAIN_ACCELERATOR  = "NVIDIA_TESLA_T4"
TRAIN_ACCEL_COUNT  = 1
WANDB_PROJECT      = "automend-model2"
WANDB_ENTITY       = "mlops-team-northeastern-university"
```

### Data configs

| Config | Samples | Purpose |
|---|---|---|
| `configs/data/track_b_chatml.yaml` | All (null) | Full training pipeline |
| `configs/data/track_b_chatml_sweep.yaml` | 2000 train / 200 val | Fast sweep trials |

Both use **90/5/5** train/val/test split.

### Sweep config

`model_2_training/configs/sweep/wandb_sweep.yaml` — Bayesian optimization over:
- `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`
- `gradient_accumulation_steps`, `lr_scheduler_type`, `warmup_ratio`
- `weight_decay`, `lora_r`, `lora_alpha`, `lora_dropout`

---

## Monitoring

| Resource | URL |
|---|---|
| Vertex AI Custom Jobs | https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=automend |
| Vertex AI Pipelines | https://console.cloud.google.com/vertex-ai/pipelines?project=automend |
| Cloud Run Logs | https://console.cloud.google.com/run/detail/us-central1/automend-webhook/logs?project=automend |
| Cloud Scheduler | https://console.cloud.google.com/cloudscheduler?project=automend |
| W&B Dashboard | https://wandb.ai/mlops-team-northeastern-university/automend-model2 |
| GCS Bucket | https://console.cloud.google.com/storage/browser/automend-model2 |

---

## Cost Estimates

| Resource | Unit Cost | Typical Usage | Estimate |
|---|---|---|---|
| T4 GPU (n1-standard-8) | ~$0.55/hr | 10 trials × 30 min | ~$2.75/sweep |
| T4 GPU (full training) | ~$0.55/hr | ~3–5 hours | ~$1.65–$2.75 |
| Cloud Run | ~$0 | Very low traffic | <$0.01 |
| GCS storage | ~$0.02/GB/mo | ~10–50 GB | <$1/mo |
| Cloud Scheduler | Free tier | 1 job/sweep | $0 |

> Costs are estimates. Check [GCP Pricing](https://cloud.google.com/pricing) for current rates.

---

## Troubleshooting

### `gcloud: command not found` inside container

The Docker image must include gcloud CLI. Rebuild after verifying `gcp/Dockerfile` installs `google-cloud-cli`:

```dockerfile
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] ..." \
    && apt-get install -y google-cloud-cli
```

Then rebuild and push: `bash gcp/build_and_push.sh`

### `WANDB_API_KEY not set`

The container fetches the API key at runtime via:
```bash
export WANDB_API_KEY=$(gcloud secrets versions access latest --secret=wandb-api-key)
```

Requires gcloud CLI to be installed in the image (see above) and the service account to have `roles/secretmanager.secretAccessor`.

### Vertex AI jobs stuck in PENDING

Usually a quota issue. Check:
```bash
gcloud compute regions describe us-central1 --format="value(quotas)" | grep NVIDIA_T4
```

Request a quota increase at: **IAM & Admin → Quotas** → `NVIDIA_TESLA_T4_GPUS` in `us-central1`.

### Cloud Scheduler fires before sweep finishes

The `/trigger` endpoint returns `503` if the sweep isn't done yet, which causes Cloud Scheduler to retry automatically. Increase `--delay-hours` to be safe (default is 6 hours for 10 × T4 trials).

### `Permission denied` on Vertex AI submission

Ensure the trainer service account has `roles/aiplatform.user`:
```bash
gcloud projects add-iam-policy-binding automend \
  --member="serviceAccount:automend-trainer@automend.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### Docker build fails on Windows

Use Git Bash or WSL2 to run `bash gcp/build_and_push.sh`. PowerShell may have path issues.

Alternatively, build manually:
```powershell
docker build --file gcp/Dockerfile --tag us-central1-docker.pkg.dev/automend/automend-images/automend-train:latest --platform linux/amd64 .
docker push us-central1-docker.pkg.dev/automend/automend-images/automend-train:latest
```

### `Billing account not associated`

```bash
gcloud billing accounts list
gcloud billing projects link automend --billing-account=XXXXXX-XXXXXX-XXXXXX
```

---

## Quick Reference

```bash
# 1. Build + push image
bash gcp/build_and_push.sh

# 2. Run sweep (10 trials, fully automated)
python gcp/jobs/submit_sweep_agent.py \
  --trials 10 \
  --cloud-run-url https://automend-webhook-XXXX-uc.a.run.app

# 3. (Manual alternative) fetch best config after sweep finishes
python model_2_training/scripts/fetch_best_config.py \
  --sweep-id mlops-team-northeastern-university/automend-model2/<id>

# 4. (Manual alternative) submit full training pipeline
python gcp/pipelines/submit_training_pipeline.py \
  --train-config best_sweep_config.yaml

# 5. Retrain on new data (no sweep needed)
python gcp/pipelines/submit_training_pipeline.py --retrain-only
```
