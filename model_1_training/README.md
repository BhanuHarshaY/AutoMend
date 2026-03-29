# Model 1 Training Pipeline — Watchdog Anomaly Detector

RoBERTa-base fine-tuned for multi-class sequence classification on infrastructure telemetry data (Track A). Detects failures in server logs and metric token sequences across 7 anomaly categories.

## Architecture

| Component | Detail |
|-----------|--------|
| **Base Model** | `roberta-base` (125M parameters) |
| **Task** | 7-class sequence classification |
| **Training** | Full fine-tuning with Focal Loss |
| **HP Tuning** | Ray Tune + Optuna (Bayesian) |
| **Tracking** | Weights & Biases |
| **Explainability** | Captum Integrated Gradients |
| **Bias Detection** | Fairlearn MetricFrame |
| **Deployment** | GCP Vertex AI Pipelines + Artifact Registry |

### Label Semantics

| Label | Class Name | Source |
|-------|-----------|--------|
| 0 | Normal | DS1, DS2 |
| 1 | Resource_Exhaustion | DS1, DS2 |
| 2 | System_Crash | DS1, DS2 |
| 3 | Network_Failure | DS1, DS2 |
| 4 | Data_Drift | DS1, DS2 |
| 5 | Auth_Failure | DS2 only |
| 6 | Permission_Denied | DS2 only |

### Custom Tokenizer

The data pipeline produces `sequence_ids` as discretized integer IDs (not natural language). This pipeline adds custom special tokens to the RoBERTa tokenizer:

- `[CPU_0]` .. `[CPU_9]` — CPU utilization buckets
- `[MEM_0]` .. `[MEM_9]` — Memory utilization buckets
- `[STS_TERMINATED]`, `[STS_FAILED]`, etc. — Status tokens
- `[EVT_ADD]`, `[EVT_FAILURE]`, etc. — Event tokens
- `[TMPL_1]` .. `[TMPL_999]` — Log event template IDs

The model's token embeddings are resized to accommodate these new tokens.

## Setup

### Prerequisites

- Python 3.10+
- CUDA GPU (recommended) or CPU
- Docker (for containerized training)

### Local Installation

```bash
cd model_1_training
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` at the repo root and set:

```
WANDB_API_KEY=your_wandb_key        # Optional: for experiment tracking
PIPELINE_DATA_MODE=sample           # dummy / sample / full
```

## Usage

### 1. Split Data

Requires `data/processed/track_A_combined.parquet` from the data pipeline.

```bash
python -m model_1_training.scripts.run_split \
    --config model_1_training/configs/data/track_a.yaml
```

### 2. Train

```bash
python -m model_1_training.scripts.run_train \
    --data-config model_1_training/configs/data/track_a.yaml \
    --model-config model_1_training/configs/model/roberta_base.yaml \
    --train-config model_1_training/configs/train/full_finetune.yaml
```

CLI overrides: `--lr 3e-5 --batch-size 32 --epochs 10 --weight-decay 0.05`

### 3. Evaluate

```bash
python -m model_1_training.scripts.run_eval \
    --checkpoint model_1_training/outputs/checkpoints/best_model \
    --split model_1_training/data/splits/val.parquet \
    --output-dir model_1_training/outputs/reports/val
```

Outputs: `metrics.json`, `confusion_matrix.png`, `per_class_f1.png`, `per_class_recall.png`

### 4. Hyperparameter Sweep

```bash
python -m model_1_training.scripts.run_sweep \
    --sweep-config model_1_training/configs/sweep/ray_optuna_sweep.yaml
```

Search space:
- Learning Rate: 1e-5 to 5e-5 (log-uniform)
- Batch Size: [16, 32, 64]
- Weight Decay: 0.01 to 0.1

### 5. Sensitivity Analysis

```bash
python -m model_1_training.scripts.run_sensitivity \
    --checkpoint model_1_training/outputs/checkpoints/best_model \
    --split model_1_training/data/splits/val.parquet \
    --output-dir model_1_training/outputs/reports/sensitivity \
    --anomaly-only
```

### 6. Bias Detection

```bash
python -m model_1_training.scripts.run_bias \
    --checkpoint model_1_training/outputs/checkpoints/best_model \
    --split model_1_training/data/splits/val.parquet \
    --output-dir model_1_training/outputs/reports/bias
```

## Docker (Local)

Build and run the training container locally with GPU access:

```bash
# Build
docker build -f model_1_training/docker/Dockerfile -t automend-train-model1 .

# Train
docker run --gpus all \
    -v ./data:/workspace/data \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    automend-train-model1 \
    -m model_1_training.scripts.run_train

# Evaluate
docker run --gpus all \
    -v ./data:/workspace/data \
    -v ./model_1_training/outputs:/workspace/model_1_training/outputs \
    automend-train-model1 \
    -m model_1_training.scripts.run_eval \
    --checkpoint /workspace/model_1_training/outputs/checkpoints/best_model \
    --split /workspace/model_1_training/data/splits/val.parquet \
    --output-dir /workspace/model_1_training/outputs/reports/val
```

## GCP Deployment

### Build and Push Image

```bash
bash model_1_training/gcp/build_and_push.sh
```

### Submit Training Pipeline (Vertex AI)

```bash
# Full pipeline
python model_1_training/gcp/pipelines/submit_training_pipeline.py

# Dry run (compile YAML only)
python model_1_training/gcp/pipelines/submit_training_pipeline.py --dry-run

# With best sweep config
python model_1_training/gcp/pipelines/submit_training_pipeline.py \
    --train-config best_sweep_config.yaml
```

### Run Sweep on GCP

```bash
python model_1_training/gcp/jobs/submit_sweep_agent.py --trials 20
```

## Project Structure

```
model_1_training/
├── scripts/           # CLI entry points
│   ├── run_split.py
│   ├── run_train.py
│   ├── run_eval.py
│   ├── run_sweep.py
│   ├── run_sensitivity.py
│   ├── run_bias.py
│   └── fetch_best_config.py
├── src/
│   ├── data/          # Data loading, tokenizer, dataset, splitting
│   ├── model/         # Model loading, focal loss
│   ├── train/         # Training loop, callbacks
│   ├── eval/          # Evaluation, metrics, visualization
│   ├── sensitivity/   # Captum Integrated Gradients
│   ├── bias/          # Fairlearn bias detection
│   ├── tracking/      # W&B experiment tracking
│   └── utils/         # Config, device, seed utilities
├── configs/           # YAML configuration files
├── docker/            # Dockerfile for training container
├── gcp/               # GCP deployment (Vertex AI, AR, GCS)
├── data/splits/       # Train/val/test splits (generated)
└── requirements.txt
```

## CI/CD

The pipeline follows the same pattern as Model 2:

1. **Build**: `build_and_push.sh` builds the Docker image and pushes to Artifact Registry
2. **Sweep**: `submit_sweep_agent.py` launches Ray Tune + Optuna on Vertex AI
3. **Train**: `submit_training_pipeline.py` runs the full pipeline on Vertex AI
4. **Validate**: Automated validation gate checks Macro F1 >= 0.90
5. **Bias Check**: Fairlearn analysis runs automatically; alerts on disparity > 10%
6. **Rollback**: New model metrics compared against previous best in W&B

Slack alerts are sent via the existing `src/utils/alerting.py` for pipeline failures.
