# Automend MLOps Monorepo

A production-ready MLOps data pipeline integrating 6 datasets for two ML tracks, built with Apache Airflow orchestration, Ray + Polars distributed processing, DVC data versioning, Polars-native validation, Fairlearn bias detection, and centralized Slack alerting.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [ML Tracks and Datasets](#ml-tracks-and-datasets)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Data Acquisition Modes](#data-acquisition-modes)
7. [Configuration Reference](#configuration-reference)
8. [Pipeline Components](#pipeline-components)
9. [Data Versioning with DVC](#data-versioning-with-dvc)
10. [Schema Validation and Statistics](#schema-validation-and-statistics)
11. [Anomaly Detection and Alerts](#anomaly-detection-and-alerts)
12. [Bias Detection and Mitigation](#bias-detection-and-mitigation)
13. [Testing](#testing)
14. [Troubleshooting](#troubleshooting)

---

## Project Overview

AutoMend is a self-healing MLOps platform, "Zapier for MLOps," that autonomously remediates production ML incidents through event-driven workflows. The end-to-end MLOps data pipeline processes and prepares data for two distinct machine learning models:

- **Track A (Trigger Engine)**: A BERT-based classifier for anomaly detection
- **Track B (Generative Architect)**: A fine-tuned Llama-3 agent for infrastructure remediation

The pipeline transforms raw data from 6 different sources into standardized "Golden Formats" ready for model training.

### Key Features

- **Airflow Orchestration**: All pipelines run as DAGs with dependency management
- **Ray + Polars Processing**: Distributed data processing with Ray tasks/actors and Polars DataFrames
- **Three Data Modes**: `dummy` (offline synthetic), `sample` (capped download), `full` (complete dataset)
- **DVC Data Versioning**: Track and version all data artifacts
- **Polars-Native Validation**: Lightweight schema and data quality checks
- **Fairlearn Bias Detection**: Data slicing and fairness analysis
- **Centralized Alerting**: Slack webhook notifications for all pipeline events
- **Comprehensive Testing**: Unit, integration, and E2E tests with pytest
- **Docker Deployment**: Complete containerized environment with Ray head node

---

## Architecture

### Orchestration Design

Airflow serves as the **sole orchestrator** for all pipelines. DVC is used **only for data versioning**, not for pipeline execution. Ray provides distributed processing within each pipeline task.

```
                         ┌─────────────────────────────────────┐
                         │      AIRFLOW (Orchestration)        │
                         └─────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
          ┌─────────────────┐                         ┌─────────────────┐
          │  master_track_a │                         │  master_track_b │
          └─────────────────┘                         └─────────────────┘
                    │                                           │
        ┌───────────┴───────────┐           ┌───────────┬───────┴───────┬───────────┐
        ▼                       ▼           ▼           ▼               ▼           ▼
  ┌───────────┐           ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
  │ds1_alibaba│           │ds2_loghub │ │ds3_stack- │ │ds4_synth- │ │ds5_glaive │ │ds6_iac    │
  │_pipeline  │           │_pipeline  │ │overflow   │ │etic_dag   │ │_pipeline  │ │_pipeline  │
  └───────────┘           └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘
        │                       │           │           │               │           │
        └───────────┬───────────┘           └───────────┴───────┬───────┴───────────┘
                    ▼                                           ▼
          ┌─────────────────┐                         ┌─────────────────┐
          │  Track A        │                         │  Track B        │
          │  Combiner       │                         │  Combiner       │
          └─────────────────┘                         └─────────────────┘
```

### Processing Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | Apache Airflow | DAG scheduling, task dependencies, retries |
| Processing | Ray (Tasks, Data, Actors) | Parallel/distributed data processing |
| DataFrames | Polars | Vectorized data transformations (replaces pandas) |
| Validation | Polars-native checks | Schema validation, data quality (replaces Great Expectations) |
| Bias Detection | Fairlearn + Polars | Data slicing, fairness metrics |
| Data Versioning | DVC | Track and version data artifacts |
| Alerting | Slack Webhooks | Pipeline event notifications |
| Containerization | Docker Compose | Airflow + Ray + Postgres |

### Individual Dataset DAG Tasks (example for DS1)

```
acquire_data ──► preprocess_data ──► validate_schema ──► schema_stats
                                                              │
                                                              ▼
      dvc_version ◄── export_to_interim ◄── bias_detection ◄── detect_anomalies
```

Each DAG's `acquire_data` task is **self-contained**: if raw data is missing, it automatically acquires it based on the `PIPELINE_DATA_MODE` environment variable.

---

## ML Tracks and Datasets

### Track A: Trigger Engine (Anomaly Classification)

**Target Model**: BERT-based Sequence Classifier (LogBERT)

| Label | Class | Description | Source |
|-------|-------|-------------|--------|
| 0 | Normal | No anomaly detected | DS1, DS2 |
| 1 | Resource_Exhaustion | High memory/CPU usage, OOM | DS1, DS2 |
| 2 | System_Crash | Failed jobs, executor failures | DS1, DS2 |
| 3 | Network_Failure | Timeout, unreachable hosts | DS1, DS2 |
| 4 | Data_Drift | Checksum errors, verification failures | DS1, DS2 |
| 5 | Auth_Failure | Authentication failures | DS2 only |
| 6 | Permission_Denied | Access denied errors | DS2 only |

**Output Format (Format A)**: Parquet with columns `sequence_ids` (List[int]) and `label` (int 0-6)

#### Dataset 1: Alibaba Cluster Trace 2017

- **Processing**: Polars LazyFrames + Ray remote tasks for parallel CSV processing
- **Pipeline**: Feature selection → Discretization → Sliding window → Label logic → Class balancing

#### Dataset 2: LogHub (LogPAI)

- **Systems**: Linux, Hadoop, HDFS, Spark, HPC
- **Processing**: Ray Data + Polars normalizers for each log system
- **Pipeline**: Normalize → Merge → Sample → Label → Validate → Format

### Track B: Generative Architect (Workflow Agent)

**Target Model**: Fine-Tuned Llama-3-8B (Instruction Tuned)

**Output Format (Format B - ChatML)**: JSONL with ChatML message structure

#### Dataset 3: StackOverflow Q&A

- **Processing**: Python generators + chunked Polars + Ray for parallel batch processing
- **Source**: StackOverflow API (sample/full mode) or synthetic CSV (dummy mode)

#### Dataset 4: Synthetic MLOps Scenarios

- **Processing**: Ray actors + asyncio for concurrent Gemini API calls
- **Source**: SQLite prompts database + Google Gemini 2.5 Flash API
- **Requires**: `GOOGLE_API_KEY` environment variable

#### Dataset 5: Glaive Function Calling v2

- **Processing**: Ray Data + HuggingFace integration for distributed loading
- **Source**: HuggingFace Hub (`glaiveai/glaive-function-calling-v2`)

#### Dataset 6: The Stack (IaC)

- **Processing**: Ray Data + distributed filtering and PII redaction
- **Source**: HuggingFace Hub (`bigcode/the-stack-dedup`, YAML sub-corpus)
- **Requires**: `HF_TOKEN` for gated dataset access (full mode)

---

## Project Structure

```
Automend/
├── data/
│   ├── raw/                        # Raw input data (DVC tracked)
│   ├── external/                   # External CSV files (DS3)
│   ├── interim/                    # Intermediate outputs for combiners
│   └── processed/                  # Final outputs (DVC tracked)
├── src/
│   ├── config/
│   │   ├── paths.py               # Centralized path configuration
│   │   ├── ray_config.py          # Ray initialization and per-dataset configs
│   │   └── data_mode.py           # PIPELINE_DATA_MODE configuration
│   ├── utils/
│   │   ├── data_acquire.py        # Shared ensure_data() orchestrator
│   │   ├── polars_validation.py   # Polars-native validation functions
│   │   ├── dvc_utils.py           # DVC utility functions
│   │   └── alerting.py            # Centralized Slack alerting
│   ├── dataset_1_alibaba/         # Track A: Alibaba pipeline
│   ├── dataset_2_loghub/          # Track A: LogHub pipeline
│   ├── dataset_3_stackoverflow/   # Track B: StackOverflow pipeline
│   ├── dataset_4_synthetic/       # Track B: Synthetic pipeline
│   ├── dataset_5_glaive/          # Track B: Glaive pipeline
│   ├── dataset_6_the_stack/       # Track B: The Stack pipeline
│   ├── combiner_track_a/          # Combines DS1 + DS2 → Parquet
│   └── combiner_track_b/          # Combines DS3-6 → JSONL
├── dags/                          # ALL Airflow DAGs (centralized)
│   ├── master_track_a.py
│   ├── master_track_b.py
│   ├── ds1_alibaba_dag.py
│   ├── ds2_loghub_dag.py
│   ├── ds3_stackoverflow_dag.py
│   ├── ds4_synthetic_dag.py
│   ├── ds5_glaive_dag.py
│   └── ds6_iac_dag.py
├── tests/                         # Root-level integration tests
├── scripts/
│   └── seed_all.py                # Master seed script (respects PIPELINE_DATA_MODE)
├── docker-compose.yaml            # Airflow + Ray + Postgres
├── requirements.txt
├── pytest.ini
└── .env.example                   # Environment variables template
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Automend

# Create and activate conda environment
conda create -n mlops_project python=3.12
conda activate mlops_project

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env`:

```bash
# Data acquisition mode (dummy / sample / full)
PIPELINE_DATA_MODE=dummy

# Required for DS4 (Gemini API) in all modes
GOOGLE_API_KEY=your_gemini_api_key

# Required for DS6 (The Stack) in full mode
HF_TOKEN=your_huggingface_token

# Optional: Slack alerting
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### 3. Seed Data (Optional)

Data is auto-acquired when DAGs run, but you can pre-seed for local development:

```bash
# Seed using current PIPELINE_DATA_MODE (default: dummy)
python scripts/seed_all.py

# Force sample-mode downloads for DS2/DS5/DS6
python scripts/seed_all.py --download

# Seed specific datasets
python scripts/seed_all.py --ds1 --ds2
```

### 4. Start Docker Services

```bash
docker-compose up -d

# Wait for initialization (~60 seconds for pip installs)
docker-compose logs -f airflow-init
```

Services:
- **Airflow UI**: http://localhost:8080 (username: `airflow`, password: `airflow`)
- **Ray Dashboard**: http://localhost:8265

### 5. Run Pipelines

In the Airflow UI:
1. Enable the desired DAG (toggle on)
2. Click "Trigger DAG"
3. Monitor in Graph or Tree view

Individual DAGs: `ds1_alibaba_pipeline`, `ds2_loghub_pipeline`, `ds3_stackoverflow_pipeline`, `ds4_synthetic_dag`, `ds5_glaive_pipeline`, `ds6_iac_pipeline`

Master DAGs: `master_track_a` (DS1 + DS2 + combiner), `master_track_b` (DS3-6 + combiner)

---

## Data Acquisition Modes

The `PIPELINE_DATA_MODE` environment variable controls how each pipeline acquires raw data when it's missing. Every DAG's acquire task follows the same pattern:

1. Check if data exists locally
2. If not, try `dvc pull` from remote
3. If still missing, acquire based on mode

### Mode Comparison

| Dataset | `dummy` (offline) | `sample` (capped download) | `full` (complete) |
|---------|-------------------|---------------------------|-------------------|
| DS1 (Alibaba) | 100-row synthetic CSVs | 1,000-row synthetic CSVs | Manual CSV placement required |
| DS2 (LogHub) | 50-row synthetic log CSVs | 2K-row CSVs from GitHub | Same as sample (max available) |
| DS3 (StackOverflow) | 50-row synthetic Q&A | 500 questions via API | Unlimited API queries |
| DS4 (Synthetic) | 15 seed prompts + Gemini | 15 prompts + Gemini | 100 expanded prompts + Gemini |
| DS5 (Glaive) | 100 synthetic JSONL records | 5,000 records from HuggingFace | Full dataset from HuggingFace |
| DS6 (The Stack) | 50 synthetic parquet records | 20,000 records from HuggingFace | Full dataset (needs HF_TOKEN) |

### When to Use Each Mode

- **`dummy`**: Local development, CI/CD, E2E testing. No network or API keys needed. Fast.
- **`sample`**: Integration testing with real data. Network required. Moderate size.
- **`full`**: Production training runs. Network + API tokens required. Full dataset size.

### Setting the Mode

```bash
# In .env file
PIPELINE_DATA_MODE=dummy

# Or inline
PIPELINE_DATA_MODE=sample python scripts/seed_all.py

# Docker Compose reads from .env automatically
docker-compose up -d
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PIPELINE_DATA_MODE` | No | `dummy` | Data acquisition mode: `dummy`, `sample`, or `full` |
| `GOOGLE_API_KEY` | For DS4 | - | Google Gemini API key for synthetic workflow generation |
| `HF_TOKEN` | For DS6 full | - | HuggingFace token for gated dataset access |
| `SLACK_WEBHOOK_URL` | No | - | Slack incoming webhook URL for alerts |
| `SLACK_CHANNEL` | No | `#automend-alerts` | Slack channel display name |
| `RAY_NUM_CPUS` | No | `4` | Number of CPUs for Ray workers |
| `RAY_OBJECT_STORE_MB` | No | `512` | Ray object store memory in MB |

### Key Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables (API keys, mode, Slack) |
| `src/config/data_mode.py` | Per-dataset acquisition parameters for each mode |
| `src/config/ray_config.py` | Ray initialization, per-dataset chunk/sample sizes |
| `src/config/paths.py` | Centralized data directory paths |
| `docker-compose.yaml` | Service definitions (Airflow, Ray, Postgres) |

---

## Pipeline Components

### Data Acquisition (`src/utils/data_acquire.py`)

All DAGs use the shared `ensure_data()` function:

```python
from src.utils.data_acquire import ensure_data

# Called in each DAG's acquire task
result = ensure_data("ds1", raw_dir, project_root)
# Returns: {"status": "cached"|"seeded"|"downloaded", "mode": "dummy"|"sample"|"full"}
```

The function checks local files → tries DVC pull → dispatches to the appropriate seed or download script based on `PIPELINE_DATA_MODE`.

### Data Processing (Ray + Polars)

Each dataset uses Ray and Polars differently based on its workload:

| Dataset | Ray Pattern | Polars Usage |
|---------|------------|-------------|
| DS1 | `@ray.remote` tasks (parallel CSV files) | LazyFrames for transforms |
| DS2 | Ray Data for distributed normalization | Expressions for log parsing |
| DS3 | `@ray.remote` tasks (parallel batch processing) | DataFrame for validation |
| DS4 | Ray Actors + asyncio (concurrent Gemini calls) | - |
| DS5 | `ray.data.from_huggingface()` | DataFrame for preprocessing |
| DS6 | `ray.data.read_parquet()` + `.map()` | DataFrame for stats/bias |

### Validation (`src/utils/polars_validation.py`)

Polars-native validation replaces Great Expectations:

```python
from src.utils.polars_validation import (
    validate_columns_present,
    validate_no_nulls,
    validate_value_range,
    validate_row_count,
    run_validation_suite,
)

checks = [
    validate_columns_present(df, ["col_a", "col_b"]),
    validate_no_nulls(df, ["col_a"]),
    validate_value_range(df, "score", min_val=0, max_val=100),
]
report = run_validation_suite(checks)
# {"total": 3, "passed": 3, "failed": 0, "all_passed": True, "results": [...]}
```

### Centralized Alerting (`src/utils/alerting.py`)

All alerts flow through a unified Slack webhook system:

```python
from src.utils.alerting import (
    alert_anomaly_detected,
    alert_validation_failure,
    alert_bias_detected,
    on_failure_callback,
)
```

| Alert Type | Severity | When Triggered |
|------------|----------|----------------|
| Pipeline Failure | CRITICAL | Any task fails (via `on_failure_callback`) |
| Anomaly Detected | WARNING/ERROR | Data anomalies found |
| Validation Failure | ERROR | Schema validation fails |
| Bias Detected | WARNING/ERROR | Data bias above threshold |

Alerts are **non-blocking** -- pipelines continue even if Slack delivery fails. All alerts are also logged to `logs/alerts.log` and `logs/alerts_history.json`.

---

## Data Versioning with DVC

DVC is used for **data tracking only** (not pipeline orchestration).

```ini
# .dvc/config
[core]
    remote = local_remote
[remote "local_remote"]
    url = ../dvc_storage
```

Each DAG automatically versions data with DVC after successful acquisition and processing. To interact manually:

```bash
dvc pull          # Pull data from remote
dvc push          # Push data to remote
dvc status        # Check tracking status
```

### Wiping Data for Fresh Start

```powershell
Remove-Item -Recurse -Force ..\dvc_storage       # Remote storage
Remove-Item -Recurse -Force data\raw\*            # Raw data
Remove-Item -Recurse -Force data\processed\*      # Processed data
Remove-Item -Recurse -Force data\interim\*         # Interim data
Remove-Item -Recurse -Force data\external\*        # External CSVs (DS3)
```

---

## Bias Detection and Mitigation

Each dataset has dedicated bias detection using Polars for slicing and Fairlearn for metrics:

| Dataset | Slicing Features | Library |
|---------|-----------------|---------|
| DS1 (Alibaba) | `status`, `task_type`, `label` | Fairlearn + Polars→pandas bridge |
| DS2 (LogHub) | `system`, `severity`, `event_type` | Polars |
| DS3 (StackOverflow) | Tags, answer quality | Polars |
| DS5 (Glaive) | Complexity tier, turn count, function calls | Polars |
| DS6 (The Stack) | IaC type, license, file size | Polars |

---

## Testing

```bash
# Run all tests
python run_all_tests.py

# Run specific dataset tests
python run_all_tests.py --ds1
python run_all_tests.py --ds2

# Run with pytest directly
pytest tests/ -v                           # Root integration tests
pytest src/dataset_1_alibaba/tests/ -v     # DS1 tests
pytest --cov=src --cov-report=html         # With coverage
```

---

## Troubleshooting

### DAGs Not Showing in Airflow UI

```bash
docker-compose exec airflow-scheduler python /opt/airflow/dags/master_track_a.py
docker-compose logs airflow-scheduler | tail -50
```

### Ray Connection Issues

Ray runs locally within each Airflow worker (no cluster connection needed in Docker Compose). If you see Ray errors:

```bash
# Check Ray head is running
docker-compose ps ray-head
docker-compose logs ray-head --tail 20

# Restart services
docker-compose restart airflow-scheduler airflow-webserver
```

### Task Failures

1. Check task logs in Airflow UI (Task → Log)
2. Common causes:
   - Missing API keys (`GOOGLE_API_KEY` for DS4, `HF_TOKEN` for DS6)
   - Network connectivity (DS2/DS5/DS6 in sample/full mode)
   - Memory limits (increase Docker Desktop memory to 6-8 GB)

### Docker Commands

```bash
docker-compose up -d              # Start all services
docker-compose down               # Stop services
docker-compose down -v            # Stop and remove volumes (full reset)
docker-compose logs -f            # Follow all logs
docker-compose restart            # Restart all services
```

---

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Ray Documentation](https://docs.ray.io/)
- [Polars Documentation](https://docs.pola.rs/)
- [DVC Documentation](https://dvc.org/doc)
- [Fairlearn Documentation](https://fairlearn.org/)
- [LogPAI/LogHub Repository](https://github.com/logpai/loghub)

---

## Dataset Licenses

| Dataset | Source | License |
|---------|--------|---------|
| DS1: Alibaba Cluster Trace 2017 | [alibaba/clusterdata](https://github.com/alibaba/clusterdata) | Research Use |
| DS2: LogHub (LogPAI) | [logpai/loghub](https://github.com/logpai/loghub) | CC BY 4.0 |
| DS3: StackOverflow | [StackOverflow](https://stackoverflow.com/) | CC BY-SA 4.0 |
| DS4: Synthetic | Generated via Gemini API | N/A |
| DS5: Glaive Function Calling v2 | [HuggingFace](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Check HuggingFace |
| DS6: The Stack | [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) | Permissive (varies) |

---

## License

This project code is developed as part of an MLOps course assignment. The datasets used have their own licenses as detailed above.
