"""
Seed script for DS6 (The Stack) - generates synthetic parquet chunks for dummy mode.

Creates chunk_0000.parquet with records matching The Stack dedup YAML
sub-corpus schema (content, ext, lang, size, hexsha, etc.).

Usage:
    python -m src.dataset_6_the_stack.scripts.download.seed_data
"""

import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
DS6_ROOT = SCRIPT_DIR.parent.parent
PROJECT_ROOT = DS6_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

random.seed(42)

YAML_TEMPLATES = [
    # Kubernetes Deployment with GPU
    """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}-inference
  labels:
    app: {name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    spec:
      containers:
      - name: model-server
        image: registry.example.com/{name}:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
""",
    # KServe InferenceService
    """\
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {name}-predictor
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: gs://models/{name}/v1
      resources:
        limits:
          nvidia.com/gpu: 1
""",
    # Kubeflow Pipeline
    """\
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: {name}-pipeline
spec:
  entrypoint: main
  templates:
  - name: main
    steps:
    - - name: preprocess
        template: preprocess-step
    - - name: train
        template: train-step
  - name: preprocess-step
    container:
      image: registry.example.com/{name}-preprocess:latest
  - name: train-step
    container:
      image: registry.example.com/{name}-train:latest
      resources:
        limits:
          nvidia.com/gpu: {replicas}
""",
    # MLflow Serving
    """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-{name}
spec:
  replicas: {replicas}
  template:
    spec:
      containers:
      - name: mlflow-server
        image: mlflow-serving:latest
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow.mlops:5000"
        - name: MODEL_URI
          value: "models:/{name}/Production"
""",
    # Seldon Deployment
    """\
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: {name}-seldon
spec:
  predictors:
  - name: default
    replicas: {replicas}
    graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/{name}
""",
]

MODEL_NAMES = [
    "fraud-detector", "churn-predictor", "recommendation-engine",
    "sentiment-analyzer", "image-classifier", "text-summarizer",
    "anomaly-detector", "demand-forecaster", "price-optimizer",
    "search-ranker",
]

REPO_NAMES = [
    "ml-platform/infra", "data-science/deployments", "mlops/manifests",
    "ai-team/k8s-configs", "ml-eng/serving", "platform/gpu-workloads",
]

LICENSES = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc"]


def _gen_record(idx: int) -> dict:
    template = YAML_TEMPLATES[idx % len(YAML_TEMPLATES)]
    name = MODEL_NAMES[idx % len(MODEL_NAMES)]
    replicas = random.randint(1, 4)
    content = template.format(name=name, replicas=replicas)

    sha = hashlib.sha1(f"{idx}-{name}".encode()).hexdigest()
    repo = random.choice(REPO_NAMES)
    ext = random.choice(["yaml", "yml"])
    license_val = random.choice(LICENSES)
    path = f"k8s/{name}.{ext}"

    return {
        "content": content,
        "ext": ext,
        "lang": "YAML",
        "size": len(content.encode()),
        "avg_line_length": round(len(content) / max(1, content.count("\n") + 1), 2),
        "max_line_length": max(len(line) for line in content.split("\n")),
        "alphanum_fraction": round(
            sum(c.isalnum() for c in content) / max(1, len(content)), 4
        ),
        "hexsha": sha,
        "max_stars_repo_path": path,
        "max_stars_repo_name": repo,
        "max_stars_repo_licenses": json.dumps([license_val]),
        "max_issues_repo_path": path,
        "max_forks_repo_path": path,
        "max_issues_repo_licenses": json.dumps([license_val]),
        "max_forks_repo_licenses": json.dumps([license_val]),
    }


def generate_all(raw_dir: Path, num_records: int = 50) -> int:
    """Generate synthetic parquet chunk(s) matching The Stack schema."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = [_gen_record(i) for i in range(num_records)]
    table = pa.Table.from_pylist(records)
    out_path = raw_dir / "chunk_0000.parquet"
    pq.write_table(table, out_path, compression="snappy")

    print(f"DS6 seed: wrote {num_records} records to {out_path}")
    return num_records


def main():
    try:
        from src.config.paths import get_ds6_raw_dir
        raw_dir = get_ds6_raw_dir()
    except ImportError:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "ds6_the_stack"

    print("Generating DS6 (The Stack) seed data...")
    generate_all(raw_dir)
    return True


if __name__ == "__main__":
    main()
