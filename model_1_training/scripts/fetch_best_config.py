"""
fetch_best_config.py

Pull the best sweep configuration from W&B and save it locally.
Optionally upload to GCS for Vertex AI pipeline consumption.

Usage:
    python -m model_1_training.scripts.fetch_best_config \
        --sweep-id <entity/project/sweep_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

M1_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = M1_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from model_1_training.src.utils.config import save_yaml


def fetch_from_wandb(sweep_id: str) -> dict:
    """Fetch the best run config from a W&B sweep."""
    import wandb

    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    best_run = sweep.best_run()
    logger.info(f"Best run: {best_run.name} (ID: {best_run.id})")
    logger.info(f"Best metric: {best_run.summary.get('eval_macro_f1', 'N/A')}")

    config = dict(best_run.config)
    return config


def upload_to_gcs(local_path: Path, gcs_path: str) -> None:
    """Upload a file to GCS."""
    try:
        from google.cloud import storage
        bucket_name = gcs_path.replace("gs://", "").split("/")[0]
        blob_name = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded -> {gcs_path}")
    except Exception as e:
        logger.warning(f"GCS upload failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch best sweep config from W&B")
    parser.add_argument("--sweep-id", required=True, help="W&B sweep ID (entity/project/sweep_id)")
    parser.add_argument("--upload-gcs", action="store_true", help="Upload to GCS")
    args = parser.parse_args()

    config = fetch_from_wandb(args.sweep_id)

    out_path = M1_ROOT / "configs" / "train" / "best_sweep_config.yaml"
    save_yaml(config, out_path)
    logger.success(f"Saved best config -> {out_path}")

    if args.upload_gcs:
        upload_to_gcs(out_path, "gs://automend-model1/configs/train/best_sweep_config.yaml")

    print(f"\nBest config saved to: {out_path}")
    print(f"Config: {config}")


if __name__ == "__main__":
    main()
