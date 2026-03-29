"""
run_train.py

CLI entry point for Model 1 training.

Usage:
    python -m model_1_training.scripts.run_train \
        --data-config configs/data/track_a.yaml \
        --model-config configs/model/roberta_base.yaml \
        --train-config configs/train/full_finetune.yaml \
        [--output-dir /path/to/checkpoints] \
        [--splits-dir /path/to/splits]
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

M1_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = M1_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

from loguru import logger
from model_1_training.src.utils.config import load_yaml
from model_1_training.src.train.train_loop import run_training

# Try loading W&B API key from GCP Secret Manager (when running on Vertex AI)
try:
    from google.cloud import secretmanager
    if os.environ.get("PROJECT_ID") and not os.environ.get("WANDB_API_KEY"):
        client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{os.environ['PROJECT_ID']}/secrets/WANDB_API_KEY/versions/latest"
        response = client.access_secret_version(request={"name": secret_name})
        os.environ["WANDB_API_KEY"] = response.payload.data.decode("UTF-8").strip()
        logger.info("Loaded WANDB_API_KEY from Secret Manager")
except Exception:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model 1 (RoBERTa anomaly classifier)")
    parser.add_argument("--data-config", default=str(M1_ROOT / "configs/data/track_a.yaml"))
    parser.add_argument("--model-config", default=str(M1_ROOT / "configs/model/roberta_base.yaml"))
    parser.add_argument("--train-config", default=str(M1_ROOT / "configs/train/full_finetune.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--artifact", default=None, help="Override path to track_A_combined.parquet")
    # CLI overrides for sweep
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    if args.lr:
        train_cfg["learning_rate"] = args.lr
    if args.batch_size:
        train_cfg["per_device_train_batch_size"] = args.batch_size
    if args.epochs:
        train_cfg["num_train_epochs"] = args.epochs
    if args.weight_decay:
        train_cfg["weight_decay"] = args.weight_decay

    best_model = run_training(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        m1_root=M1_ROOT,
        artifact_path=args.artifact,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
    )

    print(f"\nTraining complete!")
    print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()
