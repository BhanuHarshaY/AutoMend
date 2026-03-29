"""
submit_sweep_agent.py

Launch Model 1 hyperparameter sweep agents on GCP.
Each agent runs Ray Tune + Optuna trials inside the training container.

Usage:
    python model_1_training/gcp/jobs/submit_sweep_agent.py --trials 20
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROJECT_ID, REGION, IMAGE_URI, GCS_FUSE_MOUNT,
    TRAINER_SA, WANDB_PROJECT, WANDB_ENTITY,
    TRAIN_MACHINE_TYPE, TRAIN_ACCELERATOR, TRAIN_ACCEL_COUNT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Model 1 sweep on GCP")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--image-tag", default="latest")
    args = parser.parse_args()

    image = IMAGE_URI.rsplit(":", 1)[0] + f":{args.image_tag}"

    print(f"Sweep configuration:")
    print(f"  Trials     : {args.trials}")
    print(f"  Image      : {image}")
    print(f"  Machine    : {TRAIN_MACHINE_TYPE} + {TRAIN_ACCELERATOR}")
    print(f"  W&B project: {WANDB_PROJECT}")
    print()

    try:
        from google.cloud import aiplatform
    except ImportError:
        print("google-cloud-aiplatform not installed.")
        sys.exit(1)

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=f"model1-sweep-{args.trials}trials",
        container_uri=image,
        command=[
            "python", "-m", "model_1_training.scripts.run_sweep",
            "--sweep-config", "/workspace/model_1_training/configs/sweep/ray_optuna_sweep.yaml",
            "--data-config", "/workspace/model_1_training/configs/data/track_a.yaml",
            "--model-config", "/workspace/model_1_training/configs/model/roberta_base.yaml",
            "--train-config", "/workspace/model_1_training/configs/train/full_finetune.yaml",
        ],
    )

    job.run(
        machine_type=TRAIN_MACHINE_TYPE,
        accelerator_type=TRAIN_ACCELERATOR,
        accelerator_count=TRAIN_ACCEL_COUNT,
        service_account=TRAINER_SA,
        environment_variables={
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_ENTITY": WANDB_ENTITY,
            "PYTHONPATH": "/workspace",
            "PROJECT_ID": PROJECT_ID,
        },
    )

    print(f"\nSweep job submitted!")
    print(f"Track at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


if __name__ == "__main__":
    main()
