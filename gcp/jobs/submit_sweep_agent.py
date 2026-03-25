"""
submit_sweep_agent.py

Submits W&B sweep agent trials as Vertex AI Custom Training Jobs.

This is Workflow 1 — Hyperparameter Search.

How it works:
  1. This script auto-creates the W&B sweep (no manual step needed)
  2. Launches N parallel Vertex AI training jobs
  3. Each job runs: wandb agent <sweep_id> --count 1
  4. W&B Bayesian optimizer picks next hyperparameters for each trial
  5. After all trials finish, fetch the best config automatically

Usage:
    # Auto-create sweep + launch 10 trials (recommended)
    python gcp/jobs/submit_sweep_agent.py --trials 10

    # Resume an existing sweep
    python gcp/jobs/submit_sweep_agent.py --sweep-id <sweep_id> --trials 10

    # Dry-run — print job config but do NOT submit
    python gcp/jobs/submit_sweep_agent.py --trials 3 --dry-run
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROJECT_ID, REGION, IMAGE_URI,
    TRAINER_SA,
    WANDB_PROJECT, WANDB_ENTITY,
    TRAIN_MACHINE_TYPE, TRAIN_ACCELERATOR, TRAIN_ACCEL_COUNT,
)

_REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
_SWEEP_CONFIG = _REPO_ROOT / "model_2_training/configs/sweep/wandb_sweep.yaml"


def _image(tag: str) -> str:
    base = IMAGE_URI.rsplit(":", 1)[0]
    return f"{base}:{tag}"


def create_sweep() -> str:
    """
    Auto-create a W&B sweep from the sweep config YAML.

    Returns:
        Full sweep path: entity/project/sweep_id
    """
    try:
        import wandb
    except ImportError:
        print("wandb not installed. Run: pip install wandb")
        sys.exit(1)

    import yaml
    with open(_SWEEP_CONFIG) as f:
        sweep_cfg = yaml.safe_load(f)

    print(f"Creating W&B sweep from: {_SWEEP_CONFIG}")
    sweep_id = wandb.sweep(
        sweep=sweep_cfg,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
    )
    full_sweep_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}"
    print(f"Sweep created: {full_sweep_path}")
    print(f"W&B URL: https://wandb.ai/{full_sweep_path}")
    return full_sweep_path


def submit_sweep_trial(
    sweep_id: str,
    trial_index: int,
    image_tag: str,
    dry_run: bool = False,
) -> None:
    """Submit a single W&B sweep trial as a Vertex AI Custom Training Job."""
    try:
        from google.cloud import aiplatform
    except ImportError:
        print("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")
        sys.exit(1)

    image = _image(image_tag)
    job_display_name = f"automend-sweep-trial-{trial_index:02d}"

    command = [
        "python", "/workspace/gcp/scripts/run_agent.py", sweep_id
    ]

    worker_pool_spec = {
        "machine_spec": {
            "machine_type":      TRAIN_MACHINE_TYPE,
            "accelerator_type":  TRAIN_ACCELERATOR,
            "accelerator_count": TRAIN_ACCEL_COUNT,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": image,
            "command": command,
            "env": [
                {"name": "WANDB_PROJECT",               "value": WANDB_PROJECT},
                {"name": "WANDB_ENTITY",                "value": WANDB_ENTITY},
                {"name": "WANDB_START_METHOD",          "value": "thread"},
                {"name": "PYTHONUNBUFFERED",            "value": "1"},
                {"name": "PYTHONPATH",                  "value": "/workspace"},
                {"name": "PYTORCH_CUDA_ALLOC_CONF",     "value": "expandable_segments:True"},
            ],
        },
    }

    if dry_run:
        print(f"  [DRY-RUN] Would submit: {job_display_name}")
        print(f"    Image   : {image}")
        print(f"    Machine : {TRAIN_MACHINE_TYPE} + {TRAIN_ACCELERATOR} x{TRAIN_ACCEL_COUNT}")
        print(f"    Command : wandb agent --count 1 {sweep_id}")
        return

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=[worker_pool_spec],
        staging_bucket="gs://automend-model2/staging",
    )

    job.submit(service_account=TRAINER_SA)
    print(f"  Submitted: {job_display_name} — resource: {job.resource_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-create W&B sweep and launch trials on Vertex AI."
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help=(
            "Resume an existing sweep (entity/project/sweep_id). "
            "If omitted, a new sweep is created automatically."
        ),
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of parallel sweep trials to launch (default: 10)",
    )
    parser.add_argument(
        "--image-tag", default="latest",
        help="Docker image tag (default: latest)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print job configs but do NOT submit to Vertex AI",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-create sweep if no sweep-id provided
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Resuming existing sweep: {sweep_id}")
    else:
        sweep_id = create_sweep()

    print()
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Launching {args.trials} sweep trials")
    print(f"  Sweep ID : {sweep_id}")
    print(f"  Image    : {_image(args.image_tag)}")
    print(f"  Machine  : {TRAIN_MACHINE_TYPE} + {TRAIN_ACCELERATOR} x{TRAIN_ACCEL_COUNT}")
    print(f"  SA       : {TRAINER_SA}")
    print()

    for i in range(1, args.trials + 1):
        submit_sweep_trial(
            sweep_id=sweep_id,
            trial_index=i,
            image_tag=args.image_tag,
            dry_run=args.dry_run,
        )

    print()
    if not args.dry_run:
        print(f"All {args.trials} trials submitted.")
        print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
        print(f"W&B sweep:  https://wandb.ai/{sweep_id}")

        print()
        print("When all trials complete, fetch best config + train:")
        print(f"  python model_2_training/scripts/fetch_best_config.py --sweep-id {sweep_id}")
    else:
        print("Dry-run complete — no jobs submitted.")


if __name__ == "__main__":
    main()
