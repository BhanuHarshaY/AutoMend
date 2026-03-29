"""
submit_training_pipeline.py

Submits the AutoMend Model 1 production training pipeline to Vertex AI.

Pipeline steps:
  1. Split data
  2. Train (RoBERTa + Focal Loss, GPU)
  3. Evaluate on validation set
  4. Evaluate on test set
  5. Sensitivity analysis (Captum)
  6. Bias detection (Fairlearn)

Usage (run from repo root):
    python model_1_training/gcp/pipelines/submit_training_pipeline.py
    python model_1_training/gcp/pipelines/submit_training_pipeline.py --dry-run
"""

from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROJECT_ID, REGION, IMAGE_URI,
    GCS_PIPELINE_ROOT, GCS_FUSE_MOUNT, TRAINER_SA,
    WANDB_SECRET_NAME, WANDB_PROJECT, WANDB_ENTITY,
    PIPELINE_NAME, PIPELINE_DISPLAY_NAME,
    TRAIN_MACHINE_TYPE, TRAIN_ACCELERATOR, TRAIN_ACCEL_COUNT,
    EVAL_MACHINE_TYPE,
)

_CONFIGS = "/workspace/model_1_training/configs"


def _image(tag: str) -> str:
    base = IMAGE_URI.rsplit(":", 1)[0]
    return f"{base}:{tag}"


def build_pipeline(image_tag: str, train_config: str, run_id: str):
    try:
        from kfp import dsl
    except ImportError:
        print("kfp not installed. Run: pip install kfp google-cloud-aiplatform")
        sys.exit(1)

    image = _image(image_tag)
    _run_root = f"{GCS_FUSE_MOUNT}/outputs/runs/{run_id}"
    _checkpoint = f"{_run_root}/checkpoints/best_model"
    _splits = f"{GCS_FUSE_MOUNT}/data/splits"

    def _make_op(name, command, use_gpu=False):
        from kfp import dsl

        @dsl.container_component
        def _op():
            return dsl.ContainerSpec(image=image, command=command)

        task = _op()
        task.set_display_name(name)

        for k, v in {
            "WANDB_PROJECT":      WANDB_PROJECT,
            "WANDB_ENTITY":       WANDB_ENTITY,
            "WANDB_START_METHOD": "thread",
            "PYTHONUNBUFFERED":   "1",
            "PYTHONPATH":         "/workspace",
            "PROJECT_ID":         PROJECT_ID,
        }.items():
            task.set_env_variable(k, v)

        if use_gpu:
            task.set_accelerator_type(TRAIN_ACCELERATOR)
            task.set_accelerator_limit(str(TRAIN_ACCEL_COUNT))
            task.set_cpu_limit("8").set_memory_limit("32G")
        else:
            task.set_cpu_limit("4").set_memory_limit("16G")

        return task

    train_config_path = (
        train_config if train_config.startswith("/")
        else f"{_CONFIGS}/train/{train_config}"
    )

    @dsl.pipeline(
        name=PIPELINE_NAME,
        description=PIPELINE_DISPLAY_NAME,
        pipeline_root=GCS_PIPELINE_ROOT,
    )
    def training_pipeline():
        # 1. Split
        split_op = _make_op(
            "1 - Split data",
            command=[
                "python", "-m", "model_1_training.scripts.run_split",
                "--config", f"{_CONFIGS}/data/track_a.yaml",
                "--artifact", f"{GCS_FUSE_MOUNT}/data/track_A_combined.parquet",
                "--splits-dir", _splits,
            ],
        )

        # 2. Train
        train_op = _make_op(
            "2 - Train RoBERTa (GPU)",
            command=[
                "python", "-m", "model_1_training.scripts.run_train",
                "--data-config", f"{_CONFIGS}/data/track_a.yaml",
                "--model-config", f"{_CONFIGS}/model/roberta_base.yaml",
                "--train-config", train_config_path,
                "--output-dir", f"{_run_root}/checkpoints",
                "--splits-dir", _splits,
                "--artifact", f"{GCS_FUSE_MOUNT}/data/track_A_combined.parquet",
            ],
            use_gpu=True,
        )
        train_op.after(split_op)

        # 3. Eval (val)
        eval_val_op = _make_op(
            "3 - Evaluate (val)",
            command=[
                "python", "-m", "model_1_training.scripts.run_eval",
                "--checkpoint", _checkpoint,
                "--split", f"{_splits}/val.parquet",
                "--output-dir", f"{_run_root}/reports/val",
            ],
            use_gpu=True,
        )
        eval_val_op.after(train_op)

        # 4. Eval (test)
        eval_test_op = _make_op(
            "4 - Evaluate (test)",
            command=[
                "python", "-m", "model_1_training.scripts.run_eval",
                "--checkpoint", _checkpoint,
                "--split", f"{_splits}/test.parquet",
                "--output-dir", f"{_run_root}/reports/test",
            ],
            use_gpu=True,
        )
        eval_test_op.after(eval_val_op)

        # 5. Sensitivity analysis
        sensitivity_op = _make_op(
            "5 - Sensitivity (Captum IG)",
            command=[
                "python", "-m", "model_1_training.scripts.run_sensitivity",
                "--checkpoint", _checkpoint,
                "--split", f"{_splits}/val.parquet",
                "--output-dir", f"{_run_root}/reports/sensitivity",
                "--anomaly-only",
            ],
            use_gpu=True,
        )
        sensitivity_op.after(eval_test_op)

        # 6. Bias detection
        bias_op = _make_op(
            "6 - Bias detection (Fairlearn)",
            command=[
                "python", "-m", "model_1_training.scripts.run_bias",
                "--checkpoint", _checkpoint,
                "--split", f"{_splits}/val.parquet",
                "--output-dir", f"{_run_root}/reports/bias",
            ],
            use_gpu=True,
        )
        bias_op.after(sensitivity_op)

    return training_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Submit Model 1 training pipeline to Vertex AI")
    parser.add_argument("--image-tag", default="latest")
    parser.add_argument("--train-config", default="full_finetune.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-yaml", default="model_1_training/gcp/pipelines/training_pipeline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    pipeline_fn = build_pipeline(
        image_tag=args.image_tag,
        train_config=args.train_config,
        run_id=run_id,
    )

    try:
        from kfp import compiler
    except ImportError:
        print("kfp not installed. Run: pip install kfp google-cloud-aiplatform")
        sys.exit(1)

    output_yaml = Path(args.output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(pipeline_fn, str(output_yaml))
    print(f"Pipeline compiled -> {output_yaml}")

    if args.dry_run:
        print("Dry-run — skipping submission.")
        return

    try:
        from google.cloud import aiplatform
    except ImportError:
        print("google-cloud-aiplatform not installed.")
        sys.exit(1)

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.PipelineJob(
        display_name=PIPELINE_DISPLAY_NAME,
        template_path=str(output_yaml),
        pipeline_root=GCS_PIPELINE_ROOT,
        enable_caching=False,
    )

    print(f"\nSubmitting Model 1 training pipeline:")
    print(f"  Project    : {PROJECT_ID}")
    print(f"  Region     : {REGION}")
    print(f"  Image      : {_image(args.image_tag)}")
    print(f"  Run ID     : {run_id}")
    print(f"  Train cfg  : {args.train_config}")
    print(f"  SA         : {TRAINER_SA}")

    job.submit(service_account=TRAINER_SA)
    print(f"\nPipeline submitted!")
    print(f"Track at: https://console.cloud.google.com/vertex-ai/pipelines?project={PROJECT_ID}")


if __name__ == "__main__":
    main()
