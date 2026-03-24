"""
submit_training_pipeline.py

Submits the AutoMend production training pipeline to Vertex AI Pipelines.

Use this for:
  - Workflow 2: Full training with best sweep config (split→train→eval→test→benchmark→robustness)
  - Workflow 3: Retrain on new data (split→train→eval→test)

Usage (run from repo root):
    # Full pipeline (Workflow 2) — after sweep is complete
    python gcp/pipelines/submit_training_pipeline.py

    # Retrain only (Workflow 3) — skip benchmark + robustness
    python gcp/pipelines/submit_training_pipeline.py --retrain-only

    # Use best sweep config
    python gcp/pipelines/submit_training_pipeline.py \\
        --train-config configs/train/best_sweep_config.yaml

    # Dry-run — compile YAML only, do not submit
    python gcp/pipelines/submit_training_pipeline.py --dry-run
"""

from __future__ import annotations
import argparse
import sys
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

# Paths inside the container (configs are baked into the image)
_CONFIGS = "/workspace/model_2_training/configs"
_CHECKPOINT = f"{GCS_FUSE_MOUNT}/outputs/checkpoints/best_model"
_BENCHMARK  = f"{GCS_FUSE_MOUNT}/data/benchmarks/gold_benchmark.jsonl"


def _image(tag: str) -> str:
    base = IMAGE_URI.rsplit(":", 1)[0]
    return f"{base}:{tag}"


def build_pipeline(image_tag: str, retrain_only: bool, train_config: str):
    try:
        from kfp import dsl
        from kfp.dsl import PipelineTask
    except ImportError:
        print("kfp not installed. Run: pip install kfp google-cloud-aiplatform")
        sys.exit(1)

    image = _image(image_tag)

    def _make_op(
        name: str,
        command: list[str],
        use_gpu: bool = False,
    ) -> "PipelineTask":
        """Create a container task with secret injection and resource limits."""
        from kfp import dsl

        @dsl.container_component
        def _op() -> dsl.ContainerSpec:
            return dsl.ContainerSpec(
                image=image,
                command=command,
                env={
                    "WANDB_PROJECT":      WANDB_PROJECT,
                    "WANDB_ENTITY":       WANDB_ENTITY,
                    "WANDB_START_METHOD": "thread",
                    "PYTHONUNBUFFERED":   "1",
                    "PYTHONPATH":         "/workspace",
                },
            )

        task = _op()
        task.set_display_name(name)

        if use_gpu:
            task.set_accelerator_type(TRAIN_ACCELERATOR)
            task.set_accelerator_limit(str(TRAIN_ACCEL_COUNT))
            task.set_cpu_limit("8").set_memory_limit("32G")
        else:
            task.set_cpu_limit("4").set_memory_limit("16G")

        return task

    pipeline_desc = (
        f"{PIPELINE_DISPLAY_NAME} — retrain" if retrain_only
        else f"{PIPELINE_DISPLAY_NAME} — full"
    )

    @dsl.pipeline(
        name=PIPELINE_NAME,
        description=pipeline_desc,
        pipeline_root=GCS_PIPELINE_ROOT,
    )
    def training_pipeline():

        # ---- Step 1: Split ------------------------------------------------
        split_op = _make_op(
            "1 · Split data",
            command=[
                "python", "-m", "model_2_training.scripts.run_split",
                "--config", f"{_CONFIGS}/data/track_b_chatml.yaml",
            ],
        )

        # ---- Step 2: Train ------------------------------------------------
        train_op = _make_op(
            "2 · QLoRA training (GPU)",
            command=[
                "python", "-m", "model_2_training.scripts.run_train",
                "--data-config",  f"{_CONFIGS}/data/track_b_chatml.yaml",
                "--model-config", f"{_CONFIGS}/model/qwen_baseline.yaml",
                "--train-config", f"{_CONFIGS}/train/{train_config}",
            ],
            use_gpu=True,
        )
        train_op.after(split_op)

        # ---- Step 3: Eval (val) -------------------------------------------
        eval_op = _make_op(
            "3 · Evaluate — val set",
            command=[
                "python", "-m", "model_2_training.scripts.run_eval",
                "--config",     f"{_CONFIGS}/eval/json_eval.yaml",
                "--checkpoint", _CHECKPOINT,
                "--split",      f"{GCS_FUSE_MOUNT}/data/splits/val.jsonl",
                "--output-dir", f"{GCS_FUSE_MOUNT}/outputs/reports/val",
            ],
        )
        eval_op.after(train_op)

        # ---- Step 4: Test -------------------------------------------------
        test_op = _make_op(
            "4 · Evaluate — test set",
            command=[
                "python", "-m", "model_2_training.scripts.run_test",
                "--config",     f"{_CONFIGS}/eval/json_eval.yaml",
                "--checkpoint", _CHECKPOINT,
                "--split",      f"{GCS_FUSE_MOUNT}/data/splits/test.jsonl",
                "--output-dir", f"{GCS_FUSE_MOUNT}/outputs/reports/test",
            ],
        )
        test_op.after(eval_op)

        # ---- Step 5 & 6: Benchmark + Robustness (Workflow 2 only) ----------
        if not retrain_only:
            benchmark_op = _make_op(
                "5 · Benchmark — gold set",
                command=[
                    "python", "-m", "model_2_training.scripts.run_benchmark",
                    "--config",     f"{_CONFIGS}/eval/json_eval.yaml",
                    "--checkpoint", _CHECKPOINT,
                    "--benchmark",  _BENCHMARK,
                    "--output-dir", f"{GCS_FUSE_MOUNT}/outputs/reports/benchmark",
                ],
            )
            benchmark_op.after(test_op)

            robustness_op = _make_op(
                "6 · Robustness evaluation",
                command=[
                    "python", "-m", "model_2_training.scripts.run_robustness",
                    "--config",     f"{_CONFIGS}/eval/json_eval.yaml",
                    "--checkpoint", _CHECKPOINT,
                    "--benchmark",  _BENCHMARK,
                    "--output-dir", f"{GCS_FUSE_MOUNT}/outputs/reports/robustness",
                ],
            )
            robustness_op.after(benchmark_op)

    return training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit AutoMend production training pipeline to Vertex AI."
    )
    parser.add_argument(
        "--image-tag", default="latest",
        help="Docker image tag (default: latest)",
    )
    parser.add_argument(
        "--train-config", default="qlora_sft.yaml",
        help="Train config filename inside configs/train/ (default: qlora_sft.yaml). "
             "Use best_sweep_config.yaml after a sweep.",
    )
    parser.add_argument(
        "--retrain-only", action="store_true",
        help="Workflow 3: skip benchmark + robustness steps (split→train→eval→test only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compile pipeline YAML but do NOT submit",
    )
    parser.add_argument(
        "--output-yaml", default="gcp/pipelines/training_pipeline.yaml",
        help="Where to write compiled YAML (default: gcp/pipelines/training_pipeline.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline_fn = build_pipeline(
        image_tag=args.image_tag,
        retrain_only=args.retrain_only,
        train_config=args.train_config,
    )

    try:
        from kfp import compiler
    except ImportError:
        print("kfp not installed. Run: pip install kfp google-cloud-aiplatform")
        sys.exit(1)

    output_yaml = Path(args.output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(pipeline_fn, str(output_yaml))
    print(f"Pipeline compiled → {output_yaml}")

    if args.dry_run:
        print("Dry-run — skipping submission.")
        return

    try:
        from google.cloud import aiplatform
    except ImportError:
        print("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")
        sys.exit(1)

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.PipelineJob(
        display_name=PIPELINE_DISPLAY_NAME,
        template_path=str(output_yaml),
        pipeline_root=GCS_PIPELINE_ROOT,
        enable_caching=False,
    )

    workflow = "Retrain (Workflow 3)" if args.retrain_only else "Full training (Workflow 2)"
    print(f"\nSubmitting: {workflow}")
    print(f"  Project      : {PROJECT_ID}")
    print(f"  Region       : {REGION}")
    print(f"  Image        : {_image(args.image_tag)}")
    print(f"  Train config : configs/train/{args.train_config}")
    print(f"  SA           : {TRAINER_SA}")

    job.submit(service_account=TRAINER_SA)

    print(f"\nPipeline submitted!")
    print(f"Track at: https://console.cloud.google.com/vertex-ai/pipelines?project={PROJECT_ID}")


if __name__ == "__main__":
    main()
