"""
main.py — AutoMend Automation Webhook (Cloud Run)

Two endpoints:
  POST /trigger      — called by Cloud Scheduler after sweep trials finish
                       checks sweep status, fetches best config, submits pipeline
  GET  /health       — health check

Cloud Scheduler is created automatically by submit_sweep_agent.py.
No manual W&B webhook setup needed.
"""

from __future__ import annotations
import os
import json
import logging
import tempfile

from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- config ----
PROJECT_ID          = "automend"
REGION              = "us-central1"
GCS_BUCKET          = "automend-model2"
GCS_PIPELINE_ROOT   = f"gs://{GCS_BUCKET}/pipeline_runs"
TRAINER_SA          = f"automend-trainer@{PROJECT_ID}.iam.gserviceaccount.com"
PIPELINE_DISPLAY_NAME = "AutoMend QLoRA Training"
IMAGE_URI           = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/automend-images/automend-train:latest"
WANDB_PROJECT       = "automend-model2"
WANDB_ENTITY        = "mlops-team-northeastern-university"
OBJECTIVE_METRIC    = "benchmark/tax_valid_rate"
_CONFIGS            = "/workspace/model_2_training/configs"
_GCS_FUSE           = "/gcs/automend-model2"
_CHECKPOINT         = f"{_GCS_FUSE}/outputs/checkpoints/best_model"
_BENCHMARK          = f"{_GCS_FUSE}/data/benchmarks/gold_benchmark.jsonl"

_TRAIN_KEYS = {
    "learning_rate", "num_train_epochs", "per_device_train_batch_size",
    "gradient_accumulation_steps", "lr_scheduler_type", "warmup_ratio",
    "weight_decay", "lora_r", "lora_alpha", "lora_dropout",
}


def _get_secret(secret_name: str) -> str:
    from google.cloud import secretmanager
    client   = secretmanager.SecretManagerServiceClient()
    name     = f"projects/{PROJECT_ID}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("utf-8")


def _is_sweep_finished(sweep, min_runs: int = 1) -> bool:
    """Return True if sweep is done or has enough completed runs."""
    state = getattr(sweep, "state", "").lower()
    if state in ("finished", "cancelled"):
        return True
    # Also accept if all runs are finished even if sweep state is still "running"
    runs = list(sweep.runs)
    finished = [r for r in runs if r.state in ("finished", "failed", "crashed")]
    return len(runs) >= min_runs and len(finished) == len(runs)


def fetch_best_config(sweep_path: str) -> tuple[dict, object]:
    """Fetch best hyperparameters from completed W&B sweep."""
    import wandb

    os.environ["WANDB_API_KEY"] = _get_secret("WANDB_API_KEY")
    api   = wandb.Api()
    sweep = api.sweep(sweep_path)

    if not _is_sweep_finished(sweep):
        raise RuntimeError(f"Sweep {sweep_path} is not finished yet (state={sweep.state})")

    runs = sorted(
        sweep.runs,
        key=lambda r: r.summary.get(OBJECTIVE_METRIC, -1.0),
        reverse=True,
    )
    if not runs:
        raise RuntimeError(f"No runs found in sweep: {sweep_path}")

    best  = runs[0]
    score = best.summary.get(OBJECTIVE_METRIC, 0.0)
    logger.info(f"Best run: {best.name} — {OBJECTIVE_METRIC}={score:.4f}")

    # Base config defaults
    base_cfg = {
        "output_dir": "outputs/checkpoints",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": 10,
        "report_to": "wandb",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "fp16": False,
        "bf16": True,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }

    for key in _TRAIN_KEYS:
        if key in best.config:
            logger.info(f"  {key}: {base_cfg.get(key)} → {best.config[key]}")
            base_cfg[key] = best.config[key]

    return base_cfg, best


def upload_best_config(cfg: dict) -> str:
    """Upload best_sweep_config.yaml to GCS and return gs:// path."""
    import yaml
    from google.cloud import storage

    yaml_str = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
    client   = storage.Client()
    bucket   = client.bucket(GCS_BUCKET)
    blob     = bucket.blob("configs/best_sweep_config.yaml")
    blob.upload_from_string(yaml_str, content_type="text/yaml")

    path = f"gs://{GCS_BUCKET}/configs/best_sweep_config.yaml"
    logger.info(f"Best config uploaded → {path}")
    return path


def submit_training_pipeline(gcs_config_path: str) -> str:
    """Compile and submit the Vertex AI full training pipeline."""
    from google.cloud import aiplatform
    from kfp import dsl, compiler

    @dsl.pipeline(
        name="automend-training-pipeline",
        pipeline_root=GCS_PIPELINE_ROOT,
    )
    def training_pipeline():
        def _op(name, command, use_gpu=False):
            @dsl.container_component
            def _c() -> dsl.ContainerSpec:
                return dsl.ContainerSpec(
                    image=IMAGE_URI,
                    command=command,
                    env={
                        "PYTHONPATH":    "/workspace",
                        "WANDB_PROJECT": WANDB_PROJECT,
                        "WANDB_ENTITY":  WANDB_ENTITY,
                    },
                )
            t = _c()
            t.set_display_name(name)
            if use_gpu:
                t.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit("1")
                t.set_cpu_limit("8").set_memory_limit("32G")
            else:
                t.set_cpu_limit("4").set_memory_limit("16G")
            return t

        split_op = _op("1 · Split", [
            "python", "-m", "model_2_training.scripts.run_split",
            "--config", f"{_CONFIGS}/data/track_b_chatml.yaml",
        ])

        train_op = _op("2 · Train (GPU)", [
            "python", "-m", "model_2_training.scripts.run_train",
            "--data-config",  f"{_CONFIGS}/data/track_b_chatml.yaml",
            "--model-config", f"{_CONFIGS}/model/qwen_baseline.yaml",
            "--train-config", f"{_GCS_FUSE}/configs/best_sweep_config.yaml",
        ], use_gpu=True)
        train_op.after(split_op)

        eval_op = _op("3 · Eval", [
            "python", "-m", "model_2_training.scripts.run_eval",
            "--config", f"{_CONFIGS}/eval/json_eval.yaml",
            "--checkpoint", _CHECKPOINT,
            "--split", f"{_GCS_FUSE}/data/splits/val.jsonl",
            "--output-dir", f"{_GCS_FUSE}/outputs/reports/val",
        ])
        eval_op.after(train_op)

        test_op = _op("4 · Test", [
            "python", "-m", "model_2_training.scripts.run_test",
            "--config", f"{_CONFIGS}/eval/json_eval.yaml",
            "--checkpoint", _CHECKPOINT,
            "--split", f"{_GCS_FUSE}/data/splits/test.jsonl",
            "--output-dir", f"{_GCS_FUSE}/outputs/reports/test",
        ])
        test_op.after(eval_op)

        benchmark_op = _op("5 · Benchmark", [
            "python", "-m", "model_2_training.scripts.run_benchmark",
            "--config", f"{_CONFIGS}/eval/json_eval.yaml",
            "--checkpoint", _CHECKPOINT,
            "--benchmark", _BENCHMARK,
            "--output-dir", f"{_GCS_FUSE}/outputs/reports/benchmark",
        ])
        benchmark_op.after(test_op)

        robustness_op = _op("6 · Robustness", [
            "python", "-m", "model_2_training.scripts.run_robustness",
            "--config", f"{_CONFIGS}/eval/json_eval.yaml",
            "--checkpoint", _CHECKPOINT,
            "--benchmark", _BENCHMARK,
            "--output-dir", f"{_GCS_FUSE}/outputs/reports/robustness",
        ])
        robustness_op.after(benchmark_op)

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        tmp = f.name
    compiler.Compiler().compile(training_pipeline, tmp)

    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.PipelineJob(
        display_name=PIPELINE_DISPLAY_NAME,
        template_path=tmp,
        pipeline_root=GCS_PIPELINE_ROOT,
        enable_caching=False,
    )
    job.submit(service_account=TRAINER_SA)
    logger.info(f"Pipeline submitted: {job.resource_name}")
    return job.resource_name


@app.route("/trigger", methods=["POST"])
def trigger():
    """
    Called by Cloud Scheduler after sweep trials are expected to be done.
    Checks sweep status, fetches best config, submits training pipeline.
    """
    payload  = request.get_json(silent=True) or {}
    sweep_id = payload.get("sweep_id")

    if not sweep_id:
        return jsonify({"error": "missing sweep_id"}), 400

    if "/" not in sweep_id:
        sweep_id = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}"

    logger.info(f"Trigger received for sweep: {sweep_id}")

    try:
        best_cfg, best_run = fetch_best_config(sweep_id)
    except RuntimeError as e:
        if "not finished" in str(e):
            # Sweep still running — return 503 so Cloud Scheduler retries
            logger.warning(str(e))
            return jsonify({"status": "pending", "message": str(e)}), 503
        raise

    gcs_path     = upload_best_config(best_cfg)
    pipeline_job = submit_training_pipeline(gcs_path)

    return jsonify({
        "status":       "ok",
        "sweep_id":     sweep_id,
        "best_run":     best_run.name,
        "pipeline_job": pipeline_job,
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
