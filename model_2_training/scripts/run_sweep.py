"""
run_sweep.py

W&B sweep agent entry point for AutoMend Track B hyperparameter search.

This script is called by the W&B agent for each sweep trial. It:
  1. Reads the current trial's hyperparameters from wandb.config
  2. Merges them into the base training + model configs
  3. Runs training via run_training()
  4. Scores the resulting checkpoint on the locked gold benchmark
  5. Logs benchmark metrics back to W&B under the "benchmark/" prefix

Setup:
    # 1. Initialize the sweep (run once)
    wandb sweep configs/sweep/wandb_sweep.yaml --project automend-track-b

    # 2. Launch one or more agents (run on each machine / GPU)
    wandb agent <entity>/automend-track-b/<sweep_id>

    # Or run a single trial manually (useful for testing the sweep script):
    python scripts/run_sweep.py --dry-run
"""

from __future__ import annotations
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant.*")

import yaml
from loguru import logger
from dotenv import load_dotenv

_M2_ROOT = Path(__file__).resolve().parent.parent
_AUTOMEND_ROOT = _M2_ROOT.parent
sys.path.insert(0, str(_AUTOMEND_ROOT))

load_dotenv(_AUTOMEND_ROOT / ".env")

from model_2_training.src.train.train_loop import run_training
from model_2_training.src.eval.evaluator import run_evaluation

_BASE_DATA_CONFIG  = _M2_ROOT / "configs/data/track_b_chatml_sweep.yaml"
_BASE_MODEL_CONFIG = _M2_ROOT / "configs/model/qwen_baseline.yaml"
_BASE_TRAIN_CONFIG = _M2_ROOT / "configs/train/qlora_sft.yaml"
_BASE_EVAL_CONFIG  = _M2_ROOT / "configs/eval/json_eval.yaml"
_BENCHMARK_PATH    = _M2_ROOT / "data/benchmarks/gold_benchmark.jsonl"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_sweep_params(train_cfg: dict, model_cfg: dict, sweep_params: dict) -> None:
    """Merge W&B sweep trial parameters into train and model config dicts in-place."""
    train_keys = {
        "learning_rate", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "lr_scheduler_type", "warmup_ratio",
        "weight_decay", "lora_r", "lora_alpha", "lora_dropout",
    }
    for key, value in sweep_params.items():
        if key in train_keys:
            train_cfg[key] = value
            logger.info(f"  sweep override → {key} = {value}")

    # Auto-set lora_alpha = 2 × lora_r if alpha wasn't swept
    if "lora_r" in sweep_params and "lora_alpha" not in sweep_params:
        train_cfg["lora_alpha"] = sweep_params["lora_r"] * 2
        logger.info(f"  sweep auto-set → lora_alpha = {train_cfg['lora_alpha']} (2 × lora_r)")


def run_sweep_trial(dry_run: bool = False) -> None:
    """Run one sweep trial. Called by the W&B agent."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb is not installed. Install with: pip install wandb")
        sys.exit(1)

    if not _BENCHMARK_PATH.exists():
        logger.error(
            f"Gold benchmark not found: {_BENCHMARK_PATH}\n"
            "Build it first with:  python scripts/build_benchmark.py"
        )
        sys.exit(1)

    # Load configs before wandb.init() so we can compute the run name
    # in the same format used by normal training runs:
    # {model_short}_{quant}_lora-r{lora_r}_{n_train}samples_{datetime}
    data_cfg  = _load_yaml(_BASE_DATA_CONFIG)
    model_cfg = _load_yaml(_BASE_MODEL_CONFIG)
    train_cfg = _load_yaml(_BASE_TRAIN_CONFIG)
    eval_cfg  = _load_yaml(_BASE_EVAL_CONFIG)

    # W&B agent injects sweep params into the process env before calling this
    # script, but wandb.config is only available after wandb.init(). For the
    # run name we only need lora_r, which we read from the CLI args W&B passed.
    import sys as _sys
    _lora_r = train_cfg.get("lora_r", 16)
    for arg in _sys.argv:
        if arg.startswith("--lora_r="):
            try:
                _lora_r = int(arg.split("=", 1)[1])
            except ValueError:
                pass

    from datetime import datetime as _dt
    _model_short = model_cfg.get("model_name", "model").split("/")[-1]
    _quant       = model_cfg.get("quantization") or "mlx"
    _n_train     = data_cfg.get("max_train_samples") or "full"
    _run_name    = f"{_model_short}_{_quant}_lora-r{_lora_r}_{_n_train}samples_{_dt.now().strftime('%Y%m%d-%H%M')}"

    with wandb.init(name=_run_name) as run:
        sweep_params = dict(wandb.config)
        logger.info(f"Sweep trial {run.name} — params: {sweep_params}")

        # Log model name so every sweep run shows which base model was used
        wandb.config.update({"model_name": model_cfg.get("model_name", "unknown")})

        _apply_sweep_params(train_cfg, model_cfg, sweep_params)

        # Point checkpoint output to a sweep-specific subdirectory
        sweep_ckpt_dir = _M2_ROOT / "outputs/checkpoints/sweeps" / run.name
        train_cfg["output_dir"] = str(sweep_ckpt_dir)

        if dry_run:
            logger.info("DRY RUN — skipping actual training.")
            # Log dummy metrics so the sweep infrastructure can be validated
            wandb.log({
                "benchmark/json_parse_rate": 0.0,
                "benchmark/tax_valid_rate":  0.0,
                "benchmark/non_empty_rate":  0.0,
            })
            return

        # ── Train ──────────────────────────────────────────────────────────
        train_start = time.perf_counter()
        best_model_dir = run_training(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            m2_root=_M2_ROOT,
        )
        train_duration = time.perf_counter() - train_start
        logger.success(f"Training done in {train_duration:.1f}s ({train_duration / 60:.1f}min) — checkpoint: {best_model_dir}")

        # ── Benchmark eval ─────────────────────────────────────────────────
        eval_start = time.perf_counter()
        output_dir = _M2_ROOT / "outputs/reports/sweeps" / run.name
        metrics = run_evaluation(
            split_path=_BENCHMARK_PATH,
            checkpoint_path=best_model_dir,
            output_dir=output_dir,
            eval_cfg=eval_cfg,
            split_name="benchmark",
        )
        eval_duration = time.perf_counter() - eval_start
        total_duration = train_duration + eval_duration
        logger.info(f"Evaluation done in {eval_duration:.1f}s ({eval_duration / 60:.1f}min)")

        # Log to W&B under benchmark/ prefix so the sweep objective can find them.
        # Also log short-named aliases for metrics the sweep objective references
        # (objective is "benchmark/tax_valid_rate", but the full key is
        #  "benchmark/phase1_structural/tax_valid_rate").
        benchmark_metrics = {
            f"benchmark/{k}": v
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        # Short aliases for sweep objective + early stopping
        _alias_map = {
            "phase1_structural/tax_valid_rate":        "tax_valid_rate",
            "phase1_structural/json_parse_rate":       "json_parse_rate",
            "phase1_structural/non_empty_rate":        "non_empty_rate",
            "phase2a_schema/schema_valid_rate":        "schema_valid_rate",
            "phase2c_params/full_param_validity_rate": "full_param_validity_rate",
        }
        for full_key, short_key in _alias_map.items():
            if full_key in metrics:
                benchmark_metrics[f"benchmark/{short_key}"] = metrics[full_key]

        # Trial-level timing metrics
        benchmark_metrics["timing/train_duration_s"] = round(train_duration, 1)
        benchmark_metrics["timing/train_duration_min"] = round(train_duration / 60, 2)
        benchmark_metrics["timing/eval_duration_s"] = round(eval_duration, 1)
        benchmark_metrics["timing/total_trial_duration_s"] = round(total_duration, 1)
        benchmark_metrics["timing/total_trial_duration_min"] = round(total_duration / 60, 2)

        # GPU peak memory summary (if CUDA)
        try:
            import torch
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                benchmark_metrics["gpu/trial_peak_memory_gb"] = round(peak_gb, 3)
                benchmark_metrics["gpu/vram_total_gb"] = round(total_vram, 1)
                benchmark_metrics["gpu/vram_utilization_pct"] = round((peak_gb / total_vram) * 100, 1)
        except Exception:
            pass

        wandb.log(benchmark_metrics)
        logger.success(f"Sweep trial complete — {total_duration / 60:.1f}min total")
        logger.info(f"  benchmark/tax_valid_rate  : {metrics.get('phase1_structural/tax_valid_rate', 'N/A')}")
        logger.info(f"  benchmark/json_parse_rate : {metrics.get('phase1_structural/json_parse_rate', 'N/A')}")
        logger.info(f"  timing: train={train_duration / 60:.1f}min, eval={eval_duration / 60:.1f}min, total={total_duration / 60:.1f}min")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W&B sweep agent trial runner.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip training and log dummy metrics — useful for validating sweep config.",
    )
    # W&B agent injects sweep params as CLI flags — ignore them here,
    # they are read from wandb.config inside run_sweep_trial()
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_sweep_trial(dry_run=args.dry_run)
