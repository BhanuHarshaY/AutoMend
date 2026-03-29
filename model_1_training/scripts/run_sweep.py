"""
run_sweep.py

Hyperparameter sweep using Ray Tune + Optuna (Bayesian optimization).

Searches over learning rate, batch size, and weight decay. Each trial
trains a model, evaluates on the validation split, and reports Macro F1.

Usage:
    python -m model_1_training.scripts.run_sweep \
        --sweep-config configs/sweep/ray_optuna_sweep.yaml \
        --data-config configs/data/track_a.yaml \
        --model-config configs/model/roberta_base.yaml \
        --train-config configs/train/full_finetune.yaml
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
from model_1_training.src.utils.config import load_yaml, save_yaml


def _train_fn(config: dict, data_cfg: dict, model_cfg: dict, base_train_cfg: dict):
    """
    Single trial training function invoked by Ray Tune.

    `config` contains the sampled hyperparameters from Optuna.
    """
    import ray.train
    from model_1_training.src.train.train_loop import run_training

    train_cfg = {**base_train_cfg}
    train_cfg["learning_rate"] = config["learning_rate"]
    train_cfg["per_device_train_batch_size"] = config["per_device_train_batch_size"]
    train_cfg["weight_decay"] = config["weight_decay"]
    train_cfg["report_to"] = "wandb"

    for k, v in config.items():
        if k not in train_cfg:
            train_cfg[k] = v

    trial_output = M1_ROOT / "outputs" / "sweep_trials" / str(ray.train.get_context().get_trial_id())

    best_model = run_training(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        m1_root=M1_ROOT,
        output_dir=trial_output,
    )

    from model_1_training.src.eval.evaluator import evaluate
    splits_dir = M1_ROOT / data_cfg.get("splits_dir", "data/splits")
    metrics = evaluate(
        checkpoint_dir=best_model,
        split_path=splits_dir / "val.parquet",
        output_dir=trial_output / "eval",
        num_labels=model_cfg.get("num_labels", 7),
    )

    ray.train.report({
        "eval_macro_f1": metrics["macro_f1"],
        "eval_anomaly_recall": metrics["anomaly_recall"],
        "eval_accuracy": metrics["accuracy"],
    })


def run_sweep(sweep_cfg: dict, data_cfg: dict, model_cfg: dict, train_cfg: dict) -> dict:
    """Launch Ray Tune sweep with Optuna search."""
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler
    from functools import partial

    search_space_cfg = sweep_cfg["search_space"]
    search_space = {}

    for param, spec in search_space_cfg.items():
        if spec["type"] == "log_uniform":
            search_space[param] = tune.loguniform(spec["min"], spec["max"])
        elif spec["type"] == "uniform":
            search_space[param] = tune.uniform(spec["min"], spec["max"])
        elif spec["type"] == "choice":
            search_space[param] = tune.choice(spec["values"])

    fixed = sweep_cfg.get("fixed", {})
    for k, v in fixed.items():
        train_cfg[k] = v

    optuna_search = OptunaSearch(metric="eval_macro_f1", mode="max")

    scheduler_cfg = sweep_cfg.get("scheduler", {})
    scheduler = ASHAScheduler(
        max_t=scheduler_cfg.get("max_t", 5),
        grace_period=scheduler_cfg.get("grace_period", 2),
        reduction_factor=scheduler_cfg.get("reduction_factor", 2),
    )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    trainable = partial(_train_fn, data_cfg=data_cfg, model_cfg=model_cfg, base_train_cfg=train_cfg)

    analysis = tune.run(
        trainable,
        config=search_space,
        num_samples=sweep_cfg.get("num_trials", 20),
        search_alg=optuna_search,
        scheduler=scheduler,
        resources_per_trial={
            "cpu": sweep_cfg.get("cpus_per_trial", 4),
            "gpu": sweep_cfg.get("gpus_per_trial", 1),
        },
        local_dir=str(M1_ROOT / "outputs" / "ray_results"),
        name="model1_sweep",
        verbose=1,
    )

    best_config = analysis.best_config
    best_result = analysis.best_result
    logger.success(f"Best trial — Macro F1: {best_result['eval_macro_f1']:.4f}")
    logger.info(f"Best config: {best_config}")

    best_out = {**train_cfg, **best_config}
    save_yaml(best_out, M1_ROOT / "configs" / "train" / "best_sweep_config.yaml")
    logger.info("Best config saved -> configs/train/best_sweep_config.yaml")

    return {"best_config": best_config, "best_result": best_result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model 1 hyperparameter sweep")
    parser.add_argument("--sweep-config", default=str(M1_ROOT / "configs/sweep/ray_optuna_sweep.yaml"))
    parser.add_argument("--data-config", default=str(M1_ROOT / "configs/data/track_a.yaml"))
    parser.add_argument("--model-config", default=str(M1_ROOT / "configs/model/roberta_base.yaml"))
    parser.add_argument("--train-config", default=str(M1_ROOT / "configs/train/full_finetune.yaml"))
    args = parser.parse_args()

    sweep_cfg = load_yaml(args.sweep_config)
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    result = run_sweep(sweep_cfg, data_cfg, model_cfg, train_cfg)
    print(f"\nSweep complete!")
    print(f"Best Macro F1: {result['best_result']['eval_macro_f1']:.4f}")
    print(f"Best config: {result['best_config']}")


if __name__ == "__main__":
    main()
