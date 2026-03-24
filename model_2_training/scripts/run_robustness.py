"""
run_robustness.py

Phase 3 — Robustness & Slice Evaluation entry point.

Runs the full Phase 3 pipeline against a locked benchmark:
  1. Clean baseline evaluation
  2. Five perturbation types: typo, noise, truncation, case_lower, paraphrase
  3. Robustness delta table (clean vs perturbed, per perturbation type)
  4. Slice-based evaluation: archetype / dataset / input_length / complexity
  5. Failure log — every regression where clean=VALID but perturbed=not VALID

By default this runs against the gold benchmark from Phase 2.5, so the same
30 locked, stratified samples are used for clean/perturbed comparison. You
can point --benchmark at any JSONL file.

Usage
-----
    # Full run — all 5 perturbation types
    python scripts/run_robustness.py \\
        --config     configs/eval/json_eval.yaml \\
        --checkpoint outputs/checkpoints/best_model

    # Custom benchmark
    python scripts/run_robustness.py \\
        --config     configs/eval/json_eval.yaml \\
        --checkpoint outputs/checkpoints/best_model \\
        --benchmark  data/splits/test.jsonl

    # Specific perturbations only
    python scripts/run_robustness.py \\
        --config          configs/eval/json_eval.yaml \\
        --checkpoint      outputs/checkpoints/best_model \\
        --perturbations   typo noise

    # Custom output directory
    python scripts/run_robustness.py \\
        --config      configs/eval/json_eval.yaml \\
        --checkpoint  outputs/checkpoints/best_model \\
        --output-dir  outputs/reports/robustness/my_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

_M2_ROOT       = Path(__file__).resolve().parent.parent
_AUTOMEND_ROOT = _M2_ROOT.parent
sys.path.insert(0, str(_AUTOMEND_ROOT))

from model_2_training.src.data.load_jsonl import load_jsonl
from model_2_training.src.robustness.perturbations import PERTURBATION_TYPES
from model_2_training.src.robustness.robustness_runner import run_robustness_eval

_DEFAULT_BENCHMARK = _M2_ROOT / "data/benchmarks/gold_benchmark.jsonl"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Robustness & Slice Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to eval config YAML (e.g. configs/eval/json_eval.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory to evaluate.",
    )
    parser.add_argument(
        "--benchmark",
        default=str(_DEFAULT_BENCHMARK),
        help=(
            f"JSONL file to run robustness tests on.\n"
            f"Default: {_DEFAULT_BENCHMARK}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for reports.\n"
            "Default: outputs/reports/robustness/<checkpoint_name>"
        ),
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=None,
        choices=PERTURBATION_TYPES,
        metavar="PERTURBATION",
        help=(
            f"Perturbation types to run. Default: all.\n"
            f"Choices: {', '.join(PERTURBATION_TYPES)}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for perturbations (default: 42).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Config ---
    config_path = _M2_ROOT / args.config
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        eval_cfg = yaml.safe_load(f)

    # --- Benchmark ---
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        logger.error(
            f"Benchmark not found: {benchmark_path}\n"
            "Build it first with:  python scripts/build_benchmark.py"
        )
        sys.exit(1)

    # --- Checkpoint ---
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # --- Output dir ---
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _M2_ROOT / "outputs/reports/robustness" / checkpoint_path.name
    )

    # --- Load samples ---
    samples = load_jsonl(benchmark_path, malformed_strategy="skip")
    if not samples:
        logger.error(f"No valid samples loaded from {benchmark_path}")
        sys.exit(1)

    perturbation_types = args.perturbations or PERTURBATION_TYPES

    logger.info("=" * 60)
    logger.info("PHASE 3 — ROBUSTNESS & SLICE EVALUATION")
    logger.info(f"  checkpoint     : {checkpoint_path}")
    logger.info(f"  benchmark      : {benchmark_path}  ({len(samples)} samples)")
    logger.info(f"  perturbations  : {perturbation_types}")
    logger.info(f"  output_dir     : {output_dir}")
    logger.info("=" * 60)

    results = run_robustness_eval(
        samples=samples,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_cfg=eval_cfg,
        perturbation_types=perturbation_types,
        seed=args.seed,
    )

    # --- Terminal summary ---
    clean   = results["clean_metrics"]
    summary = results["robustness_summary"]
    deltas  = summary.get("delta", {})

    logger.success("Phase 3 complete.")
    logger.info("-" * 50)
    logger.info("Clean baseline:")
    for key in ["phase1_structural/tax_valid_rate", "phase2a_schema/schema_valid_rate",
                "phase2c_params/full_param_validity_rate"]:
        val = clean.get(key, "N/A")
        val_str = f"{val:.1%}" if isinstance(val, float) else str(val)
        logger.info(f"  {key.split('/')[-1]:<30} {val_str}")

    logger.info("")
    logger.info("Robustness deltas (positive = degradation):")
    for pert_type, delta in deltas.items():
        vd = delta.get("phase1_structural/tax_valid_rate", "N/A")
        sd = delta.get("phase2a_schema/schema_valid_rate", "N/A")
        vd_str = f"{vd:+.1%}" if isinstance(vd, float) else str(vd)
        sd_str = f"{sd:+.1%}" if isinstance(sd, float) else str(sd)
        logger.info(f"  {pert_type:<14}  valid_rate_drop={vd_str}  schema_drop={sd_str}")

    worst = summary.get("worst_perturbation", "N/A")
    logger.info("")
    logger.info(f"Worst perturbation: {worst}")
    logger.info(f"Reports saved in:   {output_dir}")


if __name__ == "__main__":
    main()
