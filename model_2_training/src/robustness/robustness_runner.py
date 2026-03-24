"""
robustness_runner.py

Phase 3 — Robustness evaluation orchestrator.

Coordinates the full Phase 3 pipeline in a single function call:

  1. Load the model once (reused for all perturbation runs)
  2. Run generation + metrics on clean (unperturbed) samples
  3. For each perturbation type:
       a. Perturb all samples
       b. Run generation
       c. Run all metrics
       d. Record regressions (clean=VALID → perturbed=not VALID)
  4. Build robustness delta summary across all perturbation types
  5. Run slice-based evaluation on clean predictions
  6. Save all report artifacts to output_dir

Output artifacts
----------------
  clean_predictions.jsonl      Raw generation on unperturbed samples
  clean_metrics.json           Full metric dict for clean baseline
  {pert}_predictions.jsonl     Raw generation per perturbation type
  {pert}_metrics.json          Metrics per perturbation type
  robustness_summary.json      Delta table + worst perturbation
  slice_report.json            Per-slice metrics across all 4 dimensions
  failure_log.json             Every regression with original vs perturbed I/O
  robustness_report.md         Human-readable Markdown summary

Public API
----------
  run_robustness_eval(samples, checkpoint_path, output_dir, eval_cfg, ...) -> dict
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from model_2_training.src.eval.generator import (
    load_model_for_inference,
    run_generation,
    save_predictions,
)
from model_2_training.src.eval.metrics_aggregator import run_all_metrics
from model_2_training.src.eval.error_taxonomy import _classify as _classify_output
from model_2_training.src.robustness.perturbations import perturb_dataset, PERTURBATION_TYPES
from model_2_training.src.robustness.slice_eval import (
    compute_all_slices,
    build_robustness_summary,
    KEY_METRICS,
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_robustness_eval(
    samples: list[dict],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    eval_cfg: dict,
    perturbation_types: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """
    Run Phase 3 robustness + slice evaluation.

    The model is loaded once and reused across all perturbation runs to avoid
    redundant disk I/O and to keep memory pressure consistent.

    Args:
        samples:            Clean samples (e.g. from gold_benchmark.jsonl).
        checkpoint_path:    Path to checkpoint directory.
        output_dir:         Directory to write all reports.
        eval_cfg:           Eval config dict loaded from json_eval.yaml.
        perturbation_types: Subset of PERTURBATION_TYPES to run.
                            Defaults to all 5.
        seed:               Base random seed for perturbations.

    Returns:
        Dict with keys:
          "clean_metrics"       — metrics on clean data
          "perturbed_metrics"   — {pert_type: metrics_dict}
          "robustness_summary"  — delta table + worst perturbation
          "slice_report"        — {dimension: {slice_value: metrics}}
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir      = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if perturbation_types is None:
        perturbation_types = PERTURBATION_TYPES

    gen_kwargs = dict(
        max_new_tokens=eval_cfg.get("max_new_tokens", 512),
        temperature=eval_cfg.get("temperature", 0.1),
        top_p=eval_cfg.get("top_p", 0.9),
        do_sample=eval_cfg.get("do_sample", False),
    )

    # --- Load model once ---
    logger.info("Loading model for Phase 3 robustness evaluation...")
    model, tokenizer = load_model_for_inference(checkpoint_path)

    # -------------------------------------------------------------------------
    # 1. Clean baseline
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 3 — Clean baseline")
    logger.info("=" * 60)
    clean_preds = run_generation(
        samples=samples, model=model, tokenizer=tokenizer, **gen_kwargs
    )
    save_predictions(clean_preds, output_dir / "clean_predictions.jsonl")
    clean_metrics = run_all_metrics(clean_preds)
    _save_json(clean_metrics, output_dir / "clean_metrics.json")
    _log_key_metrics("clean", clean_metrics)

    # -------------------------------------------------------------------------
    # 2. Perturbed evaluations
    # -------------------------------------------------------------------------
    perturbed_metrics: dict[str, dict] = {}
    all_regressions:   list[dict]      = []

    for pert_type in perturbation_types:
        logger.info("=" * 60)
        logger.info(f"Phase 3 — Perturbation: {pert_type}")
        logger.info("=" * 60)

        pert_samples = perturb_dataset(samples, pert_type, seed=seed)
        pert_preds   = run_generation(
            samples=pert_samples, model=model, tokenizer=tokenizer, **gen_kwargs
        )
        save_predictions(pert_preds, output_dir / f"{pert_type}_predictions.jsonl")

        pert_metrics = run_all_metrics(pert_preds)
        perturbed_metrics[pert_type] = pert_metrics
        _save_json(pert_metrics, output_dir / f"{pert_type}_metrics.json")
        _log_key_metrics(pert_type, pert_metrics)

        # Find regressions for this perturbation type
        regressions = _find_regressions(clean_preds, pert_preds, pert_type)
        all_regressions.extend(regressions)
        logger.info(f"  regressions (VALID → fail): {len(regressions)}")

    # -------------------------------------------------------------------------
    # 3. Robustness summary
    # -------------------------------------------------------------------------
    robustness_summary = build_robustness_summary(clean_metrics, perturbed_metrics)
    _save_json(robustness_summary, output_dir / "robustness_summary.json")

    # -------------------------------------------------------------------------
    # 4. Slice-based evaluation (clean predictions only)
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 3 — Slice-based evaluation")
    logger.info("=" * 60)
    slice_report = compute_all_slices(clean_preds)
    _save_json(slice_report, output_dir / "slice_report.json")

    # -------------------------------------------------------------------------
    # 5. Failure log
    # -------------------------------------------------------------------------
    _save_json(all_regressions, output_dir / "failure_log.json")
    logger.info(f"Total regressions logged: {len(all_regressions)}")

    # -------------------------------------------------------------------------
    # 6. Markdown report
    # -------------------------------------------------------------------------
    _save_markdown(
        clean_metrics=clean_metrics,
        robustness_summary=robustness_summary,
        slice_report=slice_report,
        n_regressions=len(all_regressions),
        output_path=output_dir / "robustness_report.md",
    )

    logger.success(f"Phase 3 complete. All reports in: {output_dir}")

    return {
        "clean_metrics":      clean_metrics,
        "perturbed_metrics":  perturbed_metrics,
        "robustness_summary": robustness_summary,
        "slice_report":       slice_report,
    }


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

def _find_regressions(
    clean_preds: list[dict],
    pert_preds: list[dict],
    perturbation_type: str,
) -> list[dict]:
    """
    Identify samples that were VALID on clean input but failed on perturbed input.

    Re-derives failure categories from generated text rather than relying on
    any pre-computed field, so it is always consistent with the taxonomy module.

    Args:
        clean_preds:       Predictions on unperturbed samples.
        pert_preds:        Predictions on perturbed samples (same order).
        perturbation_type: Label to include in each regression record.

    Returns:
        List of regression dicts, one per regressed sample.
    """
    regressions = []
    for clean, pert in zip(clean_preds, pert_preds):
        clean_cat = _classify_output(
            clean.get("generated", ""), clean.get("reference", "")
        )
        pert_cat = _classify_output(
            pert.get("generated", ""), pert.get("reference", "")
        )

        if clean_cat == "VALID" and pert_cat != "VALID":
            orig_user = _extract_user_message(clean.get("sample") or {})
            pert_user = _extract_user_message(pert.get("sample") or {})
            regressions.append({
                "index":             clean.get("index"),
                "perturbation_type": perturbation_type,
                "clean_category":    clean_cat,
                "pert_category":     pert_cat,
                "original_input":    orig_user[:400],
                "perturbed_input":   pert_user[:400],
                "clean_output":      (clean.get("generated") or "")[:600],
                "pert_output":       (pert.get("generated") or "")[:600],
                "reference":         (clean.get("reference") or "")[:300],
            })
    return regressions


def _extract_user_message(sample: dict) -> str:
    """Return the first user message content from a sample dict."""
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _log_key_metrics(label: str, metrics: dict) -> None:
    """Log key metric values at INFO level for quick console feedback."""
    valid = metrics.get("phase1_structural/tax_valid_rate", "N/A")
    schema = metrics.get("phase2a_schema/schema_valid_rate", "N/A")
    param = metrics.get("phase2c_params/full_param_validity_rate", "N/A")
    _fmt = lambda v: f"{v:.1%}" if isinstance(v, float) else str(v)
    logger.info(
        f"  [{label}]  valid={_fmt(valid)}  schema={_fmt(schema)}  param={_fmt(param)}"
    )


def _save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved -> {path}")


def _save_markdown(
    clean_metrics: dict,
    robustness_summary: dict,
    slice_report: dict,
    n_regressions: int,
    output_path: Path,
) -> None:
    """Write a human-readable Phase 3 summary to a Markdown file."""
    lines = [
        "# Phase 3 — Robustness & Slice Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Clean Baseline",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    _METRIC_LABELS = {
        "phase1_structural/json_parse_rate":       "JSON parse rate",
        "phase1_structural/tax_valid_rate":        "Valid output rate",
        "phase2a_schema/schema_valid_rate":        "Schema valid rate",
        "phase2b_fields/steps_count_match_rate":   "Step count match rate",
        "phase2c_params/full_param_validity_rate": "Param validity rate",
    }
    for key, label in _METRIC_LABELS.items():
        val = clean_metrics.get(key, "N/A")
        val_str = f"{val:.1%}" if isinstance(val, float) else str(val)
        lines.append(f"| {label} | {val_str} |")

    lines += [
        "",
        "---",
        "",
        "## Robustness Deltas vs Clean Baseline",
        "",
        "> Positive delta = performance drop under perturbation.",
        "> Negative = unexpected improvement (noise on small samples).",
        "",
        f"| Perturbation | {' | '.join(_METRIC_LABELS.values())} |",
        f"|{'|'.join(['-' * 14] * (len(_METRIC_LABELS) + 1))}|",
    ]

    deltas = robustness_summary.get("delta", {})
    for pert_type, delta in deltas.items():
        cells = []
        for key in _METRIC_LABELS:
            d = delta.get(key)
            cells.append(f"{d:+.1%}" if d is not None else "N/A")
        lines.append(f"| {pert_type} | {' | '.join(cells)} |")

    worst = robustness_summary.get("worst_perturbation", "N/A")
    worst_delta = robustness_summary.get("worst_perturbation_total_delta", "N/A")
    lines += [
        "",
        f"**Worst perturbation:** `{worst}` "
        f"(sum of key metric deltas: {worst_delta})",
        f"**Total regressions (VALID → fail):** {n_regressions}",
        "",
        "---",
        "",
        "## Slice-Based Evaluation (Clean Data)",
        "",
    ]

    for dim, slices in slice_report.items():
        lines += [
            f"### By {dim}",
            "",
            "| Slice | N | Valid Rate | Schema Valid | Param Valid |",
            "|-------|---|-----------|--------------|-------------|",
        ]
        _f = lambda v: f"{v:.1%}" if isinstance(v, float) else str(v)
        for slice_val, m in sorted(slices.items()):
            n       = m.get("slice_n", "?")
            valid   = _f(m.get("phase1_structural/tax_valid_rate", "N/A"))
            schema  = _f(m.get("phase2a_schema/schema_valid_rate", "N/A"))
            param   = _f(m.get("phase2c_params/full_param_validity_rate", "N/A"))
            lines.append(f"| {slice_val} | {n} | {valid} | {schema} | {param} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Artifacts",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `clean_metrics.json` | Full 4-phase metrics on unperturbed data |",
        "| `{pert}_metrics.json` | Metrics per perturbation type |",
        "| `robustness_summary.json` | Delta table + worst perturbation |",
        "| `slice_report.json` | Per-slice metrics across all 4 dimensions |",
        "| `failure_log.json` | Regression records with original vs perturbed I/O |",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Robustness report saved -> {output_path}")
