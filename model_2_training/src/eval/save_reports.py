"""
save_reports.py

Saves evaluation results to disk in multiple formats.

Outputs:
  - metrics.json        — all computed metrics (all phases)
  - metrics_summary.md  — human-readable Markdown report with per-phase tables
  - error_samples.json  — sample predictions that failed JSON parsing / schema
  - sample_outputs.json — a random selection of predictions for manual review
  - param_errors.json   — sample steps that failed parameter validation (Phase 2C)
"""

from __future__ import annotations
import json
import random
from datetime import datetime
from pathlib import Path
from loguru import logger


def save_metrics_json(metrics: dict, output_dir: Path, filename: str = "metrics.json") -> Path:
    """Save metrics dict to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved -> {path}")
    return path


def save_markdown_report(
    metrics: dict,
    output_dir: Path,
    split_name: str = "validation",
    filename: str = "metrics_summary.md",
    per_field_report: dict | None = None,
) -> Path:
    """Save a human-readable Markdown metrics report with per-phase tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Model 2 Evaluation Report — {split_name.capitalize()} Set",
        f"",
        f"**Generated:** {now}  ",
        f"**Split:** {split_name}  ",
        f"**Total samples:** {metrics.get('total_samples', 'N/A')}  ",
        f"",
    ]

    def _fmt(val: object, is_rate: bool = False) -> str:
        if isinstance(val, float) and is_rate:
            return f"{val * 100:.1f}%"
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    def _section(title: str, rows: list[tuple[str, str, bool]]) -> None:
        """Append a metric table section. rows = [(key, label, is_rate)]"""
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, label, is_rate in rows:
            val = metrics.get(key, "N/A")
            lines.append(f"| {label} | {_fmt(val, is_rate)} |")
        lines.append("")

    # --- Phase 1 — Structural ---
    _section("Phase 1 — JSON Structural Quality", [
        ("phase1_structural/non_empty_rate",        "Non-empty output rate",          True),
        ("phase1_structural/starts_with_brace_rate","Starts with `{` or `[`",         True),
        ("phase1_structural/ends_with_brace_rate",  "Ends with `}` or `]`",           True),
        ("phase1_structural/json_parse_rate",       "JSON parse success rate",         True),
        ("phase1_structural/malformed_json_rate",   "Malformed JSON rate",             True),
        ("phase1_structural/truncation_rate",       "Truncation rate",                 True),
        ("phase1_structural/quote_balance_rate",    "Quote balance rate",              True),
        ("phase1_structural/brace_balance_rate",    "Brace balance rate",              True),
        ("phase1_structural/avg_output_length",     "Average output length (chars)",   False),
    ])

    # --- Phase 2A — Schema Validity ---
    _section("Phase 2A — Schema Validity", [
        ("phase2a_schema/schema_valid_rate",   "Schema valid rate",         True),
        ("phase2a_schema/correct_shape_rate",  "Correct shape rate",        True),
        ("phase2a_schema/extra_fields_rate",   "Extra fields rate",         True),
        ("phase2a_schema/wrong_type_rate",     "Wrong type rate",           True),
    ])

    # Schema error distribution sub-table
    err_dist = metrics.get("phase2a_schema/schema_error_distribution")
    if err_dist and isinstance(err_dist, dict):
        lines.append("### Schema Error Breakdown")
        lines.append("")
        lines.append("| Error Type | Count |")
        lines.append("|------------|-------|")
        for err_type, count in err_dist.items():
            lines.append(f"| `{err_type}` | {count} |")
        lines.append("")

    # --- Phase 2B — Field Correctness ---
    _section("Phase 2B — Field-Level Correctness", [
        ("phase2b_fields/tool_name_nonempty_rate",  "Tool name non-empty rate",    True),
        ("phase2b_fields/params_nonempty_rate",     "Params non-empty rate",       True),
        ("phase2b_fields/steps_count_match_rate",   "Step count match rate",       True),
        ("phase2b_fields/message_nonempty_rate",    "Message non-empty rate",      True),
        ("phase2b_fields/steps_is_list_rate",       "Steps is list rate",          True),
    ])

    # Per-field breakdown sub-table
    if per_field_report:
        lines.append("### Per-Field Breakdown")
        lines.append("")
        lines.append("| Field | Present | Non-Empty | Correct Type |")
        lines.append("|-------|---------|-----------|--------------|")
        for field, breakdown in per_field_report.items():
            present   = _fmt(breakdown.get("present",      "N/A"), True)
            non_empty = _fmt(breakdown.get("non_empty",    "N/A"), True)
            corr_type = _fmt(breakdown.get("correct_type", breakdown.get("is_list", "N/A")), True)
            lines.append(f"| `{field}` | {present} | {non_empty} | {corr_type} |")
        lines.append("")

    # --- Phase 2C — Parameter Validation ---
    _section("Phase 2C — Parameter Validation", [
        ("phase2c_params/param_schema_coverage_rate",    "Schema coverage rate",          True),
        ("phase2c_params/param_completeness_rate",       "Required params present rate",  True),
        ("phase2c_params/param_no_extras_rate",          "No extra params rate",          True),
        ("phase2c_params/param_type_correctness_rate",   "Param type correctness rate",   True),
        ("phase2c_params/full_param_validity_rate",      "Full param validity rate",      True),
    ])

    # --- Interpretation ---
    lines.append("## Interpretation")
    lines.append("")
    json_rate   = metrics.get("phase1_structural/json_parse_rate", 0) or 0
    schema_rate = metrics.get("phase2a_schema/schema_valid_rate",  0) or 0
    param_rate  = metrics.get("phase2c_params/full_param_validity_rate", None)

    if json_rate >= 0.95:
        lines.append(f"- **Structural (Phase 1):** {json_rate*100:.1f}% JSON parse rate — excellent structural learning.")
    elif json_rate >= 0.8:
        lines.append(f"- **Structural (Phase 1):** {json_rate*100:.1f}% JSON parse rate — good, minor truncation/formatting issues.")
    else:
        lines.append(f"- **Structural (Phase 1):** {json_rate*100:.1f}% JSON parse rate — review training data or increase max_new_tokens.")

    if schema_rate >= 0.95:
        lines.append(f"- **Schema (Phase 2A):** {schema_rate*100:.1f}% schema valid rate — model reliably follows the output contract.")
    elif schema_rate >= 0.8:
        lines.append(f"- **Schema (Phase 2A):** {schema_rate*100:.1f}% schema valid rate — some structural issues, see error breakdown above.")
    else:
        lines.append(f"- **Schema (Phase 2A):** {schema_rate*100:.1f}% schema valid rate — significant schema violations, review format examples in training data.")

    if param_rate is not None:
        if param_rate >= 0.95:
            lines.append(f"- **Params (Phase 2C):** {param_rate*100:.1f}% full validity rate — model respects tool parameter contracts.")
        else:
            lines.append(f"- **Params (Phase 2C):** {param_rate*100:.1f}% full validity rate — parameter issues detected, see param_errors.json.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Markdown report saved -> {path}")
    return path


def save_error_samples(errors: list[dict], output_dir: Path, filename: str = "error_samples.json") -> Path:
    """Save sample predictions that failed JSON parsing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    logger.info(f"Error samples saved -> {path} ({len(errors)} examples)")
    return path


def save_sample_outputs(
    predictions: list[dict],
    output_dir: Path,
    n: int = 50,
    seed: int = 42,
    filename: str = "sample_outputs.json",
) -> Path:
    """Save a random selection of predictions for manual review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    rng = random.Random(seed)
    sample = rng.sample(predictions, min(n, len(predictions)))
    # Keep only key fields
    slim = [
        {
            "index": p.get("index"),
            "generated": p.get("generated", ""),
            "reference": p.get("reference", ""),
        }
        for p in sample
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    logger.info(f"Sample outputs saved -> {path} ({len(slim)} examples)")
    return path


def save_param_errors(
    param_errors: list[dict],
    output_dir: Path,
    filename: str = "param_errors.json",
) -> Path:
    """Save Phase 2C parameter validation error samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(param_errors, f, indent=2, ensure_ascii=False)
    logger.info(f"Param errors saved -> {path} ({len(param_errors)} examples)")
    return path


def save_taxonomy_report(predictions: list[dict], output_dir: Path, filename: str = "error_taxonomy.json") -> Path:
    """
    Save the labeled failure-category breakdown to a JSON report.

    Expects predictions to already have "failure_category" set by
    error_taxonomy.label_predictions() — this is called automatically by
    compute_metrics(), so running evaluator.run_evaluation() is sufficient.
    """
    from model_2_training.src.eval.error_taxonomy import get_taxonomy_errors, TAXONOMY_LABELS

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    counts: dict[str, int] = {label: 0 for label in TAXONOMY_LABELS}
    for pred in predictions:
        cat = pred.get("failure_category")
        if cat in counts:
            counts[cat] += 1

    n = len(predictions)
    summary = {
        "total": n,
        "categories": {
            label: {
                "count": counts[label],
                "rate": round(counts[label] / n, 4) if n else 0.0,
            }
            for label in TAXONOMY_LABELS
        },
        "failures": get_taxonomy_errors(predictions),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Taxonomy report saved -> {path}")
    return path


def save_all_reports(
    metrics: dict,
    predictions: list[dict],
    error_samples: list[dict],
    output_dir: Path,
    split_name: str = "validation",
    num_sample_outputs: int = 50,
    per_field_report: dict | None = None,
    param_errors: list[dict] | None = None,
) -> dict[str, Path]:
    """
    Save all report artifacts in one call.

    Returns:
        Dict of report_type -> path.
    """
    paths = {}
    paths["metrics_json"]   = save_metrics_json(metrics, output_dir)
    paths["markdown"]       = save_markdown_report(
        metrics, output_dir, split_name=split_name, per_field_report=per_field_report
    )
    paths["error_samples"]  = save_error_samples(error_samples, output_dir)
    paths["sample_outputs"] = save_sample_outputs(predictions, output_dir, n=num_sample_outputs)
    paths["taxonomy"]       = save_taxonomy_report(predictions, output_dir)
    if param_errors is not None:
        paths["param_errors"] = save_param_errors(param_errors, output_dir)
    return paths
