"""
save_reports.py

Saves evaluation results to disk in multiple formats.

Outputs:
  - metrics.json        — all computed metrics
  - metrics_summary.md  — human-readable Markdown report
  - error_samples.json  — sample predictions that failed JSON parsing
  - sample_outputs.json — a random selection of predictions for manual review
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
) -> Path:
    """Save a human-readable Markdown metrics report."""
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
        f"## JSON Structural Quality Metrics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]

    rate_metrics = [
        ("non_empty_rate", "Non-empty output rate"),
        ("starts_with_brace_rate", "Starts with brace `{` or `[`"),
        ("ends_with_brace_rate", "Ends with brace `}` or `]`"),
        ("json_parse_rate", "JSON parse success rate"),
        ("malformed_json_rate", "Malformed JSON rate"),
        ("truncation_rate", "Truncation rate"),
        ("quote_balance_rate", "Quote balance rate"),
        ("brace_balance_rate", "Brace balance rate"),
        ("avg_output_length", "Average output length (chars)"),
    ]

    for key, label in rate_metrics:
        val = metrics.get(key, "N/A")
        if isinstance(val, float) and key.endswith("_rate"):
            display = f"{val * 100:.1f}%"
        elif isinstance(val, float):
            display = f"{val:.1f}"
        else:
            display = str(val)
        lines.append(f"| {label} | {display} |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")

    json_rate = metrics.get("json_parse_rate", 0)
    if json_rate >= 0.8:
        lines.append("Model is producing valid JSON in most outputs — good structural learning.")
    elif json_rate >= 0.5:
        lines.append("Model produces valid JSON roughly half the time — further training recommended.")
    else:
        lines.append("Model is struggling to produce valid JSON — review training data or hyperparameters.")

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


def save_all_reports(
    metrics: dict,
    predictions: list[dict],
    error_samples: list[dict],
    output_dir: Path,
    split_name: str = "validation",
    num_sample_outputs: int = 50,
) -> dict[str, Path]:
    """
    Save all report artifacts in one call.

    Returns:
        Dict of report_type -> path.
    """
    paths = {}
    paths["metrics_json"] = save_metrics_json(metrics, output_dir)
    paths["markdown"] = save_markdown_report(metrics, output_dir, split_name=split_name)
    paths["error_samples"] = save_error_samples(error_samples, output_dir)
    paths["sample_outputs"] = save_sample_outputs(predictions, output_dir, n=num_sample_outputs)
    return paths
