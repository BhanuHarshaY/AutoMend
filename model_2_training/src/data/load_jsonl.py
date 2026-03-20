"""
load_jsonl.py

Responsible for loading and validating the Track B combined JSONL artifact.

Responsibilities:
  - Open JSONL file safely
  - Parse line by line
  - Validate each record against dataset_contract
  - Skip or raise on malformed rows depending on strategy
  - Return clean validated samples
  - Provide a summarize_dataset() helper

This module does NOT tokenize, split, or train.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator
from loguru import logger

from model_2_training.src.data.dataset_contract import validate_sample, summarize_violations


def _iter_jsonl(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (line_number, raw_line) for each non-empty line in a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_num, line


def load_jsonl(
    path: str | Path,
    malformed_strategy: str = "skip",
) -> list[dict]:
    """
    Load and validate a JSONL file against the Track B dataset contract.

    Args:
        path: Path to the JSONL file.
        malformed_strategy: "skip" (log and continue) or "raise" (stop on first error).

    Returns:
        List of validated sample dicts.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if malformed_strategy is invalid.
        json.JSONDecodeError / ContractViolation: if strategy is "raise" and a bad row is found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL artifact not found: {path}")
    if malformed_strategy not in ("skip", "raise"):
        raise ValueError(f"malformed_strategy must be 'skip' or 'raise', got: {malformed_strategy!r}")

    valid_samples: list[dict] = []
    invalid_count = 0
    violations: list[str] = []

    logger.info(f"Loading JSONL from: {path}")

    for line_num, raw_line in _iter_jsonl(path):
        # Parse JSON
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError as e:
            msg = f"Line {line_num}: JSON parse error — {e}"
            violations.append("json_parse_error")
            invalid_count += 1
            if malformed_strategy == "raise":
                raise
            logger.warning(msg)
            continue

        # Validate contract
        is_valid, reason = validate_sample(obj)
        if not is_valid:
            violations.append(reason)
            invalid_count += 1
            msg = f"Line {line_num}: contract violation — {reason}"
            if malformed_strategy == "raise":
                from model_2_training.src.data.dataset_contract import ContractViolation
                raise ContractViolation(msg)
            logger.warning(msg)
            continue

        valid_samples.append(obj)

    total = len(valid_samples) + invalid_count
    logger.info(
        f"Loaded {len(valid_samples)}/{total} valid samples "
        f"({invalid_count} invalid, strategy='{malformed_strategy}')"
    )

    if violations:
        summary = summarize_violations(violations)
        logger.warning(f"Violation summary: {summary}")

    return valid_samples


def summarize_dataset(path: str | Path, malformed_strategy: str = "skip") -> dict:
    """
    Load the JSONL and return a summary report without returning all samples.

    Args:
        path: Path to the JSONL file.
        malformed_strategy: "skip" or "raise".

    Returns:
        Dict with keys: total_rows, valid_rows, invalid_rows, violation_summary,
        source_counts (if metadata.source_dataset present), role_distribution.
    """
    path = Path(path)
    samples = load_jsonl(path, malformed_strategy=malformed_strategy)

    source_counts: dict[str, int] = {}
    role_dist: dict[str, int] = {}

    for s in samples:
        # Source tracking
        meta = s.get("metadata") or s  # some datasets put source_dataset at top level
        src = meta.get("source_dataset", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

        # Role distribution
        for msg in s.get("messages", []):
            role = msg.get("role", "unknown")
            role_dist[role] = role_dist.get(role, 0) + 1

    # Count raw lines for total
    total_lines = sum(1 for _ in _iter_jsonl(path))

    return {
        "total_rows": total_lines,
        "valid_rows": len(samples),
        "invalid_rows": total_lines - len(samples),
        "source_counts": source_counts,
        "role_distribution": role_dist,
    }
