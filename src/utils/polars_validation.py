"""
Polars-native data validation utilities.

Replaces Great Expectations with lightweight, pure-Polars validation
expressions. Each function returns a dict with 'success' and 'detail' keys.

Usage:
    import polars as pl
    from src.utils.polars_validation import validate_columns_present, validate_no_nulls

    df = pl.read_parquet("data.parquet")
    results = validate_columns_present(df, ["col_a", "col_b"])
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import polars as pl

logger = logging.getLogger("automend.polars_validation")


def _result(success: bool, check: str, detail: str = "") -> dict:
    return {"success": success, "check": check, "detail": detail}


def validate_columns_present(
    df: pl.DataFrame, required: Sequence[str]
) -> dict:
    """Check that all required columns exist in the DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _result(False, "columns_present", f"Missing columns: {missing}")
    return _result(True, "columns_present", f"All {len(required)} columns present")


def validate_no_nulls(df: pl.DataFrame, columns: Sequence[str]) -> list[dict]:
    """Check that specified columns contain no null values."""
    results = []
    for col in columns:
        if col not in df.columns:
            results.append(_result(False, f"no_nulls_{col}", f"Column '{col}' not found"))
            continue
        null_count = df[col].null_count()
        if null_count > 0:
            results.append(
                _result(False, f"no_nulls_{col}", f"{null_count} nulls in '{col}'")
            )
        else:
            results.append(_result(True, f"no_nulls_{col}", f"No nulls in '{col}'"))
    return results


def validate_value_range(
    df: pl.DataFrame, column: str, *, min_val: Any = None, max_val: Any = None
) -> dict:
    """Check that all values in a numeric column fall within [min_val, max_val]."""
    if column not in df.columns:
        return _result(False, f"range_{column}", f"Column '{column}' not found")

    violations = df.filter(
        (pl.col(column) < min_val) if min_val is not None else pl.lit(False)
    ).height + df.filter(
        (pl.col(column) > max_val) if max_val is not None else pl.lit(False)
    ).height

    if violations > 0:
        return _result(
            False, f"range_{column}",
            f"{violations} values outside [{min_val}, {max_val}]"
        )
    return _result(True, f"range_{column}", f"All values in [{min_val}, {max_val}]")


def validate_allowed_values(
    df: pl.DataFrame, column: str, allowed: Sequence[Any]
) -> dict:
    """Check that a column only contains values from an allowed set."""
    if column not in df.columns:
        return _result(False, f"allowed_{column}", f"Column '{column}' not found")

    actual = set(df[column].unique().to_list())
    disallowed = actual - set(allowed)
    if disallowed:
        return _result(
            False, f"allowed_{column}",
            f"Disallowed values: {disallowed}"
        )
    return _result(True, f"allowed_{column}", f"All values in allowed set")


def validate_row_count(
    df: pl.DataFrame, *, min_rows: int = 0, max_rows: int | None = None
) -> dict:
    """Check that row count is within expected range."""
    n = df.height
    if n < min_rows:
        return _result(False, "row_count", f"{n} rows < minimum {min_rows}")
    if max_rows is not None and n > max_rows:
        return _result(False, "row_count", f"{n} rows > maximum {max_rows}")
    return _result(True, "row_count", f"{n} rows within expected range")


def validate_no_empty_strings(
    df: pl.DataFrame, columns: Sequence[str]
) -> list[dict]:
    """Check that string columns have no empty values."""
    results = []
    for col in columns:
        if col not in df.columns:
            results.append(_result(False, f"no_empty_{col}", f"Column '{col}' not found"))
            continue
        empty_count = df.filter(pl.col(col).str.len_bytes() == 0).height
        if empty_count > 0:
            results.append(
                _result(False, f"no_empty_{col}", f"{empty_count} empty strings in '{col}'")
            )
        else:
            results.append(_result(True, f"no_empty_{col}", f"No empty strings in '{col}'"))
    return results


def validate_unique(df: pl.DataFrame, column: str) -> dict:
    """Check that a column has all unique values (no duplicates)."""
    if column not in df.columns:
        return _result(False, f"unique_{column}", f"Column '{column}' not found")
    n_unique = df[column].n_unique()
    n_total = df.height
    duplicates = n_total - n_unique
    if duplicates > 0:
        return _result(False, f"unique_{column}", f"{duplicates} duplicate values")
    return _result(True, f"unique_{column}", "All values unique")


def validate_regex_match(
    df: pl.DataFrame, column: str, pattern: str
) -> dict:
    """Check that all values in a string column match a regex pattern."""
    if column not in df.columns:
        return _result(False, f"regex_{column}", f"Column '{column}' not found")
    non_matching = df.filter(~pl.col(column).str.contains(pattern)).height
    if non_matching > 0:
        return _result(
            False, f"regex_{column}",
            f"{non_matching} values don't match pattern '{pattern}'"
        )
    return _result(True, f"regex_{column}", f"All values match pattern")


def run_validation_suite(checks: list[dict]) -> dict:
    """
    Run a list of pre-computed check results and produce a summary report.

    Args:
        checks: List of dicts from the validate_* functions above.
                Can be nested lists (from functions returning list[dict]).

    Returns:
        Summary dict with total/passed/failed counts and full results.
    """
    flat: list[dict] = []
    for item in checks:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    passed = sum(1 for c in flat if c["success"])
    failed = sum(1 for c in flat if not c["success"])
    return {
        "total": len(flat),
        "passed": passed,
        "failed": failed,
        "success_rate": passed / len(flat) if flat else 1.0,
        "all_passed": failed == 0,
        "results": flat,
    }
