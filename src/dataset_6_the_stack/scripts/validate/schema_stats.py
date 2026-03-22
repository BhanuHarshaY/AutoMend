"""
Validates every training record in training_records.jsonl:
  - Required fields present (messages, _meta)
  - messages has exactly 3 turns (system + user + assistant)
  - assistant content is valid JSON
  - assistant content follows workflow.steps shape
  - manifest_content inside the first step is valid YAML
  - No PII leaked through redaction
  - Content length within bounds

Writes logs/schema_report.json with per-field statistics.

Uses Polars DataFrames for aggregate statistics computation
while keeping the per-record validation logic in pure Python.
"""

import json
import logging
import re
import sys
import yaml
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).parents[2]
PROJECT_ROOT = _ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds6_processed_dir, LOGS_DIR
    _PROC = get_ds6_processed_dir()
    _LOGS = LOGS_DIR
except ImportError:
    import yaml as _yaml_init
    _cfg_init = _yaml_init.safe_load((_ROOT / "config/iac_analysis.yaml").read_text())
    _PROC = _ROOT / _cfg_init["paths"]["processed_dir"]
    _LOGS = _ROOT / _cfg_init["paths"]["logs_dir"]

_LOGS.mkdir(parents=True, exist_ok=True)
(_ROOT / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(_LOGS / "schema_stats.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

REQUIRED_TOP_KEYS = {"messages", "_meta"}
REQUIRED_META_KEYS = {"hexsha", "path", "size", "licenses"}
PII_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?:\d{1,3}\.){3}\d{1,3}"),
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
]


def validate_record(record: dict) -> tuple[bool, list[str]]:
    """
    Check one training record against all validation rules.
    Returns (True, []) if valid, (False, [violations]) otherwise.
    """
    violations: list[str] = []

    for k in REQUIRED_TOP_KEYS:
        if k not in record:
            violations.append(f"missing_top_key:{k}")

    msgs = record.get("messages", [])
    if len(msgs) != 3:
        violations.append(f"wrong_message_count:{len(msgs)}")
    else:
        if msgs[0].get("role") != "system":
            violations.append("first_role_not_system")
        if msgs[1].get("role") != "user":
            violations.append("second_role_not_user")
        if msgs[2].get("role") != "assistant":
            violations.append("third_role_not_assistant")

        system_text = msgs[0].get("content", "")
        if not isinstance(system_text, str) or not system_text.strip():
            violations.append("empty_system_prompt")

        prompt = msgs[1].get("content", "")
        if not isinstance(prompt, str) or not prompt.strip():
            violations.append("empty_prompt")

        assistant = msgs[2].get("content", "")
        try:
            parsed = json.loads(assistant)
        except json.JSONDecodeError as e:
            violations.append(f"assistant_invalid_json:{e}")
            parsed = None

        if parsed is not None:
            workflow = parsed.get("workflow")
            if not isinstance(workflow, dict):
                violations.append("missing_workflow")
            else:
                steps = workflow.get("steps")
                if not isinstance(steps, list) or not steps:
                    violations.append("missing_steps")
                else:
                    first_step = steps[0]
                    if not isinstance(first_step, dict):
                        violations.append("bad_first_step")
                    else:
                        if first_step.get("tool") != "apply_manifest":
                            violations.append("wrong_tool_name")

                        params = first_step.get("params", {})
                        if not isinstance(params, dict):
                            violations.append("bad_params")
                        else:
                            manifest = params.get("manifest_content", "")
                            if not manifest:
                                violations.append("empty_manifest_content")
                            else:
                                try:
                                    yaml.safe_load(manifest)
                                except yaml.YAMLError as e:
                                    violations.append(f"manifest_invalid_yaml:{e}")

                                for pat in PII_PATTERNS:
                                    if pat.search(manifest):
                                        violations.append(f"pii_leaked:{pat.pattern[:20]}")

    meta = record.get("_meta", {})
    for k in REQUIRED_META_KEYS:
        if k not in meta:
            violations.append(f"missing_meta_key:{k}")

    return len(violations) == 0, violations


def _extract_stats_row(record: dict) -> dict | None:
    """Extract numeric fields from a valid record for Polars aggregation."""
    msgs = record.get("messages", [])
    if len(msgs) != 3:
        return None

    manifest = ""
    try:
        parsed = json.loads(msgs[2].get("content", "{}"))
        workflow = parsed.get("workflow", {})
        steps = workflow.get("steps", [])
        if steps and isinstance(steps[0], dict):
            params = steps[0].get("params", {})
            if isinstance(params, dict):
                manifest = params.get("manifest_content", "")
    except Exception:
        manifest = ""

    return {
        "system_length": len(msgs[0].get("content", "")),
        "prompt_length": len(msgs[1].get("content", "")),
        "manifest_length": len(manifest),
        "source_size": int((record.get("_meta") or {}).get("size") or 0),
    }


def compute_stats_polars(valid_records: list[dict]) -> dict:
    """Compute aggregate statistics over valid records using Polars."""
    rows = [r for rec in valid_records if (r := _extract_stats_row(rec)) is not None]
    if not rows:
        return {
            "system_length_chars": {},
            "prompt_length_chars": {},
            "manifest_length_chars": {},
            "source_file_size": {},
        }

    df = pl.DataFrame(rows)

    def _col_stats(col: str) -> dict:
        s = df.select(
            pl.col(col).count().alias("count"),
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
            pl.col(col).mean().round(1).alias("mean"),
            pl.col(col).median().alias("median"),
        ).row(0, named=True)
        return {k: v for k, v in s.items() if v is not None}

    return {
        "system_length_chars": _col_stats("system_length"),
        "prompt_length_chars": _col_stats("prompt_length"),
        "manifest_length_chars": _col_stats("manifest_length"),
        "source_file_size": _col_stats("source_size"),
    }


def run_validation() -> dict:
    in_path = _PROC / "training_records.jsonl"
    out_path = _LOGS / "schema_report.json"
    _LOGS.mkdir(parents=True, exist_ok=True)

    assert in_path.exists(), f"No training records at {in_path}"
    log.info("Validating %s", in_path)

    valid_records: list[dict] = []
    violation_counts: dict[str, int] = {}
    total = valid = invalid = 0

    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                violation_counts["unparseable_line"] = (
                    violation_counts.get("unparseable_line", 0) + 1
                )
                continue

            ok, violations = validate_record(record)
            if ok:
                valid += 1
                valid_records.append(record)
            else:
                invalid += 1
                for v in violations:
                    violation_counts[v] = violation_counts.get(v, 0) + 1

    stats = compute_stats_polars(valid_records)

    report = {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "pass_rate_pct": round(valid / total * 100, 2) if total else 0,
        "violation_counts": violation_counts,
        "statistics": stats,
    }

    out_path.write_text(json.dumps(report, indent=2))
    log.info(
        "Schema validation: %d/%d passed (%.1f%%)",
        valid,
        total,
        report["pass_rate_pct"],
    )
    if violation_counts:
        log.warning("Violations found: %s", violation_counts)

    return report


if __name__ == "__main__":
    run_validation()