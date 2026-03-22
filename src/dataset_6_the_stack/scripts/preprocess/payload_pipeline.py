"""
Orchestrator. Reads raw parquet chunks produced by stack_iac_sample.py,
transforms each row via payload_preprocess.py, validates the output,
and writes training records to data/processed/ds6_the_stack/training_records.jsonl.

Preferred path: Ray Data distributed map/filter across all chunks.
Fallback:       sequential chunk-at-a-time processing.
"""
import json
import logging
import sys
import time
import yaml
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

PREPROCESS_DIR = Path(__file__).resolve().parent
DS6_ROOT = PREPROCESS_DIR.parent.parent
PROJECT_ROOT = DS6_ROOT.parent.parent

for p in (PREPROCESS_DIR, DS6_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from payload_preprocess import (
    build_redactors,
    build_prompt_rules,
    process_row,
)
from src.config.ray_config import init_ray

CFG = yaml.safe_load((DS6_ROOT / "config/iac_analysis.yaml").read_text())

try:
    from src.config.paths import get_ds6_raw_dir, get_ds6_processed_dir, LOGS_DIR

    RAW = get_ds6_raw_dir()
    PROC = get_ds6_processed_dir()
    LOGS = LOGS_DIR
except ImportError:
    RAW = DS6_ROOT / CFG["paths"]["raw_dir"]
    PROC = DS6_ROOT / CFG["paths"]["processed_dir"]
    LOGS = DS6_ROOT / CFG["paths"]["logs_dir"]

PROC.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "preprocess.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

REDACTORS = build_redactors(CFG)
PROMPT_RULES = build_prompt_rules(CFG)


# ── validation ────────────────────────────────────────────────────────
def validate(record: dict) -> tuple[bool, str]:
    """
    Round-trip check on the assistant content:

      1. messages must be in system / user / assistant order
      2. assistant content must be valid JSON
      3. JSON must follow workflow.steps shape
      4. manifest_content inside the first step must be valid YAML
    """
    try:
        messages = record["messages"]
        if len(messages) != 3:
            return False, "bad_message_count"

        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            return False, "bad_message_roles"

        parsed = json.loads(messages[2]["content"])

        workflow = parsed["workflow"]
        steps = workflow["steps"]
        if not isinstance(steps, list) or not steps:
            return False, "missing_steps"

        first_step = steps[0]
        if first_step["tool"] != "apply_manifest":
            return False, "wrong_tool"

        manifest = first_step["params"]["manifest_content"]
        yaml.safe_load(manifest)
        return True, "ok"

    except json.JSONDecodeError:
        return False, "invalid_json"
    except yaml.YAMLError:
        return False, "invalid_yaml_after_wrap"
    except (KeyError, IndexError, TypeError) as e:
        return False, f"missing_key:{e}"


# ── per-row transform used by both paths ──────────────────────────────
def _transform_row(row: dict) -> dict | None:
    """Process + validate a single row; return the record or None."""
    record, status = process_row(row, CFG, REDACTORS, PROMPT_RULES)
    if status != "ok":
        return None
    ok, _ = validate(record)
    return record if ok else None


# ── Ray Data path ─────────────────────────────────────────────────────
def _run_ray(out_path: Path) -> Counter:
    import ray.data

    init_ray()
    log.info("Ray Data path — reading parquet from %s", RAW)

    ray_ds = ray.data.read_parquet(str(RAW))

    processed = (
        ray_ds
        .map(lambda row: _transform_row(row))
        .filter(lambda r: r is not None)
    )

    records = processed.take_all()

    stats: Counter = Counter()
    stats["total"] = ray_ds.count()

    with open(out_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1

    stats["dropped"] = stats["total"] - stats["written"]
    return stats


# ── Sequential fallback ──────────────────────────────────────────────
def _process_chunk_seq(chunk_path: Path, out_fh, stats: Counter) -> None:
    """Read one parquet chunk and write valid training records."""
    for row in pq.read_table(chunk_path).to_pylist():
        stats["total"] += 1

        record, status = process_row(row, CFG, REDACTORS, PROMPT_RULES)
        if status != "ok":
            stats[f"drop_{status}"] += 1
            continue

        ok, v_reason = validate(record)
        if not ok:
            stats[f"drop_{v_reason}"] += 1
            continue

        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats["written"] += 1


def _run_sequential(out_path: Path) -> Counter:
    chunks = sorted(RAW.glob("chunk_*.parquet"))
    assert chunks, f"No parquet chunks in {RAW} — run download script first."

    stats: Counter = Counter()
    log.info("Sequential path — processing %d chunks → %s", len(chunks), out_path)

    with open(out_path, "w", encoding="utf-8") as fh:
        for chunk_path in tqdm(chunks, desc="Chunks"):
            _process_chunk_seq(chunk_path, fh, stats)

    return stats


# ── entry point ───────────────────────────────────────────────────────
def run() -> None:
    out_path = PROC / "training_records.jsonl"
    t0 = time.time()

    try:
        stats = _run_ray(out_path)
    except Exception as exc:
        log.warning("Ray pipeline failed (%s), falling back to sequential", exc)
        stats = _run_sequential(out_path)

    elapsed = time.time() - t0
    n, w = stats["total"], stats["written"]
    pct = lambda x: f"{x / n * 100:.1f}%" if n else "n/a"

    log.info("Pipeline complete in %.1fs", elapsed)
    log.info("Rows read: %d | Written: %d (%s yield)", n, w, pct(w))
    for k, v in sorted(stats.items()):
        if k.startswith("drop_"):
            log.info("  drop %-30s %6d  (%s)", k[5:], v, pct(v))
    log.info("Output → %s", out_path)

    log_path = LOGS / CFG["paths"]["pipeline_log"].split("/")[-1]
    log_path.write_text(json.dumps(dict(stats), indent=2))
    log.info("Stats → %s", log_path)


if __name__ == "__main__":
    run()