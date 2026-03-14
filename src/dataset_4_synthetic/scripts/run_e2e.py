"""Run the Dataset 4 pipeline end-to-end (no Airflow): fetch prompts -> Gemini -> Format B.

Supports two modes:
  - Sequential (default for small prompt sets or no Ray)
  - Parallel via Ray actors + asyncio (for larger batches)

Assumes .env has GEMINI_API_KEY and prompts DB is seeded (run scripts/seed_prompts.py first).
"""
import json
import sqlite3
import sys
from pathlib import Path

DS4_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DS4_ROOT.parent.parent
sys.path.insert(0, str(DS4_ROOT / "src"))
sys.path.insert(0, str(DS4_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from data import db_ops, gemini_gen, preprocessor, tools_loader, pipeline_logger, schema_stats, anomaly, parquet_ops
from src.config.paths import get_ds4_prompts_db, get_ds4_raw_dir, get_ds4_processed_dir

try:
    from src.config.ray_config import get_dataset_config
    DS4_CFG = get_dataset_config("ds4")
except ImportError:
    DS4_CFG = {"num_workers": 4, "max_concurrent_per_worker": 5}

DB_PATH = get_ds4_prompts_db()
RAW_DIR = get_ds4_raw_dir()
PROCESSED_DIR = get_ds4_processed_dir()
PROCESSED_FILE = PROCESSED_DIR / "dataset4_format_b.jsonl"
TOOLS_JSON = DS4_ROOT / "config" / "available_tools.json"

USE_RAY_THRESHOLD = 10


def main():
    log = pipeline_logger.get_logger("run_e2e")
    tools = tools_loader.load_available_tools(TOOLS_JSON)
    log.info("Loaded %d tools from %s: %s", len(tools), TOOLS_JSON, tools)

    log.info("Connecting to DB")
    conn = sqlite3.connect(DB_PATH)
    db_ops.setup_db(conn)
    prompts = db_ops.fetch_unprocessed_prompts(conn)
    conn.close()

    if not prompts:
        log.error("No unprocessed prompts. Run: python scripts/seed_prompts.py")
        sys.exit(1)

    tools_str = ", ".join(tools)
    log.info("Found %d prompts", len(prompts))

    raw_records = []

    if len(prompts) >= USE_RAY_THRESHOLD:
        log.info("Using Ray actors for parallel Gemini generation...")
        try:
            workflows = gemini_gen.generate_workflows_parallel(
                prompts, tools,
                num_workers=DS4_CFG.get("num_workers", 4),
                max_concurrent=DS4_CFG.get("max_concurrent_per_worker", 5),
            )
            for row, wf in zip(prompts, workflows):
                if wf is not None:
                    raw_records.append({
                        "user_intent": row["user_intent"],
                        "tool_context": tools_str,
                        "workflow": wf,
                    })
            log.info("Ray parallel generation produced %d records", len(raw_records))
        except Exception as e:
            log.warning("Ray parallel generation failed (%s), falling back to sequential", e)
            raw_records = []

    if not raw_records:
        log.info("Using sequential Gemini generation...")
        for i, row in enumerate(prompts):
            log.info("  [%d/%d] %s", i + 1, len(prompts),
                     row["user_intent"][:50] + "..." if len(row["user_intent"]) > 50 else row["user_intent"])
            workflow = gemini_gen.generate_workflow(row["user_intent"], tools)
            raw_records.append({
                "user_intent": row["user_intent"],
                "tool_context": tools_str,
                "workflow": workflow.model_dump(),
            })

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "raw_batch.json"
    raw_path.write_text(json.dumps(raw_records, indent=2), encoding="utf-8")
    log.info("Wrote raw to %s", raw_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    parquet_records = [
        {
            "user_intent": r["user_intent"],
            "tool_context": r["tool_context"],
            "workflow_json": json.dumps(r["workflow"]),
        }
        for r in raw_records
    ]
    parquet_path = PROCESSED_DIR / "raw_batch.parquet"
    parquet_ops.write_raw_parquet(parquet_records, str(parquet_path))
    log.info("Wrote Parquet checkpoint to %s", parquet_path)

    preprocessor.write_format_b_jsonl(raw_records, PROCESSED_FILE)
    log.info("Wrote Format B JSONL to %s", PROCESSED_FILE)

    records = [json.loads(line) for line in PROCESSED_FILE.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
    schema_stats_dir = PROCESSED_DIR / "schema_stats"
    schema_stats_dir.mkdir(parents=True, exist_ok=True)
    schema_stats.generate_schema_and_stats(records, schema_stats_dir)
    validation = schema_stats.validate_schema(records)
    log.info("Schema/stats written to %s; validation valid=%s", schema_stats_dir, validation["valid"])
    if not validation["valid"]:
        log.error("Schema validation errors: %s", validation["errors"])

    anomalies = anomaly.check_and_alert(records)
    if anomalies:
        log.error("Anomalies: %s", anomalies)
    else:
        log.info("No anomalies detected")

    log.info("E2E done")


if __name__ == "__main__":
    main()
