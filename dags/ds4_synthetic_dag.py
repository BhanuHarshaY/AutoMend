"""
Dataset 4 Synthetic MLOps Pipeline DAG
DVC pull -> fetch prompts -> Gemini -> Format B -> DVC push

Uses centralized Slack alerting via src/utils/alerting.py

Moved from src/dataset_4_synthetic/dags/dataset4_synthetic_dag.py
"""
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)

# Project paths - handle both local dev and Docker
PROJECT_ROOT = Path("/opt/airflow")
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure project root is on path when Airflow loads this DAG
DS4_ROOT = PROJECT_ROOT / "src" / "dataset_4_synthetic"
DS4_SRC = DS4_ROOT / "src"
if str(DS4_SRC) not in sys.path:
    sys.path.insert(0, str(DS4_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow.decorators import dag, task

# Import after path fix - use local modules for data operations
from data import db_ops, gemini_gen, preprocessor, tools_loader, pipeline_logger, schema_stats, anomaly, parquet_ops
# Use shared DVC utils instead of local dvc_ops
from src.utils.dvc_utils import dvc_pull, dvc_version_path
# Centralized alerting
from src.utils.alerting import alert_anomaly_detected, on_failure_callback

DEFAULT_ARGS = {"retries": 1, "on_failure_callback": on_failure_callback}

# Use centralized data paths
DATA_ROOT = PROJECT_ROOT / "data"
DS4_RAW = DATA_ROOT / "raw" / "ds4_synthetic"
DS4_PROCESSED = DATA_ROOT / "processed" / "ds4_synthetic"
DB_PATH = DS4_RAW / "prompts.db"  # DB in raw/ since it contains input prompts, not outputs
PROCESSED_FILE = DS4_PROCESSED / "dataset4_format_b.jsonl"
SCHEMA_STATS_DIR = DS4_PROCESSED / "schema_stats"
TOOLS_JSON = DS4_ROOT / "config" / "available_tools.json"


@dag(
    dag_id="ds4_synthetic_dag",
    default_args=DEFAULT_ARGS,
    schedule=None,
    tags=["ds4_synthetic", "track_b"],
)
def ds4_synthetic_dag():
    @task
    def pull_data():
        log = pipeline_logger.get_logger("ds4_dag.pull_data")
        log.info("Starting DVC pull")
        dvc_pull(cwd=str(PROJECT_ROOT))
        log.info("DVC pull done")

    @task
    def ensure_prompts():
        from src.utils.data_acquire import ensure_data
        log = pipeline_logger.get_logger("ds4_dag.ensure_prompts")
        result = ensure_data("ds4", DS4_RAW, PROJECT_ROOT)
        log.info("DS4 acquire: %s", result)
        return result

    @task
    def fetch_prompts():
        import sqlite3
        log = pipeline_logger.get_logger("ds4_dag.fetch_prompts")
        log.info("Fetching unprocessed prompts from DB")
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        try:
            db_ops.setup_db(conn)
            prompts = db_ops.fetch_unprocessed_prompts(conn)
            log.info("Fetched %d prompts", len(prompts))
            return prompts
        finally:
            conn.close()

    @task
    def generate_synthetic_data(prompts: list):
        import json
        log = pipeline_logger.get_logger("ds4_dag.generate_synthetic_data")
        tools = tools_loader.load_available_tools(TOOLS_JSON)
        tools_str = ", ".join(tools)
        DS4_RAW.mkdir(parents=True, exist_ok=True)
        raw_records = []
        prompt_ids = []  # Track which prompts we're processing
        for i, row in enumerate(prompts or []):
            workflow = gemini_gen.generate_workflow(row["user_intent"], tools)
            raw_records.append({
                "user_intent": row["user_intent"],
                "tool_context": tools_str,
                "workflow": workflow.model_dump(),
            })
            prompt_ids.append(row["id"])  # Store the prompt ID
        raw_path = DS4_RAW / "raw_batch.json"
        raw_path.write_text(json.dumps(raw_records, indent=2), encoding="utf-8")
        log.info("Generated %d raw records, wrote %s", len(raw_records), raw_path)

        DS4_PROCESSED.mkdir(parents=True, exist_ok=True)
        parquet_records = [
            {
                "user_intent": r["user_intent"],
                "tool_context": r["tool_context"],
                "workflow_json": json.dumps(r["workflow"]),
            }
            for r in raw_records
        ]
        parquet_path = DS4_PROCESSED / "raw_batch.parquet"
        parquet_ops.write_raw_parquet(parquet_records, str(parquet_path))
        log.info("Parquet checkpoint written to %s", parquet_path)

        # Return both records and IDs for downstream processing
        return {"records": raw_records, "prompt_ids": prompt_ids}

    @task
    def format_data(generation_result: dict):
        import sqlite3
        log = pipeline_logger.get_logger("ds4_dag.format_data")
        raw_records = generation_result.get("records", [])
        prompt_ids = generation_result.get("prompt_ids", [])
        
        DS4_PROCESSED.mkdir(parents=True, exist_ok=True)
        preprocessor.write_format_b_jsonl(raw_records or [], PROCESSED_FILE)
        log.info("Format B JSONL written to %s", PROCESSED_FILE)
        
        # Mark prompts as processed after successful formatting
        if prompt_ids:
            conn = sqlite3.connect(DB_PATH)
            try:
                updated = db_ops.mark_prompts_processed(conn, prompt_ids)
                log.info("Marked %d prompts as processed", updated)
            finally:
                conn.close()
        
        return str(PROCESSED_FILE)

    @task
    def generate_schema_and_stats(processed_path: str):
        import json
        log = pipeline_logger.get_logger("ds4_dag.generate_schema_and_stats")
        path = Path(processed_path)
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
        SCHEMA_STATS_DIR.mkdir(parents=True, exist_ok=True)
        schema_stats.generate_schema_and_stats(records, SCHEMA_STATS_DIR)
        validation = schema_stats.validate_schema(records)
        log.info("Schema and stats written to %s; validation valid=%s", SCHEMA_STATS_DIR, validation["valid"])
        if not validation["valid"]:
            log.error("Schema validation errors: %s", validation["errors"])
        return processed_path

    @task
    def validate_and_alert(processed_path: str):
        import json
        log = pipeline_logger.get_logger("ds4_dag.validate_and_alert")
        path = Path(processed_path)
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
        anomalies = anomaly.detect_anomalies(records)
        if anomalies:
            log.error("Anomalies detected: %s", anomalies)
            # Send centralized alert
            alert_anomaly_detected(
                pipeline_name="ds4_synthetic",
                anomaly_count=len(anomalies),
                anomaly_types=["structural", "missing_field"],
                details={"sample_issues": anomalies[:3]}
            )
        else:
            log.info("No anomalies detected")
        return processed_path

    @task
    def export_to_interim(processed_path: str):
        import shutil
        log = pipeline_logger.get_logger("ds4_dag.export_to_interim")
        
        input_path = Path(processed_path)
        output_path = PROJECT_ROOT / "data" / "interim" / "ds4_synthetic.jsonl"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        
        log.info("Exported to interim: %s", output_path)
        return str(output_path)

    @task
    def commit_and_push(_: str):
        log = pipeline_logger.get_logger("ds4_dag.commit_and_push")
        log.info("DVC add and push starting")
        result = dvc_version_path(str(DS4_PROCESSED), cwd=str(PROJECT_ROOT), push=True)
        if result["add_result"]["returncode"] == 0:
            log.info("DS4 Synthetic output versioned successfully")
        else:
            log.warning("DVC versioning had issues: %s", result["add_result"]["stderr"])
        return result

    t1 = pull_data()
    t1b = ensure_prompts()
    t2 = fetch_prompts()
    t3 = generate_synthetic_data(t2)
    t4 = format_data(t3)
    t5 = generate_schema_and_stats(t4)
    t6 = validate_and_alert(t5)
    t7 = export_to_interim(t6)
    t8 = commit_and_push(t7)
    t1 >> t1b >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8


dag = ds4_synthetic_dag()
