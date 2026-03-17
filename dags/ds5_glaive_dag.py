"""
Airflow DAG for Glaive Function Calling v2 Data Pipeline.
Orchestrates the full pipeline from acquisition to bias detection.

Uses centralized Slack alerting via src/utils/alerting.py

Moved from src/dataset_5_glaive/dags/glaive_pipeline_dag.py
"""

from datetime import datetime, timedelta
from airflow import DAG
try:
    from airflow.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import logging
import sys
import os
from pathlib import Path

# Project paths - handle both local dev and Docker
PROJECT_ROOT = Path("/opt/airflow")
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
DS5_RAW = DATA_ROOT / "raw" / "ds5_glaive"
DS5_PROCESSED = DATA_ROOT / "processed" / "ds5_glaive"
DS5_SCRIPTS = PROJECT_ROOT / "src" / "dataset_5_glaive" / "scripts"

sys.path.insert(0, str(DS5_SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = str(DS5_SCRIPTS) + ":" + os.environ.get("PYTHONPATH", "")

logger = logging.getLogger(__name__)

# Centralized alerting
from src.utils.alerting import (
    alert_anomaly_detected,
    alert_validation_failure,
    alert_bias_detected,
    on_failure_callback,
)

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="ds5_glaive_pipeline",
    default_args=DEFAULT_ARGS,
    description="Dataset 5 - Glaive Function Calling v2 Data Pipeline",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["automend", "ds5_glaive", "track_b"],
) as dag:

    def task_data_acquisition(**context):
        from src.utils.data_acquire import ensure_data
        result = ensure_data("ds5", DS5_RAW, PROJECT_ROOT)
        logger.info("DS5 acquire: %s", result)

        output_file = DS5_RAW / "glaive_raw.jsonl"
        count = result.get("record_count", 0)
        if count == 0 and output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)

        context["ti"].xcom_push(key="record_count", value=count)
        context["ti"].xcom_push(key="source", value=result.get("status", "unknown"))
        return count

    def task_preprocessing(**context):
        from preprocessing import run_preprocessing
        DS5_PROCESSED.mkdir(parents=True, exist_ok=True)
        df = run_preprocessing(
            raw_file=DS5_RAW / "glaive_raw.jsonl",
            output_file=DS5_PROCESSED / "glaive_processed.jsonl",
        )
        logger.info("Preprocessing complete. Shape: %s", df.shape)
        context["ti"].xcom_push(key="processed_count", value=len(df))
        return len(df)

    def task_schema_validation(**context):
        import json
        from schema_validation import load_processed_data, run_validation
        df = load_processed_data(DS5_PROCESSED / "glaive_processed.jsonl")
        report = run_validation(df)
        passed = report["passed"]
        failed = report["failed"]
        logger.info("Validation: %d/%d passed", passed, report["total"])
        if failed > 0:
            failed_checks = [r["check"] for r in report["results"] if not r["success"]]
            logger.error("Failed checks: %s", failed_checks)
            for r in report["results"]:
                if not r["success"]:
                    logger.error("  %s: %s", r["check"], r.get("detail", ""))
            alert_validation_failure(
                pipeline_name="ds5_glaive",
                failed_checks=failed_checks,
            )
            raise ValueError(f"Schema validation failed: {failed} checks failed: {failed_checks}")
        context["ti"].xcom_push(key="validation_passed", value=passed)
        return passed

    def task_anomaly_detection(**context):
        from anomaly_detection import run_anomaly_detection
        report = run_anomaly_detection(filepath=DS5_PROCESSED / "glaive_processed.jsonl")
        logger.info("Anomaly detection: %d anomalies found", report["anomalies_found"])
        
        # Send centralized alert if anomalies detected
        if report["anomalies_found"] > 0:
            anomaly_types = [c["check"] for c in report.get("checks", []) if c.get("is_anomaly")]
            alert_anomaly_detected(
                pipeline_name="ds5_glaive",
                anomaly_count=report["anomalies_found"],
                anomaly_types=anomaly_types,
            )
        
        context["ti"].xcom_push(key="anomalies_found", value=report["anomalies_found"])
        return report["anomalies_found"]

    def task_bias_detection(**context):
        from bias_detection import run_bias_detection
        report = run_bias_detection(filepath=DS5_PROCESSED / "glaive_processed.jsonl")
        logger.info("Bias detection: %d findings across %d slices", report["findings_count"], report["slices_analyzed"])
        
        # Send centralized alert if bias detected
        if report["findings_count"] > 0:
            biased_slices = [f.get("slice", "unknown") for f in report.get("findings", [])]
            alert_bias_detected(
                pipeline_name="ds5_glaive",
                biased_slices=biased_slices,
                severity_counts={"high": report["findings_count"]}
            )
        
        context["ti"].xcom_push(key="bias_findings", value=report["findings_count"])
        return report["findings_count"]

    def task_export_to_interim(**context):
        import shutil
        logger.info("Exporting DS5 Glaive to interim directory")
        
        input_path = DS5_PROCESSED / "glaive_chatml.jsonl"
        output_path = DATA_ROOT / "interim" / "ds5_glaive.jsonl"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        
        logger.info("Exported to interim: %s", output_path)
        return str(output_path)

    def task_dvc_versioning(**context):
        from src.utils.dvc_utils import dvc_version_path
        logger.info("DVC versioning DS5 Glaive processed output folder")
        result = dvc_version_path(str(DS5_PROCESSED), cwd=str(PROJECT_ROOT), push=True)
        if result["add_result"]["returncode"] == 0:
            logger.info("DS5 Glaive output versioned successfully")
        else:
            logger.warning("DVC versioning had issues: %s", result["add_result"]["stderr"])
        context["ti"].xcom_push(key="dvc_result", value=result)
        return result

    def task_pipeline_summary(**context):
        ti = context["ti"]
        record_count = ti.xcom_pull(task_ids="data_acquisition", key="record_count")
        processed_count = ti.xcom_pull(task_ids="preprocessing", key="processed_count")
        validation_passed = ti.xcom_pull(task_ids="schema_validation", key="validation_passed")
        anomalies_found = ti.xcom_pull(task_ids="anomaly_detection", key="anomalies_found")
        bias_findings = ti.xcom_pull(task_ids="bias_detection", key="bias_findings")

        summary = {
            "pipeline": "Glaive Function Calling v2",
            "run_date": str(datetime.now()),
            "record_count": record_count,
            "processed_count": processed_count,
            "validation_passed": validation_passed,
            "anomalies_found": anomalies_found,
            "bias_findings": bias_findings,
            "status": "SUCCESS",
        }
        logger.info("Pipeline Summary: %s", summary)
        print("\n" + "=" * 50)
        print("       PIPELINE SUMMARY")
        print("=" * 50)
        for k, v in summary.items():
            print(f"  {k:<25} {v}")
        print("=" * 50)
        return summary

    t1_acquisition = PythonOperator(task_id="data_acquisition", python_callable=task_data_acquisition)
    t2_preprocessing = PythonOperator(task_id="preprocessing", python_callable=task_preprocessing)
    t3_schema_validation = PythonOperator(task_id="schema_validation", python_callable=task_schema_validation)
    t4_anomaly_detection = PythonOperator(task_id="anomaly_detection", python_callable=task_anomaly_detection)
    t5_bias_detection = PythonOperator(task_id="bias_detection", python_callable=task_bias_detection)
    t6_export_interim = PythonOperator(task_id="export_to_interim", python_callable=task_export_to_interim)
    t7_dvc_versioning = PythonOperator(task_id="dvc_versioning", python_callable=task_dvc_versioning)
    t8_summary = PythonOperator(task_id="pipeline_summary", python_callable=task_pipeline_summary)

    t1_acquisition >> t2_preprocessing >> t3_schema_validation >> t4_anomaly_detection >> t5_bias_detection >> t6_export_interim >> t7_dvc_versioning >> t8_summary
