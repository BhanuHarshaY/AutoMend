"""
Airflow DAG - Alibaba Cluster Trace 2017 Pipeline
Track A: Trigger Engine (Anomaly Classification)
acquire -> preprocess -> validate -> schema_stats -> detect_anomalies -> bias_detection -> export_to_interim -> dvc_version

Moved from src/dataset_1_alibaba/dags/alibaba_pipeline.py
"""

from airflow import DAG
try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Project paths - handle both local dev and Docker
PROJECT_ROOT = Path("/opt/airflow")
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
DS1_RAW = DATA_ROOT / "raw" / "ds1_alibaba"
DS1_PROCESSED = DATA_ROOT / "processed" / "ds1_alibaba"
DS1_SCRIPTS = PROJECT_ROOT / "src" / "dataset_1_alibaba" / "scripts"

sys.path.insert(0, str(DS1_SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized alerting
from src.utils.alerting import (
    alert_anomaly_detected,
    alert_validation_failure,
    alert_bias_detected,
    on_failure_callback,
)

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="ds1_alibaba_pipeline",
    description="Dataset 1 - Alibaba 2017 Cluster Trace - Track A Trigger Engine Pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "automend", "ds1_alibaba", "track_a"],
) as dag:

    def acquire():
        from src.utils.data_acquire import ensure_data
        result = ensure_data("ds1", DS1_RAW, PROJECT_ROOT)
        logger.info("[ACQUIRE] %s", result)
        return result

    def preprocess():
        from preprocess import run_preprocessing
        seqs = run_preprocessing()
        print(f"[PREPROCESS] {len(seqs)} Format A sequences created")

    def validate():
        from validate_schema import validate_schema
        passed = validate_schema()
        if not passed:
            raise ValueError("Schema validation failed")
        print("[VALIDATE] Passed")

    def schema_stats():
        from schema_stats import run_schema_stats
        stats = run_schema_stats()
        print(f"[STATS] {stats['total_sequences']} sequences | {stats['label_distribution']}")

    def anomaly_detect():
        from anomaly_detection import detect_anomalies
        anomalies = detect_anomalies()
        
        # Send alert via centralized alerting system
        if anomalies:
            anomaly_types = list(set(a.get("label_name", "unknown") for a in anomalies))
            alert_anomaly_detected(
                pipeline_name="ds1_alibaba",
                anomaly_count=len(anomalies),
                anomaly_types=anomaly_types,
                details={"critical_count": len([a for a in anomalies if a.get("label") in [1, 2]])}
            )
        
        print(f"[ANOMALY] {len(anomalies)} anomalies detected")

    def bias_detect():
        from bias_detection import run_bias_detection
        report = run_bias_detection()
        print(f"[BIAS] Status bias: {report['raw_data_slicing']['status_slice']['bias_detected']}")
        print(f"[BIAS] Sequence bias: {report['sequence_bias']['bias_detected']}")
        print(f"[BIAS] Mitigation: {report['mitigation']['techniques_applied']}")

    def export_interim():
        from export_to_interim import export_to_interim
        output_path = export_to_interim()
        print(f"[EXPORT] Exported to interim: {output_path}")

    def dvc_version():
        from src.utils.dvc_utils import dvc_version_path
        logger.info("DVC versioning DS1 Alibaba processed output")
        result = dvc_version_path(str(DS1_PROCESSED), cwd=str(PROJECT_ROOT), push=True)
        if result["add_result"]["returncode"] == 0:
            logger.info("DS1 Alibaba output versioned successfully")
        else:
            logger.warning("DVC versioning had issues: %s", result["add_result"]["stderr"])
        return result

    t1 = PythonOperator(task_id="acquire_data", python_callable=acquire)
    t2 = PythonOperator(task_id="preprocess_data", python_callable=preprocess)
    t3 = PythonOperator(task_id="validate_schema", python_callable=validate)
    t4 = PythonOperator(task_id="schema_stats", python_callable=schema_stats)
    t5 = PythonOperator(task_id="detect_anomalies", python_callable=anomaly_detect)
    t6 = PythonOperator(task_id="bias_detection", python_callable=bias_detect)
    t7 = PythonOperator(task_id="export_to_interim", python_callable=export_interim)
    t8 = PythonOperator(task_id="dvc_version", python_callable=dvc_version)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8
