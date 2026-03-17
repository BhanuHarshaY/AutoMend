"""
AutoMend Data Pipeline DAG - StackOverflow Dataset
Orchestrates the complete data pipeline from acquisition through preprocessing,
validation, bias detection, and DVC versioning.

Uses centralized Slack alerting via src/utils/alerting.py

Moved from src/dataset_3_stackoverflow/dags/automend_pipeline_dag.py
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import os

from airflow import DAG
try:
    from airflow.operators.python import PythonOperator, BranchPythonOperator
except ImportError:
    from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# Project paths - handle both local dev and Docker
PROJECT_ROOT = Path("/opt/airflow")
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
DS3_RAW = DATA_ROOT / "raw" / "ds3_stackoverflow"
DS3_PROCESSED = DATA_ROOT / "processed" / "ds3_stackoverflow"
DS3_VALIDATED = DS3_PROCESSED / "validated"
DS3_TRAINING = DS3_PROCESSED / "training"
DS3_SCRIPTS = PROJECT_ROOT / "src" / "dataset_3_stackoverflow" / "scripts"

# Add scripts directory to path
sys.path.insert(0, str(DS3_SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variable for config
os.environ["DATA_ROOT"] = str(DATA_ROOT)
os.environ["DS3_RAW"] = str(DS3_RAW)
os.environ["DS3_PROCESSED"] = str(DS3_PROCESSED)

# Import centralized alerting
from src.utils.alerting import (
    alert_pipeline_start,
    alert_pipeline_success,
    alert_pipeline_failure,
    alert_validation_failure,
    alert_anomaly_detected as alert_anomalies_detected,
    alert_bias_detected,
    on_failure_callback,
)

# Import pipeline modules - with error handling for Airflow
try:
    from data_acquisition import run_acquisition
    from data_preprocessing import run_preprocessing
    from data_validation import run_validation
    from bias_detection import run_bias_detection
    MODULES_LOADED = True
except ImportError as e:
    import logging
    logging.error(f"Failed to import pipeline modules: {e}")
    MODULES_LOADED = False

    def run_acquisition(**kwargs):
        raise ImportError("Module not loaded. Check PYTHONPATH includes scripts directory.")
    def run_preprocessing(**kwargs):
        raise ImportError("Module not loaded")
    def run_validation(**kwargs):
        raise ImportError("Module not loaded")
    def run_bias_detection(**kwargs):
        raise ImportError("Module not loaded")

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
    'on_failure_callback': on_failure_callback,
}

dag = DAG(
    'ds3_stackoverflow_pipeline',
    default_args=default_args,
    description='Dataset 3 - AutoMend StackOverflow Data Pipeline for MLOps Training',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'ds3_stackoverflow', 'track_b'],
)


def task_start_pipeline(**context):
    dag_id = context['dag'].dag_id
    run_id = context['run_id']
    alert_pipeline_start(dag_id, run_id)
    return {
        "start_time": datetime.now().isoformat(),
        "dag_id": dag_id,
        "run_id": run_id
    }


def task_acquire_data(**context):
    import logging
    logger = logging.getLogger(__name__)

    try:
        from src.utils.data_acquire import ensure_data
        result = ensure_data("ds3", DS3_RAW, PROJECT_ROOT)
        logger.info("DS3 acquire: %s", result)
        context['ti'].xcom_push(key='acquisition_stats', value=result)
        return result
    except Exception as e:
        alert_pipeline_failure(
            context['dag'].dag_id, context['run_id'], 'acquire_data', str(e)
        )
        raise


def task_preprocess_data(**context):
    try:
        stats = run_preprocessing()
        context['ti'].xcom_push(key='preprocessing_stats', value=stats)
        if stats.get("status") != "success":
            raise Exception(f"Preprocessing failed: {stats.get('error')}")
        return stats
    except Exception as e:
        alert_pipeline_failure(
            context['dag'].dag_id, context['run_id'], 'preprocess_data', str(e)
        )
        raise


def task_validate_data(**context):
    try:
        results = run_validation()
        context['ti'].xcom_push(key='validation_results', value=results)
        anomaly_count = results.get("statistics", {}).get("total_anomalies", 0)
        if anomaly_count > 0:
            alert_anomalies_detected(
                pipeline_name="ds3_stackoverflow",
                anomaly_count=anomaly_count,
                anomaly_types=["outlier"]
            )
        return results
    except Exception as e:
        alert_pipeline_failure(
            context['dag'].dag_id, context['run_id'], 'validate_data', str(e)
        )
        raise


def task_check_validation(**context):
    ti = context['ti']
    validation_results = ti.xcom_pull(key='validation_results', task_ids='validate_data')
    if validation_results and validation_results.get("is_valid"):
        return 'detect_bias'
    else:
        alert_validation_failure(
            pipeline_name="ds3_stackoverflow",
            validation_results=validation_results or {}
        )
        return 'handle_validation_failure'


def task_handle_validation_failure(**context):
    ti = context['ti']
    validation_results = ti.xcom_pull(key='validation_results', task_ids='validate_data')
    failed_checks = validation_results.get("failed", []) if validation_results else []
    error_msg = f"Validation failed with {len(failed_checks)} issues"
    print(f"WARNING: {error_msg}")
    print(f"Failed checks: {json.dumps(failed_checks, indent=2)}")
    return {"status": "validation_failed", "issues": len(failed_checks)}


def task_detect_bias(**context):
    try:
        report = run_bias_detection()
        context['ti'].xcom_push(key='bias_report', value=report)
        biased_slices = report.get("biased_slices", [])
        if biased_slices:
            summary = report.get("summary", {})
            alert_bias_detected(
                pipeline_name="ds3_stackoverflow",
                biased_slices=biased_slices,
                severity_counts={
                    "high": summary.get("high_severity_count", 0),
                    "medium": summary.get("medium_severity_count", 0),
                    "low": summary.get("low_severity_count", 0),
                }
            )
        return report
    except Exception as e:
        alert_pipeline_failure(
            context['dag'].dag_id, context['run_id'], 'detect_bias', str(e)
        )
        raise


def task_generate_training_data(**context):
    try:
        input_path = DS3_VALIDATED / "qa_pairs_validated.json"
        if not input_path.exists():
            input_path = DS3_PROCESSED / "qa_pairs_processed.json"

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        training_data = []
        for record in data:
            training_data.append({
                "messages": [
                    {"role": "user", "content": record.get("question_body", "")},
                    {"role": "assistant", "content": record.get("answer_body", "")}
                ],
                "metadata": {
                    "question_id": record.get("question_id"),
                    "tags": record.get("tags", []),
                    "error_signatures": record.get("error_signatures", []),
                    "infra_components": record.get("infra_components", []),
                    "quality_score": record.get("quality_score", 0),
                }
            })

        DS3_TRAINING.mkdir(parents=True, exist_ok=True)
        output_path = DS3_TRAINING / "training_data.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2)

        jsonl_path = DS3_TRAINING / "training_data.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for example in training_data:
                f.write(json.dumps(example) + "\n")

        stats = {
            "total_examples": len(training_data),
            "output_path": str(output_path),
            "jsonl_path": str(jsonl_path),
        }
        context['ti'].xcom_push(key='training_stats', value=stats)
        return stats
    except Exception as e:
        alert_pipeline_failure(
            context['dag'].dag_id, context['run_id'], 'generate_training_data', str(e)
        )
        raise


def task_export_to_interim(**context):
    import shutil
    import logging
    logger = logging.getLogger(__name__)
    
    input_path = DS3_TRAINING / "training_data.jsonl"
    output_path = DATA_ROOT / "interim" / "ds3_stackoverflow.jsonl"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    
    logger.info(f"Exported to interim: {output_path}")
    return str(output_path)


def task_dvc_version(**context):
    import logging
    from src.utils.dvc_utils import dvc_version_path
    logger = logging.getLogger(__name__)
    logger.info("DVC versioning DS3 StackOverflow processed output")
    result = dvc_version_path(str(DS3_PROCESSED), cwd=str(PROJECT_ROOT), push=True)
    if result["add_result"]["returncode"] == 0:
        logger.info("DS3 StackOverflow output versioned successfully")
    else:
        logger.warning("DVC versioning had issues: %s", result["add_result"]["stderr"])
    context['ti'].xcom_push(key='dvc_result', value=result)
    return result


def task_end_pipeline(**context):
    ti = context['ti']
    acquisition_stats = ti.xcom_pull(key='acquisition_stats', task_ids='acquire_data') or {}
    preprocessing_stats = ti.xcom_pull(key='preprocessing_stats', task_ids='preprocess_data') or {}
    training_stats = ti.xcom_pull(key='training_stats', task_ids='generate_training_data') or {}
    start_info = ti.xcom_pull(task_ids='start_pipeline') or {}
    start_time = datetime.fromisoformat(start_info.get("start_time", datetime.now().isoformat()))
    duration = (datetime.now() - start_time).total_seconds()

    alert_pipeline_success(
        context['dag'].dag_id, context['run_id'], duration,
        {
            "records_acquired": acquisition_stats.get("total_pairs", "N/A"),
            "records_processed": preprocessing_stats.get("output_records", "N/A"),
            "training_examples": training_stats.get("total_examples", "N/A"),
        }
    )
    return {
        "status": "success",
        "duration_seconds": duration,
        "training_examples": training_stats.get("total_examples", 0)
    }


with dag:
    start = PythonOperator(task_id='start_pipeline', python_callable=task_start_pipeline)
    acquire = PythonOperator(task_id='acquire_data', python_callable=task_acquire_data)
    preprocess = PythonOperator(task_id='preprocess_data', python_callable=task_preprocess_data)
    validate = PythonOperator(task_id='validate_data', python_callable=task_validate_data)
    check_validation = BranchPythonOperator(task_id='check_validation', python_callable=task_check_validation)
    handle_failure = PythonOperator(task_id='handle_validation_failure', python_callable=task_handle_validation_failure)
    bias_detection = PythonOperator(task_id='detect_bias', python_callable=task_detect_bias)
    generate_training = PythonOperator(task_id='generate_training_data', python_callable=task_generate_training_data, trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    export_interim = PythonOperator(task_id='export_to_interim', python_callable=task_export_to_interim, trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    dvc_version = PythonOperator(task_id='dvc_version', python_callable=task_dvc_version, trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    end = PythonOperator(task_id='end_pipeline', python_callable=task_end_pipeline, trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    join = EmptyOperator(task_id='join', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    start >> acquire >> preprocess >> validate >> check_validation
    check_validation >> [bias_detection, handle_failure]
    bias_detection >> join
    handle_failure >> join
    join >> generate_training >> export_interim >> dvc_version >> end
