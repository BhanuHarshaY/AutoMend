"""Airflow DAG: LogHub MLOps Pipeline (Dataset 2)

Tasks (in order):
  1.  verify_inputs        — confirm all 10 raw CSV files are present
  2a-e. normalize_*        — normalize each system's logs (parallel)
  3.  merge_events         — merge + deterministic 10% sample
  4.  filter_templates     — keep templates for sampled EventIds
  5.  label_event_types    — add event_type column
  6.  generate_statistics  — GE schema validation + pandas stats report
  7.  aggregate_metrics    — generate trigger-ready aggregates
  8.  validate_quality     — run all data quality checks (fails DAG if bad)
  9.  detect_bias          — data slicing bias analysis (informational)
 10.  format_response      — convert events to Format A sequences (7 labels, padded to 512)
 11.  export_to_interim    — copy processed parquet to data/interim/ for combiner
 12.  dvc_version          — version processed output with DVC

All tasks use PythonOperator calling the same functions used in standalone scripts,
so the pipeline can also be run directly without Airflow.

Uses centralized Slack alerting via src/utils/alerting.py

Moved from src/dataset_2_loghub/dags/loghub_mlops_pipeline.py
"""
import logging
import sys
from pathlib import Path

from airflow import DAG
try:
    from airflow.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

# Project paths - handle both local dev and Docker
PROJECT_ROOT = Path("/opt/airflow")
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
DS2_RAW = DATA_ROOT / "raw" / "ds2_loghub"
DS2_PROCESSED = DATA_ROOT / "processed" / "ds2_loghub"

DS2_ROOT = PROJECT_ROOT / "src" / "dataset_2_loghub"
sys.path.insert(0, str(DS2_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

def _setup_ds2_path():
    """Ensure DS2 paths are in sys.path for task execution.

    DS2 has a local ``utils`` package that collides with the project-level
    ``src/utils``.  We force DS2's src dir to the front of sys.path and
    evict any cached ``utils`` module so the next ``import utils`` resolves
    to DS2's copy.
    """
    import sys
    from pathlib import Path
    project_root = Path("/opt/airflow")
    if not (project_root / "src").exists():
        project_root = Path(__file__).resolve().parent.parent
    ds2_src = project_root / "src" / "dataset_2_loghub" / "src"

    # Always force DS2 src to front of path
    if str(ds2_src) in sys.path:
        sys.path.remove(str(ds2_src))
    sys.path.insert(0, str(ds2_src))

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Evict cached utils module so DS2's local utils is found
    for mod_name in list(sys.modules):
        if mod_name == "utils" or mod_name.startswith("utils."):
            del sys.modules[mod_name]

_alert_logger = logging.getLogger("airflow.alert")

# Import centralized alerting
from src.utils.alerting import (
    on_failure_callback,
    alert_bias_detected,
    alert_validation_failure,
)

default_args = {
    "owner": "mlops",
    "retries": 1,
    "email_on_failure": False,
    "on_failure_callback": on_failure_callback,
}

dag = DAG(
    dag_id="ds2_loghub_pipeline",
    default_args=default_args,
    description="Dataset 2 - End-to-end LogHub MLOps data pipeline",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "ds2_loghub", "track_a"],
)


def task_verify_inputs():
    _setup_ds2_path()
    from src.utils.data_acquire import ensure_data
    result = ensure_data("ds2", DS2_RAW, PROJECT_ROOT)
    _alert_logger.info("DS2 acquire: %s", result)

    from ingest.verify_inputs import verify_inputs
    if not verify_inputs():
        raise RuntimeError("Input verification failed — missing raw CSV files after acquire.")
    return result


def task_normalize_linux():
    _setup_ds2_path()
    from normalize.normalize_linux import normalize_linux
    normalize_linux()


def task_normalize_hpc():
    _setup_ds2_path()
    from normalize.normalize_hpc import normalize_hpc
    normalize_hpc()


def task_normalize_hdfs():
    _setup_ds2_path()
    from normalize.normalize_hdfs import normalize_hdfs
    normalize_hdfs()


def task_normalize_hadoop():
    _setup_ds2_path()
    from normalize.normalize_hadoop import normalize_hadoop
    normalize_hadoop()


def task_normalize_spark():
    _setup_ds2_path()
    from normalize.normalize_spark import normalize_spark
    normalize_spark()


def task_merge_events():
    _setup_ds2_path()
    from sample.merge_events import merge_events
    merge_events()


def task_filter_templates():
    _setup_ds2_path()
    from sample.filter_templates import filter_templates
    filter_templates()


def task_label_event_types():
    _setup_ds2_path()
    from label.label_event_types import label_event_types
    label_event_types()


def task_generate_statistics():
    _setup_ds2_path()
    from validate.generate_statistics import generate_statistics
    result = generate_statistics()
    if result["polars_validation"]["success"] is False:
        raise RuntimeError(
            "Polars schema validation FAILED — check statistics_report.json"
        )


def task_aggregate_metrics():
    _setup_ds2_path()
    from aggregate.aggregates import aggregate_metrics
    aggregate_metrics()


def task_validate_quality():
    _setup_ds2_path()
    from validate.validate_quality import validate_quality
    passed = validate_quality()
    if not passed:
        alert_validation_failure(
            pipeline_name="ds2_loghub",
            failed_checks=["quality_check"],
        )
        raise RuntimeError("Data quality validation failed — check validation_report.json")


def task_detect_bias():
    _setup_ds2_path()
    from bias.detect_bias import detect_bias
    report = detect_bias()
    if report["bias_detected"]:
        _alert_logger.warning(
            "Bias detected in %d slice(s) — review bias_report.json for details.",
            len(report["flags"])
        )
        # Send centralized alert
        alert_bias_detected(
            pipeline_name="ds2_loghub",
            biased_slices=[f["slice"] for f in report.get("flags", [])],
            severity_counts={"high": len(report.get("flags", []))}
        )


def task_format_response():
    _setup_ds2_path()
    from format.format_response import format_response
    format_response()


def task_export_to_interim():
    _setup_ds2_path()
    import shutil
    from pathlib import Path
    
    project_root = Path("/opt/airflow")
    if not (project_root / "src").exists():
        project_root = Path(__file__).resolve().parent.parent
    
    input_path = project_root / "data" / "processed" / "ds2_loghub" / "data_ready" / "event_sequences.parquet"
    output_path = project_root / "data" / "interim" / "ds2_loghub.parquet"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    
    _alert_logger.info(f"Exported to interim: {output_path}")
    return str(output_path)


def task_dvc_version():
    _setup_ds2_path()
    from src.utils.dvc_utils import dvc_version_path
    _alert_logger.info("DVC versioning DS2 Loghub processed output")
    result = dvc_version_path(str(DS2_PROCESSED), cwd=str(PROJECT_ROOT), push=True)
    if result["add_result"]["returncode"] == 0:
        _alert_logger.info("DS2 Loghub output versioned successfully")
    else:
        _alert_logger.warning("DVC versioning had issues: %s", result["add_result"]["stderr"])
    return result


with dag:
    t_verify = PythonOperator(task_id="verify_inputs", python_callable=task_verify_inputs)

    t_norm_linux = PythonOperator(task_id="normalize_linux", python_callable=task_normalize_linux)
    t_norm_hpc = PythonOperator(task_id="normalize_hpc", python_callable=task_normalize_hpc)
    t_norm_hdfs = PythonOperator(task_id="normalize_hdfs", python_callable=task_normalize_hdfs)
    t_norm_hadoop = PythonOperator(task_id="normalize_hadoop", python_callable=task_normalize_hadoop)
    t_norm_spark = PythonOperator(task_id="normalize_spark", python_callable=task_normalize_spark)

    t_sample = PythonOperator(task_id="merge_events", python_callable=task_merge_events)
    t_filter_tmpl = PythonOperator(task_id="filter_templates", python_callable=task_filter_templates)
    t_label = PythonOperator(task_id="label_event_types", python_callable=task_label_event_types)
    t_stats = PythonOperator(task_id="generate_statistics", python_callable=task_generate_statistics)
    t_agg = PythonOperator(task_id="aggregate_metrics", python_callable=task_aggregate_metrics)
    t_validate = PythonOperator(task_id="validate_quality", python_callable=task_validate_quality)
    t_bias = PythonOperator(task_id="detect_bias", python_callable=task_detect_bias, trigger_rule=TriggerRule.ALL_DONE)
    t_format_response = PythonOperator(task_id="format_response", python_callable=task_format_response, trigger_rule=TriggerRule.ALL_DONE)
    t_export_interim = PythonOperator(task_id="export_to_interim", python_callable=task_export_to_interim)
    t_dvc_version = PythonOperator(task_id="dvc_version", python_callable=task_dvc_version)

    t_verify >> [t_norm_linux, t_norm_hpc, t_norm_hdfs, t_norm_hadoop, t_norm_spark]
    [t_norm_linux, t_norm_hpc, t_norm_hdfs, t_norm_hadoop, t_norm_spark] >> t_sample
    t_sample >> t_filter_tmpl >> t_label >> t_stats >> t_agg >> t_validate >> t_bias >> t_format_response >> t_export_interim >> t_dvc_version
