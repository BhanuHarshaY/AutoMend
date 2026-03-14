"""
Unit tests for Alibaba 2017 Pipeline - Track A
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import json
import sys
from pathlib import Path
import polars as pl

# Setup paths
TEST_DIR = Path(__file__).resolve().parent
DS1_ROOT = TEST_DIR.parent
PROJECT_ROOT = DS1_ROOT.parent.parent
sys.path.insert(0, str(DS1_ROOT / "scripts"))  # Add scripts dir
sys.path.insert(0, str(DS1_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds1_raw_dir, get_ds1_processed_dir
    RAW_DIR = get_ds1_raw_dir()
    PROCESSED_DIR = get_ds1_processed_dir()
except ImportError:
    RAW_DIR = DS1_ROOT / "data" / "raw"
    PROCESSED_DIR = DS1_ROOT / "data" / "processed"

from preprocess import (
    load_server_usage,
    load_batch_task,
    load_server_event,
    discretize_cpu,
    discretize_mem,
    assign_label,
    balance_classes
)
from validate_schema import validate_schema
from schema_stats import run_schema_stats
from bias_detection import detect_slice_bias, detect_sequence_bias


# ── Discretization Tests ──────────────────────────────────────────────────────

def test_discretize_cpu_low():
    """CPU 0-10% should map to token 100"""
    assert discretize_cpu(5.0) == 100

def test_discretize_cpu_high():
    """CPU 90-100% should map to token 109"""
    assert discretize_cpu(95.0) == 109

def test_discretize_cpu_mid():
    """CPU 50-60% should map to token 105"""
    assert discretize_cpu(55.0) == 105

def test_discretize_mem_low():
    """Mem 0-10% should map to token 200"""
    assert discretize_mem(5.0) == 200

def test_discretize_mem_high():
    """Mem 90-100% should map to token 209"""
    assert discretize_mem(95.0) == 209

def test_discretize_cpu_invalid():
    """Invalid CPU value should return default token 104"""
    assert discretize_cpu("invalid") == 104


# ── Label Logic Tests ─────────────────────────────────────────────────────────

def test_label_normal():
    """Terminated job with low memory = Normal"""
    assert assign_label("Terminated", 102, 202) == 0

def test_label_resource_exhaustion():
    """Terminated job with high memory (>=70%) = Resource_Exhaustion"""
    assert assign_label("Terminated", 102, 207) == 1

def test_label_system_crash():
    """Failed job with low CPU (<=30%) = System_Crash"""
    assert assign_label("Failed", 102, 202) == 2

def test_label_data_drift():
    """Waiting job = Data_Drift"""
    assert assign_label("Waiting", 102, 202) == 4


# ── Data Loading Tests ────────────────────────────────────────────────────────

def test_load_server_usage():
    """server_usage should load with correct columns"""
    df = load_server_usage()
    assert "cpu_util_percent" in df.columns
    assert "mem_util_percent" in df.columns
    assert df.height > 0

def test_load_batch_task():
    """batch_task should load with correct columns"""
    df = load_batch_task()
    assert "status" in df.columns
    assert "plan_cpu" in df.columns
    assert df.height > 0

def test_load_server_event():
    """server_event should load with correct columns"""
    df = load_server_event()
    assert "event_type" in df.columns
    assert "machine_id" in df.columns
    assert df.height > 0


# ── Class Balancing Tests ─────────────────────────────────────────────────────

def test_balance_classes_reduces_normal():
    """Normal class should be reduced after balancing"""
    sequences = [{"sequence_ids": [100], "label": 0}] * 100
    sequences += [{"sequence_ids": [100], "label": 1}] * 5
    balanced = balance_classes(sequences)
    normal_count = sum(1 for s in balanced if s["label"] == 0)
    assert normal_count <= 15  # max 3x failure count


# ── Schema Validation Tests ───────────────────────────────────────────────────

def test_schema_validation_passes():
    """Schema validation should pass on processed output"""
    path = PROCESSED_DIR / "format_a_sequences.json"
    if path.exists():
        assert validate_schema(path) is True
    else:
        pytest.skip("Processed file not found")


# ── Bias Detection Tests ──────────────────────────────────────────────────────

def test_bias_detection_runs():
    """Bias detection should return a report with expected keys"""
    csv_path = RAW_DIR / "batch_task_sample.csv"
    if not csv_path.exists():
        pytest.skip(f"Raw file not found: {csv_path}")
    df = pl.read_csv(csv_path, has_header=False, new_columns=[
        "start_time", "end_time", "inst_num", "task_type",
        "job_id", "status", "plan_cpu", "plan_mem"
    ])
    report = detect_slice_bias(df)
    assert "status_slice" in report
    assert "task_type_slice" in report
    assert "failure_rate_by_task_type" in report

def test_sequence_bias_detection():
    """Sequence bias detection should return label distribution"""
    seq_path = PROCESSED_DIR / "format_a_sequences.json"
    if not seq_path.exists():
        pytest.skip("Processed sequences not found")
    result = detect_sequence_bias()
    assert "label_distribution" in result
    assert "bias_detected" in result