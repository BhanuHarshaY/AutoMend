"""Unit tests for data quality checks and pipeline components.

Run with: conda run -n mlops pytest tests/ -v
"""
import sys
from pathlib import Path

DS2_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DS2_ROOT.parent.parent
# Add DS2_ROOT first so "from src.utils..." works in source modules
sys.path.insert(0, str(DS2_ROOT))
# Add DS2's src directory for direct imports like "from utils..."  
sys.path.insert(0, str(DS2_ROOT / "src"))

# Use centralized data paths, fallback to legacy
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ds2_loghub" / "mlops_processed"
if not PROCESSED_DIR.exists():
    PROCESSED_DIR = DS2_ROOT / "data_processed" / "mlops_processed"

import json
import polars as pl
import pytest

from utils.hashing import stable_hash, should_keep
from utils.timeparse import combine_timestamp, safe_str
from label.label_event_types import label_event_type
from normalize.normalize_linux import normalize_severity as linux_severity
from normalize.normalize_hpc import normalize_severity as hpc_severity
from normalize.normalize_hdfs import normalize_severity as hdfs_severity


# ── Hashing tests ──────────────────────────────────────────────────────────────

class TestHashing:
    def test_stable_hash_deterministic(self):
        """Same input always yields same hash."""
        assert stable_hash("linux:1") == stable_hash("linux:1")

    def test_stable_hash_different_inputs(self):
        """Different inputs yield different hashes (very high probability)."""
        assert stable_hash("linux:1") != stable_hash("linux:2")

    def test_should_keep_reproducible(self):
        """should_keep is deterministic across calls."""
        assert should_keep("linux", "1") == should_keep("linux", "1")

    def test_should_keep_roughly_10pct(self):
        """Roughly 10% of 1000 rows should be kept."""
        kept = sum(should_keep("linux", str(i)) for i in range(1000))
        assert 50 <= kept <= 150, f"Expected ~100 kept, got {kept}"


# ── Timestamp tests ────────────────────────────────────────────────────────────

class TestTimeparse:
    def test_combine_timestamp_linux(self):
        assert combine_timestamp("Jun", "14", "15:16:01") == "Jun 14 15:16:01"

    def test_combine_timestamp_hdfs(self):
        assert combine_timestamp("081109", "203615") == "081109 203615"

    def test_safe_str_none(self):
        assert safe_str(None) == ""

    def test_safe_str_nan(self):
        import math
        assert safe_str(float("nan")) == ""

    def test_safe_str_normal(self):
        assert safe_str("hello") == "hello"


# ── Severity normalization tests ───────────────────────────────────────────────

class TestLinuxSeverity:
    def test_auth_failure_is_error(self):
        assert linux_severity("authentication failure; logname= uid=0") == "ERROR"

    def test_failed_password_is_error(self):
        assert linux_severity("Failed password for user root") == "ERROR"

    def test_warn_keyword(self):
        assert linux_severity("connection timed out after retry") == "WARN"

    def test_normal_message_is_info(self):
        assert linux_severity("check pass; user unknown") == "INFO"


class TestHpcSeverity:
    def test_flag_1_is_error(self):
        assert hpc_severity("active", "1", "some message") == "ERROR"

    def test_unavailable_state_is_error(self):
        assert hpc_severity("state_change.unavailable", "0", "some message") == "ERROR"

    def test_active_state_is_info(self):
        assert hpc_severity("active", "0", "normal boot sequence") == "INFO"


class TestHdfsSeverity:
    def test_info_maps_to_info(self):
        assert hdfs_severity("INFO", "normal log") == "INFO"

    def test_fatal_maps_to_error(self):
        assert hdfs_severity("FATAL", "something bad") == "ERROR"

    def test_debug_maps_to_info(self):
        assert hdfs_severity("DEBUG", "debug output") == "INFO"

    def test_keyword_fallback(self):
        assert hdfs_severity("", "exception in thread main") == "ERROR"


# ── Event-type labeling tests ──────────────────────────────────────────────────

class TestLabeling:
    def test_auth_failure(self):
        assert label_event_type("authentication failure logname=0", "", "ERROR") == "auth_failure"

    def test_permission_denied(self):
        assert label_event_type("permission denied for user", "", "ERROR") == "permission_denied"

    def test_storage_unavailable(self):
        assert label_event_type("Component is in the unavailable state", "", "ERROR") == "storage_unavailable"

    def test_compute_oom(self):
        assert label_event_type("java heap space OutOfMemory error", "", "ERROR") == "compute_oom"

    def test_network_issue(self):
        assert label_event_type("connection refused by host", "", "ERROR") == "network_issue"

    def test_executor_failure(self):
        assert label_event_type("executor lost on node", "", "ERROR") == "executor_failure"

    def test_job_failed(self):
        assert label_event_type("task exception in job", "", "ERROR") == "job_failed"

    def test_normal_ops(self):
        assert label_event_type("normal startup completed", "", "INFO") == "normal_ops"

    def test_unknown(self):
        assert label_event_type("some random message", "", "WARN") == "unknown"


# ── Integration test: check output files exist and pass validation ─────────────

class TestOutputFiles:
    PROCESSED = PROCESSED_DIR

    def test_events_parquet_exists(self):
        assert (self.PROCESSED / "mlops_events.parquet").exists()

    def test_templates_csv_exists(self):
        assert (self.PROCESSED / "mlops_templates.csv").exists()

    def test_event_counts_exists(self):
        assert (self.PROCESSED / "event_counts_by_window.csv").exists()

    def test_error_rate_exists(self):
        assert (self.PROCESSED / "error_rate_by_system.csv").exists()

    def test_top_templates_exists(self):
        assert (self.PROCESSED / "top_templates.csv").exists()

    def test_validation_report_passed(self):
        report_path = self.PROCESSED / "validation_report.json"
        assert report_path.exists(), "validation_report.json not found"
        with open(report_path) as f:
            report = json.load(f)
        assert report["passed"], f"Validation failed: {report['errors']}"

    def test_events_schema(self):
        df = pl.read_parquet(self.PROCESSED / "mlops_events.parquet")
        required = ["system", "timestamp", "severity", "source",
                    "event_id", "event_template", "message", "raw_id",
                    "extras", "event_type"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_events_severity_values(self):
        df = pl.read_parquet(self.PROCESSED / "mlops_events.parquet")
        assert set(df["severity"].unique().to_list()).issubset({"INFO", "WARN", "ERROR"})

    def test_events_all_5_systems(self):
        df = pl.read_parquet(self.PROCESSED / "mlops_events.parquet")
        assert set(df["system"].unique().to_list()) == {"linux", "hpc", "hdfs", "hadoop", "spark"}

    def test_sample_pct_in_range(self):
        df = pl.read_parquet(self.PROCESSED / "mlops_events.parquet")
        pct = df.height / 10_000 * 100
        assert 95 <= pct <= 100, f"Sample pct {pct:.1f}% out of expected 95-100% range"
