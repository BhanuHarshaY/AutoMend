"""Unit tests for the aggregation component."""
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

import polars as pl
import pytest

from aggregate.aggregates import make_time_bucket

SYSTEMS = ["linux", "hpc", "hdfs", "hadoop", "spark"]


def _make_labeled_df() -> pl.DataFrame:
    """Create a synthetic labeled events DataFrame."""
    rows = []
    event_types = ["job_failed", "normal_ops", "auth_failure", "network_issue"]
    timestamps = {
        "linux":  "Jun 14 15:16:01",
        "hpc":    "1077804742",
        "hdfs":   "081109 203615",
        "hadoop": "2015-10-18 18:01:47,978",
        "spark":  "17/06/09 20:10:40",
    }
    for system in SYSTEMS:
        for i in range(20):
            rows.append({
                "system": system,
                "timestamp": timestamps[system],
                "severity": "ERROR" if i % 4 == 0 else "INFO",
                "source": "test",
                "event_id": f"E{i % 5 + 1}",
                "event_template": f"Template {i % 5}",
                "message": f"message {i}",
                "raw_id": str(i),
                "extras": "{}",
                "event_type": event_types[i % len(event_types)],
            })
    return pl.DataFrame(rows)


# ── make_time_bucket tests ─────────────────────────────────────────────────────

class TestMakeTimeBucket:
    def test_hpc_epoch_uses_first_7_chars(self):
        result = make_time_bucket("1077804742", "hpc")
        assert result == "1077804"

    def test_hpc_short_timestamp(self):
        result = make_time_bucket("1077", "hpc")
        assert result == "1077"

    def test_hdfs_truncates_time_to_4_digits(self):
        result = make_time_bucket("081109 203615", "hdfs")
        assert result == "081109 2036"

    def test_hadoop_truncates_to_minute(self):
        result = make_time_bucket("2015-10-18 18:01:47,978", "hadoop")
        assert result == "2015-10-18 18:01"

    def test_linux_truncates_to_minute(self):
        result = make_time_bucket("Jun 14 15:16:01", "linux")
        assert result == "Jun 14 15:16"

    def test_spark_truncates_to_minute(self):
        result = make_time_bucket("17/06/09 20:10:40", "spark")
        assert result == "17/06/09 20:10"


# ── aggregate_metrics output tests ────────────────────────────────────────────

class TestAggregateMetrics:
    def _run_aggregate(self, tmp_path):
        from aggregate.aggregates import aggregate_metrics
        df = _make_labeled_df()
        events_path = tmp_path / "events.parquet"
        df.write_parquet(events_path)
        counts, error_rate, top_templates = aggregate_metrics(
            events_path=events_path,
            out_dir=tmp_path,
        )
        return counts, error_rate, top_templates

    def test_event_counts_schema(self, tmp_path):
        counts, _, _ = self._run_aggregate(tmp_path)
        for col in ["system", "severity", "event_type", "time_bucket", "count"]:
            assert col in counts.columns

    def test_error_rate_between_0_and_1(self, tmp_path):
        _, error_rate, _ = self._run_aggregate(tmp_path)
        assert (error_rate["error_rate"] >= 0).all()
        assert (error_rate["error_rate"] <= 1).all()

    def test_error_count_not_exceeds_total(self, tmp_path):
        _, error_rate, _ = self._run_aggregate(tmp_path)
        assert (error_rate["error_count"] <= error_rate["total"]).all()

    def test_top_templates_per_system_max_10(self, tmp_path):
        _, _, top_templates = self._run_aggregate(tmp_path)
        for system in SYSTEMS:
            count = top_templates.filter(pl.col("system") == system).height
            assert count <= 10, f"{system} has {count} templates (max 10)"

    def test_all_systems_in_event_counts(self, tmp_path):
        counts, _, _ = self._run_aggregate(tmp_path)
        assert set(counts["system"].unique().to_list()) == set(SYSTEMS)

    def test_error_rate_output_files_created(self, tmp_path):
        self._run_aggregate(tmp_path)
        assert (tmp_path / "event_counts_by_window.csv").exists()
        assert (tmp_path / "error_rate_by_system.csv").exists()
        assert (tmp_path / "top_templates.csv").exists()


# ── Integration: aggregate output files exist ──────────────────────────────────

class TestAggregateOutputExists:
    PROCESSED = PROCESSED_DIR

    def test_event_counts_exists(self):
        assert (self.PROCESSED / "event_counts_by_window.csv").exists()

    def test_error_rate_exists(self):
        assert (self.PROCESSED / "error_rate_by_system.csv").exists()

    def test_top_templates_exists(self):
        assert (self.PROCESSED / "top_templates.csv").exists()
