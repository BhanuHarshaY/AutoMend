"""Unit tests for the data ingestion and statistics components."""
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
from unittest.mock import patch, MagicMock
import polars as pl
import pytest


# ── verify_inputs tests ────────────────────────────────────────────────────────

class TestVerifyInputs:
    def test_passes_when_all_files_present(self, tmp_path):
        """verify_inputs returns True when all required files exist."""
        from ingest.verify_inputs import verify_inputs, REQUIRED

        # Create the expected directory structure
        for system, files in REQUIRED.items():
            system_dir = tmp_path / system
            system_dir.mkdir()
            for fname in files:
                (system_dir / fname).write_text("header\nrow1\n")

        result = verify_inputs(data_raw=tmp_path)
        assert result is True

    def test_fails_when_file_missing(self, tmp_path):
        """verify_inputs returns False when a required file is missing."""
        from ingest.verify_inputs import verify_inputs

        # Create only partial structure (missing most files)
        (tmp_path / "Linux").mkdir()
        # Leave the file absent

        result = verify_inputs(data_raw=tmp_path)
        assert result is False

    def test_fails_when_directory_missing(self, tmp_path):
        """verify_inputs returns False when a system directory is absent."""
        from ingest.verify_inputs import verify_inputs
        # Empty tmp_path — no system subdirs
        result = verify_inputs(data_raw=tmp_path)
        assert result is False

    def test_required_has_all_5_systems(self):
        """REQUIRED dict covers all 5 systems."""
        from ingest.verify_inputs import REQUIRED
        assert set(REQUIRED.keys()) == {"Linux", "HPC", "HDFS", "Hadoop", "Spark"}

    def test_each_system_has_structured_and_templates(self):
        """Each system needs both structured CSV and templates CSV."""
        from ingest.verify_inputs import REQUIRED
        for system, files in REQUIRED.items():
            assert any("structured" in f for f in files), f"{system} missing structured CSV"
            assert any("templates" in f for f in files), f"{system} missing templates CSV"


# ── generate_statistics tests ──────────────────────────────────────────────────

class TestGenerateStatistics:
    def _make_events_df(self) -> pl.DataFrame:
        rows = []
        for system in ["linux", "hpc", "hdfs", "hadoop", "spark"]:
            for i in range(10):
                rows.append({
                    "system": system,
                    "timestamp": "Jun 14 15:16:01",
                    "severity": "INFO",
                    "source": "test",
                    "event_id": f"E{i+1}",
                    "event_template": f"Template {i}",
                    "message": f"msg {i}",
                    "raw_id": str(i),
                    "extras": "{}",
                    "event_type": "normal_ops",
                })
        return pl.DataFrame(rows)

    def test_statistics_report_created(self, tmp_path):
        """generate_statistics writes a statistics_report.json."""
        from validate.generate_statistics import generate_statistics

        df = self._make_events_df()
        events_path = tmp_path / "events.parquet"
        report_path = tmp_path / "statistics_report.json"
        df.write_parquet(events_path)

        result = generate_statistics(events_path=events_path, report_path=report_path)
        assert report_path.exists()

    def test_report_contains_statistics_key(self, tmp_path):
        """Report JSON has 'statistics' and 'polars_validation' keys."""
        from validate.generate_statistics import generate_statistics

        df = self._make_events_df()
        events_path = tmp_path / "events.parquet"
        report_path = tmp_path / "statistics_report.json"
        df.write_parquet(events_path)

        result = generate_statistics(events_path=events_path, report_path=report_path)
        assert "statistics" in result
        assert "polars_validation" in result

    def test_statistics_row_count_correct(self, tmp_path):
        """Statistics report reflects actual row count."""
        from validate.generate_statistics import generate_statistics

        df = self._make_events_df()
        events_path = tmp_path / "events.parquet"
        report_path = tmp_path / "statistics_report.json"
        df.write_parquet(events_path)

        result = generate_statistics(events_path=events_path, report_path=report_path)
        assert result["statistics"]["total_rows"] == df.height

    def test_statistics_all_systems_present(self, tmp_path):
        """rows_per_system covers all 5 systems."""
        from validate.generate_statistics import generate_statistics

        df = self._make_events_df()
        events_path = tmp_path / "events.parquet"
        report_path = tmp_path / "statistics_report.json"
        df.write_parquet(events_path)

        result = generate_statistics(events_path=events_path, report_path=report_path)
        systems = set(result["statistics"]["rows_per_system"].keys())
        assert systems == {"linux", "hpc", "hdfs", "hadoop", "spark"}


# ── Integration: statistics report exists after pipeline ──────────────────────

class TestStatisticsOutputExists:
    PROCESSED = PROCESSED_DIR

    def test_statistics_report_exists(self):
        path = self.PROCESSED / "statistics_report.json"
        if not path.exists():
            pytest.skip("Pipeline not yet run (generate_statistics task)")
        with open(path) as f:
            report = json.load(f)
        assert "statistics" in report
        assert report["statistics"]["total_rows"] > 0

    def test_bias_report_exists(self):
        path = self.PROCESSED / "bias_report.json"
        if not path.exists():
            pytest.skip("Pipeline not yet run (detect_bias task)")
        with open(path) as f:
            report = json.load(f)
        assert "slices" in report
        assert "flags" in report
        assert "bias_detected" in report
