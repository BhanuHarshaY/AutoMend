"""
Integration Tests for Pipeline Data Flow.

Tests the complete data flow from interim to processed:
- Track A: DS1 + DS2 exports -> combiner -> combined Parquet
- Track B: DS3-6 exports -> combiner -> combined JSONL

Run with: pytest tests/test_integration.py -v -m integration
"""

import json
import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Track A Integration Tests
# =============================================================================

class TestTrackAIntegration:
    """Integration tests for Track A pipeline (Parquet)."""

    @pytest.mark.integration
    def test_track_a_pipeline_integration(
        self, sample_parquet_ds1, sample_parquet_ds2, tmp_processed_dir, monkeypatch
    ):
        """Full Track A pipeline: DS1 + DS2 -> combiner -> combined output."""
        from combiner_track_a import combine

        # Setup paths
        interim_dir = sample_parquet_ds1.parent
        output_path = tmp_processed_dir / "track_A_combined.parquet"

        monkeypatch.setattr(combine, "INTERIM_DIR", interim_dir)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        # Get input row counts
        ds1_rows = pl.read_parquet(sample_parquet_ds1).height
        ds2_rows = pl.read_parquet(sample_parquet_ds2).height

        # Run combiner
        result_df = combine.combine_track_a()

        # Verify output
        assert output_path.exists(), "Combined output file should exist"
        assert result_df.height == ds1_rows + ds2_rows, "Row count should equal sum of inputs"
        
        # Verify schema preserved
        assert "sequence_ids" in result_df.columns
        assert "label" in result_df.columns
        assert "source_dataset" in result_df.columns

        # Verify sources tracked
        sources = set(result_df["source_dataset"].unique())
        assert "ds1_alibaba" in sources
        assert "ds2_loghub" in sources

    @pytest.mark.integration
    def test_track_a_data_integrity(
        self, sample_parquet_ds1, sample_parquet_ds2, tmp_processed_dir, monkeypatch
    ):
        """Verify data integrity after Track A combination."""
        from combiner_track_a import combine

        output_path = tmp_processed_dir / "track_A_combined.parquet"

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        # Get original labels
        ds1_labels = set(pl.read_parquet(sample_parquet_ds1)["label"].unique().to_list())
        ds2_labels = set(pl.read_parquet(sample_parquet_ds2)["label"].unique().to_list())
        expected_labels = ds1_labels | ds2_labels

        # Run combiner
        result_df = combine.combine_track_a()

        # Verify all labels preserved
        result_labels = set(result_df["label"].unique().to_list())
        assert result_labels == expected_labels, (
            f"Labels mismatch. Expected: {expected_labels}, Got: {result_labels}"
        )


# =============================================================================
# Track B Integration Tests
# =============================================================================

class TestTrackBIntegration:
    """Integration tests for Track B pipeline (JSONL ChatML)."""

    @pytest.mark.integration
    def test_track_b_pipeline_integration(
        self, sample_track_b_files, tmp_processed_dir, monkeypatch
    ):
        """Full Track B pipeline: DS3-6 -> combiner -> combined output."""
        from combiner_track_b import combine

        interim_dir = sample_track_b_files[0].parent
        output_path = tmp_processed_dir / "track_B_combined.jsonl"

        monkeypatch.setattr(combine, "INTERIM_DIR", interim_dir)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        # Count input records
        expected_total = 0
        for file_path in sample_track_b_files:
            with open(file_path, "r", encoding="utf-8") as f:
                expected_total += sum(1 for line in f if line.strip())

        # Run combiner
        total, by_source = combine.combine_track_b()

        # Verify output
        assert output_path.exists(), "Combined output file should exist"
        assert total == expected_total, f"Record count should be {expected_total}, got {total}"

        # Verify all sources present
        expected_sources = {p.stem for p in sample_track_b_files}
        assert set(by_source.keys()) == expected_sources

    @pytest.mark.integration
    def test_track_b_data_integrity(
        self, sample_track_b_files, tmp_processed_dir, monkeypatch
    ):
        """Verify data integrity after Track B combination."""
        from combiner_track_b import combine

        output_path = tmp_processed_dir / "track_B_combined.jsonl"

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_track_b_files[0].parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        # Run combiner
        combine.combine_track_b()

        # Read and verify each record
        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                
                # Must have messages
                assert "messages" in record, f"Line {line_num}: missing 'messages'"
                
                # Must have source_dataset
                assert "source_dataset" in record, f"Line {line_num}: missing 'source_dataset'"
                
                # Messages must be valid ChatML
                for msg in record["messages"]:
                    assert "role" in msg
                    assert "content" in msg


# =============================================================================
# Cross-Track Tests
# =============================================================================

class TestInterimToProcessedFlow:
    """Tests for overall data flow from interim to processed."""

    @pytest.mark.integration
    def test_interim_dir_structure(self, tmp_data_dir):
        """Interim directory has expected structure."""
        interim_dir = tmp_data_dir / "interim"
        assert interim_dir.exists()
        assert interim_dir.is_dir()

    @pytest.mark.integration
    def test_processed_dir_structure(self, tmp_data_dir):
        """Processed directory has expected structure."""
        processed_dir = tmp_data_dir / "processed"
        assert processed_dir.exists()
        assert processed_dir.is_dir()

    @pytest.mark.integration
    def test_both_tracks_can_run_independently(
        self,
        sample_track_a_files,
        sample_track_b_files,
        tmp_processed_dir,
        monkeypatch,
    ):
        """Track A and Track B pipelines can run independently."""
        from combiner_track_a import combine as combine_a
        from combiner_track_b import combine as combine_b

        # Setup Track A
        monkeypatch.setattr(combine_a, "INTERIM_DIR", sample_track_a_files[0].parent)
        monkeypatch.setattr(combine_a, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine_a, "INPUT_FILES", sample_track_a_files)
        monkeypatch.setattr(
            combine_a, "OUTPUT_FILE", tmp_processed_dir / "track_A_combined.parquet"
        )

        # Setup Track B
        monkeypatch.setattr(combine_b, "INTERIM_DIR", sample_track_b_files[0].parent)
        monkeypatch.setattr(combine_b, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine_b, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(
            combine_b, "OUTPUT_FILE", tmp_processed_dir / "track_B_combined.jsonl"
        )

        # Run both
        df_a = combine_a.combine_track_a()
        total_b, _ = combine_b.combine_track_b()

        # Both should succeed
        assert df_a.height > 0
        assert total_b > 0

        # Both outputs should exist
        assert (tmp_processed_dir / "track_A_combined.parquet").exists()
        assert (tmp_processed_dir / "track_B_combined.jsonl").exists()
