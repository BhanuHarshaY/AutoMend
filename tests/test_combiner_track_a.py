"""
Unit tests for Track A Combiner (Trigger Engine).

Tests the combine_track_a function that concatenates Parquet files
from Dataset 1 (Alibaba) and Dataset 2 (Loghub).

Run with: pytest tests/test_combiner_track_a.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestCombineTrackA:
    """Unit tests for Track A combiner."""

    @pytest.mark.unit
    def test_combine_track_a_concatenates_dataframes(
        self, sample_parquet_ds1, sample_parquet_ds2, tmp_processed_dir, monkeypatch
    ):
        """Verifies row counts sum correctly after concatenation."""
        from combiner_track_a import combine

        # Read expected row counts
        ds1_rows = pl.read_parquet(sample_parquet_ds1).height
        ds2_rows = pl.read_parquet(sample_parquet_ds2).height
        expected_total = ds1_rows + ds2_rows

        # Patch the paths in the module
        monkeypatch.setattr(combine, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", tmp_processed_dir / "track_A_combined.parquet")

        # Run combiner
        result_df = combine.combine_track_a()

        assert result_df.height == expected_total

    @pytest.mark.unit
    def test_combine_track_a_adds_source_column(
        self, sample_parquet_ds1, sample_parquet_ds2, tmp_processed_dir, monkeypatch
    ):
        """Verifies source_dataset column is added to track data origin."""
        from combiner_track_a import combine

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", tmp_processed_dir / "track_A_combined.parquet")

        result_df = combine.combine_track_a()

        assert "source_dataset" in result_df.columns
        assert set(result_df["source_dataset"].unique().to_list()) == {"ds1_alibaba", "ds2_loghub"}

    @pytest.mark.unit
    def test_combine_track_a_preserves_schema(
        self, sample_parquet_ds1, sample_parquet_ds2, tmp_processed_dir, monkeypatch
    ):
        """Verifies sequence_ids and label columns are preserved."""
        from combiner_track_a import combine

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", tmp_processed_dir / "track_A_combined.parquet")

        result_df = combine.combine_track_a()

        assert "sequence_ids" in result_df.columns
        assert "label" in result_df.columns

    @pytest.mark.unit
    def test_combine_track_a_handles_missing_file(
        self, sample_parquet_ds1, tmp_interim_dir, tmp_processed_dir, monkeypatch
    ):
        """Verifies graceful handling when one input file is missing."""
        from combiner_track_a import combine

        nonexistent_file = tmp_interim_dir / "nonexistent.parquet"

        monkeypatch.setattr(combine, "INTERIM_DIR", tmp_interim_dir)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, nonexistent_file])
        monkeypatch.setattr(combine, "OUTPUT_FILE", tmp_processed_dir / "track_A_combined.parquet")

        # Should still work with just one file
        result_df = combine.combine_track_a()

        ds1_rows = pl.read_parquet(sample_parquet_ds1).height
        assert result_df.height == ds1_rows

    @pytest.mark.unit
    def test_combine_track_a_raises_on_no_files(
        self, tmp_interim_dir, tmp_processed_dir, monkeypatch
    ):
        """Verifies FileNotFoundError is raised when all input files are missing."""
        from combiner_track_a import combine

        nonexistent1 = tmp_interim_dir / "missing1.parquet"
        nonexistent2 = tmp_interim_dir / "missing2.parquet"

        monkeypatch.setattr(combine, "INTERIM_DIR", tmp_interim_dir)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [nonexistent1, nonexistent2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", tmp_processed_dir / "track_A_combined.parquet")

        with pytest.raises(FileNotFoundError):
            combine.combine_track_a()

    @pytest.mark.unit
    def test_combine_track_a_output_is_valid_parquet(
        self, sample_parquet_ds1, sample_parquet_ds2, tmp_processed_dir, monkeypatch
    ):
        """Verifies output file is a valid, readable Parquet file."""
        from combiner_track_a import combine

        output_path = tmp_processed_dir / "track_A_combined.parquet"

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        combine.combine_track_a()

        # Verify file exists and is readable
        assert output_path.exists()
        df = pl.read_parquet(output_path)
        assert df.height > 0
