"""
End-to-End Tests for Complete Pipeline Execution.

Tests the full pipeline from sample data through to final combined outputs
with comprehensive validation of data quality and statistics.

Run with: pytest tests/test_e2e.py -v -m e2e
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
# Track A End-to-End Tests
# =============================================================================

class TestTrackAEndToEnd:
    """End-to-end tests for Track A (Trigger Engine) pipeline."""

    @pytest.mark.e2e
    def test_e2e_track_a_with_sample_data(
        self, sample_parquet_ds1, sample_parquet_ds2, tmp_processed_dir, monkeypatch
    ):
        """Full end-to-end Track A pipeline with sample fixture data."""
        from combiner_track_a import combine

        output_path = tmp_processed_dir / "track_A_combined.parquet"

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        # Run complete pipeline
        result_df = combine.combine_track_a()

        # Verify output exists and is valid
        assert output_path.exists()
        
        # Read back and verify
        loaded_df = pl.read_parquet(output_path)
        assert loaded_df.height == result_df.height
        
        # Schema validation
        assert "sequence_ids" in loaded_df.columns
        assert "label" in loaded_df.columns
        assert "source_dataset" in loaded_df.columns

    @pytest.mark.e2e
    def test_e2e_track_a_output_statistics(
        self, sample_parquet_ds1, sample_parquet_ds2, tmp_processed_dir, monkeypatch
    ):
        """Verify Track A output has expected statistical properties."""
        from combiner_track_a import combine

        output_path = tmp_processed_dir / "track_A_combined.parquet"

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        result_df = combine.combine_track_a()

        # Check statistics (combiner returns Polars DataFrame)
        stats = {
            "total_rows": result_df.height,
            "unique_labels": result_df["label"].n_unique(),
            "sources": result_df["source_dataset"].n_unique(),
            "avg_sequence_length": result_df["sequence_ids"].list.len().mean(),
        }

        assert stats["total_rows"] > 0, "Should have rows"
        assert stats["unique_labels"] >= 1, "Should have at least one label"
        assert stats["sources"] == 2, "Should have 2 source datasets"
        assert stats["avg_sequence_length"] > 0, "Sequences should not be empty"

    @pytest.mark.e2e
    def test_e2e_track_a_label_distribution(
        self, sample_parquet_ds1, sample_parquet_ds2, tmp_processed_dir, monkeypatch
    ):
        """Verify label distribution is preserved after combination."""
        from combiner_track_a import combine

        output_path = tmp_processed_dir / "track_A_combined.parquet"

        # Get expected label counts
        ds1_df = pl.read_parquet(sample_parquet_ds1)
        ds2_df = pl.read_parquet(sample_parquet_ds2)
        combined = pl.concat([ds1_df, ds2_df])
        expected_vc = combined["label"].value_counts()
        expected_dict = dict(zip(expected_vc["label"].to_list(), expected_vc["count"].to_list()))

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        result_df = combine.combine_track_a()
        result_vc = result_df["label"].value_counts()
        result_dict = dict(zip(result_vc["label"].to_list(), result_vc["count"].to_list()))

        # Compare distributions
        for label in expected_dict:
            assert label in result_dict, f"Missing label: {label}"
            assert result_dict[label] == expected_dict[label], (
                f"Label {label} count mismatch"
            )


# =============================================================================
# Track B End-to-End Tests
# =============================================================================

class TestTrackBEndToEnd:
    """End-to-end tests for Track B (Generative Architect) pipeline."""

    @pytest.mark.e2e
    def test_e2e_track_b_with_sample_data(
        self, sample_track_b_files, tmp_processed_dir, monkeypatch
    ):
        """Full end-to-end Track B pipeline with sample fixture data."""
        from combiner_track_b import combine

        output_path = tmp_processed_dir / "track_B_combined.jsonl"

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_track_b_files[0].parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        # Run complete pipeline
        total, by_source = combine.combine_track_b()

        # Verify output exists and is valid
        assert output_path.exists()
        
        # Read back and count
        with open(output_path, "r", encoding="utf-8") as f:
            loaded_count = sum(1 for line in f if line.strip())
        
        assert loaded_count == total

    @pytest.mark.e2e
    def test_e2e_track_b_output_statistics(
        self, sample_track_b_files, tmp_processed_dir, monkeypatch
    ):
        """Verify Track B output has expected statistical properties."""
        from combiner_track_b import combine

        output_path = tmp_processed_dir / "track_B_combined.jsonl"

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_track_b_files[0].parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        combine.combine_track_b()

        # Collect statistics
        stats = {
            "total_records": 0,
            "total_messages": 0,
            "unique_sources": set(),
            "roles_used": set(),
        }

        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                stats["total_records"] += 1
                stats["total_messages"] += len(record["messages"])
                stats["unique_sources"].add(record["source_dataset"])
                for msg in record["messages"]:
                    stats["roles_used"].add(msg["role"])

        assert stats["total_records"] > 0, "Should have records"
        assert stats["total_messages"] > 0, "Should have messages"
        assert len(stats["unique_sources"]) == 4, "Should have 4 sources"
        assert stats["roles_used"].issubset({"system", "user", "assistant"})

    @pytest.mark.e2e
    def test_e2e_track_b_chatm_format_valid(
        self, sample_track_b_files, tmp_processed_dir, monkeypatch
    ):
        """Verify all output records are valid ChatML format."""
        from combiner_track_b import combine

        output_path = tmp_processed_dir / "track_B_combined.jsonl"

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_track_b_files[0].parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        combine.combine_track_b()

        # Validate each record
        valid_roles = {"system", "user", "assistant"}
        
        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                
                # Must have messages list
                assert isinstance(record.get("messages"), list), (
                    f"Line {line_num}: 'messages' must be a list"
                )
                
                # Each message must have valid structure
                for msg_idx, msg in enumerate(record["messages"]):
                    assert msg.get("role") in valid_roles, (
                        f"Line {line_num}, msg {msg_idx}: invalid role"
                    )
                    assert isinstance(msg.get("content"), str), (
                        f"Line {line_num}, msg {msg_idx}: content must be string"
                    )


# =============================================================================
# Combined Statistics Tests
# =============================================================================

class TestCombinedOutputStatistics:
    """Tests for verifying combined output statistics."""

    @pytest.mark.e2e
    def test_e2e_combined_output_file_sizes(
        self,
        sample_parquet_ds1,
        sample_parquet_ds2,
        sample_track_b_files,
        tmp_processed_dir,
        monkeypatch,
    ):
        """Verify combined output files have non-zero sizes."""
        from combiner_track_a import combine as combine_a
        from combiner_track_b import combine as combine_b

        track_a_output = tmp_processed_dir / "track_A_combined.parquet"
        track_b_output = tmp_processed_dir / "track_B_combined.jsonl"

        # Setup Track A
        monkeypatch.setattr(combine_a, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine_a, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine_a, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine_a, "OUTPUT_FILE", track_a_output)

        # Setup Track B
        monkeypatch.setattr(combine_b, "INTERIM_DIR", sample_track_b_files[0].parent)
        monkeypatch.setattr(combine_b, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine_b, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine_b, "OUTPUT_FILE", track_b_output)

        # Run both pipelines
        combine_a.combine_track_a()
        combine_b.combine_track_b()

        # Verify file sizes
        assert track_a_output.stat().st_size > 0, "Track A output should have content"
        assert track_b_output.stat().st_size > 0, "Track B output should have content"

    @pytest.mark.e2e
    def test_e2e_no_data_loss(
        self,
        sample_parquet_ds1,
        sample_parquet_ds2,
        sample_track_b_files,
        tmp_processed_dir,
        monkeypatch,
    ):
        """Verify no data is lost during combination."""
        from combiner_track_a import combine as combine_a
        from combiner_track_b import combine as combine_b

        # Count Track A input rows
        ds1_rows = pl.read_parquet(sample_parquet_ds1).height
        ds2_rows = pl.read_parquet(sample_parquet_ds2).height
        expected_track_a_rows = ds1_rows + ds2_rows

        # Count Track B input records
        expected_track_b_records = 0
        for file_path in sample_track_b_files:
            with open(file_path, "r", encoding="utf-8") as f:
                expected_track_b_records += sum(1 for line in f if line.strip())

        track_a_output = tmp_processed_dir / "track_A_combined.parquet"
        track_b_output = tmp_processed_dir / "track_B_combined.jsonl"

        # Setup and run Track A
        monkeypatch.setattr(combine_a, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine_a, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine_a, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine_a, "OUTPUT_FILE", track_a_output)

        # Setup and run Track B
        monkeypatch.setattr(combine_b, "INTERIM_DIR", sample_track_b_files[0].parent)
        monkeypatch.setattr(combine_b, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine_b, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine_b, "OUTPUT_FILE", track_b_output)

        result_a = combine_a.combine_track_a()
        total_b, _ = combine_b.combine_track_b()

        # Verify no data loss
        assert result_a.height == expected_track_a_rows, (
            f"Track A data loss: expected {expected_track_a_rows}, got {len(result_a)}"
        )
        assert total_b == expected_track_b_records, (
            f"Track B data loss: expected {expected_track_b_records}, got {total_b}"
        )
