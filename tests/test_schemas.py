"""
Schema Validation Tests for Format A (Parquet) and Format B (JSONL ChatML).

Validates that output data conforms to expected schemas:
- Format A: {"sequence_ids": List[int], "label": int}
- Format B: {"messages": [{"role": str, "content": str}, ...]}

Run with: pytest tests/test_schemas.py -v
"""

import json
import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Format A Schema Validation (Track A - Parquet)
# =============================================================================

class TestFormatASchema:
    """Schema validation tests for Format A (Parquet)."""

    @pytest.mark.unit
    def test_format_a_has_sequence_ids_column(self, sample_parquet_ds1):
        """Parquet file has sequence_ids column."""
        df = pl.read_parquet(sample_parquet_ds1)
        assert "sequence_ids" in df.columns

    @pytest.mark.unit
    def test_format_a_has_label_column(self, sample_parquet_ds1):
        """Parquet file has label column."""
        df = pl.read_parquet(sample_parquet_ds1)
        assert "label" in df.columns

    @pytest.mark.unit
    def test_format_a_sequence_ids_is_list(self, sample_parquet_ds1):
        """sequence_ids column contains list-like sequences."""
        df = pl.read_parquet(sample_parquet_ds1)
        for idx, row in enumerate(df.iter_rows(named=True)):
            seq = row["sequence_ids"]
            assert isinstance(seq, (list, tuple)) or hasattr(seq, "__iter__"), (
                f"Row {idx}: sequence_ids should be list-like, got {type(seq)}"
            )

    @pytest.mark.unit
    def test_format_a_sequence_ids_contains_integers(self, sample_parquet_ds1):
        """sequence_ids lists contain only integer values."""
        import numpy as np
        df = pl.read_parquet(sample_parquet_ds1)
        for idx, row in enumerate(df.iter_rows(named=True)):
            for i, token in enumerate(row["sequence_ids"]):
                is_int = isinstance(token, (int, np.integer)) or (
                    isinstance(token, (float, np.floating)) and float(token).is_integer()
                )
                assert is_int, (
                    f"Row {idx}, token {i}: expected integer value, got {type(token)}"
                )

    @pytest.mark.unit
    def test_format_a_label_is_numeric(self, sample_parquet_ds1):
        """label column contains numeric values."""
        df = pl.read_parquet(sample_parquet_ds1)
        assert df["label"].dtype in (pl.Int64, pl.UInt64, pl.Float64, pl.Int32, pl.Float32), (
            f"label should be numeric, got {df['label'].dtype}"
        )

    @pytest.mark.unit
    def test_format_a_no_null_sequence_ids(self, sample_parquet_ds1):
        """sequence_ids column has no null values."""
        df = pl.read_parquet(sample_parquet_ds1)
        assert df["sequence_ids"].null_count() == 0, "Found null values in sequence_ids"

    @pytest.mark.unit
    def test_format_a_no_null_labels(self, sample_parquet_ds1):
        """label column has no null values."""
        df = pl.read_parquet(sample_parquet_ds1)
        assert df["label"].null_count() == 0, "Found null values in label"

    @pytest.mark.unit
    def test_format_a_sequence_ids_not_empty(self, sample_parquet_ds1):
        """sequence_ids lists are not empty."""
        df = pl.read_parquet(sample_parquet_ds1)
        for idx, row in enumerate(df.iter_rows(named=True)):
            assert len(row["sequence_ids"]) > 0, f"Row {idx}: sequence_ids is empty"


# =============================================================================
# Format B Schema Validation (Track B - JSONL ChatML)
# =============================================================================

class TestFormatBSchema:
    """Schema validation tests for Format B (JSONL ChatML)."""

    @pytest.mark.unit
    def test_format_b_has_messages_key(self, sample_jsonl_ds3):
        """Each JSONL record has a 'messages' key."""
        with open(sample_jsonl_ds3, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                assert "messages" in record, f"Line {line_num}: missing 'messages' key"

    @pytest.mark.unit
    def test_format_b_messages_is_list(self, sample_jsonl_ds3):
        """'messages' field is a list."""
        with open(sample_jsonl_ds3, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                assert isinstance(record["messages"], list), (
                    f"Line {line_num}: 'messages' should be list, got {type(record['messages'])}"
                )

    @pytest.mark.unit
    def test_format_b_messages_have_role_and_content(self, sample_jsonl_ds3):
        """Each message has 'role' and 'content' keys."""
        with open(sample_jsonl_ds3, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                for msg_idx, msg in enumerate(record["messages"]):
                    assert "role" in msg, (
                        f"Line {line_num}, msg {msg_idx}: missing 'role'"
                    )
                    assert "content" in msg, (
                        f"Line {line_num}, msg {msg_idx}: missing 'content'"
                    )

    @pytest.mark.unit
    def test_format_b_roles_are_valid(self, sample_jsonl_ds3):
        """Message roles are one of: system, user, assistant."""
        valid_roles = {"system", "user", "assistant"}
        with open(sample_jsonl_ds3, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                for msg_idx, msg in enumerate(record["messages"]):
                    assert msg["role"] in valid_roles, (
                        f"Line {line_num}, msg {msg_idx}: invalid role '{msg['role']}'"
                    )

    @pytest.mark.unit
    def test_format_b_content_is_string(self, sample_jsonl_ds3):
        """Message content is a string."""
        with open(sample_jsonl_ds3, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                for msg_idx, msg in enumerate(record["messages"]):
                    assert isinstance(msg["content"], str), (
                        f"Line {line_num}, msg {msg_idx}: content should be str, "
                        f"got {type(msg['content'])}"
                    )

    @pytest.mark.unit
    def test_format_b_messages_not_empty(self, sample_jsonl_ds3):
        """'messages' list is not empty."""
        with open(sample_jsonl_ds3, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                assert len(record["messages"]) > 0, (
                    f"Line {line_num}: 'messages' list is empty"
                )

    @pytest.mark.unit
    def test_format_b_content_not_empty(self, sample_jsonl_ds3):
        """Message content is not empty string."""
        with open(sample_jsonl_ds3, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                for msg_idx, msg in enumerate(record["messages"]):
                    assert len(msg["content"].strip()) > 0, (
                        f"Line {line_num}, msg {msg_idx}: content is empty"
                    )


# =============================================================================
# Combined Output Schema Tests
# =============================================================================

class TestCombinedOutputSchemas:
    """Schema validation for combined output files."""

    @pytest.mark.unit
    def test_combined_track_a_has_source_column(
        self, sample_parquet_ds1, sample_parquet_ds2, tmp_processed_dir, monkeypatch
    ):
        """Combined Track A output has source_dataset column."""
        from combiner_track_a import combine

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_parquet_ds1.parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [sample_parquet_ds1, sample_parquet_ds2])
        monkeypatch.setattr(combine, "OUTPUT_FILE", tmp_processed_dir / "track_A_combined.parquet")

        df = combine.combine_track_a()
        assert "source_dataset" in df.columns

    @pytest.mark.unit
    def test_combined_track_b_has_source_field(
        self, sample_track_b_files, tmp_processed_dir, monkeypatch
    ):
        """Combined Track B output has source_dataset field in each record."""
        from combiner_track_b import combine

        output_path = tmp_processed_dir / "track_B_combined.jsonl"

        monkeypatch.setattr(combine, "INTERIM_DIR", sample_track_b_files[0].parent)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        combine.combine_track_b()

        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert "source_dataset" in record
