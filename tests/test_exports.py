"""
Export Script Tests.

Tests the export_to_interim.py scripts for each dataset and
the convert_to_parquet.py script for Dataset 1.

Run with: pytest tests/test_exports.py -v
"""

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Dataset 1 - Convert to Parquet
# =============================================================================

class TestDs1ConvertToParquet:
    """Tests for Dataset 1 JSON to Parquet conversion."""

    @pytest.mark.unit
    def test_ds1_convert_creates_parquet(self, tmp_path, sample_format_a_records):
        """Conversion creates a valid Parquet file from JSON."""
        # Setup: create input JSON
        input_dir = tmp_path / "dataset_1_alibaba" / "data" / "processed"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "format_a_sequences.json"
        
        with open(input_path, "w") as f:
            json.dump(sample_format_a_records, f)
        
        output_dir = tmp_path / "data" / "interim"
        output_dir.mkdir(parents=True)
        output_path = output_dir / "ds1_alibaba.parquet"
        
        # Import and patch
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "dataset_1_alibaba" / "scripts"))
        from dataset_1_alibaba.scripts import convert_to_parquet
        
        with patch.object(convert_to_parquet, "INPUT_PATH", input_path):
            with patch.object(convert_to_parquet, "OUTPUT_PATH", output_path):
                df = convert_to_parquet.convert_to_parquet()
        
        assert output_path.exists()
        assert df.height == len(sample_format_a_records)

    @pytest.mark.unit
    def test_ds1_convert_preserves_schema(self, tmp_path, sample_format_a_records):
        """Conversion preserves sequence_ids and label columns."""
        input_dir = tmp_path / "dataset_1_alibaba" / "data" / "processed"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "format_a_sequences.json"
        
        with open(input_path, "w") as f:
            json.dump(sample_format_a_records, f)
        
        output_dir = tmp_path / "data" / "interim"
        output_dir.mkdir(parents=True)
        output_path = output_dir / "ds1_alibaba.parquet"
        
        from dataset_1_alibaba.scripts import convert_to_parquet
        
        with patch.object(convert_to_parquet, "INPUT_PATH", input_path):
            with patch.object(convert_to_parquet, "OUTPUT_PATH", output_path):
                df = convert_to_parquet.convert_to_parquet()
        
        assert "sequence_ids" in df.columns
        assert "label" in df.columns

    @pytest.mark.unit
    def test_ds1_convert_raises_on_missing_input(self, tmp_path):
        """Conversion raises FileNotFoundError if input doesn't exist."""
        nonexistent = tmp_path / "nonexistent.json"
        output_path = tmp_path / "output.parquet"
        
        from dataset_1_alibaba.scripts import convert_to_parquet
        
        with patch.object(convert_to_parquet, "INPUT_PATH", nonexistent):
            with patch.object(convert_to_parquet, "OUTPUT_PATH", output_path):
                with pytest.raises(FileNotFoundError):
                    convert_to_parquet.convert_to_parquet()


# =============================================================================
# Export to Interim Tests (DS2-DS6)
# =============================================================================

class TestExportToInterim:
    """Tests for export_to_interim.py scripts."""

    @pytest.mark.unit
    def test_ds2_export_copies_file(self, tmp_path):
        """DS2 export copies Parquet file to interim directory."""
        # Create source file
        source_dir = tmp_path / "dataset_2_loghub" / "data_processed" / "data_ready"
        source_dir.mkdir(parents=True)
        source_file = source_dir / "event_sequences.parquet"
        
        df = pl.DataFrame({"sequence_ids": [[1, 2, 3]], "label": [0]})
        df.write_parquet(source_file)
        
        # Create interim directory
        interim_dir = tmp_path / "data" / "interim"
        interim_dir.mkdir(parents=True)
        output_file = interim_dir / "ds2_loghub.parquet"
        
        from dataset_2_loghub.scripts import export_to_interim
        
        with patch.object(export_to_interim, "INPUT_PATH", source_file):
            with patch.object(export_to_interim, "OUTPUT_PATH", output_file):
                export_to_interim.export_to_interim()
        
        assert output_file.exists()
        result = pl.read_parquet(output_file)
        assert result.height == 1

    @pytest.mark.unit
    def test_ds3_export_copies_file(self, tmp_path, sample_format_b_records):
        """DS3 export copies JSONL file to interim directory."""
        source_dir = tmp_path / "dataset_3_stackoverflow" / "data" / "training"
        source_dir.mkdir(parents=True)
        source_file = source_dir / "training_data.jsonl"
        
        with open(source_file, "w", encoding="utf-8") as f:
            for record in sample_format_b_records:
                f.write(json.dumps(record) + "\n")
        
        interim_dir = tmp_path / "data" / "interim"
        interim_dir.mkdir(parents=True)
        output_file = interim_dir / "ds3_stackoverflow.jsonl"
        
        from dataset_3_stackoverflow.scripts import export_to_interim
        
        with patch.object(export_to_interim, "INPUT_PATH", source_file):
            with patch.object(export_to_interim, "OUTPUT_PATH", output_file):
                export_to_interim.export_to_interim()
        
        assert output_file.exists()

    @pytest.mark.unit
    def test_ds4_export_copies_file(self, tmp_path, sample_format_b_records):
        """DS4 export copies JSONL file to interim directory."""
        source_dir = tmp_path / "dataset_4_synthetic" / "data" / "processed"
        source_dir.mkdir(parents=True)
        source_file = source_dir / "dataset4_format_b.jsonl"
        
        with open(source_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_format_b_records[0]) + "\n")
        
        interim_dir = tmp_path / "data" / "interim"
        interim_dir.mkdir(parents=True)
        output_file = interim_dir / "ds4_synthetic.jsonl"
        
        from dataset_4_synthetic.scripts import export_to_interim
        
        with patch.object(export_to_interim, "INPUT_PATH", source_file):
            with patch.object(export_to_interim, "OUTPUT_PATH", output_file):
                export_to_interim.export_to_interim()
        
        assert output_file.exists()

    @pytest.mark.unit
    def test_ds5_export_copies_file(self, tmp_path, sample_format_b_records):
        """DS5 export copies JSONL file to interim directory."""
        source_dir = tmp_path / "dataset_5_glaive" / "data" / "processed"
        source_dir.mkdir(parents=True)
        source_file = source_dir / "glaive_chatml.jsonl"
        
        with open(source_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_format_b_records[0]) + "\n")
        
        interim_dir = tmp_path / "data" / "interim"
        interim_dir.mkdir(parents=True)
        output_file = interim_dir / "ds5_glaive.jsonl"
        
        from dataset_5_glaive.scripts import export_to_interim
        
        with patch.object(export_to_interim, "INPUT_PATH", source_file):
            with patch.object(export_to_interim, "OUTPUT_PATH", output_file):
                export_to_interim.export_to_interim()
        
        assert output_file.exists()

    @pytest.mark.unit
    def test_ds6_export_copies_file(self, tmp_path, sample_format_b_records):
        """DS6 export copies JSONL file to interim directory."""
        source_dir = tmp_path / "dataset_6_the_stack" / "data" / "processed"
        source_dir.mkdir(parents=True)
        source_file = source_dir / "training_records.jsonl"
        
        with open(source_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_format_b_records[0]) + "\n")
        
        interim_dir = tmp_path / "data" / "interim"
        interim_dir.mkdir(parents=True)
        output_file = interim_dir / "ds6_the_stack.jsonl"
        
        from dataset_6_the_stack.scripts import export_to_interim
        
        with patch.object(export_to_interim, "INPUT_PATH", source_file):
            with patch.object(export_to_interim, "OUTPUT_PATH", output_file):
                export_to_interim.export_to_interim()
        
        assert output_file.exists()

    @pytest.mark.unit
    def test_export_raises_on_missing_source(self, tmp_path):
        """Export raises FileNotFoundError if source doesn't exist."""
        nonexistent = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "output.jsonl"
        
        from dataset_3_stackoverflow.scripts import export_to_interim
        
        with patch.object(export_to_interim, "INPUT_PATH", nonexistent):
            with patch.object(export_to_interim, "OUTPUT_PATH", output_file):
                with pytest.raises(FileNotFoundError):
                    export_to_interim.export_to_interim()
