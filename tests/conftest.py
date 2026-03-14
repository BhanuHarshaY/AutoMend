"""
Shared pytest fixtures for Automend MLOps Monorepo tests.

Provides:
- Temporary directories for interim and processed data
- Sample Parquet files for Track A (Format A schema)
- Sample JSONL files for Track B (ChatML schema)
- Project path fixtures
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import polars as pl
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Add each dataset directory to path so their tests can find their modules
for dataset_dir in (PROJECT_ROOT / "src").glob("dataset_*"):
    sys.path.insert(0, str(dataset_dir))


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def src_dir(project_root):
    """Return the src directory."""
    return project_root / "src"


@pytest.fixture
def dags_dir(project_root):
    """Return the dags directory."""
    return project_root / "dags"


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "interim").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    return data_dir


@pytest.fixture
def tmp_interim_dir(tmp_data_dir):
    """Return temporary interim directory."""
    return tmp_data_dir / "interim"


@pytest.fixture
def tmp_processed_dir(tmp_data_dir):
    """Return temporary processed directory."""
    return tmp_data_dir / "processed"


# =============================================================================
# Format A Sample Data (Track A - Parquet)
# =============================================================================

@pytest.fixture
def sample_format_a_records():
    """Sample records in Format A schema."""
    return [
        {"sequence_ids": [100, 200, 101, 201, 102, 202], "label": 0},
        {"sequence_ids": [105, 207, 106, 208, 107, 209], "label": 1},
        {"sequence_ids": [102, 202, 301, 103, 203], "label": 2},
        {"sequence_ids": [400, 104, 204, 401, 105, 205], "label": 3},
        {"sequence_ids": [103, 203, 302, 104, 204], "label": 4},
    ]


@pytest.fixture
def sample_parquet_ds1(tmp_interim_dir, sample_format_a_records):
    """Create sample Parquet file for Dataset 1 (Alibaba)."""
    df = pl.DataFrame(sample_format_a_records[:3])
    path = tmp_interim_dir / "ds1_alibaba.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def sample_parquet_ds2(tmp_interim_dir, sample_format_a_records):
    """Create sample Parquet file for Dataset 2 (Loghub)."""
    df = pl.DataFrame(sample_format_a_records[2:])
    path = tmp_interim_dir / "ds2_loghub.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def sample_track_a_files(sample_parquet_ds1, sample_parquet_ds2):
    """Return both Track A sample files."""
    return [sample_parquet_ds1, sample_parquet_ds2]


# =============================================================================
# Format B Sample Data (Track B - JSONL ChatML)
# =============================================================================

@pytest.fixture
def sample_format_b_records():
    """Sample records in Format B (ChatML) schema."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are an MLOps expert."},
                {"role": "user", "content": "How do I fix OOM errors in Kubernetes?"},
                {"role": "assistant", "content": "To fix OOM errors, increase memory limits..."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a DevOps assistant."},
                {"role": "user", "content": "Write a Terraform module for AWS S3."},
                {"role": "assistant", "content": "resource \"aws_s3_bucket\" \"main\" {...}"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain Docker networking."},
                {"role": "assistant", "content": "Docker provides several network drivers..."}
            ]
        },
    ]


@pytest.fixture
def sample_jsonl_ds3(tmp_interim_dir, sample_format_b_records):
    """Create sample JSONL file for Dataset 3 (StackOverflow)."""
    path = tmp_interim_dir / "ds3_stackoverflow.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for record in sample_format_b_records[:2]:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def sample_jsonl_ds4(tmp_interim_dir, sample_format_b_records):
    """Create sample JSONL file for Dataset 4 (Synthetic)."""
    path = tmp_interim_dir / "ds4_synthetic.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for record in sample_format_b_records[1:]:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def sample_jsonl_ds5(tmp_interim_dir, sample_format_b_records):
    """Create sample JSONL file for Dataset 5 (Glaive)."""
    path = tmp_interim_dir / "ds5_glaive.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_format_b_records[0]) + "\n")
    return path


@pytest.fixture
def sample_jsonl_ds6(tmp_interim_dir, sample_format_b_records):
    """Create sample JSONL file for Dataset 6 (The Stack)."""
    path = tmp_interim_dir / "ds6_the_stack.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_format_b_records[2]) + "\n")
    return path


@pytest.fixture
def sample_track_b_files(sample_jsonl_ds3, sample_jsonl_ds4, sample_jsonl_ds5, sample_jsonl_ds6):
    """Return all Track B sample files."""
    return [sample_jsonl_ds3, sample_jsonl_ds4, sample_jsonl_ds5, sample_jsonl_ds6]


# =============================================================================
# Malformed Data Fixtures (for error handling tests)
# =============================================================================

@pytest.fixture
def malformed_jsonl_file(tmp_interim_dir):
    """Create a JSONL file with some malformed lines."""
    path = tmp_interim_dir / "malformed.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"messages": [{"role": "user", "content": "valid"}]}\n')
        f.write('this is not valid json\n')
        f.write('{"messages": [{"role": "assistant", "content": "also valid"}]}\n')
        f.write('{"incomplete": \n')
    return path


@pytest.fixture
def empty_parquet_file(tmp_interim_dir):
    """Create an empty Parquet file."""
    path = tmp_interim_dir / "empty.parquet"
    df = pl.DataFrame(schema={"sequence_ids": pl.List(pl.Int64), "label": pl.Int64})
    df.write_parquet(path)
    return path


# =============================================================================
# DVC Config Fixtures
# =============================================================================

@pytest.fixture
def combiner_track_a_dir(src_dir):
    """Return the combiner_track_a directory."""
    return src_dir / "combiner_track_a"


@pytest.fixture
def combiner_track_b_dir(src_dir):
    """Return the combiner_track_b directory."""
    return src_dir / "combiner_track_b"
