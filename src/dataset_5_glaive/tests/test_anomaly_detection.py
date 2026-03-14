"""
Unit tests for anomaly detection pipeline.
"""

import pytest
import polars as pl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from anomaly_detection import (
    check_malformed_rate,
    check_none_complexity_rate,
    check_record_count,
    check_avg_turns,
    check_avg_calls,
    check_defined_functions_coverage,
    THRESHOLDS,
)


@pytest.fixture
def healthy_df():
    """Fixture providing a healthy DataFrame with no anomalies."""
    return pl.DataFrame({
        "has_malformed":         [False] * 4500 + [True] * 100,
        "complexity_tier":       ["simple"] * 2000 + ["none"] * 1500 + ["complex"] * 1000 + ["moderate"] * 100,
        "num_turns":             [2] * 4600,
        "num_calls":             [1] * 4600,
        "num_defined_functions": [1] * 2500 + [0] * 2100,
    })


@pytest.fixture
def anomalous_df():
    """Fixture providing a DataFrame with anomalies."""
    return pl.DataFrame({
        "has_malformed":         [True] * 500 + [False] * 500,
        "complexity_tier":       ["none"] * 900 + ["simple"] * 100,
        "num_turns":             [15] * 1000,
        "num_calls":             [8] * 1000,
        "num_defined_functions": [0] * 1000,
    })


class TestCheckMalformedRate:

    def test_healthy_df_passes(self, healthy_df):
        result = check_malformed_rate(healthy_df)
        assert result["is_anomaly"]  == False 

    def test_anomalous_df_fails(self, anomalous_df):
        result = check_malformed_rate(anomalous_df)
        assert result["is_anomaly"]  == True

    def test_result_has_required_keys(self, healthy_df):
        result = check_malformed_rate(healthy_df)
        assert "check" in result
        assert "value" in result
        assert "threshold" in result
        assert "is_anomaly" in result


class TestCheckRecordCount:

    def test_sufficient_records_passes(self, healthy_df):
        result = check_record_count(healthy_df)
        assert result["is_anomaly"]  == False 

    def test_insufficient_records_fails(self):
        small_df = pl.DataFrame({"has_malformed": [False] * 100})
        result = check_record_count(small_df)
        assert result["is_anomaly"]  == True


class TestCheckAvgTurns:

    def test_normal_turns_passes(self, healthy_df):
        result = check_avg_turns(healthy_df)
        assert result["is_anomaly"]  == False 

    def test_high_turns_fails(self, anomalous_df):
        result = check_avg_turns(anomalous_df)
        assert result["is_anomaly"]  == True


class TestCheckAvgCalls:

    def test_normal_calls_passes(self, healthy_df):
        result = check_avg_calls(healthy_df)
        assert result["is_anomaly"]  == False 

    def test_high_calls_fails(self, anomalous_df):
        result = check_avg_calls(anomalous_df)
        assert result["is_anomaly"]  == True


class TestCheckDefinedFunctionsCoverage:

    def test_sufficient_coverage_passes(self, healthy_df):
        result = check_defined_functions_coverage(healthy_df)
        assert result["is_anomaly"]  == False 

    def test_insufficient_coverage_fails(self, anomalous_df):
        result = check_defined_functions_coverage(anomalous_df)
        assert result["is_anomaly"]  == True