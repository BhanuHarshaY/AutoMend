"""
Unit tests for bias detection pipeline.
"""

import pytest
import polars as pl
import sys
from pathlib import Path
import importlib.util

DS5_ROOT = Path(__file__).resolve().parent.parent
DS5_SCRIPTS = DS5_ROOT / "scripts"
sys.path.insert(0, str(DS5_SCRIPTS))

spec = importlib.util.spec_from_file_location("glaive_bias_detection", DS5_SCRIPTS / "bias_detection.py")
glaive_bias_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(glaive_bias_detection)
add_slice_features = glaive_bias_detection.add_slice_features
analyze_slice = glaive_bias_detection.analyze_slice
detect_representation_bias = glaive_bias_detection.detect_representation_bias
suggest_mitigation = glaive_bias_detection.suggest_mitigation


@pytest.fixture
def sample_df():
    """Fixture providing a representative processed DataFrame."""
    return pl.DataFrame({
        "num_turns":             [1, 2, 3, 4, 5, 6, 7, 2, 1, 3] * 100,
        "num_calls":             [0, 1, 2, 3, 0, 1, 2, 0, 1, 0] * 100,
        "complexity_tier":       ["none", "simple", "complex", "moderate",
                                  "none", "simple", "complex", "none",
                                  "simple", "none"] * 100,
        "has_malformed":         [False] * 950 + [True] * 50,
        "has_error_handling":    [False] * 900 + [True] * 100,
        "has_parallel":          [False] * 900 + [True] * 100,
        "num_defined_functions": [0] * 400 + [1] * 600,
    })


class TestAddSliceFeatures:

    def test_adds_turn_bucket(self, sample_df):
        """Test turn_bucket column is added."""
        result = add_slice_features(sample_df)
        assert "turn_bucket" in result.columns

    def test_adds_call_bucket(self, sample_df):
        """Test call_bucket column is added."""
        result = add_slice_features(sample_df)
        assert "call_bucket" in result.columns

    def test_adds_has_defined_functions(self, sample_df):
        """Test has_defined_functions column is added."""
        result = add_slice_features(sample_df)
        assert "has_defined_functions" in result.columns

    def test_turn_bucket_values(self, sample_df):
        """Test turn_bucket contains expected category values."""
        result = add_slice_features(sample_df)
        valid_buckets = {"single", "short", "medium", "long"}
        actual = set(result["turn_bucket"].drop_nulls().unique().to_list())
        assert actual.issubset(valid_buckets)

    def test_call_bucket_values(self, sample_df):
        """Test call_bucket contains expected category values."""
        result = add_slice_features(sample_df)
        valid_buckets = {"no_calls", "one_call", "two_calls", "many_calls"}
        actual = set(result["call_bucket"].drop_nulls().unique().to_list())
        assert actual.issubset(valid_buckets)

    def test_original_df_not_modified(self, sample_df):
        """Test original DataFrame is not modified."""
        original_cols = list(sample_df.columns)
        add_slice_features(sample_df)
        assert list(sample_df.columns) == original_cols


class TestAnalyzeSlice:

    def test_returns_dataframe(self, sample_df):
        """Test analyze_slice returns a DataFrame."""
        result = analyze_slice(sample_df, "complexity_tier")
        assert isinstance(result, pl.DataFrame)

    def test_has_required_columns(self, sample_df):
        """Test result has all required columns."""
        result = analyze_slice(sample_df, "complexity_tier")
        required = [
            "slice_column", "slice_value", "count",
            "proportion", "avg_turns", "avg_calls",
            "is_underrepresented"
        ]
        for col in required:
            assert col in result.columns

    def test_proportions_sum_to_one(self, sample_df):
        """Test proportions across all slices sum to 1."""
        result = analyze_slice(sample_df, "complexity_tier")
        assert abs(result["proportion"].sum() - 1.0) < 0.01

    def test_count_matches_total(self, sample_df):
        """Test counts across slices sum to total records."""
        result = analyze_slice(sample_df, "complexity_tier")
        assert result["count"].sum() == sample_df.height

    def test_underrepresented_flagged_correctly(self, sample_df):
        """Test slices below threshold are flagged as underrepresented."""
        result = analyze_slice(sample_df, "complexity_tier")
        for row in result.iter_rows(named=True):
            if row["proportion"] < 0.05:
                assert row["is_underrepresented"] == True
            else:
                assert row["is_underrepresented"] == False


class TestDetectRepresentationBias:

    def test_returns_list(self, sample_df):
        """Test returns a list."""
        df_with_slices = add_slice_features(sample_df)
        slice_df = analyze_slice(df_with_slices, "complexity_tier")
        findings = detect_representation_bias(slice_df)
        assert isinstance(findings, list)

    def test_finding_has_required_keys(self, sample_df):
        """Test each finding has required keys."""
        df_with_slices = add_slice_features(sample_df)
        slice_df = analyze_slice(df_with_slices, "complexity_tier")
        findings = detect_representation_bias(slice_df)
        for finding in findings:
            assert "slice_column" in finding
            assert "slice_value" in finding
            assert "proportion" in finding
            assert "severity" in finding
            assert "recommendation" in finding

    def test_high_severity_for_very_small_slices(self):
        """Test very small slices get high severity."""
        slice_df = pl.DataFrame([{
            "slice_column":        "test_col",
            "slice_value":         "rare_value",
            "proportion":          0.005,
            "count":               5,
            "is_underrepresented": True,
            "avg_turns":           1.0,
            "avg_calls":           0.0,
        }])
        findings = detect_representation_bias(slice_df)
        assert findings[0]["severity"] == "high"

    def test_medium_severity_for_medium_slices(self):
        """Test medium slices get medium severity."""
        slice_df = pl.DataFrame([{
            "slice_column":        "test_col",
            "slice_value":         "medium_value",
            "proportion":          0.03,
            "count":               30,
            "is_underrepresented": True,
            "avg_turns":           2.0,
            "avg_calls":           1.0,
        }])
        findings = detect_representation_bias(slice_df)
        assert findings[0]["severity"] == "medium"

    def test_no_findings_for_balanced_data(self):
        """Test balanced data returns no findings."""
        slice_df = pl.DataFrame([
            {
                "slice_column": "test", "slice_value": "a",
                "proportion": 0.5, "count": 500,
                "is_underrepresented": False,
                "avg_turns": 2.0, "avg_calls": 1.0,
            },
            {
                "slice_column": "test", "slice_value": "b",
                "proportion": 0.5, "count": 500,
                "is_underrepresented": False,
                "avg_turns": 2.0, "avg_calls": 1.0,
            },
        ])
        findings = detect_representation_bias(slice_df)
        assert findings == []


class TestSuggestMitigation:

    def test_returns_list(self):
        """Test returns a list."""
        findings = [{
            "slice_column": "complexity_tier",
            "slice_value":  "malformed",
            "proportion":   0.001,
            "count":        5,
            "severity":     "high",
            "recommendation": "oversample"
        }]
        mitigations = suggest_mitigation(findings)
        assert isinstance(mitigations, list)

    def test_high_severity_gets_oversample(self):
        """Test high severity findings get oversample strategy."""
        findings = [{
            "slice_column": "test",
            "slice_value":  "rare",
            "proportion":   0.001,
            "count":        5,
            "severity":     "high",
            "recommendation": "test"
        }]
        mitigations = suggest_mitigation(findings)
        assert mitigations[0]["strategy"] == "oversample"

    def test_medium_severity_gets_collect_more(self):
        """Test medium severity findings get collect_more strategy."""
        findings = [{
            "slice_column": "test",
            "slice_value":  "medium",
            "proportion":   0.03,
            "count":        30,
            "severity":     "medium",
            "recommendation": "test"
        }]
        mitigations = suggest_mitigation(findings)
        assert mitigations[0]["strategy"] == "collect_more"

    def test_mitigation_has_required_keys(self):
        """Test each mitigation has required keys."""
        findings = [{
            "slice_column": "test",
            "slice_value":  "rare",
            "proportion":   0.001,
            "count":        5,
            "severity":     "high",
            "recommendation": "test"
        }]
        mitigations = suggest_mitigation(findings)
        assert "slice_column" in mitigations[0]
        assert "slice_value" in mitigations[0]
        assert "strategy" in mitigations[0]
        assert "detail" in mitigations[0]

    def test_empty_findings_returns_empty(self):
        """Test empty findings returns empty mitigations."""
        mitigations = suggest_mitigation([])
        assert mitigations == []