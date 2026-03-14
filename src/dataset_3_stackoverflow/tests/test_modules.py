"""
Test Modules for AutoMend Pipeline
===================================
Comprehensive unit tests for all pipeline components.
Run with: pytest tests/ -v --cov=scripts
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from data_acquisition import (
    clean_html, extract_error_signatures, extract_infra_components,
    RateLimitedRequester
)
from data_preprocessing import (
    normalize_whitespace, extract_code_blocks, clean_text,
    calculate_text_metrics, calculate_quality_score, classify_complexity,
    detect_question_type, preprocess_single_record
)
from data_validation import (
    ValidationResult, validate_schema, validate_data_quality,
    EXPECTED_SCHEMA
)
from bias_detection import (
    create_slices_by_column, calculate_slice_metrics, detect_bias
)
import polars as pl


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_qa_record():
    """Sample Q&A record for testing."""
    return {
        "question_id": 12345,
        "title": "How to fix Kubernetes CrashLoopBackOff error?",
        "question_body": "My pod keeps crashing with CrashLoopBackOff. "
                        "I've checked the logs but can't find the issue. "
                        "Running on GKE with 2GB memory limit.",
        "answer_body": "CrashLoopBackOff usually means your container is "
                      "failing to start. Check these steps:\n"
                      "1. Run `kubectl logs <pod-name>` to see error messages\n"
                      "2. Check if your memory limit is sufficient\n"
                      "3. Verify your container's entrypoint is correct\n"
                      "```bash\nkubectl describe pod <pod-name>\n```",
        "tags": ["kubernetes", "docker", "gke"],
        "score": 15,
        "answer_score": 25,
        "view_count": 5000,
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for validation tests."""
    # Create 100 UNIQUE records to pass duplicate check
    records = []
    for i in range(100):
        records.append({
            "question_id": i + 1,  # Unique IDs
            "title": f"Test question {i + 1}",
            "question_body": f"This is test question {i + 1} about Kubernetes pods and troubleshooting.",
            "answer_body": f"Here is a detailed answer {i + 1} about fixing pod issues with comprehensive steps.",
            "tags": ["kubernetes"] if i % 2 == 0 else ["terraform"],
            "score": 10 + (i % 20),
            "answer_score": 15 + (i % 15),
            "quality_score": 25.0 + (i % 10),
            "error_signatures": ["CrashLoopBackOff"] if i % 3 == 0 else [],
            "infra_components": ["kubernetes"] if i % 2 == 0 else ["terraform"],
            "question_type": "troubleshooting" if i % 2 == 0 else "how_to",
            "complexity": "moderate" if i % 3 == 0 else "simple",
        })
    return pl.DataFrame(records)


# =============================================================================
# Data Acquisition Tests
# =============================================================================

class TestDataAcquisition:
    """Tests for data acquisition module."""
    
    def test_clean_html_removes_tags(self):
        """Test HTML tag removal."""
        html = "<p>Hello <strong>World</strong></p>"
        result = clean_html(html)
        assert "<" not in result
        assert ">" not in result
        assert "Hello" in result
        assert "World" in result
    
    def test_clean_html_handles_empty(self):
        """Test empty input handling."""
        assert clean_html("") == ""
        assert clean_html(None) == ""
    
    def test_clean_html_decodes_entities(self):
        """Test HTML entity decoding."""
        html = "&lt;code&gt; &amp; &quot;test&quot;"
        result = clean_html(html)
        assert "&lt;" not in result
        assert "&amp;" not in result
    
    def test_extract_error_signatures_finds_patterns(self):
        """Test error signature extraction."""
        text = "Pod is in CrashLoopBackOff state due to OOMKilled"
        signatures = extract_error_signatures(text)
        assert "CrashLoopBackOff" in signatures
        assert "OOMKilled" in signatures
    
    def test_extract_error_signatures_empty(self):
        """Test with no errors present."""
        text = "Everything is working fine"
        signatures = extract_error_signatures(text)
        assert len(signatures) == 0
    
    def test_extract_infra_components(self):
        """Test infrastructure component detection."""
        text = "Deploy using kubectl to kubernetes cluster with terraform"
        components = extract_infra_components(text)
        assert "kubernetes" in components
        assert "terraform" in components
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        requester = RateLimitedRequester()
        assert requester.consecutive_failures == 0
        assert requester.total_requests == 0


# =============================================================================
# Data Preprocessing Tests
# =============================================================================

class TestDataPreprocessing:
    """Tests for data preprocessing module."""
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    World\n\nTest"
        result = normalize_whitespace(text)
        assert result == "Hello World Test"
    
    def test_extract_code_blocks(self):
        """Test code block extraction."""
        text = "Use this code: ```python\nprint('hello')\n``` end"
        clean_text, code_blocks = extract_code_blocks(text)
        assert len(code_blocks) == 1
        assert "print" in code_blocks[0]
        assert "[CODE]" in clean_text
    
    def test_clean_text_removes_urls(self):
        """Test URL removal."""
        text = "Check https://example.com for more info"
        result = clean_text(text)
        assert "https://" not in result
        assert "[URL]" in result
    
    def test_calculate_text_metrics(self):
        """Test text metrics calculation."""
        text = "Hello world. How are you? I am fine."
        metrics = calculate_text_metrics(text)
        assert metrics["word_count"] == 8
        assert metrics["sentence_count"] == 3
        assert metrics["char_count"] > 0
    
    def test_calculate_text_metrics_empty(self):
        """Test metrics for empty text."""
        metrics = calculate_text_metrics("")
        assert metrics["word_count"] == 0
        assert metrics["char_count"] == 0
    
    def test_calculate_quality_score(self, sample_qa_record):
        """Test quality score calculation."""
        score = calculate_quality_score(sample_qa_record)
        assert score > 0
        assert isinstance(score, float)
    
    def test_classify_complexity_simple(self):
        """Test simple complexity classification."""
        text = "Short text without code"
        assert classify_complexity(text) == "simple"
    
    def test_classify_complexity_complex(self):
        """Test complex text classification."""
        text = "A " * 300 + "```code```" * 3  # Long with code
        assert classify_complexity(text) == "complex"
    
    def test_detect_question_type_how_to(self):
        """Test how-to question detection."""
        assert detect_question_type("How do I configure X?") == "how_to"
    
    def test_detect_question_type_troubleshooting(self):
        """Test troubleshooting question detection."""
        # "How to fix" matches "how_to" pattern first, so use a clearer troubleshooting phrase
        assert detect_question_type("I need to fix this crash") == "troubleshooting"
        assert detect_question_type("Debug this error please") == "troubleshooting"
    
    def test_detect_question_type_explanation(self):
        """Test explanation question detection."""
        assert detect_question_type("Why is this happening?") == "explanation"
    
    def test_preprocess_single_record_valid(self, sample_qa_record):
        """Test preprocessing a valid record."""
        result = preprocess_single_record(sample_qa_record)
        assert result is not None
        assert "question_id" in result
        assert "error_signatures" in result
        assert "quality_score" in result
    
    def test_preprocess_single_record_missing_fields(self):
        """Test preprocessing rejects incomplete records."""
        invalid_record = {"question_id": 1}
        result = preprocess_single_record(invalid_record)
        assert result is None
    
    def test_preprocess_single_record_short_content(self):
        """Test preprocessing rejects short content."""
        record = {
            "question_id": 1,
            "question_body": "Hi",
            "answer_body": "Hello"
        }
        result = preprocess_single_record(record)
        assert result is None


# =============================================================================
# Data Validation Tests
# =============================================================================

class TestDataValidation:
    """Tests for data validation module."""
    
    def test_validation_result_tracking(self):
        """Test ValidationResult tracks checks correctly."""
        result = ValidationResult()
        result.add_pass("test1", "Passed")
        result.add_failure("test2", "Failed")
        result.add_warning("test3", "Warning")
        
        assert len(result.passed) == 1
        assert len(result.failed) == 1
        assert len(result.warnings) == 1
    
    def test_validation_result_is_valid(self):
        """Test is_valid property."""
        result = ValidationResult()
        result.add_pass("test", "OK")
        assert result.is_valid is True
        
        result.add_failure("test2", "Error", severity="error")
        assert result.is_valid is False
    
    def test_validate_schema_all_columns(self, sample_dataframe):
        """Test schema validation with all required columns."""
        result = ValidationResult()
        validate_schema(sample_dataframe, result)
        
        # Check that required columns pass
        passes = [p["check"] for p in result.passed]
        assert "schema_required_columns" in passes
    
    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        df = pl.DataFrame({"question_id": [1, 2]})
        result = ValidationResult()
        validate_schema(df, result)
        
        failures = [f["check"] for f in result.failed]
        assert "schema_required_columns" in failures
    
    def test_validate_data_quality_passes(self, sample_dataframe):
        """Test data quality validation passes for good data."""
        result = ValidationResult()
        validate_data_quality(sample_dataframe, result)
        
        # Should pass minimum rows check
        passes = [p["check"] for p in result.passed]
        assert "min_rows" in passes
    
    def test_validate_data_quality_fails_low_rows(self):
        """Test data quality fails for insufficient rows."""
        df = pl.DataFrame({
            "question_id": [1],
            "question_body": ["test"],
            "answer_body": ["answer"]
        })
        result = ValidationResult()
        validate_data_quality(df, result)
        
        failures = [f["check"] for f in result.failed]
        assert "min_rows" in failures


# =============================================================================
# Bias Detection Tests
# =============================================================================

class TestBiasDetection:
    """Tests for bias detection module."""
    
    def test_create_slices_by_column(self, sample_dataframe):
        """Test slice creation by column."""
        slices = create_slices_by_column(sample_dataframe, "question_type", min_samples=10)
        assert len(slices) > 0
        assert "troubleshooting" in slices or "how_to" in slices
    
    def test_create_slices_minimum_samples(self, sample_dataframe):
        """Test that slices below minimum are excluded."""
        # Add a rare category - modify first row to have rare_type
        df = sample_dataframe.with_row_index("_idx").with_columns(
            pl.when(pl.col("_idx") == 0).then(pl.lit("rare_type")).otherwise(pl.col("question_type")).alias("question_type")
        ).drop("_idx")
        
        slices = create_slices_by_column(df, "question_type", min_samples=100)
        assert "rare_type" not in slices
    
    def test_calculate_slice_metrics(self, sample_dataframe):
        """Test slice metrics calculation."""
        metrics = calculate_slice_metrics(sample_dataframe, sample_dataframe)
        
        assert "count" in metrics
        assert "proportion" in metrics
        assert metrics["proportion"] == 1.0  # Same as overall
    
    def test_detect_bias_no_bias(self):
        """Test bias detection with no bias."""
        slice_metrics = {"quality_score": {"mean": 10.0}, "proportion": 0.5}
        overall_metrics = {"quality_score": {"mean": 10.0}}
        
        result = detect_bias(slice_metrics, overall_metrics)
        assert result["is_biased"] is False
    
    def test_detect_bias_with_bias(self):
        """Test bias detection identifies bias."""
        # Need a larger difference to trigger bias (threshold is now 25%)
        slice_metrics = {"quality_score": {"mean": 5.0}, "proportion": 0.5, "count": 100}
        overall_metrics = {"quality_score": {"mean": 15.0}}  # 66% difference
        
        result = detect_bias(slice_metrics, overall_metrics, threshold=0.25)
        assert result["is_biased"] is True
        assert len(result["bias_factors"]) > 0
    
    def test_detect_bias_underrepresentation(self):
        """Test underrepresentation detection."""
        # Must be BOTH low proportion AND low count
        slice_metrics = {"quality_score": {"mean": 10.0}, "proportion": 0.003, "count": 10}
        overall_metrics = {"quality_score": {"mean": 10.0}}
        
        result = detect_bias(slice_metrics, overall_metrics)
        assert result["is_biased"] is True
        
        factors = [f["factor"] for f in result["bias_factors"]]
        assert "underrepresentation" in factors


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pl.DataFrame()
        result = ValidationResult()
        validate_schema(df, result)
        
        # Should fail on missing columns
        assert len(result.failed) > 0
    
    def test_unicode_handling(self):
        """Test Unicode text handling."""
        text = "Hello 世界! Привет мир! 🚀"
        result = clean_text(text)
        assert "世界" in result or "Hello" in result
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        text = "word " * 10000
        metrics = calculate_text_metrics(text)
        assert metrics["word_count"] == 10000
    
    def test_special_characters(self):
        """Test handling of special characters."""
        text = "Test @#$%^&*() special chars"
        result = clean_text(text)
        assert "Test" in result
    
    def test_none_values_in_dataframe(self):
        """Test handling of None values."""
        df = pl.DataFrame({
            "question_id": [1, 2, None],
            "question_body": ["test", None, "test2"],
            "answer_body": ["ans", "ans2", "ans3"]
        })
        result = ValidationResult()
        validate_data_quality(df, result)
        
        # Should detect missing values
        assert any("missing" in str(w).lower() for w in result.warnings + result.failed)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for pipeline components."""
    
    def test_full_preprocessing_flow(self, sample_qa_record):
        """Test complete preprocessing of a record."""
        # Preprocess
        result = preprocess_single_record(sample_qa_record)
        
        # Verify all expected fields
        assert result is not None
        assert "question_id" in result
        assert "error_signatures" in result
        assert "infra_components" in result
        assert "quality_score" in result
        assert "question_type" in result
        assert "complexity" in result
        
        # Verify error detection worked
        assert "CrashLoopBackOff" in result["error_signatures"]
        
        # Verify infra detection worked
        assert "kubernetes" in result["infra_components"]
    
    def test_validation_to_bias_flow(self, sample_dataframe):
        """Test flow from validation to bias detection."""
        # Validate
        val_result = ValidationResult()
        validate_schema(sample_dataframe, val_result)
        validate_data_quality(sample_dataframe, val_result)
        
        assert val_result.is_valid
        
        # Run bias detection
        slices = create_slices_by_column(sample_dataframe, "question_type", min_samples=10)
        
        overall_metrics = calculate_slice_metrics(sample_dataframe, sample_dataframe)
        
        for slice_name, slice_df in slices.items():
            slice_metrics = calculate_slice_metrics(slice_df, sample_dataframe)
            bias_result = detect_bias(slice_metrics, overall_metrics)
            
            # Verify bias detection returns expected structure
            assert "is_biased" in bias_result
            assert "severity" in bias_result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])