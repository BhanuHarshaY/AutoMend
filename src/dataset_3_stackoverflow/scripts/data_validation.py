"""
Data Validation Module
=======================
Validates data quality, generates schema and statistics using
Polars-native validation and custom validation rules.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import Counter

import polars as pl
import polars.selectors as cs

from config import config

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.polars_validation import (
    validate_columns_present,
    validate_no_nulls,
    validate_row_count,
    run_validation_suite,
)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'validation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Schema Definition
# =============================================================================

EXPECTED_SCHEMA = {
    "required_columns": [
        "question_id", "title", "question_body", "answer_body",
        "tags", "score", "quality_score"
    ],
    "column_types": {
        "question_id": "int",
        "title": "str",
        "question_body": "str",
        "answer_body": "str",
        "tags": "list",
        "score": "int",
        "answer_score": "int",
        "view_count": "int",
        "quality_score": "float",
        "error_signatures": "list",
        "infra_components": "list",
        "question_type": "str",
        "complexity": "str",
    },
    "constraints": {
        "question_id": {"min": 1},
        "score": {"min": 0},
        "quality_score": {"min": 0},
        "question_body": {"min_length": 10},
        "answer_body": {"min_length": 20},
    }
}


# =============================================================================
# Validation Functions
# =============================================================================

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        self.statistics = {}
    
    def add_pass(self, check_name: str, message: str):
        self.passed.append({"check": check_name, "message": message})
        logger.info(f"[PASS] {check_name} - {message}")
    
    def add_failure(self, check_name: str, message: str, severity: str = "error"):
        self.failed.append({"check": check_name, "message": message, "severity": severity})
        logger.error(f"[FAIL] {check_name} - {message}")
    
    def add_warning(self, check_name: str, message: str):
        self.warnings.append({"check": check_name, "message": message})
        logger.warning(f"[WARN] {check_name} - {message}")
    
    @property
    def is_valid(self) -> bool:
        """Returns True if no critical failures."""
        critical_failures = [f for f in self.failed if f.get("severity") == "error"]
        return len(critical_failures) == 0
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "total_checks": len(self.passed) + len(self.failed),
            "passed_count": len(self.passed),
            "failed_count": len(self.failed),
            "warning_count": len(self.warnings),
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "statistics": self.statistics,
        }


def validate_schema(df: pl.DataFrame, result: ValidationResult) -> None:
    """Validate DataFrame schema against expected schema."""
    
    missing_cols = set(EXPECTED_SCHEMA["required_columns"]) - set(df.columns)
    if missing_cols:
        result.add_failure("schema_required_columns",
                          f"Missing required columns: {missing_cols}")
    else:
        result.add_pass("schema_required_columns", "All required columns present")
    
    expected_cols = set(EXPECTED_SCHEMA["column_types"].keys())
    extra_cols = set(df.columns) - expected_cols - {"processed_at", "question_metrics", "answer_metrics"}
    if extra_cols:
        result.add_warning("schema_extra_columns", f"Unexpected columns found: {extra_cols}")


def validate_data_quality(df: pl.DataFrame, result: ValidationResult) -> None:
    """Validate data quality metrics."""
    
    total_rows = df.height
    result.statistics["total_rows"] = total_rows
    
    if total_rows < config.MIN_ROWS:
        result.add_failure("min_rows",
                          f"Only {total_rows} rows, minimum required: {config.MIN_ROWS}")
    else:
        result.add_pass("min_rows", f"Row count ({total_rows}) meets minimum ({config.MIN_ROWS})")
    
    critical_cols = ["question_id", "question_body", "answer_body"]
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].null_count()
            missing_ratio = missing / total_rows
            result.statistics[f"{col}_missing_ratio"] = missing_ratio
            
            if missing_ratio > config.MAX_MISSING_RATIO:
                result.add_failure("missing_values",
                                  f"Column '{col}' has {missing_ratio:.1%} missing values")
            elif missing > 0:
                result.add_warning("missing_values",
                                  f"Column '{col}' has {missing} missing values ({missing_ratio:.1%})")
            else:
                result.add_pass("missing_values", f"Column '{col}' has no missing values")
    
    if "question_id" in df.columns:
        n_total = df.height
        n_unique = df["question_id"].n_unique()
        duplicates = n_total - n_unique
        dup_ratio = duplicates / total_rows
        result.statistics["duplicate_ratio"] = dup_ratio
        
        if dup_ratio > config.MAX_DUPLICATE_RATIO:
            result.add_failure("duplicates",
                              f"Found {duplicates} duplicate question_ids ({dup_ratio:.1%})")
        elif duplicates > 0:
            result.add_warning("duplicates", f"Found {duplicates} duplicates")
        else:
            result.add_pass("duplicates", "No duplicate question_ids found")
    
    text_cols = ["question_body", "answer_body", "title"]
    for col in text_cols:
        if col in df.columns:
            empty = df.filter(pl.col(col).str.strip_chars() == "").height
            if empty > 0:
                result.add_warning("empty_strings", f"Column '{col}' has {empty} empty strings")


def validate_text_quality(df: pl.DataFrame, result: ValidationResult) -> None:
    """Validate text content quality."""
    
    if "question_body" in df.columns:
        q_lengths = df["question_body"].str.len_bytes()
        result.statistics["question_length"] = {
            "min": int(q_lengths.min()),
            "max": int(q_lengths.max()),
            "mean": float(q_lengths.mean()),
            "median": float(q_lengths.median())
        }
        
        short_questions = df.filter(pl.col("question_body").str.len_bytes() < 50).height
        if short_questions > df.height * 0.1:
            result.add_warning("text_quality",
                              f"{short_questions} questions are very short (<50 chars)")
        else:
            result.add_pass("text_quality", "Question lengths are acceptable")
    
    if "answer_body" in df.columns:
        a_lengths = df["answer_body"].str.len_bytes()
        result.statistics["answer_length"] = {
            "min": int(a_lengths.min()),
            "max": int(a_lengths.max()),
            "mean": float(a_lengths.mean()),
            "median": float(a_lengths.median())
        }
        
        short_answers = df.filter(pl.col("answer_body").str.len_bytes() < 100).height
        if short_answers > df.height * 0.1:
            result.add_warning("text_quality",
                              f"{short_answers} answers are very short (<100 chars)")


def validate_feature_distribution(df: pl.DataFrame, result: ValidationResult) -> None:
    """Validate feature distributions for potential issues."""
    
    if "score" in df.columns:
        score_stats = {
            "count": int(df["score"].len()),
            "mean": float(df["score"].mean()),
            "std": float(df["score"].std()),
            "min": float(df["score"].min()),
            "max": float(df["score"].max()),
        }
        result.statistics["score_distribution"] = score_stats
        
        if score_stats["std"] == 0:
            result.add_warning("feature_distribution", "All scores are identical")
    
    if "quality_score" in df.columns:
        qs_stats = {
            "count": int(df["quality_score"].len()),
            "mean": float(df["quality_score"].mean()),
            "std": float(df["quality_score"].std()),
            "min": float(df["quality_score"].min()),
            "max": float(df["quality_score"].max()),
        }
        result.statistics["quality_score_distribution"] = qs_stats
    
    if "tags" in df.columns:
        all_tags = []
        for tags in df["tags"].to_list():
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        tag_counts = Counter(all_tags)
        result.statistics["tag_distribution"] = dict(tag_counts.most_common(20))
        
        if len(tag_counts) < 3:
            result.add_warning("feature_distribution", "Very few unique tags found")
    
    if "error_signatures" in df.columns:
        all_errors = []
        for errors in df["error_signatures"].to_list():
            if isinstance(errors, list):
                all_errors.extend(errors)
        
        error_counts = Counter(all_errors)
        result.statistics["error_signature_distribution"] = dict(error_counts)


def validate_anomalies(df: pl.DataFrame, result: ValidationResult) -> list:
    """Detect anomalies in the data. Returns list of anomalous records."""
    
    anomalies = []
    
    numeric_cols = ["score", "answer_score", "view_count", "quality_score"]
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_df = df.filter(
            (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
        )
        
        if outlier_df.height > 0:
            result.statistics[f"{col}_outliers"] = outlier_df.height
            
            if outlier_df.height > df.height * 0.05:
                result.add_warning("anomaly_detection",
                                  f"Found {outlier_df.height} outliers in '{col}'")
            
            for row in outlier_df.iter_rows(named=True):
                anomalies.append({
                    "type": "outlier",
                    "column": col,
                    "question_id": row.get("question_id"),
                    "value": row[col],
                    "bounds": [lower_bound, upper_bound]
                })
    
    if "question_body" in df.columns:
        html_count = df.filter(
            pl.col("question_body").str.contains(r'<[a-z]+>')
        ).height
        if html_count > 0:
            result.add_warning("anomaly_detection",
                              f"Found {html_count} records with uncleaned HTML")
    
    result.statistics["total_anomalies"] = len(anomalies)
    return anomalies


# =============================================================================
# Statistics Generation
# =============================================================================

def generate_statistics(df: pl.DataFrame) -> dict:
    """Generate comprehensive statistics for the dataset."""
    
    stats = {
        "generated_at": datetime.now().isoformat(),
        "row_count": df.height,
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }
    
    numeric_df = df.select(cs.numeric())
    numeric_stats = {}
    for col in numeric_df.columns:
        numeric_stats[col] = {
            "count": int(df[col].drop_nulls().len()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "25%": float(df[col].quantile(0.25)),
            "50%": float(df[col].quantile(0.50)),
            "75%": float(df[col].quantile(0.75)),
            "max": float(df[col].max()),
        }
    stats["numeric_statistics"] = numeric_stats
    
    text_cols = ["title", "question_body", "answer_body"]
    text_stats = {}
    for col in text_cols:
        if col in df.columns:
            lengths = df[col].str.len_bytes()
            text_stats[col] = {
                "avg_length": float(lengths.mean()),
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "total_chars": int(lengths.sum()),
            }
    stats["text_statistics"] = text_stats
    
    if "question_type" in df.columns:
        vc = df["question_type"].value_counts()
        stats["question_type_distribution"] = dict(
            zip(vc["question_type"].to_list(), vc["count"].to_list())
        )
    
    if "complexity" in df.columns:
        vc = df["complexity"].value_counts()
        stats["complexity_distribution"] = dict(
            zip(vc["complexity"].to_list(), vc["count"].to_list())
        )
    
    return stats


# =============================================================================
# Main Validation Pipeline
# =============================================================================

def run_validation() -> dict:
    """
    Main validation function - entry point for Airflow task.
    
    Returns:
        dict: Validation results and statistics
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA VALIDATION")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    result = ValidationResult()
    
    try:
        input_path = config.processed_dir / "qa_pairs_processed.json"
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pl.DataFrame(data)
        logger.info(f"Loaded {df.height} records for validation")
        
        # Run polars_validation suite checks
        suite_checks = [
            validate_columns_present(df, EXPECTED_SCHEMA["required_columns"]),
            validate_no_nulls(df, ["question_id", "question_body", "answer_body"]),
            validate_row_count(df, min_rows=config.MIN_ROWS),
        ]
        suite_report = run_validation_suite(suite_checks)
        result.statistics["polars_validation_suite"] = suite_report
        
        for check in suite_report["results"]:
            if check["success"]:
                result.add_pass(check["check"], check["detail"])
            else:
                result.add_failure(check["check"], check["detail"])
        
        # Run custom validations
        validate_schema(df, result)
        validate_data_quality(df, result)
        validate_text_quality(df, result)
        validate_feature_distribution(df, result)
        
        anomalies = validate_anomalies(df, result)
        
        statistics = generate_statistics(df)
        result.statistics.update(statistics)
        
        validation_output = result.to_dict()
        validation_output["start_time"] = start_time.isoformat()
        validation_output["end_time"] = datetime.now().isoformat()
        
        report_path = config.REPORTS_DIR / "validation" / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(validation_output, f, indent=2, default=str)
        
        stats_path = config.REPORTS_DIR / "statistics" / "data_statistics.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(statistics, f, indent=2, default=str)
        
        metrics = {
            "is_valid": result.is_valid,
            "passed_checks": len(result.passed),
            "failed_checks": len(result.failed),
            "warning_checks": len(result.warnings),
            "total_records": df.height,
            "anomaly_count": result.statistics.get("total_anomalies", 0),
            "duplicate_count": result.statistics.get("duplicate_ratio", 0) * df.height,
            "timestamp": datetime.now().isoformat()
        }
        
        metrics_path = config.REPORTS_DIR / "validation" / "validation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        if anomalies:
            anomaly_path = config.REPORTS_DIR / "validation" / "anomalies.json"
            with open(anomaly_path, "w") as f:
                json.dump(anomalies, f, indent=2, default=str)
        
        if result.is_valid:
            validated_path = config.validated_dir / "qa_pairs_validated.json"
            with open(validated_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Data validated and saved to {validated_path}")
        else:
            logger.error("Validation failed - data not promoted to validated directory")
        
        logger.info(f"Validation complete: {len(result.passed)} passed, "
                   f"{len(result.failed)} failed, {len(result.warnings)} warnings")
        
        return validation_output
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        result.add_failure("execution", str(e))
        raise


if __name__ == "__main__":
    run_validation()
