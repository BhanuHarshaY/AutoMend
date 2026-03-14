"""
Anomaly Detection & Alerting for Glaive Function Calling v2
Uses Polars for threshold checks and centralized alerting.
"""

import json
import logging
import sys
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "glaive_processed.jsonl"
ANOMALY_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "validation"

THRESHOLDS = {
    "max_malformed_pct": 0.05,
    "max_none_complexity_pct": 0.60,
    "min_records": 4000,
    "max_avg_turns": 10.0,
    "max_avg_calls": 5.0,
    "min_defined_fn_pct": 0.30,
}


def send_slack_alert(message: str) -> None:
    from src.utils.alerting import alert_anomaly_detected
    import re

    logger.warning("ALERT MESSAGE: %s", message)
    match = re.search(r"Anomalies detected: (\d+)", message)
    anomaly_count = int(match.group(1)) if match else 1
    alert_anomaly_detected(
        pipeline_name="ds5_glaive",
        anomaly_count=anomaly_count,
        anomaly_types=["threshold_exceeded"],
        details={"raw_message": message[:200]},
    )


def _check(check_name: str, value: float, threshold: float, is_anomaly: bool, msg: str) -> dict:
    return {
        "check": check_name,
        "value": round(value, 4),
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "message": msg if is_anomaly else "OK",
    }


def check_malformed_rate(df: pl.DataFrame) -> dict:
    pct = df.filter(pl.col("has_malformed")).height / df.height
    return _check("malformed_rate", pct, THRESHOLDS["max_malformed_pct"],
                  pct > THRESHOLDS["max_malformed_pct"], f"Malformed rate {pct:.2%} exceeds threshold")


def check_none_complexity_rate(df: pl.DataFrame) -> dict:
    pct = df.filter(pl.col("complexity_tier") == "none").height / df.height
    return _check("none_complexity_rate", pct, THRESHOLDS["max_none_complexity_pct"],
                  pct > THRESHOLDS["max_none_complexity_pct"], f"None complexity rate {pct:.2%} exceeds threshold")


def check_record_count(df: pl.DataFrame) -> dict:
    n = df.height
    return _check("record_count", n, THRESHOLDS["min_records"],
                  n < THRESHOLDS["min_records"], f"Record count {n} below minimum {THRESHOLDS['min_records']}")


def check_avg_turns(df: pl.DataFrame) -> dict:
    avg = float(df["num_turns"].mean())
    return _check("avg_turns", avg, THRESHOLDS["max_avg_turns"],
                  avg > THRESHOLDS["max_avg_turns"], f"Avg turns {avg:.2f} exceeds threshold")


def check_avg_calls(df: pl.DataFrame) -> dict:
    avg = float(df["num_calls"].mean())
    return _check("avg_calls", avg, THRESHOLDS["max_avg_calls"],
                  avg > THRESHOLDS["max_avg_calls"], f"Avg calls {avg:.2f} exceeds threshold")


def check_defined_functions_coverage(df: pl.DataFrame) -> dict:
    pct = df.filter(pl.col("num_defined_functions") > 0).height / df.height
    return _check("defined_functions_coverage", pct, THRESHOLDS["min_defined_fn_pct"],
                  pct < THRESHOLDS["min_defined_fn_pct"], f"Defined function coverage {pct:.2%} below threshold")


def run_anomaly_detection(filepath: Path = PROCESSED_FILE) -> dict:
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed data from %s", filepath)
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pl.DataFrame(records)
    logger.info("Loaded %d records", df.height)

    n = df.height
    malformed_pct = df.filter(pl.col("has_malformed")).height / n
    none_pct = df.filter(pl.col("complexity_tier") == "none").height / n
    avg_turns = df["num_turns"].mean()
    avg_calls = df["num_calls"].mean()
    has_fn_pct = df.filter(pl.col("num_defined_functions") > 0).height / n

    checks = [
        _check("malformed_rate", malformed_pct, THRESHOLDS["max_malformed_pct"],
               malformed_pct > THRESHOLDS["max_malformed_pct"],
               f"Malformed rate {malformed_pct:.2%} exceeds threshold"),
        _check("none_complexity_rate", none_pct, THRESHOLDS["max_none_complexity_pct"],
               none_pct > THRESHOLDS["max_none_complexity_pct"],
               f"None complexity rate {none_pct:.2%} exceeds threshold"),
        _check("record_count", n, THRESHOLDS["min_records"],
               n < THRESHOLDS["min_records"],
               f"Record count {n} below minimum {THRESHOLDS['min_records']}"),
        _check("avg_turns", avg_turns, THRESHOLDS["max_avg_turns"],
               avg_turns > THRESHOLDS["max_avg_turns"],
               f"Avg turns {avg_turns:.2f} exceeds threshold"),
        _check("avg_calls", avg_calls, THRESHOLDS["max_avg_calls"],
               avg_calls > THRESHOLDS["max_avg_calls"],
               f"Avg calls {avg_calls:.2f} exceeds threshold"),
        _check("defined_functions_coverage", has_fn_pct, THRESHOLDS["min_defined_fn_pct"],
               has_fn_pct < THRESHOLDS["min_defined_fn_pct"],
               f"Defined function coverage {has_fn_pct:.2%} below threshold"),
    ]

    anomalies = [c for c in checks if c["is_anomaly"]]
    if anomalies:
        alert_msg = (
            f"AutoMend Data Pipeline Alert\n"
            f"Dataset: Glaive Function Calling v2\n"
            f"Anomalies detected: {len(anomalies)}\n\n"
        )
        for a in anomalies:
            alert_msg += f"- {a['check']}: {a['message']}\n"
        send_slack_alert(alert_msg)
    else:
        logger.info("No anomalies detected — pipeline healthy")

    report = {
        "total_checks": len(checks),
        "anomalies_found": len(anomalies),
        "pipeline_healthy": len(anomalies) == 0,
        "checks": checks,
    }
    report_path = ANOMALY_DIR / "anomaly_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: bool(x) if hasattr(x, "item") else str(x))
    logger.info("Anomaly report saved to %s", report_path)
    return report


if __name__ == "__main__":
    report = run_anomaly_detection()
    print(f"\n{'Pipeline Healthy' if report['pipeline_healthy'] else 'Anomalies Found'}")
