"""
Anomaly Detection - Flag suspicious sequences from Format A output.
Uses centralized alerting for notifications.
"""
import json
import logging
import sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds1_processed_dir, LOGS_DIR
    PROCESSED_DIR = get_ds1_processed_dir()
    LOG_DIR = LOGS_DIR
except ImportError:
    PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"
    LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "anomaly.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

INPUT_PATH = PROCESSED_DIR / "format_a_sequences.json"
LABEL_NAMES = {0: "Normal", 1: "Resource_Exhaustion", 2: "System_Crash", 3: "Network_Failure", 4: "Data_Drift"}


def detect_anomalies(path=INPUT_PATH):
    log.info("Running anomaly detection on %s", path)

    with open(path, "r") as f:
        sequences = json.load(f)

    anomalies = []
    for i, seq in enumerate(sequences):
        label = seq["label"]
        if label != 0:
            anomalies.append({
                "index": i,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "sequence_ids": seq["sequence_ids"],
            })

    log.warning("ALERT: %d anomalies detected out of %d sequences", len(anomalies), len(sequences))
    dist = Counter(a["label_name"] for a in anomalies)
    for name, count in dist.items():
        log.warning("  %s: %d", name, count)
    return anomalies


def send_alert(anomalies):
    from src.utils.alerting import alert_anomaly_detected

    critical = [a for a in anomalies if a["label"] in [1, 2]]
    if len(critical) > 0:
        log.critical("CRITICAL: %d Resource_Exhaustion/System_Crash detected!", len(critical))
        anomaly_types = list(set(a["label_name"] for a in anomalies))
        alert_anomaly_detected(
            pipeline_name="ds1_alibaba",
            anomaly_count=len(anomalies),
            anomaly_types=anomaly_types,
            details={
                "critical_count": len(critical),
                "breakdown": dict(Counter(a["label_name"] for a in anomalies)),
            },
        )
    else:
        log.info("No critical anomalies — no alert sent")


if __name__ == "__main__":
    anomalies = detect_anomalies()
    send_alert(anomalies)
    print(f"\nAnomalies detected: {len(anomalies)}")
