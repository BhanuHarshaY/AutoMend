"""
Schema & Statistics Generation for Format A sequences (Polars-based).
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
        logging.FileHandler(LOG_DIR / "schema_stats.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

INPUT_PATH = PROCESSED_DIR / "format_a_sequences.json"
STATS_PATH = PROCESSED_DIR / "schema_stats.json"

LABEL_NAMES = {
    0: "Normal",
    1: "Resource_Exhaustion",
    2: "System_Crash",
    3: "Network_Failure",
    4: "Data_Drift",
}


def run_schema_stats(path=INPUT_PATH, output_path=STATS_PATH):
    log.info("Generating schema and statistics for %s", path)

    with open(path, "r") as f:
        sequences = json.load(f)

    label_dist = Counter(s["label"] for s in sequences)
    named_dist = {LABEL_NAMES[k]: v for k, v in label_dist.items()}

    lengths = [len(s["sequence_ids"]) for s in sequences]
    all_tokens = [t for s in sequences for t in s["sequence_ids"]]
    cpu_tokens = [t for t in all_tokens if 100 <= t <= 109]
    mem_tokens = [t for t in all_tokens if 200 <= t <= 209]

    stats = {
        "total_sequences": len(sequences),
        "label_distribution": named_dist,
        "sequence_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
        },
        "token_stats": {
            "total_tokens": len(all_tokens),
            "cpu_tokens": len(cpu_tokens),
            "memory_tokens": len(mem_tokens),
            "unique_tokens": len(set(all_tokens)),
        },
        "quality_checks": {
            "empty_sequences": sum(1 for s in sequences if len(s["sequence_ids"]) == 0),
            "invalid_labels": sum(1 for s in sequences if s["label"] not in range(5)),
            "missing_fields": sum(
                1 for s in sequences if "sequence_ids" not in s or "label" not in s
            ),
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Stats saved -> %s", output_path)
    log.info("  Total sequences : %d", stats["total_sequences"])
    log.info("  Label dist      : %s", named_dist)
    return stats


if __name__ == "__main__":
    stats = run_schema_stats()
    print(f"\nTotal sequences : {stats['total_sequences']}")
    print(f"Label dist      : {stats['label_distribution']}")
