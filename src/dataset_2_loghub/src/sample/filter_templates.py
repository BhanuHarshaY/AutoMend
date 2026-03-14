"""Filter templates to keep only EventIds present in the sampled events.

Reads: data/processed/ds2_loghub/mlops_processed/mlops_events.parquet
Reads: data/raw/ds2_loghub/loghub/<System>/*_templates.csv  (for each system)
Output: data/processed/ds2_loghub/mlops_processed/mlops_templates.csv
"""
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_raw_dir, get_ds2_processed_dir, get_legacy_raw_dir

import polars as pl
from utils.io import read_csv, read_parquet, write_csv
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()
RAW_DIR = get_ds2_raw_dir() / "loghub"
if not RAW_DIR.exists():
    RAW_DIR = get_legacy_raw_dir() / "loghub"

EVENTS_PATH = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"
OUTPUT      = PROCESSED_DIR / "mlops_processed" / "mlops_templates.csv"

TEMPLATE_FILES = {
    "linux":  RAW_DIR / "Linux"  / "Linux_2k.log_templates.csv",
    "hpc":    RAW_DIR / "HPC"    / "HPC_2k.log_templates.csv",
    "hdfs":   RAW_DIR / "HDFS"   / "HDFS_2k.log_templates.csv",
    "hadoop": RAW_DIR / "Hadoop" / "Hadoop_2k.log_templates.csv",
    "spark":  RAW_DIR / "Spark"  / "Spark_2k.log_templates.csv",
}


def filter_templates(events_path: Path = EVENTS_PATH, output_path: Path = OUTPUT):
    events = read_parquet(str(events_path))
    used_ids = events["event_id"].unique().to_list()
    logger.info("Unique EventIds in sample: %d", len(used_ids))

    frames = []
    for system, tpath in TEMPLATE_FILES.items():
        if not tpath.exists():
            raise FileNotFoundError(f"Missing template file: {tpath}")
        df = read_csv(str(tpath))
        df = df.with_columns(pl.lit(system).alias("system"))
        frames.append(df)
        logger.info("Loaded %s templates: %d rows", system, df.height)

    all_templates = pl.concat(frames)

    filtered = (
        all_templates
        .filter(pl.col("EventId").is_in(used_ids))
        .unique(subset=["EventId", "system"])
    )

    logger.info("Filtered to %d unique templates (from %d total)", filtered.height, all_templates.height)

    write_csv(filtered, str(output_path))
    logger.info("Done.")
    return filtered


if __name__ == "__main__":
    filter_templates()
