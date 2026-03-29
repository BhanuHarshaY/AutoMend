"""
Track A Combiner - Trigger Engine
=================================
Combines Dataset 1 (Alibaba) and Dataset 2 (Loghub) Parquet files
into a single combined Parquet file for the Trigger Engine ML track.

Input files (from data/interim/):
    - ds1_alibaba.parquet
    - ds2_loghub.parquet

Output file (to data/processed/):
    - track_A_combined.parquet

Format A schema:
    {"sequence_ids": List[int], "label": int}
"""

import logging
import sys
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

try:
    from src.config.paths import (
        get_ds1_interim_path, get_ds2_interim_path,
        get_track_a_output, INTERIM_DIR, PROCESSED_DIR
    )
    INPUT_FILES = [get_ds1_interim_path(), get_ds2_interim_path()]
    OUTPUT_FILE = get_track_a_output()
except ImportError:
    INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    INPUT_FILES = [
        INTERIM_DIR / "ds1_alibaba.parquet",
        INTERIM_DIR / "ds2_loghub.parquet",
    ]
    OUTPUT_FILE = PROCESSED_DIR / "track_A_combined.parquet"


def combine_track_a():
    """Concatenate all Track A Parquet files into one."""
    log.info("=" * 60)
    log.info("TRACK A COMBINER - Trigger Engine")
    log.info("=" * 60)
    
    dataframes = []
    
    for input_path in INPUT_FILES:
        log.info(f"Reading: {input_path}")
        
        if not input_path.exists():
            log.warning(f"  File not found: {input_path} - skipping")
            continue
        
        df = pl.read_parquet(input_path)
        log.info(f"  Loaded {df.height} rows")
        log.info(f"  Columns: {df.columns}")
        
        _UPCAST = {
            pl.Int32: pl.Int64,
            pl.Int16: pl.Int64,
            pl.Int8: pl.Int64,
            pl.List(pl.Int32): pl.List(pl.Int64),
            pl.List(pl.Int16): pl.List(pl.Int64),
            pl.List(pl.Int8): pl.List(pl.Int64),
        }
        casts = [
            pl.col(c).cast(_UPCAST[df[c].dtype])
            for c in df.columns if df[c].dtype in _UPCAST
        ]
        if casts:
            df = df.with_columns(casts)
        df = df.with_columns(pl.lit(input_path.stem).alias("source_dataset"))
        dataframes.append(df)
    
    if not dataframes:
        raise FileNotFoundError("No input files found. Run dataset pipelines first.")
    
    combined = pl.concat(dataframes)
    log.info(f"Combined total: {combined.height} rows")
    
    if "label" in combined.columns:
        log.info(f"Label distribution:\n{combined['label'].value_counts()}")
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUTPUT_FILE)
    
    log.info(f"Saved to: {OUTPUT_FILE}")
    log.info(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")
    log.info("=" * 60)
    
    return combined


if __name__ == "__main__":
    df = combine_track_a()
    print(f"\nCombination complete!")
    print(f"Total rows: {df.height}")
    print(f"Output: {OUTPUT_FILE}")
