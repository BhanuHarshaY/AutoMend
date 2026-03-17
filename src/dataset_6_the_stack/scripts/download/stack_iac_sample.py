"""
Stream rows from bigcode/the-stack-dedup (YAML sub-corpus) and write
them to compressed parquet chunks under data/raw/.

This is the ONLY script that touches the network.
Everything downstream reads local parquet, fully repeatable offline.

Preferred path: Ray Data distributed download via from_huggingface().
Fallback:       sequential HuggingFace streaming (original behaviour).
"""

import logging
import sys
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DS6_ROOT = Path(__file__).parents[2]
PROJECT_ROOT = DS6_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.ray_config import init_ray, get_dataset_config

CFG = yaml.safe_load((DS6_ROOT / "config/iac_analysis.yaml").read_text())

DS = CFG["dataset"]
S  = CFG["sampling"]
F  = CFG["fields"]

ds6_cfg       = get_dataset_config("ds6")
_DEFAULT_SIZE = ds6_cfg.get("sample_size", S["sample_size"])
SAMPLE_SIZE   = _DEFAULT_SIZE
CHUNK_SIZE    = ds6_cfg.get("chunk_size", S["chunk_size"])

try:
    from src.config.paths import get_ds6_raw_dir, LOGS_DIR
    RAW     = get_ds6_raw_dir()
    LOG_DIR = LOGS_DIR
except ImportError:
    RAW     = DS6_ROOT / CFG["paths"]["raw_dir"]
    LOG_DIR = DS6_ROOT / CFG["paths"]["logs_dir"]

RAW.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "download.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

PULL = list(F.values())


# ── helpers ───────────────────────────────────────────────────────────
def _write_chunk(rows: list[dict], idx: int) -> None:
    path = RAW / f"chunk_{idx:04d}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")


# ── Ray Data path ─────────────────────────────────────────────────────
def _download_ray() -> None:
    """Use Ray Data for distributed download and dedup."""
    import ray
    import ray.data

    init_ray()

    log.info("Ray Data path — loading HuggingFace dataset")
    hf_ds = load_dataset(
        DS["repo"],
        data_dir=f"data/{DS['lang']}",
        split=DS["split"],
        streaming=False,
    )

    ray_ds = ray.data.from_huggingface(hf_ds).limit(SAMPLE_SIZE)

    rows = ray_ds.select_columns(PULL).take_all()

    seen: set[str] = set()
    chunk: list[dict] = []
    chunk_idx = 0

    for row in rows:
        sha = row.get(F["hexsha"], "")
        if sha in seen:
            continue
        seen.add(sha)

        chunk.append({col: row.get(col) for col in PULL})
        if len(chunk) >= CHUNK_SIZE:
            _write_chunk(chunk, chunk_idx)
            chunk_idx += 1
            chunk = []

    if chunk:
        _write_chunk(chunk, chunk_idx)
        chunk_idx += 1

    log.info(
        "Ray download complete — %d unique rows → %d parquet chunks in %s",
        len(seen), chunk_idx, RAW,
    )


# ── Sequential fallback ──────────────────────────────────────────────
def _download_sequential() -> None:
    """Original streaming download — no Ray dependency."""
    ds = load_dataset(
        DS["repo"],
        data_dir=f"data/{DS['lang']}",
        split=DS["split"],
        streaming=DS["streaming"],
    )

    chunk: list[dict] = []
    chunk_idx = 0
    seen: set[str] = set()

    for row in tqdm(ds, total=SAMPLE_SIZE, desc="Streaming"):
        if len(seen) >= SAMPLE_SIZE:
            break

        sha = row.get(F["hexsha"], "")
        if sha in seen:
            continue
        seen.add(sha)

        chunk.append({col: row.get(col) for col in PULL})
        if len(chunk) >= CHUNK_SIZE:
            _write_chunk(chunk, chunk_idx)
            chunk_idx += 1
            chunk = []

    if chunk:
        _write_chunk(chunk, chunk_idx)
        chunk_idx += 1

    log.info(
        "Sequential download complete — %d rows → %d parquet chunks in %s",
        len(seen), chunk_idx, RAW,
    )


# ── entry point ───────────────────────────────────────────────────────
def download(sample_size: int | None = None) -> None:
    """Download The Stack YAML sub-corpus.

    Parameters
    ----------
    sample_size : int | None
        Override the default sample size.  ``0`` means no limit (full dataset).
        ``None`` uses the configured default.
    """
    global SAMPLE_SIZE
    if sample_size is not None:
        SAMPLE_SIZE = sample_size if sample_size > 0 else float("inf")

    try:
        _download_ray()
    except Exception as exc:
        log.warning("Ray Data download failed (%s), falling back to sequential", exc)
        _download_sequential()


if __name__ == "__main__":
    download()
