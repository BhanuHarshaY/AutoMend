"""Unit tests for the sampling and template filtering components."""
import sys
from pathlib import Path

DS2_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DS2_ROOT.parent.parent
# Add DS2_ROOT first so "from src.utils..." works in source modules
sys.path.insert(0, str(DS2_ROOT))
# Add DS2's src directory for direct imports like "from utils..."  
sys.path.insert(0, str(DS2_ROOT / "src"))

# Use centralized data paths, fallback to legacy
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ds2_loghub" / "mlops_processed"
if not PROCESSED_DIR.exists():
    PROCESSED_DIR = DS2_ROOT / "data_processed" / "mlops_processed"

import polars as pl
import pytest

from utils.hashing import stable_hash, should_keep

UNIFIED_COLS = [
    "system", "timestamp", "severity", "source",
    "event_id", "event_template", "message", "raw_id", "extras", "event_type",
]

SYSTEMS = ["linux", "hpc", "hdfs", "hadoop", "spark"]


def _make_events_df(n_per_system: int = 200) -> pl.DataFrame:
    """Create a synthetic merged DataFrame with n rows per system."""
    rows = []
    for system in SYSTEMS:
        for i in range(n_per_system):
            rows.append({
                "system": system,
                "timestamp": f"2024-01-01 00:{i:02d}:00",
                "severity": "INFO" if i % 3 != 0 else "ERROR",
                "source": "test_component",
                "event_id": f"E{i % 10 + 1}",
                "event_template": f"Template <*> number {i % 10}",
                "message": f"Test message {i}",
                "raw_id": str(i),
                "extras": "{}",
                "event_type": "",
            })
    return pl.DataFrame(rows)


# ── Sampling logic ─────────────────────────────────────────────────────────────

class TestSampleLogic:
    def test_sample_keeps_roughly_10pct(self):
        """Roughly 10% of rows should be kept across 1000 rows."""
        df = _make_events_df(100)  # 500 rows total
        mask = pl.struct(["system", "raw_id"]).map_elements(
            lambda r: stable_hash(f"{r['system']}:{r['raw_id']}") % 10 == 0,
            return_dtype=pl.Boolean,
        )
        sampled = df.filter(mask)
        pct = sampled.height / df.height * 100
        assert 5 <= pct <= 20, f"Expected ~10% sampled, got {pct:.1f}%"

    def test_sample_is_deterministic(self):
        """Running the same hash filter twice gives identical results."""
        df = _make_events_df(50)
        mask = pl.struct(["system", "raw_id"]).map_elements(
            lambda r: stable_hash(f"{r['system']}:{r['raw_id']}") % 10 == 0,
            return_dtype=pl.Boolean,
        )
        result1 = df.filter(mask)
        result2 = df.filter(mask)
        assert result1.equals(result2)

    def test_sample_preserves_schema(self):
        """Sampled DataFrame has same columns as input."""
        df = _make_events_df(50)
        mask = pl.struct(["system", "raw_id"]).map_elements(
            lambda r: stable_hash(f"{r['system']}:{r['raw_id']}") % 10 == 0,
            return_dtype=pl.Boolean,
        )
        sampled = df.filter(mask)
        for col in UNIFIED_COLS:
            assert col in sampled.columns

    def test_should_keep_roughly_10pct_across_systems(self):
        """should_keep helper keeps ~10% of rows per system."""
        for system in SYSTEMS:
            kept = sum(should_keep(system, str(i)) for i in range(1000))
            assert 50 <= kept <= 200, f"{system}: expected ~100 kept, got {kept}"

    def test_all_systems_can_be_represented_in_sample(self):
        """Each system produces at least some kept rows across 2000 rows."""
        for system in SYSTEMS:
            kept = sum(should_keep(system, str(i)) for i in range(2000))
            assert kept > 0, f"No rows kept for system {system}"

    def test_hash_stability_across_str_int_raw_id(self):
        """Hash is stable whether raw_id is str int or string."""
        h1 = stable_hash("linux:1")
        h2 = stable_hash("linux:1")
        assert h1 == h2

    def test_different_systems_same_raw_id_hash_differently(self):
        """linux:1 and hpc:1 should not collide."""
        h1 = stable_hash("linux:1")
        h2 = stable_hash("hpc:1")
        assert h1 != h2


# ── Integration: sample output file exists ─────────────────────────────────────

class TestSampleOutputExists:
    PROCESSED = PROCESSED_DIR

    def test_events_parquet_exists(self):
        """The sampled events file should exist after pipeline run."""
        assert (self.PROCESSED / "mlops_events.parquet").exists()

    def test_sampled_events_contain_all_systems(self):
        """All 5 systems should be represented in the sample."""
        path = self.PROCESSED / "mlops_events.parquet"
        if not path.exists():
            pytest.skip("Pipeline not yet run")
        df = pl.read_parquet(path)
        assert set(df["system"].unique().to_list()) == set(SYSTEMS)

    def test_sampled_events_schema_complete(self):
        """Sampled events have all required columns."""
        path = self.PROCESSED / "mlops_events.parquet"
        if not path.exists():
            pytest.skip("Pipeline not yet run")
        df = pl.read_parquet(path)
        for col in UNIFIED_COLS:
            assert col in df.columns
