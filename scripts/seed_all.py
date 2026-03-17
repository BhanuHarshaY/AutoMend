#!/usr/bin/env python
"""
Master Seed Script - Unified data seeding for E2E testing
=========================================================
Runs all dataset seed scripts to prepare the monorepo for E2E testing with Airflow.

Respects PIPELINE_DATA_MODE environment variable:
  dummy  (default) - generate synthetic data only (offline, fast)
  sample           - download capped subsets from real sources (network required)
  full             - download full datasets (network + API tokens may be required)

The --download flag is kept as a convenience override: it forces 'sample' mode
for DS2/DS5/DS6 regardless of the env var.

Usage:
    python scripts/seed_all.py              # Seed using PIPELINE_DATA_MODE (default: dummy)
    python scripts/seed_all.py --download   # Force sample-mode downloads for DS2/DS5/DS6
    python scripts/seed_all.py --ds1        # Seed only DS1
    python scripts/seed_all.py --ds1 --ds3  # Seed DS1 and DS3
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import ensure_dirs_exist
    HAS_PATHS = True
except ImportError:
    HAS_PATHS = False

try:
    from src.config.data_mode import get_data_mode
except ImportError:
    def get_data_mode():
        return os.environ.get("PIPELINE_DATA_MODE", "dummy").strip().lower()


def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script:  {script_path}")
    print('='*60)

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            print(f"SUCCESS: {description}")
            return True
        else:
            print(f"FAILED: {description} (exit code {result.returncode})")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def seed_ds1(mode: str = "dummy") -> bool:
    """Seed Dataset 1 (Alibaba)."""
    if mode == "full":
        print("\nDS1 (Alibaba): full mode — place real CSVs in data/raw/ds1_alibaba/ manually.")
        return True
    return run_script(
        PROJECT_ROOT / "src" / "dataset_1_alibaba" / "scripts" / "seed_data.py",
        f"DS1 (Alibaba) - Generate seed CSVs [{mode}]"
    )


def seed_ds2(mode: str = "dummy") -> bool:
    """Seed Dataset 2 (LogHub)."""
    if mode == "dummy":
        return run_script(
            PROJECT_ROOT / "src" / "dataset_2_loghub" / "src" / "ingest" / "seed_data.py",
            "DS2 (LogHub) - Generate synthetic log CSVs [dummy]"
        )
    else:
        return run_script(
            PROJECT_ROOT / "src" / "dataset_2_loghub" / "src" / "ingest" / "download_data.py",
            f"DS2 (LogHub) - Download from GitHub [{mode}]"
        )


def seed_ds3(mode: str = "dummy") -> bool:
    """Seed Dataset 3 (StackOverflow)."""
    if mode == "dummy":
        return run_script(
            PROJECT_ROOT / "src" / "dataset_3_stackoverflow" / "scripts" / "seed_data.py",
            "DS3 (StackOverflow) - Generate sample CSVs [dummy]"
        )
    else:
        print(f"\nDS3 (StackOverflow): {mode} mode — use DAG to acquire via API.")
        return run_script(
            PROJECT_ROOT / "src" / "dataset_3_stackoverflow" / "scripts" / "seed_data.py",
            "DS3 (StackOverflow) - Generate seed CSVs as baseline"
        )


def seed_ds4(mode: str = "dummy") -> bool:
    """Seed Dataset 4 (Synthetic) - seeds prompts database."""
    return run_script(
        PROJECT_ROOT / "src" / "dataset_4_synthetic" / "scripts" / "seed_prompts.py",
        f"DS4 (Synthetic) - Seed prompts database [{mode}]"
    )


def seed_ds5(mode: str = "dummy") -> bool:
    """Seed Dataset 5 (Glaive)."""
    if mode == "dummy":
        return run_script(
            PROJECT_ROOT / "src" / "dataset_5_glaive" / "scripts" / "seed_data.py",
            "DS5 (Glaive) - Generate synthetic JSONL [dummy]"
        )
    else:
        return run_script(
            PROJECT_ROOT / "src" / "dataset_5_glaive" / "scripts" / "data_acquisition.py",
            f"DS5 (Glaive) - Download from HuggingFace [{mode}]"
        )


def seed_ds6(mode: str = "dummy") -> bool:
    """Seed Dataset 6 (The Stack)."""
    if mode == "dummy":
        return run_script(
            PROJECT_ROOT / "src" / "dataset_6_the_stack" / "scripts" / "download" / "seed_data.py",
            "DS6 (The Stack) - Generate synthetic parquet [dummy]"
        )
    else:
        return run_script(
            PROJECT_ROOT / "src" / "dataset_6_the_stack" / "scripts" / "download" / "stack_iac_sample.py",
            f"DS6 (The Stack) - Download from HuggingFace [{mode}]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Seed all datasets for E2E testing with Airflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Respects PIPELINE_DATA_MODE env var (dummy / sample / full).
The --download flag overrides to 'sample' mode for network-dependent datasets.

Examples:
    python scripts/seed_all.py                          # Seed using env var (default: dummy)
    PIPELINE_DATA_MODE=sample python scripts/seed_all.py  # Download capped subsets
    python scripts/seed_all.py --download               # Same as sample mode for DS2/DS5/DS6
    python scripts/seed_all.py --ds1 --ds4              # Seed only specific datasets
        """
    )

    parser.add_argument("--ds1", action="store_true", help="Seed DS1 (Alibaba)")
    parser.add_argument("--ds2", action="store_true", help="Seed DS2 (LogHub)")
    parser.add_argument("--ds3", action="store_true", help="Seed DS3 (StackOverflow)")
    parser.add_argument("--ds4", action="store_true", help="Seed DS4 (Synthetic)")
    parser.add_argument("--ds5", action="store_true", help="Seed DS5 (Glaive)")
    parser.add_argument("--ds6", action="store_true", help="Seed DS6 (The Stack)")
    parser.add_argument("--download", action="store_true",
                       help="Force 'sample' mode for DS2/DS5/DS6 (overrides PIPELINE_DATA_MODE)")

    args = parser.parse_args()

    mode = get_data_mode()
    if args.download and mode == "dummy":
        mode = "sample"

    seed_specific = any([args.ds1, args.ds2, args.ds3, args.ds4, args.ds5, args.ds6])

    print("="*60)
    print("AUTOMEND E2E TEST DATA SEEDING")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"PIPELINE_DATA_MODE: {mode}")

    if HAS_PATHS:
        print("\nCreating data directories...")
        ensure_dirs_exist()
        print("  Data directories ready")

    results = {}

    if not seed_specific or args.ds1:
        results["DS1 (Alibaba)"] = seed_ds1(mode)

    if not seed_specific or args.ds2:
        results["DS2 (LogHub)"] = seed_ds2(mode)

    if not seed_specific or args.ds3:
        results["DS3 (StackOverflow)"] = seed_ds3(mode)

    if not seed_specific or args.ds4:
        results["DS4 (Synthetic)"] = seed_ds4(mode)

    if not seed_specific or args.ds5:
        results["DS5 (Glaive)"] = seed_ds5(mode)

    if not seed_specific or args.ds6:
        results["DS6 (The Stack)"] = seed_ds6(mode)

    print("\n" + "="*60)
    print("SEEDING SUMMARY")
    print("="*60)

    success_count = 0
    for dataset, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {dataset}: {status}")
        if success:
            success_count += 1

    print(f"\nTotal: {success_count}/{len(results)} succeeded")
    print(f"Mode: {mode}")

    if success_count == len(results):
        print("\nNext steps:")
        print("  1. Copy .env.example to .env and set your API keys")
        print("  2. Start Airflow: docker-compose up -d")
        print("  3. Access UI: http://localhost:8080 (airflow/airflow)")
        print("  4. Trigger master_track_a or master_track_b DAG")
        return 0
    else:
        print("\nSome seeds failed - check logs above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
