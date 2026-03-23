"""
build_benchmark.py

Curates a locked gold benchmark set from test.jsonl.

Selection strategy — picks samples that cover all structural archetypes:
  1. Tool-call responses  — workflow.steps is non-empty list
  2. Refusal responses    — workflow.steps is [] (model should say no)
  3. Multi-step workflows — workflow.steps has 2+ entries
  4. Single-step workflows — workflow.steps has exactly 1 entry

The benchmark is written to data/benchmarks/gold_benchmark.jsonl and
accompanied by a manifest (gold_benchmark_manifest.json) that records
the selection criteria and counts.

Once built, this file is LOCKED — never regenerate from a different seed
or with different counts, as it would invalidate historical comparisons.

Usage:
    python scripts/build_benchmark.py
    python scripts/build_benchmark.py --source data/splits/test.jsonl --size 50
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

_M2_ROOT = Path(__file__).resolve().parent.parent
_AUTOMEND_ROOT = _M2_ROOT.parent
sys.path.insert(0, str(_AUTOMEND_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _get_assistant_content(sample: dict) -> str:
    for msg in reversed(sample.get("messages", [])):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _classify_sample(sample: dict) -> str:
    """Classify a sample into an archetype based on its reference answer."""
    content = _get_assistant_content(sample)
    try:
        parsed = json.loads(content)
        steps = parsed.get("workflow", {}).get("steps", None)
        if steps is None:
            return "no_workflow"
        if not isinstance(steps, list):
            return "invalid_steps"
        if len(steps) == 0:
            return "refusal"
        if len(steps) == 1:
            return "single_step"
        return "multi_step"
    except (json.JSONDecodeError, ValueError):
        return "invalid_json"


def _select_balanced(
    samples: list[dict],
    per_archetype: int,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    """
    Select up to `per_archetype` samples from each archetype bucket.
    Returns (selected_samples, archetype_counts).
    """
    buckets: dict[str, list[dict]] = {}
    for s in samples:
        arch = _classify_sample(s)
        buckets.setdefault(arch, []).append(s)

    rng = random.Random(seed)
    selected = []
    counts: dict[str, int] = {}

    for arch, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        chosen = bucket[:per_archetype]
        selected.extend(chosen)
        counts[arch] = len(chosen)
        logger.info(f"  {arch:<20} {len(bucket):>4} available → {len(chosen):>3} selected")

    return selected, counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the locked gold benchmark from test.jsonl."
    )
    parser.add_argument(
        "--source",
        default=str(_M2_ROOT / "data/splits/test.jsonl"),
        help="Source JSONL file (default: data/splits/test.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_M2_ROOT / "data/benchmarks"),
        help="Output directory (default: data/benchmarks/).",
    )
    parser.add_argument(
        "--per-archetype",
        type=int,
        default=10,
        help="Max samples to select per archetype (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42). NEVER change after first build.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing benchmark file. Use with caution — breaks historical comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_path = Path(args.source)
    output_dir  = Path(args.output_dir)
    benchmark_path  = output_dir / "gold_benchmark.jsonl"
    manifest_path   = output_dir / "gold_benchmark_manifest.json"

    if benchmark_path.exists() and not args.force:
        logger.warning(
            f"Benchmark already exists at {benchmark_path}. "
            "The gold benchmark is LOCKED — use --force only if you intentionally want to rebuild it. "
            "Rebuilding breaks historical metric comparisons."
        )
        sys.exit(0)

    if not source_path.exists():
        logger.error(f"Source not found: {source_path}")
        sys.exit(1)

    logger.info(f"Loading source: {source_path}")
    samples = _load_jsonl(source_path)
    logger.info(f"Loaded {len(samples)} samples from source.")

    logger.info(f"Selecting up to {args.per_archetype} samples per archetype (seed={args.seed}):")
    selected, counts = _select_balanced(samples, args.per_archetype, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(benchmark_path, "w", encoding="utf-8") as f:
        for sample in selected:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    manifest = {
        "created_at":    datetime.utcnow().isoformat() + "Z",
        "source":        str(source_path),
        "seed":          args.seed,
        "per_archetype": args.per_archetype,
        "total_selected": len(selected),
        "archetype_counts": counts,
        "benchmark_path": str(benchmark_path),
        "locked": True,
        "warning": "DO NOT rebuild without incrementing version. Rebuilding invalidates historical comparisons.",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.success(f"Gold benchmark written → {benchmark_path} ({len(selected)} samples)")
    logger.success(f"Manifest written       → {manifest_path}")
    logger.warning("This benchmark is now LOCKED. Do not rebuild unless absolutely necessary.")


if __name__ == "__main__":
    main()
