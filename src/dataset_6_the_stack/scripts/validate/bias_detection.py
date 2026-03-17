"""
Slices the processed training records across meaningful dimensions,
reports representation imbalance, and applies downsampling mitigation
to produce a balanced output dataset.

Detection:
  Slices records by iac_type, license, size_bucket, and prompt_type.
  Flags any slice below MIN_SLICE_PCT of total as underrepresented.

Mitigation strategy — capped downsampling on iac_type:
  The iac_type dimension is the primary driver of imbalance because
  generic k8s_workload manifests dominate the raw corpus. We cap each
  iac_type slice at MAX_SLICE_PCT of the total to prevent the model
  learning a strong prior toward one manifest type.

Uses Polars DataFrames for vectorised counting and slice analysis.
Classify functions remain pure Python (they operate on single values).
"""

import json
import logging
import random
import re
import sys
import yaml
from collections import defaultdict
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).parents[2]
PROJECT_ROOT = _ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds6_processed_dir, LOGS_DIR
    _PROC = get_ds6_processed_dir()
    _LOGS = LOGS_DIR
except ImportError:
    import yaml as _yaml_init
    _cfg_init = _yaml_init.safe_load((_ROOT / "config/iac_analysis.yaml").read_text())
    _PROC = _ROOT / _cfg_init["paths"]["processed_dir"]
    _LOGS = _ROOT / _cfg_init["paths"]["logs_dir"]

_LOGS.mkdir(parents=True, exist_ok=True)
(_ROOT / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(_LOGS / "bias_detection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

MIN_SLICE_PCT = 5.0
MAX_SLICE_PCT = 40.0
RANDOM_SEED   = 42


# ── slice classifiers (pure functions, unchanged) ─────────────────────
def classify_iac_type(manifest: str) -> str:
    c = manifest[:2000]
    if re.search(r"InferenceService|kserve",                        c): return "kserve"
    if re.search(r"seldon|SeldonDeployment",                        c): return "seldon"
    if re.search(r"kind:\s*(Deployment|StatefulSet|DaemonSet|Job)", c): return "k8s_workload"
    if re.search(r"kind:\s*(Service|Ingress|ConfigMap|Secret)",     c): return "k8s_config"
    if re.search(r"apiVersion",                                      c): return "k8s_other"
    return "other"


def classify_size_bucket(size: int) -> str:
    if size < 1_000:   return "<1KB"
    if size < 10_000:  return "1-10KB"
    if size < 100_000: return "10-100KB"
    return ">100KB"


def classify_prompt_type(prompt: str) -> str:
    p = prompt.lower()
    if "deploy"    in p: return "deploy"
    if "gpu"       in p: return "gpu"
    if "inference" in p: return "inference"
    if "service"   in p: return "service"
    if "train"     in p: return "train"
    if "pipeline"  in p: return "pipeline"
    if "ingress"   in p: return "ingress"
    if "secret"    in p: return "secret"
    if "config"    in p: return "config"
    return "fallback"


def classify_license(licenses: list) -> str:
    if not licenses:
        return "unknown"
    known = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
             "isc", "cc0-1.0", "unlicense", "wtfpl", "0bsd"}
    for lic in licenses:
        l = str(lic).lower().strip()
        if l in known:
            return l
    return "other"


# ── Polars-based slicing ──────────────────────────────────────────────
def _extract_record_fields(record: dict) -> dict | None:
    """Pull flat fields from a training record for the slice DataFrame."""
    msgs = record.get("messages", [])
    meta = record.get("_meta", {})
    if len(msgs) != 2:
        return None
    prompt = msgs[0].get("content", "")
    try:
        parsed   = json.loads(msgs[1].get("content", "{}"))
        manifest = parsed.get("params", {}).get("manifest_content", "")
    except Exception:
        return None

    size     = int(meta.get("size") or 0)
    licenses = meta.get("licenses") or []

    return {
        "iac_type":        classify_iac_type(manifest),
        "license":         classify_license(licenses),
        "size_bucket":     classify_size_bucket(size),
        "prompt_type":     classify_prompt_type(prompt),
        "manifest_length": len(manifest),
    }


def build_slices(records: list[dict]) -> dict:
    """
    Build slice counts using a Polars DataFrame for vectorised aggregation.
    Returns the same nested dict structure as the original (keyed by dimension → value).
    """
    rows = [r for rec in records if (r := _extract_record_fields(rec)) is not None]
    if not rows:
        return {d: {} for d in ("iac_type", "license", "size_bucket", "prompt_type")}

    df = pl.DataFrame(rows)
    slices: dict[str, dict] = {}

    for dim in ("iac_type", "license", "size_bucket", "prompt_type"):
        agg = df.group_by(dim).agg(
            pl.col("manifest_length").count().alias("count"),
            pl.col("manifest_length").alias("manifest_lengths"),
        )
        dim_slices: dict[str, dict] = {}
        for row in agg.iter_rows(named=True):
            val = row[dim]
            dim_slices[val] = {
                "count":            row["count"],
                "manifest_lengths": row["manifest_lengths"],
            }
        slices[dim] = dim_slices

    return slices


def summarise_slices(slices: dict, total: int) -> dict:
    """Convert raw slice data into a reportable summary with imbalance flags."""
    summary: dict = {}
    for dim, groups in slices.items():
        if not groups:
            summary[dim] = {}
            continue

        rows = []
        for val, data in groups.items():
            count = data["count"]
            mlens = data["manifest_lengths"]
            rows.append({
                "value":          val,
                "count":          count,
                "mean_manifest":  round(sum(mlens) / len(mlens), 1) if mlens else 0,
            })

        df = pl.DataFrame(rows)
        df = df.with_columns(
            (pl.col("count") / total * 100).round(2).alias("pct_of_total")
        ).with_columns([
            (pl.col("pct_of_total") < MIN_SLICE_PCT).alias("underrepresented"),
            (pl.col("pct_of_total") > MAX_SLICE_PCT).alias("overrepresented"),
        ])

        dim_summary: dict = {}
        for row in df.iter_rows(named=True):
            dim_summary[row["value"]] = {
                "count":               row["count"],
                "pct_of_total":        row["pct_of_total"],
                "mean_manifest_chars": row["mean_manifest"],
                "underrepresented":    row["underrepresented"],
                "overrepresented":     row["overrepresented"],
            }
        summary[dim] = dim_summary

    return summary


def detect_imbalances(summary: dict) -> list[str]:
    """Return plain-english messages describing any imbalances found."""
    messages = []
    for dim, groups in summary.items():
        under = [v for v, s in groups.items() if s["underrepresented"]]
        over  = [v for v, s in groups.items() if s["overrepresented"]]
        if under:
            messages.append(f"[{dim}] Underrepresented (<{MIN_SLICE_PCT}%): {under}")
        if over:
            messages.append(f"[{dim}] Overrepresented (>{MAX_SLICE_PCT}%): {over}")
    return messages


# ── bias mitigation ───────────────────────────────────────────────────
def compute_sampling_weight(iac_type_pct: float) -> float:
    if iac_type_pct <= 0:
        return 1.0
    target_pct = min(iac_type_pct, MAX_SLICE_PCT)
    weight = target_pct / iac_type_pct
    return round(max(0.1, min(5.0, weight)), 4)


def apply_mitigation(
    records: list[dict],
    slices:  dict,
    total:   int,
    out_path: Path,
) -> dict:
    """
    Downsamples overrepresented iac_type slices to MAX_SLICE_PCT of total.
    Adds a sampling_weight field to every kept record.
    """
    rng = random.Random(RANDOM_SEED)

    iac_counts = {
        val: data["count"]
        for val, data in slices.get("iac_type", {}).items()
    }
    keep_ratios: dict[str, float] = {}
    for val, count in iac_counts.items():
        actual_pct = count / total * 100 if total else 0
        keep_ratios[val] = MAX_SLICE_PCT / actual_pct if actual_pct > MAX_SLICE_PCT else 1.0

    iac_pcts = {
        val: (count / total * 100 if total else 0)
        for val, count in iac_counts.items()
    }

    kept = 0
    dropped = 0
    kept_by_type: dict[str, int] = defaultdict(int)
    dropped_by_type: dict[str, int] = defaultdict(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            msgs = record.get("messages", [])
            if len(msgs) != 2:
                continue
            try:
                parsed   = json.loads(msgs[1].get("content", "{}"))
                manifest = parsed.get("params", {}).get("manifest_content", "")
            except Exception:
                continue

            iac = classify_iac_type(manifest)
            ratio = keep_ratios.get(iac, 1.0)

            if rng.random() > ratio:
                dropped += 1
                dropped_by_type[iac] += 1
                continue

            record_out = dict(record)
            meta = dict(record_out.get("_meta", {}))
            meta["sampling_weight"] = compute_sampling_weight(iac_pcts.get(iac, 100.0))
            meta["iac_type"]        = iac
            record_out["_meta"]     = meta

            f.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            kept += 1
            kept_by_type[iac] += 1

    return {
        "strategy":            "capped_downsample_on_iac_type",
        "max_slice_pct":       MAX_SLICE_PCT,
        "random_seed":         RANDOM_SEED,
        "original_total":      total,
        "balanced_total":      kept,
        "dropped":             dropped,
        "kept_by_iac_type":    dict(kept_by_type),
        "dropped_by_iac_type": dict(dropped_by_type),
        "keep_ratios":         {k: round(v, 4) for k, v in keep_ratios.items()},
        "output_path":         str(out_path),
    }


def run_bias_detection() -> dict:
    in_path       = _PROC / "training_records.jsonl"
    balanced_path = _PROC / "training_records_balanced.jsonl"
    out_path      = _LOGS / "bias_report.json"

    assert in_path.exists(), f"No training records at {in_path}"
    log.info("Loading records from %s", in_path)

    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(records)
    log.info("Analysing %d records across 4 slice dimensions", total)

    slices     = build_slices(records)
    summary    = summarise_slices(slices, total)
    imbalances = detect_imbalances(summary)

    log.info("Applying capped-downsample mitigation (max_slice_pct=%.0f%%)", MAX_SLICE_PCT)
    mitigation = apply_mitigation(records, slices, total, balanced_path)
    log.info(
        "Mitigation complete — kept %d / %d records → %s",
        mitigation["balanced_total"], total, balanced_path,
    )

    report = {
        "total_records":    total,
        "min_slice_pct":    MIN_SLICE_PCT,
        "max_slice_pct":    MAX_SLICE_PCT,
        "imbalances_found": len(imbalances),
        "imbalances":       imbalances,
        "slices":           summary,
        "mitigation":       mitigation,
    }

    out_path.write_text(json.dumps(report, indent=2))

    if imbalances:
        log.warning("Bias detection found %d imbalances:", len(imbalances))
        for msg in imbalances:
            log.warning("  %s", msg)
    else:
        log.info("No imbalances detected")

    return report


if __name__ == "__main__":
    run_bias_detection()
