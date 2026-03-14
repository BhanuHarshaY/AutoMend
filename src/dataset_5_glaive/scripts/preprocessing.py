"""
Preprocessing Script for Glaive Function Calling v2
Parses raw JSONL, extracts function calls, cleans and engineers features.
Uses Polars for tabular operations and Ray for distributed per-record parsing.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from src.config.paths import get_ds5_raw_dir, get_ds5_processed_dir
    from src.config.ray_config import init_ray
    RAW_DIR = get_ds5_raw_dir()
    PROCESSED_DIR = get_ds5_processed_dir()
except ImportError:
    RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
    PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

    def init_ray(**kw):
        pass

RAW_FILE = RAW_DIR / "glaive_raw.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "glaive_processed.jsonl"

ERROR_KEYWORDS = [
    "error", "invalid", "failed", "null", "none",
    "exception", "traceback", "undefined", "not found",
]


def extract_function_signatures(system: str) -> dict:
    signatures = {}
    if not system or not isinstance(system, str):
        return signatures
    try:
        json_pattern = r"\{.*\}"
        match = re.search(json_pattern, system, re.DOTALL)
        if match:
            func = json.loads(match.group())
            if isinstance(func, dict) and "name" in func:
                name = func.get("name", "unknown")
                params = func.get("parameters", {})
                signatures[name] = {
                    "description": func.get("description", ""),
                    "parameters": list(params.get("properties", {}).keys()) if isinstance(params, dict) else [],
                    "required": params.get("required", []) if isinstance(params, dict) else [],
                }
    except (json.JSONDecodeError, AttributeError):
        pass
    return signatures


def extract_function_calls(text: str) -> list:
    calls = []
    pattern = r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            fixed = re.sub(r"'(\{.*?\})'", r"\1", match, flags=re.DOTALL)
            parsed = json.loads(fixed)
            if isinstance(parsed.get("arguments"), str):
                try:
                    parsed["arguments"] = json.loads(parsed["arguments"])
                except json.JSONDecodeError:
                    pass
            calls.append(parsed)
        except json.JSONDecodeError:
            calls.append({"__malformed__": match[:100]})
    return calls


def detect_error_handling(chat: str) -> dict:
    if not chat or not isinstance(chat, str):
        return {"has_error_handling": False, "error_keywords_found": []}
    chat_lower = chat.lower()
    found_keywords = [kw for kw in ERROR_KEYWORDS if kw in chat_lower]
    has_function_error = bool(
        re.search(r"<functionresponse>.*?(error|failed|invalid).*?</functionresponse>", chat_lower, re.DOTALL)
    )
    has_conditional_error = bool(re.search(r"(if|when).{0,30}(error|fail|invalid)", chat_lower))
    return {
        "has_error_handling": len(found_keywords) > 0,
        "has_function_error_response": has_function_error,
        "has_conditional_error": has_conditional_error,
        "error_keywords_found": found_keywords,
    }


def count_turns(chat: str) -> int:
    """Count USER turns in chat string."""
    return len(re.findall(r"USER:", chat)) if chat else 0


def has_malformed_calls(calls: list) -> bool:
    """Check if any call in the list is malformed."""
    return any("__malformed__" in c for c in calls)


def classify_complexity(calls: list) -> str:
    if not calls:
        return "none"
    if len(calls) == 1:
        if "__malformed__" in calls[0]:
            return "malformed"
        args = calls[0].get("arguments", {})
        if isinstance(args, dict) and len(args) <= 2:
            return "simple"
        return "moderate"
    return "complex"


def process_record(record: dict) -> Optional[dict]:
    system = record.get("system", "")
    chat = record.get("chat", "")
    if not chat or not isinstance(chat, str):
        return None

    calls = extract_function_calls(chat)
    turn_count = count_turns(chat)
    complexity = classify_complexity(calls)
    signatures = extract_function_signatures(system)
    error_info = detect_error_handling(chat)

    return {
        "system": system,
        "chat": chat,
        "num_turns": turn_count,
        "num_calls": len(calls),
        "complexity_tier": complexity,
        "has_parallel": len(calls) > 1,
        "has_malformed": has_malformed_calls(calls),
        "function_calls": json.dumps(calls),
        "num_defined_functions": len(signatures),
        "defined_function_names": json.dumps(list(signatures.keys())),
        "function_signatures": json.dumps(signatures),
        "has_error_handling": error_info["has_error_handling"],
        "has_function_error_response": error_info["has_function_error_response"],
        "has_conditional_error": error_info["has_conditional_error"],
        "error_keywords_found": json.dumps(error_info["error_keywords_found"]),
    }


def remap_to_chatml(record: dict) -> dict:
    system = record.get("system", "")
    chat = record.get("chat", "")
    calls = json.loads(record.get("function_calls", "[]"))

    user_match = re.search(r"USER:\s*(.*?)(?=ASSISTANT:|$)", chat, re.DOTALL)
    user_content = user_match.group(1).strip() if user_match else chat[:200]

    tool_definitions = record.get("function_signatures", "{}")
    system_content = (
        f"You are AutoMend, an MLOps remediation engine. "
        f"Convert user requests into valid JSON workflow definitions.\n"
        f"Available Tools: {tool_definitions}"
    )

    if calls and not has_malformed_calls(calls):
        steps = [{"tool": c.get("name", "unknown"), "params": c.get("arguments", {})} for c in calls]
        assistant_content = json.dumps({"workflow": {"steps": steps}}, indent=2)
    else:
        assistant_match = re.search(r"ASSISTANT:\s*(.*?)(?=USER:|<\|endoftext\|>|$)", chat, re.DOTALL)
        raw_response = assistant_match.group(1).strip() if assistant_match else ""
        assistant_content = json.dumps({"workflow": {"steps": []}, "message": raw_response[:200]})

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "complexity_tier": record.get("complexity_tier"),
        "has_error_handling": record.get("has_error_handling"),
        "num_turns": record.get("num_turns"),
        "num_calls": record.get("num_calls"),
    }


def run_preprocessing(
    raw_file: Path = RAW_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pl.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data from %s", raw_file)
    records = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d raw records", len(records))

    try:
        import ray
        init_ray()

        @ray.remote
        def _process_batch(batch: list[dict]) -> list[dict]:
            return [r for rec in batch if (r := process_record(rec)) is not None]

        batch_size = max(1, len(records) // 8)
        batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
        refs = [_process_batch.remote(b) for b in batches]
        processed = []
        for batch_result in ray.get(refs):
            processed.extend(batch_result)
        logger.info("Processed %d records via Ray", len(processed))
    except Exception as e:
        logger.warning("Ray processing failed (%s), falling back to sequential", e)
        processed = [r for rec in records if (r := process_record(rec)) is not None]

    df = pl.DataFrame(processed)
    logger.info("--- Dataset Statistics ---")
    logger.info("Total records:           %d", df.height)
    logger.info("Avg turns:               %.2f", df["num_turns"].mean())
    logger.info("Avg calls:               %.2f", df["num_calls"].mean())
    logger.info("Malformed calls:         %d", df["has_malformed"].sum())

    logger.info("Saving processed data to %s", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for record in processed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Remapping %d records to ChatML format...", len(processed))
    chatml_records = [remap_to_chatml(r) for r in processed]
    chatml_file = PROCESSED_DIR / "glaive_chatml.jsonl"
    with open(chatml_file, "w", encoding="utf-8") as f:
        for record in chatml_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("ChatML file saved: %d records to %s", len(chatml_records), chatml_file)

    return df


if __name__ == "__main__":
    df = run_preprocessing()
    print(f"\nPreprocessing complete. Shape: {df.shape}")
