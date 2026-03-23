"""
context_tool_parser.py

Phase 2C — Per-sample tool schema extractor.

Tools are provided to the model at runtime via RAG — there is no global registry.
Each eval sample carries its own tool context embedded in its system message.
This module extracts those tool schemas from the system message so param validation
can use the same tool context the model actually saw during generation.

Handles all three formats found in the training data:

  Format 1 — Names only (no schema, no validation possible):
    "Available Tools: scale_service, restart_pod"

  Format 2 — Simple JSON dict (parameters as list of names):
    "Available Tools: {"generate_invoice": {"description": "...",
                        "parameters": ["customer_name", "items"],
                        "required": ["customer_name", "items"]}}"

  Format 3 — JSON Schema dict (parameters as properties dict with types):
    "Available Tools: {"get_weather": {"parameters": {
                          "properties": {"city": {"type": "string"}},
                          "required": ["city"]}}}"

  Format 4 — Empty:
    "Available Tools: {}"

Returns a normalised dict keyed by tool name:
  {
    "tool_name": {
      "all_params":      list[str],   # all known param names ([] if unknown)
      "required_params": list[str],   # required param names  ([] if unknown)
      "param_types":     dict[str, str], # {param: type_str}  ({} if unknown)
      "has_schema":      bool,        # False for name-only format
    }
  }
"""

from __future__ import annotations

import json
import re
from loguru import logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_tools_from_system_message(system_content: str) -> dict[str, dict]:
    """
    Parse tool definitions from a system message string.

    Returns normalised {tool_name: schema_dict} or {} if no tools found.
    """
    if not system_content or not system_content.strip():
        return {}

    # Locate "Available Tools:" marker (case-insensitive)
    match = re.search(r"available tools\s*:\s*", system_content, re.IGNORECASE)
    if not match:
        return {}

    after = system_content[match.end():].strip()

    if not after:
        return {}

    # --- Format 1: comma-separated names (no JSON) ---
    if not after.startswith("{") and not after.startswith("["):
        names = [n.strip() for n in after.split(",") if n.strip()]
        if names:
            return {
                name: {
                    "all_params": [],
                    "required_params": [],
                    "param_types": {},
                    "has_schema": False,
                }
                for name in names
            }
        return {}

    # --- Formats 2/3/4: JSON dict ---
    try:
        data = json.loads(after)
    except json.JSONDecodeError:
        # Try to extract the first complete JSON object if there's trailing text
        data = _extract_first_json_object(after)
        if data is None:
            return {}

    if not isinstance(data, dict) or not data:
        return {}  # empty {} or wrong type

    result: dict[str, dict] = {}
    for tool_name, tool_def in data.items():
        if not isinstance(tool_def, dict):
            result[tool_name] = {
                "all_params": [],
                "required_params": [],
                "param_types": {},
                "has_schema": False,
            }
            continue

        params_field = tool_def.get("parameters", [])
        required_field = tool_def.get("required", [])

        # Format 2: parameters is a list of names
        if isinstance(params_field, list):
            all_params = [str(p) for p in params_field if p]
            required_params = [str(r) for r in required_field if r] if isinstance(required_field, list) else []
            result[tool_name] = {
                "all_params": all_params,
                "required_params": required_params,
                "param_types": {},
                "has_schema": True,
            }

        # Format 3: parameters is a JSON Schema dict with "properties"
        elif isinstance(params_field, dict):
            properties = params_field.get("properties", {})
            if isinstance(properties, dict):
                all_params = list(properties.keys())
                param_types = {
                    k: v.get("type", "unknown")
                    for k, v in properties.items()
                    if isinstance(v, dict)
                }
            else:
                all_params = []
                param_types = {}

            # required can live in params_field or top-level
            req = params_field.get("required", required_field)
            required_params = [str(r) for r in req if r] if isinstance(req, list) else []

            result[tool_name] = {
                "all_params": all_params,
                "required_params": required_params,
                "param_types": param_types,
                "has_schema": True,
            }

        else:
            result[tool_name] = {
                "all_params": [],
                "required_params": [],
                "param_types": {},
                "has_schema": False,
            }

    return result


def get_sample_tool_schemas(sample: dict) -> dict[str, dict]:
    """
    Extract tool schemas from a full prediction sample dict.

    Looks in sample["sample"]["messages"] for the first system-role message.
    Falls back to sample["messages"] if the nested structure is absent.

    Returns normalised tool schema dict (empty if no tools defined for this sample).
    """
    # predictions carry original sample under "sample" key (from generator.py)
    inner = sample.get("sample") or sample
    messages = inner.get("messages", [])

    for msg in messages:
        if msg.get("role") == "system":
            return extract_tools_from_system_message(msg.get("content", ""))

    return {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_first_json_object(text: str) -> dict | None:
    """
    Try to extract the first complete {...} JSON object from text that may
    have trailing characters after the closing brace.
    """
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
