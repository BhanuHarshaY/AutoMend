"""
dataset_contract.py

Defines and enforces the training data contract for Track B ChatML samples.

A valid sample must conform to:
  {
    "messages": [
      {"role": "system",    "content": "..."},   # optional
      {"role": "user",      "content": "..."},   # required
      {"role": "assistant", "content": "..."}    # required, non-empty
    ],
    "metadata": { ... }  # optional
  }

This module is intentionally free of ML dependencies.
"""

from __future__ import annotations
from typing import Any
from collections import Counter


VALID_ROLES = {"system", "user", "assistant"}


class ContractViolation(Exception):
    """Raised when a training sample fails the dataset contract."""


def validate_sample(sample: Any) -> tuple[bool, str]:
    """
    Validate a single training sample against the Track B contract.

    Args:
        sample: The parsed JSON object to validate.

    Returns:
        (True, "") if valid.
        (False, reason) if invalid, where reason is a short error category string.
    """
    if not isinstance(sample, dict):
        return False, "not_a_dict"

    messages = sample.get("messages")
    if messages is None:
        return False, "missing_messages_key"
    if not isinstance(messages, list):
        return False, "messages_not_a_list"
    if len(messages) == 0:
        return False, "messages_empty"

    has_assistant = False
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"message_{i}_not_a_dict"
        if "role" not in msg:
            return False, f"message_{i}_missing_role"
        if "content" not in msg:
            return False, f"message_{i}_missing_content"
        role = msg["role"]
        if not isinstance(role, str):
            return False, f"message_{i}_role_not_string"
        if role not in VALID_ROLES:
            return False, f"message_{i}_invalid_role:{role}"
        content = msg["content"]
        if not isinstance(content, str):
            return False, f"message_{i}_content_not_string"
        if role == "assistant":
            if len(content.strip()) == 0:
                return False, "assistant_content_empty"
            has_assistant = True

    if not has_assistant:
        return False, "no_assistant_message"

    metadata = sample.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return False, "metadata_not_a_dict"

    return True, ""


def assert_sample(sample: Any) -> None:
    """
    Assert that a sample is valid. Raises ContractViolation if not.

    Args:
        sample: The parsed JSON object to validate.

    Raises:
        ContractViolation: with a descriptive message.
    """
    is_valid, reason = validate_sample(sample)
    if not is_valid:
        raise ContractViolation(
            f"Sample failed contract: {reason}. "
            f"Sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}"
        )


def summarize_violations(violations: list[str]) -> dict[str, int]:
    """
    Summarize a list of violation reason strings into counts by category.

    Args:
        violations: List of reason strings returned by validate_sample.

    Returns:
        Dict mapping error category -> count, sorted by frequency.
    """
    # Normalize: strip message index from per-message errors
    normalized = []
    for v in violations:
        if v.startswith("message_") and "_not_a_dict" in v:
            normalized.append("message_not_a_dict")
        elif v.startswith("message_") and "_missing_role" in v:
            normalized.append("message_missing_role")
        elif v.startswith("message_") and "_missing_content" in v:
            normalized.append("message_missing_content")
        elif v.startswith("message_") and "_role_not_string" in v:
            normalized.append("message_role_not_string")
        elif v.startswith("message_") and "_content_not_string" in v:
            normalized.append("message_content_not_string")
        elif v.startswith("message_") and "_invalid_role" in v:
            normalized.append("message_invalid_role")
        else:
            normalized.append(v)

    counts = Counter(normalized)
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
