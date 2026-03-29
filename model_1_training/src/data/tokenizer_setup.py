"""
tokenizer_setup.py

Builds a custom MLOps infrastructure token vocabulary and adds the tokens
as special tokens to the RoBERTa tokenizer.

The Track A data pipeline produces `sequence_ids` that are *not* natural-language
subword tokens — they are discretized integer IDs representing infrastructure
telemetry signals:

  DS1 (Alibaba):
    100-109  CPU utilization buckets    -> [CPU_0] .. [CPU_9]
    200-209  Memory utilization buckets -> [MEM_0] .. [MEM_9]
    300-304  Status tokens              -> [STS_TERMINATED] .. [STS_UNKNOWN]
    400-403  Event tokens               -> [EVT_ADD] .. [EVT_UNKNOWN]

  DS2 (Loghub):
    1-999    Event template IDs         -> [TMPL_1] .. [TMPL_999]
    0        Padding value              -> [PAD_TOK]

This module provides:
  - build_token_vocab()         → int-to-string mapping
  - setup_tokenizer(model_name) → (tokenizer, int_to_token_id)
"""

from __future__ import annotations
from typing import Dict, Tuple

from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# DS1 semantic token names
# ---------------------------------------------------------------------------
_STATUS_NAMES = {300: "TERMINATED", 301: "FAILED", 302: "WAITING", 303: "RUNNING", 304: "UNKNOWN"}
_EVENT_NAMES = {400: "ADD", 401: "REMOVE", 402: "FAILURE", 403: "UNKNOWN"}

# Upper bound for DS2 event template IDs (E-codes parsed as ints)
_MAX_TEMPLATE_ID = 999


def build_token_vocab() -> Dict[int, str]:
    """
    Build the complete integer-ID → special-token-string mapping.

    Returns a dict like {0: "[PAD_TOK]", 100: "[CPU_0]", ..., 55: "[TMPL_55]", ...}.
    """
    vocab: Dict[int, str] = {}

    vocab[0] = "[PAD_TOK]"

    for i in range(10):
        vocab[100 + i] = f"[CPU_{i}]"

    for i in range(10):
        vocab[200 + i] = f"[MEM_{i}]"

    for code, name in _STATUS_NAMES.items():
        vocab[code] = f"[STS_{name}]"

    for code, name in _EVENT_NAMES.items():
        vocab[code] = f"[EVT_{name}]"

    for i in range(1, _MAX_TEMPLATE_ID + 1):
        if i not in vocab:
            vocab[i] = f"[TMPL_{i}]"

    logger.info(f"Built MLOps token vocabulary with {len(vocab)} entries")
    return vocab


def setup_tokenizer(
    model_name: str = "roberta-base",
) -> Tuple[PreTrainedTokenizerFast, Dict[int, str], Dict[int, int]]:
    """
    Load the base tokenizer, add all custom infrastructure tokens, and
    build a mapping from raw integer IDs to tokenizer input IDs.

    Args:
        model_name: HuggingFace model identifier for the tokenizer.

    Returns:
        tokenizer:       The modified tokenizer with added special tokens.
        int_to_token:     Dict mapping raw int ID → special token string.
        int_to_token_id:  Dict mapping raw int ID → tokenizer input_id (for
                          direct embedding lookup without re-tokenization).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    int_to_token = build_token_vocab()

    new_tokens = sorted(set(int_to_token.values()))
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    logger.info(
        f"Added {num_added} special tokens to '{model_name}' tokenizer "
        f"(vocab size: {len(tokenizer)})"
    )

    int_to_token_id: Dict[int, int] = {}
    for raw_id, token_str in int_to_token.items():
        tid = tokenizer.convert_tokens_to_ids(token_str)
        int_to_token_id[raw_id] = tid

    return tokenizer, int_to_token, int_to_token_id


def sequence_ids_to_string(sequence_ids: list[int], int_to_token: Dict[int, str]) -> str:
    """
    Convert a list of raw integer IDs into a space-separated string of
    special token names for tokenizer encoding.

    Unknown IDs are mapped to [PAD_TOK].
    """
    pad = int_to_token.get(0, "[PAD_TOK]")
    return " ".join(int_to_token.get(sid, pad) for sid in sequence_ids)
