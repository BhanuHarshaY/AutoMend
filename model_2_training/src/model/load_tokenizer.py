"""
load_tokenizer.py

Loads and configures the tokenizer for Model 2 training.

Responsibilities:
  - Load tokenizer from HuggingFace Hub
  - Configure pad token if not set
  - Set padding side to right for causal LM training
  - Return configured tokenizer

This module is shared between training, evaluation, and testing.
"""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from loguru import logger


def load_tokenizer(
    tokenizer_name: str,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizerBase:
    """
    Load and configure the tokenizer for the given model.

    The function performs three post-load configuration steps that are
    critical for stable causal-LM supervised fine-tuning:

    1. **Pad token** — many instruction-tuned models (including Qwen2.5)
       ship without a dedicated ``pad_token``.  When it is absent we alias
       it to ``eos_token`` so that padding calls do not fail and the padding
       id is a valid vocabulary index.  The model's embedding table may need
       to be resized if the vocabulary was not already expecting a pad id; that
       step is handled separately in the model-loading module.

    2. **Padding side** — HuggingFace defaults to left-padding for some
       tokenizers.  Right-padding is required for causal LM training because
       loss is computed left-to-right; left-padded batches would shift the
       label positions relative to the padding mask.

    Args:
        tokenizer_name:   HuggingFace Hub model identifier or local path,
                          e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``.
        trust_remote_code: Whether to allow the tokenizer to execute custom
                          code shipped with the model repository.  Required
                          for Qwen models.  Defaults to ``True``.

    Returns:
        A fully configured :class:`~transformers.PreTrainedTokenizerBase`
        instance ready for use in dataset building, training, and evaluation.

    Raises:
        OSError: if the tokenizer cannot be found locally or on the Hub.

    Example::

        tokenizer = load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer.apply_chat_template(messages, tokenize=False)
    """
    logger.info(
        f"Loading tokenizer: '{tokenizer_name}' "
        f"(trust_remote_code={trust_remote_code})"
    )

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )

    logger.info(
        f"Tokenizer loaded — vocab_size={tokenizer.vocab_size}, "
        f"model_max_length={tokenizer.model_max_length}"
    )

    # ------------------------------------------------------------------
    # 1. Ensure pad_token is set
    # ------------------------------------------------------------------
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(
            f"pad_token was not set; aliased to eos_token "
            f"('{tokenizer.eos_token}', id={tokenizer.eos_token_id})"
        )
    else:
        logger.info(
            f"pad_token already set: '{tokenizer.pad_token}' "
            f"(id={tokenizer.pad_token_id})"
        )

    # ------------------------------------------------------------------
    # 2. Enforce right-padding for causal LM training
    # ------------------------------------------------------------------
    if tokenizer.padding_side != "right":
        logger.info(
            f"Changing padding_side from '{tokenizer.padding_side}' to 'right' "
            f"(required for causal LM training)."
        )
        tokenizer.padding_side = "right"
    else:
        logger.info("padding_side is already 'right' — no change needed.")

    return tokenizer
