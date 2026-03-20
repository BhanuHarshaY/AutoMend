"""
dataset_builder.py

Converts validated ChatML samples into model-consumable PyTorch Dataset objects.

Responsibilities:
  - Format message sequences using the tokenizer's chat template
  - Tokenize formatted strings with truncation and padding
  - Return structured examples for the training collator
  - Provide a class-based interface extensible to RAG variants

Does NOT apply loss masking (handled by collators.py).
Does NOT load data from disk (handled by load_jsonl.py).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from loguru import logger


class ChatMLSupervisedDataset(Dataset):
    """
    PyTorch Dataset for supervised fine-tuning on ChatML-formatted conversations.

    At construction time every sample is formatted via the tokenizer's chat
    template and then tokenized (truncation + padding to max_seq_length).  The
    raw ``messages`` list is retained alongside the tokenised tensors so that
    the collator can re-derive per-turn boundaries for assistant-only loss
    masking without needing the original text.

    Args:
        samples:        List of validated sample dicts, each containing at
                        minimum a ``"messages"`` key whose value is a list of
                        ``{"role": str, "content": str}`` dicts.
        tokenizer:      A HuggingFace ``PreTrainedTokenizer`` (or Fast variant)
                        that has been configured for the target model.
        max_seq_length: Maximum token sequence length.  Sequences that are
                        longer are truncated; shorter ones are right-padded.

    Example::

        dataset = ChatMLSupervisedDataset(samples, tokenizer, max_seq_length=2048)
        example = dataset[0]
        # example.keys() -> {"input_ids", "attention_mask", "messages"}
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        logger.info(
            f"Building ChatMLSupervisedDataset: {len(samples)} samples, "
            f"max_seq_length={max_seq_length}"
        )

        self._examples: list[dict[str, Any]] = []
        skipped = 0

        for idx, sample in enumerate(samples):
            messages: list[dict] = sample.get("messages", [])

            # ----------------------------------------------------------------
            # Format the full conversation with the model's chat template.
            # add_generation_prompt=False because this is a full turn (including
            # the assistant reply) used for SFT — we do NOT want a trailing
            # assistant prompt header.
            # ----------------------------------------------------------------
            try:
                formatted_text: str = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as exc:
                logger.warning(
                    f"Sample {idx}: apply_chat_template failed ({exc}); skipping."
                )
                skipped += 1
                continue

            # ----------------------------------------------------------------
            # Tokenize with fixed-length padding so every batch element has
            # the same shape.  The batch dimension added by return_tensors="pt"
            # is squeezed away so each example is a 1-D tensor.
            # ----------------------------------------------------------------
            encoding = tokenizer(
                formatted_text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            self._examples.append(
                {
                    "input_ids": encoding["input_ids"].squeeze(0),        # (seq_len,)
                    "attention_mask": encoding["attention_mask"].squeeze(0),  # (seq_len,)
                    "messages": messages,   # kept raw for collator label masking
                }
            )

        logger.info(
            f"Dataset ready: {len(self._examples)} examples "
            f"({skipped} skipped due to template errors)."
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Return a single example.

        Returns:
            dict with keys:
              - ``"input_ids"``       : ``torch.LongTensor`` of shape ``(max_seq_length,)``
              - ``"attention_mask"``  : ``torch.LongTensor`` of shape ``(max_seq_length,)``
              - ``"messages"``        : raw list of ``{"role", "content"}`` dicts
        """
        return self._examples[idx]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_dataset(
    samples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> ChatMLSupervisedDataset:
    """
    Factory function that constructs a :class:`ChatMLSupervisedDataset`.

    This thin wrapper exists so call-sites and configuration-driven pipelines
    can obtain a dataset without importing the class directly.

    Args:
        samples:        Validated sample dicts (from ``load_jsonl``).
        tokenizer:      Configured tokenizer for the target model.
        max_seq_length: Sequence length cap for truncation / padding.

    Returns:
        A ready-to-use :class:`ChatMLSupervisedDataset`.
    """
    return ChatMLSupervisedDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
