"""
collators.py

Implements assistant-only loss masking for supervised fine-tuning.

The collator ensures the model learns only from the assistant response tokens,
not from the system/user prompt tokens. This is achieved by setting labels
to -100 for all non-assistant tokens.

Masking strategy:
  For each example, compute the prompt length by formatting the conversation
  up to (but not including) the last assistant turn with add_generation_prompt=True.
  Everything before that boundary gets label=-100.
  For multi-turn conversations, each assistant turn is handled in sequence.

This module is intentionally decoupled from the dataset and trainer so it can
be tested independently.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase
from loguru import logger

# Sentinel used by cross-entropy to ignore a position during loss computation.
_IGNORE_INDEX: int = -100


class AssistantOnlyCollator:
    """
    Data collator that applies assistant-only loss masking.

    Given a batch of examples from :class:`~model_2_training.src.data.dataset_builder.ChatMLSupervisedDataset`
    (each containing ``input_ids``, ``attention_mask``, and the raw
    ``messages`` list), this collator:

    1. Copies ``input_ids`` into a ``labels`` tensor.
    2. For each example, identifies every token position that belongs to an
       assistant turn and sets all *other* positions to ``-100`` in
       ``labels``.  The cross-entropy loss then only back-propagates through
       assistant response tokens.
    3. Stacks all per-example tensors into batch tensors and returns a dict
       suitable for passing directly to a HuggingFace ``Trainer`` or a manual
       training loop.

    **How assistant boundaries are found (multi-turn safe)**

    For each assistant turn *k* in the conversation:

    * Build a *prefix* message list that contains every turn *up to and
      including the user message that precedes turn k*, then call
      ``apply_chat_template(..., add_generation_prompt=True)`` so the
      tokenizer appends the assistant-start tokens (``<|im_start|>assistant``
      for ChatML / Qwen).
    * Tokenise that prefix — its token count is the ``prompt_len`` where the
      assistant reply begins in the full sequence.
    * Build the same prefix *plus* the assistant reply, tokenise it — its
      length is the ``reply_end`` where that turn ends.
    * Mark ``labels[prompt_len : reply_end]`` as active; everything else
      remains ``-100``.

    This approach is robust to variable-length system/user prompts and works
    correctly for multi-turn dialogues.

    Args:
        tokenizer:      The same tokenizer used to build the dataset.
        max_seq_length: The fixed sequence length used during dataset
                        construction.  Needed so that tokenisation calls inside
                        the collator are consistent with the dataset's tensors.

    Example::

        collator = AssistantOnlyCollator(tokenizer, max_seq_length=2048)
        batch = collator(dataset_examples)
        # batch.keys() -> {"input_ids", "attention_mask", "labels"}
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize_for_length(self, text: str) -> int:
        """
        Return the number of non-padded tokens in ``text`` after applying the
        same truncation settings used during dataset construction.

        We deliberately do NOT pad here — we only need the length of the real
        content so we can locate turn boundaries inside the already-padded
        ``input_ids`` tensor.
        """
        ids = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]
        return len(ids)

    def _find_assistant_spans(
        self,
        messages: list[dict],
    ) -> list[tuple[int, int]]:
        """
        Compute ``(start, end)`` token index pairs for every assistant turn
        in the conversation, relative to the beginning of the full tokenised
        sequence.

        The returned spans are *exclusive* on the right — i.e. the slice
        ``labels[start:end]`` covers exactly the assistant reply tokens
        including the end-of-turn token appended by the chat template.

        Args:
            messages: Raw list of ``{"role": str, "content": str}`` dicts.

        Returns:
            List of ``(start_token, end_token)`` tuples, one per assistant
            turn.  Returns an empty list if no assistant turn is found.
        """
        spans: list[tuple[int, int]] = []

        # Walk the messages; when we encounter an assistant turn, compute the
        # boundary positions in the full token sequence.
        for turn_idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # ---- Prompt length (everything up to & including the assistant
            #      start marker) -------------------------------------------
            # Build prefix = all messages before this assistant turn, then ask
            # the template to add the assistant-start prompt.
            prefix_messages = messages[:turn_idx]

            try:
                prefix_text: str = self.tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=True,   # appends <|im_start|>assistant\n
                )
            except Exception as exc:
                logger.warning(
                    f"apply_chat_template(prefix, add_generation_prompt=True) "
                    f"failed for turn {turn_idx}: {exc}. Skipping turn."
                )
                continue

            prompt_len = self._tokenize_for_length(prefix_text)

            # ---- Reply end (prefix + this assistant turn) -----------------
            # Include the assistant turn itself (without add_generation_prompt
            # so the template appends the end-of-turn token after the reply).
            prefix_plus_reply_messages = messages[: turn_idx + 1]

            try:
                prefix_plus_reply_text: str = self.tokenizer.apply_chat_template(
                    prefix_plus_reply_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as exc:
                logger.warning(
                    f"apply_chat_template(prefix+reply) failed for turn "
                    f"{turn_idx}: {exc}. Skipping turn."
                )
                continue

            reply_end = self._tokenize_for_length(prefix_plus_reply_text)

            if reply_end <= prompt_len:
                logger.warning(
                    f"Turn {turn_idx}: reply_end ({reply_end}) <= prompt_len "
                    f"({prompt_len}).  Assistant turn appears empty after "
                    f"tokenisation; skipping."
                )
                continue

            spans.append((prompt_len, reply_end))

        return spans

    # ------------------------------------------------------------------
    # Collator entry point
    # ------------------------------------------------------------------

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate a list of dataset examples into a batch.

        Each element of ``batch`` must contain:
          - ``"input_ids"``      : 1-D ``LongTensor`` of length ``max_seq_length``
          - ``"attention_mask"`` : 1-D ``LongTensor`` of length ``max_seq_length``
          - ``"messages"``       : list of ``{"role", "content"}`` dicts

        Returns:
            Dict with keys ``"input_ids"``, ``"attention_mask"``, ``"labels"``,
            each a 2-D ``LongTensor`` of shape ``(batch_size, max_seq_length)``.
        """
        all_input_ids: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for example_idx, example in enumerate(batch):
            input_ids: torch.Tensor = example["input_ids"]        # (seq_len,)
            attention_mask: torch.Tensor = example["attention_mask"]  # (seq_len,)
            messages: list[dict] = example["messages"]

            seq_len = input_ids.size(0)

            # Start with labels = input_ids; we will mask out non-assistant
            # positions by setting them to _IGNORE_INDEX.
            labels: torch.Tensor = torch.full(
                (seq_len,), fill_value=_IGNORE_INDEX, dtype=torch.long
            )

            # Find the token spans corresponding to assistant turns.
            spans = self._find_assistant_spans(messages)

            if not spans:
                # No assistant turn found — mask the entire sequence.
                # The loss will be zero for this example, but we don't crash.
                logger.warning(
                    f"Batch example {example_idx}: no assistant spans found in "
                    f"conversation with {len(messages)} messages. "
                    f"Entire sequence will be masked (labels=-100)."
                )
            else:
                # Activate only the assistant reply positions.
                for start, end in spans:
                    # Clamp to the actual (possibly truncated) sequence length.
                    clamped_start = min(start, seq_len)
                    clamped_end = min(end, seq_len)

                    if clamped_start >= clamped_end:
                        logger.warning(
                            f"Batch example {example_idx}: span ({start}, {end}) "
                            f"is fully outside the tokenised length {seq_len} "
                            f"(likely truncated). Skipping span."
                        )
                        continue

                    labels[clamped_start:clamped_end] = input_ids[clamped_start:clamped_end]

            # Padding positions should also be masked — cross-entropy must not
            # see pad tokens.  attention_mask==0 means padding.
            labels[attention_mask == 0] = _IGNORE_INDEX

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": torch.stack(all_input_ids, dim=0),          # (B, seq_len)
            "attention_mask": torch.stack(all_attention_masks, dim=0),  # (B, seq_len)
            "labels": torch.stack(all_labels, dim=0),                 # (B, seq_len)
        }
