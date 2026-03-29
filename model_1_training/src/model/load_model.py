"""
load_model.py

Load roberta-base for sequence classification and resize token embeddings
to accommodate the custom MLOps infrastructure tokens.
"""

from __future__ import annotations

from loguru import logger
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast


def load_model(
    model_name: str = "roberta-base",
    num_labels: int = 7,
    tokenizer: PreTrainedTokenizerFast | None = None,
    device_map: str | None = None,
) -> AutoModelForSequenceClassification:
    """
    Load a pre-trained model for sequence classification and resize embeddings.

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output classes.
        tokenizer: If provided, resize embeddings to match tokenizer vocab size.
        device_map: Device placement strategy (None for manual, "auto" for accelerate).

    Returns:
        Model ready for fine-tuning.
    """
    logger.info(f"Loading model '{model_name}' with {num_labels} labels")

    kwargs = {"num_labels": num_labels}
    if device_map:
        kwargs["device_map"] = device_map

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)

    if tokenizer is not None:
        old_size = model.get_input_embeddings().weight.shape[0]
        new_size = len(tokenizer)
        if new_size != old_size:
            model.resize_token_embeddings(new_size)
            logger.info(
                f"Resized token embeddings: {old_size} -> {new_size} "
                f"(+{new_size - old_size} custom MLOps tokens)"
            )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters — total: {total_params:,}, trainable: {trainable:,}")

    return model
