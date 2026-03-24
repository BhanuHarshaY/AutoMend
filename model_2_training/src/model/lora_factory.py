"""
lora_factory.py

Builds and attaches LoRA/QLoRA adapters to a loaded model.

Responsibilities:
  - Read LoRA hyperparameters from a config dict
  - Build a LoraConfig
  - Call get_peft_model to attach adapters
  - Enable gradient checkpointing if quantized (for memory efficiency)
  - Log trainable parameter counts before and after

Typical LoRA settings for Qwen SFT:
  rank=16, alpha=32, dropout=0.05, target all projection layers
"""

from __future__ import annotations
from loguru import logger

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def _count_trainable_params(model) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    bias: str = "none",
    total_layers: int = 28,
    lora_layers: int | None = None,
) -> LoraConfig:
    """
    Build a LoraConfig for supervised fine-tuning.

    Args:
        r: LoRA rank. Higher = more capacity, more params.
        lora_alpha: LoRA scaling factor. Effective scale = alpha / r.
        lora_dropout: Dropout on adapter weights.
        target_modules: List of module names to apply LoRA to.
                        Defaults to standard Qwen projection layers.
        bias: Bias training mode: "none", "all", or "lora_only".
        total_layers: Total number of transformer layers in the model.
        lora_layers: Number of layers (from the end) to apply LoRA to.
                     None = all layers (default PEFT behaviour).
                     16   = last 16 layers only (matches MLX default).

    Returns:
        Configured LoraConfig.
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    layers_to_transform = None
    if lora_layers is not None:
        start = max(0, total_layers - lora_layers)
        layers_to_transform = list(range(start, total_layers))
        logger.info(
            f"LoRA restricted to last {lora_layers} layers "
            f"(layers {start}–{total_layers - 1})"
        )
    else:
        logger.info("LoRA applied to all layers")

    logger.info(
        f"Building LoraConfig — r={r}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}, targets={target_modules}"
    )

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        inference_mode=False,
        layers_to_transform=layers_to_transform,
    )


def attach_lora(model, lora_config: LoraConfig, is_quantized: bool = False):
    """
    Attach LoRA adapters to a loaded model.

    Args:
        model: The base model returned by load_model.
        lora_config: LoraConfig built by build_lora_config.
        is_quantized: If True, calls prepare_model_for_kbit_training first.

    Returns:
        PEFT model with LoRA adapters attached and only adapter weights trainable.
    """
    if is_quantized:
        logger.info("Preparing quantized model for k-bit training (gradient checkpointing enabled)")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    trainable_before, total_before = _count_trainable_params(model)
    logger.info(f"Before LoRA — trainable: {trainable_before:,} / {total_before:,} params")

    model = get_peft_model(model, lora_config)

    trainable_after, total_after = _count_trainable_params(model)
    pct = 100.0 * trainable_after / total_after if total_after > 0 else 0.0
    logger.info(
        f"After LoRA  — trainable: {trainable_after:,} / {total_after:,} params "
        f"({pct:.2f}% trainable)"
    )

    return model


def build_and_attach_lora(model, cfg: dict, is_quantized: bool = False):
    """
    Convenience function: build LoRA config from a config dict and attach it.

    Args:
        model: The base model.
        cfg: Dict with keys matching LoraConfig args:
             lora_r, lora_alpha, lora_dropout, lora_target_modules (or from model config),
             lora_bias.
        is_quantized: Whether the model was loaded with BnB quantization.

    Returns:
        PEFT model with adapters attached.
    """
    # Detect actual layer count from the model so this works for any architecture.
    # Falls back to cfg["total_layers"] (if set), then 28 (Qwen2.5-1.5B default).
    total_layers = (
        getattr(getattr(model, "config", None), "num_hidden_layers", None)
        or cfg.get("total_layers", 28)
    )
    logger.info(f"Model has {total_layers} transformer layers")

    lora_config = build_lora_config(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules"),
        bias=cfg.get("lora_bias", "none"),
        total_layers=total_layers,
        lora_layers=cfg.get("lora_layers"),
    )
    return attach_lora(model, lora_config, is_quantized=is_quantized)
