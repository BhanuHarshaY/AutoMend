"""
load_model.py

Loads a HuggingFace causal LM for supervised fine-tuning.

Device routing
--------------
  CUDA   bitsandbytes 4-bit / 8-bit quantization, bf16 or fp16
  CPU    Full precision fp32, no quantization
  MPS    This module is NOT called on MPS — the mlx_train backend handles
         model loading directly via mlx_lm.load(). If called from MPS by
         mistake, a clear error is raised.

Does NOT attach LoRA adapters (handled by lora_factory.py).
Does NOT load the tokenizer (handled by load_tokenizer.py).
"""

from __future__ import annotations
from loguru import logger

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from model_2_training.src.utils.device import detect_device, get_device_config


def build_bnb_config(quantization: str) -> BitsAndBytesConfig:
    """
    Build a BitsAndBytesConfig for 4-bit or 8-bit quantization (CUDA only).

    Args:
        quantization: "4bit" or "8bit".

    Returns:
        Configured BitsAndBytesConfig.

    Raises:
        ValueError: if quantization string is not recognized.
    """
    if quantization == "4bit":
        logger.info("Quantization: 4-bit NF4 (double quant, bf16 compute dtype)")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8bit":
        logger.info("Quantization: 8-bit (bitsandbytes LLM.int8)")
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Unsupported quantization: {quantization!r}. Choose '4bit' or '8bit'."
        )


def load_model(
    model_name: str,
    quantization: str | None = "4bit",
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> AutoModelForCausalLM:
    """
    Load a causal LM from HuggingFace Hub with device-appropriate settings.

    Detects the active device via device.py and adjusts:
      - Quantization: 4-bit/8-bit on CUDA, disabled on CPU
      - dtype: bf16 on Ampere+ CUDA, fp32 on CPU
      - device_map: "auto" on CUDA, explicit CPU dict on CPU

    This function should NOT be called when running on MPS — the mlx_train
    backend handles model loading for Apple Silicon via mlx_lm.load().

    Args:
        model_name:        HuggingFace model ID, e.g. "Qwen/Qwen2.5-1.5B-Instruct"
                           or "meta-llama/Llama-3.1-8B-Instruct".
        quantization:      "4bit", "8bit", or None. Ignored on CPU.
        device_map:        Config hint. Overridden per detected device.
        trust_remote_code: Set True for models that require it (e.g. Qwen).

    Returns:
        Loaded model ready for LoRA attachment.

    Raises:
        RuntimeError: if called on MPS (use mlx_train backend instead).
    """
    device = detect_device()

    if device == "mps":
        raise RuntimeError(
            "load_model() was called on MPS but Apple Silicon uses the MLX backend. "
            "Model loading is handled by mlx_lm.load() inside mlx_train.py. "
            "This is a bug in the training dispatcher — check train_loop.py."
        )

    dev_cfg = get_device_config(device, quantization)
    effective_quant = dev_cfg["effective_quantization"]
    effective_dtype = dev_cfg["torch_dtype"]
    effective_dmap  = dev_cfg["device_map"]

    logger.info(f"Loading model: {model_name}")
    logger.info(
        f"  device={device}  quantization={effective_quant}  "
        f"dtype={effective_dtype}  device_map={effective_dmap}"
    )

    kwargs: dict = {
        "device_map":        effective_dmap,
        "trust_remote_code": trust_remote_code,
    }

    if effective_quant:
        # dtype is embedded inside the BnB config; don't set torch_dtype separately
        kwargs["quantization_config"] = build_bnb_config(effective_quant)
    else:
        kwargs["torch_dtype"] = effective_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded — {total_params:,} total parameters")

    return model
