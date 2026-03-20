"""
device.py

Central device detection and per-device configuration for the Model 2 pipeline.

Single source of truth for all device-specific settings. No other module should
call torch.cuda.is_available() or check for MPS independently — import from here.

Supported backends
------------------
  cuda  Windows / Linux with NVIDIA GPU
          bitsandbytes 4-bit / 8-bit quantization
          bf16 or fp16 mixed precision
          Backend: HuggingFace Transformers + PEFT + Trainer

  mps   macOS with Apple Silicon (M1 / M2 / M3 / M4)
          bitsandbytes NOT supported (CUDA kernels only)
          Native 4-bit quantization via Apple's MLX + mlx-lm on Metal
          Backend: mlx-lm (training) + mlx-lm generate (inference)

  cpu   Any system with no GPU
          No quantization, fp32 only
          Backend: HuggingFace Transformers fp32 (very slow for large models)
"""

from __future__ import annotations

import torch
from loguru import logger


def detect_device() -> str:
    """
    Detect the best available compute device.

    Priority: CUDA → MPS → CPU.

    Returns:
        One of "cuda", "mps", "cpu"
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Device detected: CUDA — {name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Device detected: MPS — Apple Silicon Metal GPU")
        logger.info("  Training backend: mlx-lm  |  Inference backend: mlx-lm generate")
        return "mps"

    logger.info("Device detected: CPU — no GPU found")
    logger.warning("Training on CPU is extremely slow. A GPU is strongly recommended.")
    return "cpu"


def get_device_config(device: str, quantization: str | None = None) -> dict:
    """
    Return device-appropriate settings for model loading and TrainingArguments.

    Args:
        device:       "cuda", "mps", or "cpu"
        quantization: Requested quantization from config ("4bit", "8bit", None).
                      On MPS, quantization is handled by MLX natively (not bitsandbytes).
                      On CPU, quantization is unsupported and silently disabled.

    Returns:
        dict with keys:
            effective_quantization  str | None   quantization to actually pass to BnB
            torch_dtype             torch.dtype  weight dtype for from_pretrained
            device_map              str | dict   passed to from_pretrained
            bf16                    bool         TrainingArguments.bf16
            fp16                    bool         TrainingArguments.fp16
    """
    if device == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
        return {
            "effective_quantization": quantization,
            "torch_dtype": torch.bfloat16 if bf16_ok else torch.float16,
            "device_map": "auto",
            "bf16": bf16_ok,
            "fp16": not bf16_ok,
        }

    if device == "mps":
        if quantization:
            logger.info(
                f"Config requests '{quantization}' quantization. On MPS, MLX handles "
                "quantization natively via Metal — bitsandbytes is not used."
            )
        return {
            "effective_quantization": None,    # MLX manages its own quantization
            "torch_dtype": torch.float32,
            "device_map": {"": "mps"},
            "bf16": False,
            "fp16": False,
        }

    # cpu
    if quantization:
        logger.warning(
            f"Quantization '{quantization}' requested but CPU has no GPU kernels. "
            "Loading in fp32."
        )
    return {
        "effective_quantization": None,
        "torch_dtype": torch.float32,
        "device_map": {"": "cpu"},
        "bf16": False,
        "fp16": False,
    }
