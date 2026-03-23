"""
generator.py

Loads a saved checkpoint and runs generation on a list of samples.

Device dispatch
---------------
  CUDA / CPU  → HuggingFace generate (AutoModelForCausalLM + PEFT)
  MPS         → mlx-lm generate (mlx_lm.load + mlx_lm.generate)

The device is detected once at the top of load_model_for_inference().
Both paths return the same output schema so the rest of the eval pipeline
(metrics_json.py, save_reports.py) is device-agnostic.

Does NOT compute metrics.
"""

from __future__ import annotations
import json
from pathlib import Path

import torch
from loguru import logger

from model_2_training.src.utils.device import detect_device, get_device_config


# ---------------------------------------------------------------------------
# Unified entry point — dispatches per device
# ---------------------------------------------------------------------------

def load_model_for_inference(checkpoint_path: str | Path):
    """
    Load a saved model/adapter and tokenizer, routing to the right backend.

    CUDA / CPU  → loads via HuggingFace AutoModelForCausalLM + PeftModel
    MPS         → loads via mlx_lm.load() (returns MLX model + tokenizer)

    Args:
        checkpoint_path: Path to the checkpoint directory (best_model/).

    Returns:
        (model, tokenizer) — types differ by backend but both work with
        run_generation() which also dispatches per device.
    """
    device = detect_device()

    if device == "mps":
        return _load_mlx(checkpoint_path)
    else:
        return _load_hf(checkpoint_path, device)


def run_generation(
    samples: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
    do_sample: bool = False,
) -> list[dict]:
    """
    Run generation on a list of samples, routing to the right backend.

    CUDA / CPU  → HuggingFace model.generate()
    MPS         → mlx_lm.generate()

    Args:
        samples:        List of validated sample dicts.
        model:          Loaded model (HF or MLX).
        tokenizer:      Loaded tokenizer.
        max_new_tokens: Maximum tokens to generate per sample.
        temperature:    Sampling temperature (do_sample=True only).
        top_p:          Top-p nucleus sampling (do_sample=True only).
        do_sample:      If False, greedy decoding.

    Returns:
        List of dicts: {index, prompt, generated, reference, sample}
    """
    device = detect_device()

    if device == "mps":
        return _generate_mlx(samples, model, tokenizer, max_new_tokens)
    else:
        return _generate_hf(samples, model, tokenizer, max_new_tokens, temperature, top_p, do_sample)


# ---------------------------------------------------------------------------
# HuggingFace backend  (CUDA / CPU)
# ---------------------------------------------------------------------------

def _load_hf(checkpoint_path: str | Path, device: str):
    """Load a HuggingFace (full or PEFT) checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dev_cfg = get_device_config(device)
    dtype      = dev_cfg["torch_dtype"]
    device_map = dev_cfg["device_map"]

    logger.info(f"Loading tokenizer from: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    adapter_config = checkpoint_path / "adapter_config.json"
    if adapter_config.exists():
        logger.info("Detected PEFT adapter — loading base model + LoRA weights")
        with open(adapter_config) as f:
            ac = json.load(f)
        base_model_name = ac.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
        logger.info(f"Base model: {base_model_name}  dtype={dtype}  device_map={device_map}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    else:
        logger.info(f"Loading full HF checkpoint from {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    model.eval()
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"HF model loaded for inference — {total:,} parameters")
    return model, tokenizer


def _generate_hf(
    samples: list[dict],
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> list[dict]:
    """HuggingFace generation loop (CUDA / CPU)."""
    results = []
    device = next(model.parameters()).device

    for i, sample in enumerate(samples):
        try:
            prompt    = _build_prompt(sample, tokenizer)
            reference = _extract_reference(sample)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            ).to(device)

            gen_kwargs: dict = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs.update(temperature=temperature, top_p=top_p, do_sample=True)
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            prompt_len    = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][prompt_len:]
            generated     = tokenizer.decode(generated_ids, skip_special_tokens=True)

            results.append({
                "index":     i,
                "prompt":    prompt,
                "generated": generated,
                "reference": reference,
                "sample":    sample,
            })

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"HF generated {i + 1}/{len(samples)}")

        except Exception as e:
            logger.warning(f"Sample {i} HF generation failed: {e}")
            results.append({
                "index": i, "prompt": "", "generated": "",
                "reference": "", "sample": sample, "error": str(e),
            })

    logger.info(f"HF generation complete — {len(results)} samples processed")
    return results


# ---------------------------------------------------------------------------
# MLX backend  (MPS / Apple Silicon)
# ---------------------------------------------------------------------------

def _load_mlx(checkpoint_path: str | Path):
    """Load an MLX model + tokenizer via mlx_lm."""
    try:
        from mlx_lm import load
    except ImportError:
        raise ImportError(
            "mlx-lm is not installed. Install with:\n"
            "    pip install mlx mlx-lm"
        )

    checkpoint_path = Path(checkpoint_path)
    adapter_config  = checkpoint_path / "adapter_config.json"

    if adapter_config.exists():
        with open(adapter_config) as f:
            ac = json.load(f)
        # PEFT saves "base_model_name_or_path"; mlx-lm saves "base_model" or "model"
        base_name = (
            ac.get("base_model_name_or_path")
            or ac.get("base_model")
            or ac.get("model")
            or "Qwen/Qwen2.5-1.5B-Instruct"
        )
        logger.info(f"Loading MLX base model '{base_name}' + adapter from {checkpoint_path}")
        model, tokenizer = load(base_name, adapter_path=str(checkpoint_path))
    else:
        logger.info(f"Loading MLX model from {checkpoint_path}")
        model, tokenizer = load(str(checkpoint_path))

    def _count_mlx_params(node):
        if isinstance(node, dict):
            return sum(_count_mlx_params(v) for v in node.values())
        if isinstance(node, list):
            return sum(_count_mlx_params(v) for v in node)
        return node.size if hasattr(node, "size") else 0

    total = _count_mlx_params(model.parameters()) if hasattr(model, "parameters") else 0
    logger.info(f"MLX model loaded for inference — {total:,} parameters")
    return model, tokenizer


def _generate_mlx(
    samples: list[dict],
    model,
    tokenizer,
    max_new_tokens: int,
) -> list[dict]:
    """MLX generation loop (MPS / Apple Silicon)."""
    try:
        from mlx_lm import generate
    except ImportError:
        raise ImportError("mlx-lm is not installed. Install with: pip install mlx mlx-lm")

    results = []
    for i, sample in enumerate(samples):
        try:
            prompt    = _build_prompt(sample, tokenizer)
            reference = _extract_reference(sample)

            generated = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False,
            )

            results.append({
                "index":     i,
                "prompt":    prompt,
                "generated": generated,
                "reference": reference,
                "sample":    sample,
            })

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"MLX generated {i + 1}/{len(samples)}")

        except Exception as e:
            logger.warning(f"Sample {i} MLX generation failed: {e}")
            results.append({
                "index": i, "prompt": "", "generated": "",
                "reference": "", "sample": sample, "error": str(e),
            })

    logger.info(f"MLX generation complete — {len(results)} samples processed")
    return results


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_prompt(sample: dict, tokenizer) -> str:
    """
    Format the inference prompt from a sample — strips the last assistant turn
    and sets add_generation_prompt=True so the model knows to generate a reply.
    """
    messages = sample["messages"]
    last_user_idx = max(j for j, m in enumerate(messages) if m["role"] == "user")
    prompt_messages = messages[: last_user_idx + 1]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _extract_reference(sample: dict) -> str:
    """Extract the last assistant message content as the reference string."""
    for msg in reversed(sample["messages"]):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


# ---------------------------------------------------------------------------
# Save helper (device-agnostic)
# ---------------------------------------------------------------------------

def save_predictions(predictions: list[dict], output_path: str | Path) -> None:
    """
    Save raw generation results to a JSONL file.

    Args:
        predictions: List of prediction dicts from run_generation.
        output_path: File path to save to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            row = {
                "index":     pred["index"],
                "generated": pred["generated"],
                "reference": pred["reference"],
                "error":     pred.get("error"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved → {output_path} ({len(predictions)} rows)")
