# Model 2 Training Pipeline — QLoRA SFT on Qwen2.5

> **Part of AutoMend** — an end-to-end MLOps platform for anomaly detection (Track A) and generative workflow architecture (Track B).
> This module handles **Track B**: supervised fine-tuning of Qwen2.5-1.5B-Instruct to generate structured JSON workflow responses.

---

## Table of Contents

1. [What This Pipeline Does](#1-what-this-pipeline-does)
2. [Multi-Device Architecture](#2-multi-device-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Prerequisites by Platform](#4-prerequisites-by-platform)
5. [Environment Setup](#5-environment-setup)
   - [CUDA — Windows / Linux (NVIDIA)](#51-cuda--windows--linux-nvidia)
   - [MPS — macOS Apple Silicon](#52-mps--macos-apple-silicon)
   - [CPU — Any platform, no GPU](#53-cpu--any-platform-no-gpu)
6. [CUDA Blackwell Note (RTX 50xx)](#6-cuda-blackwell-note-rtx-50xx)
7. [Data Pipeline Dependency](#7-data-pipeline-dependency)
8. [Running the Pipeline — Step by Step](#8-running-the-pipeline--step-by-step)
9. [Configuration Reference](#9-configuration-reference)
10. [Model & LoRA Architecture](#10-model--lora-architecture)
11. [Training Details by Backend](#11-training-details-by-backend)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Weights & Biases Tracking](#13-weights--biases-tracking)
14. [Output Structure](#14-output-structure)
15. [Sample Results](#15-sample-results)
16. [Known Issues & Fixes Applied](#16-known-issues--fixes-applied)
17. [Reproducing the Smoke Test Run (CUDA)](#17-reproducing-the-smoke-test-run-cuda)
18. [Reproducing on Apple Silicon (MPS)](#18-reproducing-on-apple-silicon-mps)
19. [Full Training Run](#19-full-training-run)

---

## 1. What This Pipeline Does

AutoMend Track B trains a small generative model to take a user question (with available tools listed as context) and produce a structured JSON response in one of two forms:

```json
// Tool call response
{
  "workflow": {
    "steps": [
      { "tool": "calculate_bmi", "params": { "height": 1.75, "weight": 70 } }
    ]
  }
}

// Refusal / message response
{
  "workflow": { "steps": [] },
  "message": "I'm sorry, I can't assist with that using the available tools."
}
```

The pipeline fine-tunes **Qwen2.5-1.5B-Instruct** using **LoRA** on the Track B combined dataset. Only assistant tokens contribute to the loss — system and user tokens are masked to `-100`.

---

## 2. Multi-Device Architecture

The pipeline detects the compute device at startup via `src/utils/device.py` and routes automatically to the correct backend. You run the **same commands** on every platform.

```
detect_device()   →   "cuda"  /  "mps"  /  "cpu"
                            │           │           │
                     ┌──────┘    ┌──────┘    ┌──────┘
                     ▼           ▼           ▼
               HF Backend    MLX Backend  HF Backend
               (bitsandbytes  (mlx-lm      (fp32,
                4-bit NF4,    Metal GPU,   no quant,
                PEFT LoRA,    native quant, very slow)
                HF Trainer)   mlx_lm.lora)
```

| | CUDA (NVIDIA) | MPS (Apple Silicon) | CPU |
|---|---|---|---|
| Platforms | Windows, Linux | macOS M1/M2/M3/M4 | Any |
| Quantization | 4-bit NF4 via bitsandbytes | Native via mlx-lm Metal | None (fp32) |
| LoRA framework | PEFT | mlx-lm built-in | PEFT |
| Training engine | HuggingFace Trainer | `mlx_lm.lora` subprocess | HuggingFace Trainer |
| Inference engine | HF `model.generate()` | `mlx_lm.generate()` | HF `model.generate()` |
| Mixed precision | bf16 (Ampere+) or fp16 | Not applicable (Metal) | fp32 |
| VRAM / RAM needed | ~4 GB (4-bit) | ~6 GB unified RAM (fp32) | ~6 GB system RAM |

### Why MLX instead of bitsandbytes on Apple Silicon?

`bitsandbytes` implements quantization as **CUDA C++ kernels** — they are compiled for NVIDIA GPUs and cannot run on Apple's Metal GPU. MLX is Apple's own open-source ML framework, purpose-built for M-series chips. It provides:

- Native 4-bit quantization on Metal
- LoRA fine-tuning via `mlx-lm` (supports Qwen, Llama, Mistral, etc.)
- Both training **and** inference — no need for a separate framework
- Full compatibility with HuggingFace model weights (converts on the fly)

> **GGUF note:** GGUF is an inference-only format (llama.cpp, Ollama). It cannot do LoRA training. MLX is the correct choice for training on Apple Silicon.

---

## 3. Folder Structure

```
model_2_training/
│
├── configs/
│   ├── data/track_b_chatml.yaml        # Data: paths, ratios, seq length, sample caps
│   ├── model/qwen_baseline.yaml        # Model: name, quantization, LoRA targets
│   ├── train/qlora_sft.yaml            # Training: lr, epochs, batch, LoRA r/alpha
│   └── eval/json_eval.yaml             # Eval: max_new_tokens, split name
│
├── scripts/
│   ├── run_split.py                    # CLI: create train/val/test splits
│   ├── run_train.py                    # CLI: run training (auto-detects device)
│   ├── run_eval.py                     # CLI: evaluate on val set
│   └── run_test.py                     # CLI: evaluate on held-out test set
│
├── src/
│   ├── utils/
│   │   ├── device.py                   # ★ Device detection & per-device config
│   │   ├── config.py                   # YAML loader
│   │   ├── seed.py                     # Reproducible seeding
│   │   ├── io.py                       # Atomic file I/O
│   │   └── paths.py                    # Paths class
│   │
│   ├── data/
│   │   ├── dataset_contract.py         # ChatML schema validation
│   │   ├── load_jsonl.py               # JSONL loader with malformed-row handling
│   │   ├── split_data.py               # Train/val/test splitter + stratification
│   │   ├── dataset_builder.py          # ChatMLSupervisedDataset (torch Dataset)
│   │   └── collators.py                # AssistantOnlyCollator (loss masking)
│   │
│   ├── model/
│   │   ├── load_tokenizer.py           # Tokenizer loader
│   │   ├── load_qwen.py                # Qwen loader — CUDA/CPU only (MPS uses MLX)
│   │   └── lora_factory.py             # PEFT LoRA attachment — CUDA/CPU only
│   │
│   ├── train/
│   │   ├── train_loop.py               # ★ Orchestrator — dispatches to HF or MLX
│   │   ├── trainer_factory.py          # HF TrainingArguments + Trainer builder
│   │   ├── mlx_train.py                # ★ MLX backend: data prep + training + inference
│   │   └── callbacks.py                # Custom Trainer callbacks
│   │
│   ├── eval/
│   │   ├── generator.py                # ★ Device-aware inference (HF or MLX)
│   │   ├── metrics_json.py             # 9 JSON structural quality metrics
│   │   ├── save_reports.py             # Save metrics, markdown, sample outputs
│   │   └── evaluator.py                # Full eval pipeline orchestrator
│   │
│   ├── test/run_testset.py             # Test set eval wrapper
│   │
│   └── tracking/
│       ├── wandb_logger.py             # W&B logging (graceful degradation)
│       └── artifact_logger.py          # Local run archive
│
├── data/splits/
│   ├── train.jsonl                     # Training split (from run_split.py)
│   ├── val.jsonl                       # Validation split
│   ├── test.jsonl                      # Test split
│   └── split_summary.json
│
├── outputs/
│   ├── checkpoints/
│   │   ├── best_model/                 # Final checkpoint / adapter
│   │   ├── checkpoint-N/               # Intermediate checkpoints (CUDA/CPU)
│   │   ├── mlx_data/                   # MLX-formatted data (MPS only)
│   │   ├── mlx_lora_config.yaml        # mlx-lm training config (MPS only)
│   │   └── run_config_snapshot.json    # Full config + detected device
│   │
│   └── reports/val/
│       ├── metrics.json
│       ├── metrics_summary.md
│       ├── sample_outputs.json
│       └── validation_predictions.jsonl
│
└── requirements.txt
```

Files marked ★ are where device dispatch logic lives.

---

## 4. Prerequisites by Platform

### CUDA — Windows / Linux (NVIDIA GPU)

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12 | conda env `mlops_project_model` |
| CUDA toolkit | 12.8 | Required for Blackwell (RTX 50xx) |
| PyTorch | nightly cu128 | See §6 for Blackwell GPUs |
| bitsandbytes | ≥ 0.43 | Quantization kernels |
| PEFT | ≥ 0.10 | LoRA adapter management |
| HuggingFace Transformers | ≥ 4.40 | Training + inference |

### MPS — macOS Apple Silicon (M1/M2/M3/M4)

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12 | conda env `mlops_project_model` |
| macOS | 13.5+ | Required for MLX Metal support |
| PyTorch | ≥ 2.1 | For device detection only (not used for training) |
| mlx | ≥ 0.16 | Apple's ML framework |
| mlx-lm | ≥ 0.19 | LLM LoRA training + inference on MLX |
| HuggingFace Transformers | ≥ 4.40 | Tokenizer loading only |

### CPU — Any platform, no GPU

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12 | |
| PyTorch | ≥ 2.1 | CPU-only build is fine |
| HuggingFace Transformers | ≥ 4.40 | |

> CPU training is extremely slow (~100x slower than CUDA). Only use for debugging.

---

## 5. Environment Setup

### 5.1 CUDA — Windows / Linux (NVIDIA)

```bash
# Activate conda env
conda activate mlops_project_model

# Install all dependencies
pip install -r requirements.txt

# If you have a Blackwell GPU (RTX 50xx): install PyTorch nightly cu128 FIRST
# See §6 before running pip install -r requirements.txt
```

**HuggingFace login** (needed to download Qwen):
```bash
huggingface-cli login
# Paste your HF token when prompted
```

**W&B login** (optional):
```bash
wandb login
# Paste your W&B API key when prompted
```

---

### 5.2 MPS — macOS Apple Silicon

```bash
# Activate conda env
conda activate mlops_project_model

# Install base dependencies (torch for device detection, HF for tokenizer)
pip install torch transformers accelerate pyyaml loguru wandb numpy pytest

# Install the MLX backend
pip install mlx mlx-lm

# Verify MLX sees your GPU
python -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0)
```

**HuggingFace login** (same as CUDA):
```bash
huggingface-cli login
```

> You do **not** need `bitsandbytes`, `peft`, or `trl` on Apple Silicon.
> The MLX backend handles quantization and LoRA natively.

---

### 5.3 CPU — Any platform, no GPU

```bash
conda activate mlops_project_model
pip install torch transformers accelerate peft pyyaml loguru wandb numpy pytest
```

> Skip `bitsandbytes` — quantization is not available on CPU.

---

## 6. CUDA Blackwell Note (RTX 50xx)

> Skip this section if you don't have an RTX 5070 / 5080 / 5090.

Blackwell GPUs use compute capability **sm_120** which is not supported by the stable PyTorch release (cu126 or earlier). You will get:

```
CUDA error: no kernel image is available for execution on the device
```

**Fix — install PyTorch nightly cu128 before anything else:**

```bash
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# Then install bitsandbytes
pip install bitsandbytes --upgrade

# Then install the rest
pip install transformers peft trl accelerate pyyaml loguru wandb numpy pytest
```

**Verify:**
```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
# Expected: NVIDIA GeForce RTX 5070 Laptop GPU
#           True
```

---

## 7. Data Pipeline Dependency

This module consumes the output of the AutoMend data pipeline:

```
AutoMend/data/processed/track_B_combined.jsonl   ← required input
```

This file is produced by the combiner step (ds3 + ds4 + ds5 + ds6 → 5,118 samples).

**Sample format:**
```json
{
  "messages": [
    { "role": "system",    "content": "You are a helpful assistant. Available tools: ..." },
    { "role": "user",      "content": "Calculate my BMI. I weigh 70kg and am 1.75m tall." },
    { "role": "assistant", "content": "{\"workflow\": {\"steps\": [...]}}" }
  ],
  "metadata": { "source_dataset": "ds5", "task_type": "tool_call" }
}
```

If the artifact is missing, regenerate it from the AutoMend root:
```bash
python -c "from src.combiner_track_b.combine import combine_track_b; combine_track_b()"
```

---

## 8. Running the Pipeline — Step by Step

> All commands run from the **`model_2_training/`** directory with `mlops_project_model` active.
> The **same commands work on all platforms** — device detection is automatic.

### Step 1 — Create Splits

```bash
python scripts/run_split.py --config configs/data/track_b_chatml.yaml
```

Creates `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl` and a summary JSON.

On **MPS**, this step also indirectly prepares the raw splits that `mlx_train.py` will convert to MLX chat format during training.

**Expected output (full dataset):**
```
Split result — train: 4094, val: 511, test: 513 (total: 5118)
```

**For a quick smoke test**, set caps in `configs/data/track_b_chatml.yaml`:
```yaml
max_train_samples: 200
max_val_samples:    50
max_test_samples:   50
```

---

### Step 2 — Train

```bash
python scripts/run_train.py \
  --data-config  configs/data/track_b_chatml.yaml \
  --model-config configs/model/qwen_baseline.yaml \
  --train-config configs/train/qlora_sft.yaml
```

**What happens automatically by device:**

| Step | CUDA | MPS | CPU |
|------|------|-----|-----|
| Seed | ✓ | ✓ | ✓ |
| Config snapshot | ✓ (includes `"device": "cuda"`) | ✓ (includes `"device": "mps"`) | ✓ |
| Tokenizer | HF `AutoTokenizer` | HF `AutoTokenizer` | HF `AutoTokenizer` |
| Data prep | `ChatMLSupervisedDataset` | MLX chat JSONL conversion | `ChatMLSupervisedDataset` |
| Model load | Qwen + 4-bit NF4 | mlx-lm loads base model | Qwen fp32 |
| LoRA attach | PEFT (`peft.get_peft_model`) | mlx-lm built-in | PEFT |
| Training | HuggingFace Trainer | `mlx_lm.lora` subprocess | HuggingFace Trainer |
| Precision | bf16 (Ampere+) / fp16 | Metal (handled by MLX) | fp32 |
| Adapter saved | `best_model/` PEFT format | `best_model/` MLX format | `best_model/` PEFT format |

**CLI overrides (all platforms):**
```bash
python scripts/run_train.py \
  --data-config  configs/data/track_b_chatml.yaml \
  --model-config configs/model/qwen_baseline.yaml \
  --train-config configs/train/qlora_sft.yaml \
  --epochs 3 \
  --lr 2e-4 \
  --batch-size 2
```

---

### Step 3 — Evaluate (Validation Set)

```bash
python scripts/run_eval.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

Auto-detects CUDA/MPS/CPU and uses the appropriate inference backend.

Saves to `outputs/reports/val/`:
- `metrics.json` — 9 structural quality metrics
- `metrics_summary.md` — human-readable table
- `sample_outputs.json` — generated vs. reference pairs
- `error_samples.json` — samples where JSON parse failed

---

### Step 4 — Test Set Evaluation (Final / Hold-Out)

```bash
python scripts/run_test.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

> Run only once, after all hyperparameter decisions are final. Do not use test results to guide tuning.

---

## 9. Configuration Reference

### `configs/data/track_b_chatml.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `artifact_path` | `data/processed/track_B_combined.jsonl` | Raw artifact (relative to AutoMend root) |
| `splits_dir` | `data/splits` | Split output dir (relative to `model_2_training/`) |
| `train_ratio` | `0.80` | Training fraction |
| `val_ratio` | `0.10` | Validation fraction |
| `test_ratio` | `0.10` | Test fraction |
| `shuffle_seed` | `42` | Reproducibility seed |
| `max_seq_length` | `2048` | Token length cap |
| `malformed_row_strategy` | `"skip"` | `"skip"` or `"raise"` |
| `stratify_by` | `[source_dataset]` | Stratify splits by this metadata field |
| `max_train_samples` | `null` | Cap training samples (`200` for smoke tests) |
| `max_val_samples` | `null` | Cap val samples (`50` for smoke tests) |
| `max_test_samples` | `null` | Cap test samples (`50` for smoke tests) |

### `configs/model/qwen_baseline.yaml`

| Key | Value | Description |
|-----|-------|-------------|
| `model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID |
| `tokenizer_name` | `Qwen/Qwen2.5-1.5B-Instruct` | Tokenizer (usually same as model) |
| `quantization` | `"4bit"` | CUDA: `"4bit"` / `"8bit"` / `null`. MPS: ignored (MLX handles it). CPU: ignored. |
| `device_map` | `"auto"` | CUDA: used as-is. MPS/CPU: overridden by device.py |
| `lora_target_modules` | 7 projection layers | CUDA/CPU only — MLX targets layers by count |

### `configs/train/qlora_sft.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `num_train_epochs` | `3` | Full training; `1` for smoke test |
| `per_device_train_batch_size` | `2` | Reduce to `1` if OOM |
| `gradient_accumulation_steps` | `8` | Effective batch = 2 × 8 = 16 |
| `learning_rate` | `2e-4` | Initial LR (cosine decay) |
| `warmup_ratio` | `0.05` | 5% warmup steps |
| `lr_scheduler_type` | `"cosine"` | Cosine annealing |
| `bf16` | `true` | Overridden by device.py (ignored on CPU/MPS) |
| `lora_r` | `16` | LoRA rank — used by both CUDA (PEFT) and MPS (mlx-lm) |
| `lora_alpha` | `32` | LoRA scaling — used by both backends |
| `lora_dropout` | `0.05` | Dropout on LoRA layers |
| `report_to` | `"wandb"` | Set to `"none"` to disable W&B |
| `eval_steps` | `100` | Evaluate every N steps (CUDA/CPU) |
| `save_steps` | `100` | Save checkpoint every N steps (CUDA/CPU) |

---

## 10. Model & LoRA Architecture

### Base Model

| Property | Value |
|----------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Total parameters | ~907M |
| Context window | 32,768 tokens (capped at 2,048 during training) |
| HuggingFace ID | `Qwen/Qwen2.5-1.5B-Instruct` |

### LoRA Configuration

| Property | CUDA / CPU (PEFT) | MPS (mlx-lm) |
|----------|-------------------|--------------|
| Rank (r) | 16 | 16 |
| Alpha (α) | 32 | 32 |
| Dropout | 0.05 | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Last 16 transformer layers |
| Trainable params | ~18M (~2.04% of 907M) | ~18M (approx.) |

### Loss Masking (CUDA / CPU only)

`AssistantOnlyCollator` scans each tokenized sequence for assistant spans and sets `labels = -100` for all system and user tokens. Only assistant response tokens contribute to cross-entropy loss.

On MPS, `mlx_lm.lora` handles loss masking internally using the chat format.

---

## 11. Training Details by Backend

### CUDA / CPU — HuggingFace Trainer

1. Tokenize with `apply_chat_template()`, truncate/pad to 2048
2. `AssistantOnlyCollator` masks non-assistant tokens to -100
3. AdamW + cosine LR schedule + 5% warmup
4. bitsandbytes 4-bit NF4 quantization (CUDA only)
5. Best checkpoint restored by eval_loss at end of training

**Smoke test results (200 samples, 1 epoch, CUDA RTX 5070):**

| Step | Loss | LR |
|------|------|----|
| 10 | 0.578 | 1.58e-4 |
| 20 | 0.428 | 3.17e-5 |

Loss dropped 26% over 25 steps — model is learning the JSON schema.

### MPS — mlx-lm

1. `prepare_mlx_data()` copies splits to `outputs/checkpoints/mlx_data/` in MLX chat format
2. `build_mlx_lora_config()` writes a YAML config for `mlx_lm.lora`
3. `python -m mlx_lm lora --config mlx_lora_config.yaml` runs on Metal GPU
4. Adapter saved to `outputs/checkpoints/best_model/` in MLX format
5. Validation loss reported every `eval_steps` iterations

**mlx-lm iteration calculation:**
```
iters = ceil(n_samples / batch_size) × num_epochs
e.g.: ceil(200 / 2) × 1 = 100 iters for smoke test
```

---

## 12. Evaluation Metrics

9 JSON structural quality metrics computed on generated outputs (device-agnostic):

| Metric | Smoke Test (50 val samples, CUDA) | Interpretation |
|--------|-----------------------------------|----------------|
| Non-empty output rate | **100%** | Model always produces output |
| Starts with `{` or `[` | **100%** | JSON structure always initiated |
| Ends with `}` or `]` | **98%** | 1 sample slightly truncated |
| JSON parse success rate | **96%** | 48/50 outputs are valid JSON |
| Malformed JSON rate | 4% | 2 samples: unclosed strings from long code gen |
| Truncation rate | 2% | 1 sample hit the max token limit |
| Quote balance rate | **98%** | Strings properly closed |
| Brace balance rate | **94%** | Nested braces mostly balanced |
| Average output length | 392 chars | Not too terse, not runaway |

**Main failure mode with 200 samples:** over-calling (generates 2 tool steps when 1 is needed). Resolves significantly with full dataset training.

---

## 13. Weights & Biases Tracking

W&B is integrated for CUDA and CPU runs. Each run is auto-named:

```
{model_short}_{quant}_lora-r{r}_{n_samples}samples_{datetime}
# CUDA example: Qwen2.5-1.5B-Instruct_4bit_lora-r16_200samples_20260319-2008
# MPS example:  Qwen2.5-1.5B-Instruct_mlx_lora-r16_200samples_20260319-2008
# CPU example:  Qwen2.5-1.5B-Instruct_fp32_lora-r16_200samples_20260319-2008
```

### Setup

```bash
wandb login   # paste API key once — saved permanently

# Or set in AutoMend/.env:
WANDB_API_KEY=your_api_key
WANDB_PROJECT=automend-model2
WANDB_ENTITY=your-entity-name
```

### Disable W&B

```yaml
# configs/train/qlora_sft.yaml
report_to: "none"
```

> W&B tracking for MPS runs is logged via the subprocess stdout. For full W&B integration in MLX, `mlx_lm.lora` supports a `--wandb-project` flag — this can be added to `build_mlx_lora_config()` in `mlx_train.py`.

---

## 14. Output Structure

```
outputs/
├── checkpoints/
│   ├── best_model/                          # Final adapter (PEFT or MLX format)
│   │   ├── adapter_config.json              # LoRA config
│   │   ├── adapter_model.safetensors        # PEFT adapter weights (CUDA/CPU)
│   │   │   — OR —
│   │   ├── adapters.npz                     # MLX adapter weights (MPS)
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── chat_template.jinja
│   │
│   ├── checkpoint-N/                        # Intermediate checkpoints (CUDA/CPU)
│   ├── mlx_data/                            # MLX-formatted JSONL data (MPS only)
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   ├── mlx_lora_config.yaml                 # mlx-lm training config (MPS only)
│   └── run_config_snapshot.json             # All configs + detected device
│
└── reports/val/
    ├── metrics.json
    ├── metrics_summary.md
    ├── sample_outputs.json
    ├── error_samples.json
    └── validation_predictions.jsonl
```

### Loading the trained adapter for inference

**CUDA / CPU (PEFT):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(base, "outputs/checkpoints/best_model")
tokenizer = AutoTokenizer.from_pretrained("outputs/checkpoints/best_model")
```

**MPS (MLX):**
```python
from mlx_lm import load, generate

model, tokenizer = load(
    "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path="outputs/checkpoints/best_model"
)
response = generate(model, tokenizer, prompt="...", max_tokens=512)
```

---

## 15. Sample Results

### Exact match — multi-step tool call (index 21)

| | Content |
|--|---------|
| **Generated** | `{"workflow": {"steps": [{"tool": "get_stock_price", "params": {"symbol": "AAPL"}}, {"tool": "get_stock_price", "params": {"symbol": "MSFT"}}]}}` |
| **Reference** | *(identical)* |
| **Result** | ✅ Exact match |

### Correct refusal (index 1)

| | Content |
|--|---------|
| **Generated** | `{"workflow": {"steps": []}, "message": "I'm sorry, but I can't assist with that. My current capabilities allow me to translate words and phrases between languages."}` |
| **Reference** | `{"workflow": {"steps": []}, "message": "I'm sorry, but I'm unable to perform external tasks like ordering a pizza..."}` |
| **Result** | ✅ Semantically correct (different wording, same intent and structure) |

### Over-calling (index 7) — main failure mode with 200 training samples

| | Steps |
|--|-------|
| **Generated** | 2 calls: `calculate_tip(50, 15%)` and `calculate_tip(75, 20%)` |
| **Reference** | 1 call: `calculate_tip(50, 15%)` |
| **Result** | ⚠️ Extra tool call — resolves with full dataset training |

---

## 16. Known Issues & Fixes Applied

### 1. Blackwell GPU — CUDA kernel error

**Error:** `CUDA error: no kernel image is available for execution on the device`
**Cause:** Stable PyTorch (cu126) has no kernels for sm_120 (RTX 5070/5080/5090).
**Fix:** Install PyTorch nightly cu128. See §6.

### 2. Transformers 5.x — `evaluation_strategy` renamed

**Error:** `TypeError: TrainingArguments.__init__() got unexpected keyword argument 'evaluation_strategy'`
**Fix:** `trainer_factory.py` uses `eval_strategy=` (the new name). The YAML key remains `evaluation_strategy` — the factory maps it.

### 3. Transformers 5.x — `tokenizer` renamed in Trainer

**Error:** `TypeError: Trainer.__init__() got unexpected keyword argument 'tokenizer'`
**Fix:** `trainer_factory.py` passes `processing_class=tokenizer` (the new name).

### 4. W&B API key not loaded from `.env`

**Symptom:** W&B prompts for login even when `.env` has `WANDB_API_KEY`.
**Cause:** W&B does not auto-read `.env` files.
**Workaround:** `wandb login` once in the terminal — credentials persist in the conda env.

### 5. bitsandbytes on MPS / CPU

**Error:** `CUDA not found` or similar when trying to use BnB on non-CUDA devices.
**Fix:** `device.py` + `load_qwen.py` detect MPS/CPU and skip quantization entirely. MPS routes to the MLX backend; CPU loads in fp32.

---

## 17. Reproducing the Smoke Test Run (CUDA)

Exact reproduction of the 2026-03-19 run: 200 train / 50 val / 50 test, 1 epoch, RTX 5070.

**Step 1 — Verify CUDA environment:**
```bash
conda activate mlops_project_model
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
# Expected: 2.x.x+cu128  |  NVIDIA GeForce RTX 5070 Laptop GPU
```

**Step 2 — Set smoke test caps** in `configs/data/track_b_chatml.yaml`:
```yaml
max_train_samples: 200
max_val_samples:    50
max_test_samples:   50
```

And in `configs/train/qlora_sft.yaml`:
```yaml
num_train_epochs: 1
per_device_train_batch_size: 1
```

**Step 3 — Verify the artifact:**
```bash
wc -l ../data/processed/track_B_combined.jsonl
# Expected: 5118
```

**Step 4 — Run:**
```bash
python scripts/run_split.py --config configs/data/track_b_chatml.yaml

python scripts/run_train.py \
  --data-config  configs/data/track_b_chatml.yaml \
  --model-config configs/model/qwen_baseline.yaml \
  --train-config configs/train/qlora_sft.yaml

python scripts/run_eval.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

**Expected results:**
- Step 10 loss: ~0.578, Step 20 loss: ~0.428
- JSON parse rate: ~96%, Non-empty rate: 100%
- Training time: ~5 minutes

---

## 18. Reproducing on Apple Silicon (MPS)

```bash
# 1. Verify environment
conda activate mlops_project_model
python -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0)

# 2. Verify PyTorch sees MPS
python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True

# 3. Set smoke test caps (same as §17 Step 2)

# 4. Run — exact same commands as CUDA
python scripts/run_split.py --config configs/data/track_b_chatml.yaml

python scripts/run_train.py \
  --data-config  configs/data/track_b_chatml.yaml \
  --model-config configs/model/qwen_baseline.yaml \
  --train-config configs/train/qlora_sft.yaml

python scripts/run_eval.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

**What happens differently on MPS:**
- `device.py` returns `"mps"` → `train_loop.py` routes to `_run_mlx_training()`
- Data is converted to MLX chat format in `outputs/checkpoints/mlx_data/`
- `mlx_lora_config.yaml` is written and `python -m mlx_lm lora` is invoked
- Adapter is saved in MLX format to `best_model/`
- Eval inference uses `mlx_lm.load()` + `mlx_lm.generate()` instead of HF

**Memory:** 1.5B model in fp32 ≈ 6 GB unified memory — works on M1 16 GB and up.

---

## 19. Full Training Run

Set all caps to `null` and use 3 epochs for production-quality results.

**`configs/data/track_b_chatml.yaml`:**
```yaml
max_train_samples: null
max_val_samples:   null
max_test_samples:  null
```

**`configs/train/qlora_sft.yaml`:**
```yaml
num_train_epochs: 3
per_device_train_batch_size: 2
```

```bash
python scripts/run_split.py --config configs/data/track_b_chatml.yaml

python scripts/run_train.py \
  --data-config  configs/data/track_b_chatml.yaml \
  --model-config configs/model/qwen_baseline.yaml \
  --train-config configs/train/qlora_sft.yaml \
  --epochs 3

python scripts/run_eval.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model

# Final test evaluation — run only once
python scripts/run_test.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

**Expected improvements over smoke test:**
- Over-calling significantly reduced (more single-step training examples seen)
- Tool name accuracy improves
- JSON parse rate > 98%
- Better parameter completeness in tool calls

---

## Appendix A — Data Contract

Each sample in `track_B_combined.jsonl` must satisfy:

```python
{
  "messages": [                          # Required: list of message dicts
    { "role": "system",    "content": "..." },   # Required
    { "role": "user",      "content": "..." },   # Required
    { "role": "assistant", "content": "..." }    # Must end with assistant
  ],
  "metadata": {                          # Optional: source info
    "source_dataset": "ds5",
    "task_type": "tool_call"
  }
}
```

Samples failing this contract are handled by `malformed_row_strategy`:
- `"skip"` (default): logged and dropped silently
- `"raise"`: raises `ContractViolation` immediately

---

## Appendix B — Switching to Llama Models

When moving from Qwen2.5 to Llama models in future iterations, only the model config needs to change:

```yaml
# configs/model/qwen_baseline.yaml  →  configs/model/llama_baseline.yaml
model_name:      "meta-llama/Llama-3.1-8B-Instruct"
tokenizer_name:  "meta-llama/Llama-3.1-8B-Instruct"
quantization:    "4bit"   # CUDA: bitsandbytes | MPS: mlx-lm native
lora_target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"
```

Both backends (CUDA bitsandbytes and MPS mlx-lm) support Llama 3 natively. No code changes needed.

---

*AutoMend — Model 2 Training Pipeline*
*Backends: HuggingFace Transformers + PEFT (CUDA/CPU) · MLX + mlx-lm (Apple Silicon)*
