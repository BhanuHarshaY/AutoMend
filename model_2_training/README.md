# AutoMend — Track B: QLoRA Fine-Tuning & Evaluation Pipeline

> End-to-end supervised fine-tuning of **Qwen2.5-1.5B-Instruct** to generate structured JSON remediation workflows from MLOps incident descriptions.
> Supports NVIDIA CUDA, Apple Silicon (MLX/Metal), and CPU — same commands on every platform.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. System Requirements](#2-system-requirements)
- [3. Installation](#3-installation)
- [4. Repository Structure](#4-repository-structure)
- [5. Data](#5-data)
- [6. Phase 1 — Fine-Tuning](#6-phase-1--fine-tuning)
- [7. Phase 2 — Structured Evaluation](#7-phase-2--structured-evaluation)
- [8. Phase 2.5 — Gold Benchmark & Error Taxonomy](#8-phase-25--gold-benchmark--error-taxonomy)
- [9. Phase 2.75 — Hyperparameter Sweep](#9-phase-275--hyperparameter-sweep)
- [10. Phase 3 — Robustness & Slice Evaluation](#10-phase-3--robustness--slice-evaluation)
- [11. GCP Cloud Deployment](#11-gcp-cloud-deployment)
- [12. Configuration Reference](#12-configuration-reference)
- [13. Experiment Tracking](#13-experiment-tracking)
- [14. Output Artifacts](#14-output-artifacts)
- [15. Inference](#15-inference)
- [16. Sample Predictions](#16-sample-predictions)
- [17. Reproducing Results](#17-reproducing-results)
- [18. Troubleshooting](#18-troubleshooting)
- [Appendix A — Data Contract](#appendix-a--data-contract)
- [Appendix B — Switching Base Models](#appendix-b--switching-base-models)

---

## 1. Project Overview

AutoMend Track B fine-tunes a small generative model to produce structured JSON remediation workflows from MLOps incident descriptions. Given a user message and a list of available tools in the system prompt, the model outputs one of two response shapes:

**Shape A — Tool workflow** (one or more tool calls):

```json
{
  "workflow": {
    "steps": [
      { "tool": "restart_pod",   "params": { "namespace": "prod", "pod": "inference-worker-7" } },
      { "tool": "scale_service", "params": { "service": "inference-worker", "replicas": 3 } }
    ]
  }
}
```

**Shape B — Refusal** (no applicable tool available):

```json
{
  "workflow": { "steps": [] },
  "message": "I'm sorry, I cannot assist with that using the available tools."
}
```

### Phase Roadmap

| Phase | Name | Purpose | Key Output |
|-------|------|---------|------------|
| **1** | Fine-Tuning | Train QLoRA adapter, run 9 structural metrics | Trained adapter at `outputs/checkpoints/best_model/` |
| **2** | Structured Evaluation | Validate JSON correctness at schema, field, and parameter level | 54-metric evaluation report |
| **2.5** | Gold Benchmark | Lock 30 stratified test questions, label failures with 9-category taxonomy | `tax_valid_rate` — reproducible cross-model baseline |
| **2.75** | Hyperparameter Sweep | Bayesian search over 10 training hyperparameters (21 trials) | Winning config: r16, lr=1.19e-4, eff. batch=64 |
| **3** | Robustness & Slices | Stress-test with 5 perturbation types; slice metrics by archetype, length, complexity | Robustness delta table, failure log, slice report |

---

## 2. System Requirements

### CUDA — Windows / Linux (NVIDIA GPU)

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12 | `conda env mlops_project_model` |
| CUDA Toolkit | 12.8 | Required for Blackwell (RTX 50xx); 12.x for older GPUs |
| PyTorch | nightly cu128 | See [Blackwell Note](#blackwell-gpus-rtx-50xx-series) |
| bitsandbytes | ≥ 0.43 | 4-bit NF4 quantization kernels |
| PEFT | ≥ 0.10 | LoRA adapter management |
| HuggingFace Transformers | ≥ 4.40 | Training and inference |
| VRAM | ≥ 4 GB | 4-bit quantized inference; ~6 GB for training |

### MPS — macOS Apple Silicon (M1 / M2 / M3 / M4)

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12 | `conda env mlops_project_model` |
| macOS | 13.5+ | Required for MLX Metal GPU support |
| PyTorch | ≥ 2.1 | Device detection only — not used for training |
| mlx | ≥ 0.16 | Apple's open-source ML framework |
| mlx-lm | ≥ 0.19 | LLM LoRA training and inference on Metal |
| HuggingFace Transformers | ≥ 4.40 | Tokenizer loading only |
| Unified RAM | ≥ 6 GB | 1.5B model in fp32 |

> **Why MLX and not bitsandbytes on Apple Silicon?** `bitsandbytes` is implemented as CUDA C++ kernels compiled for NVIDIA GPUs — they cannot execute on Apple Metal. MLX is Apple's purpose-built ML framework for M-series chips with native 4-bit quantization, LoRA training, and inference. GGUF (llama.cpp, Ollama) is inference-only and cannot fine-tune. MLX is the correct choice.

### CPU — Any Platform

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12 | |
| PyTorch | ≥ 2.1 | CPU-only wheel is fine |
| HuggingFace Transformers | ≥ 4.40 | |
| System RAM | ≥ 6 GB | fp32 model weights |

> CPU training is approximately 100× slower than CUDA. Use only for import/module verification.

---

## 3. Installation

### CUDA — Windows / Linux

#### Blackwell GPUs (RTX 50xx series)

Blackwell GPUs use compute capability **sm_120**, which is unsupported by stable PyTorch (cu126). You will see `CUDA error: no kernel image is available for execution on the device`. Install PyTorch nightly **first**, before all other packages:

```bash
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

pip install bitsandbytes --upgrade
pip install transformers peft trl accelerate pyyaml loguru wandb numpy pytest
```

Verify:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
# Expected:
# NVIDIA GeForce RTX 5070 Laptop GPU
# True
```

#### All Other NVIDIA GPUs

```bash
conda activate mlops_project_model
pip install -r requirements.txt
```

#### HuggingFace Authentication

Required to download Qwen2.5-1.5B-Instruct:

```bash
huggingface-cli login
# Paste your token from huggingface.co/settings/tokens
```

#### Weights & Biases Setup

Copy the example env file and fill in credentials:

```bash
cp AutoMend/.env.example AutoMend/.env
```

```bash
# AutoMend/.env
WANDB_API_KEY=<your key from wandb.ai/settings>
WANDB_PROJECT=automend-model2
WANDB_ENTITY=<your username or team>
```

All training and evaluation scripts call `load_dotenv()` at startup — no manual `wandb login` required.

---

### MPS — macOS Apple Silicon

```bash
conda activate mlops_project_model

# Base dependencies (torch for device detection; HF for tokenizer)
pip install torch transformers accelerate pyyaml loguru wandb numpy pytest

# MLX backend
pip install mlx mlx-lm
```

Verify the Metal GPU is visible:

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0)

python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True
```

> Do **not** install `bitsandbytes`, `peft`, or `trl` on Apple Silicon. The MLX backend handles quantization and LoRA natively.

---

### CPU — Any Platform

```bash
conda activate mlops_project_model
pip install torch transformers accelerate peft pyyaml loguru wandb numpy pytest
# Skip bitsandbytes — quantization is not available on CPU
```

---

## 4. Repository Structure

```
model_2_training/
│
├── configs/
│   ├── data/
│   │   └── track_b_chatml.yaml         # Data paths, split ratios, sequence length, sample caps
│   ├── model/
│   │   ├── qwen_baseline.yaml          # Qwen2.5-1.5B-Instruct — quantization, LoRA targets
│   │   └── qwen_3b_baseline.yaml       # Scale-up variant (3B parameters)
│   ├── train/
│   │   └── qlora_sft.yaml              # LR, epochs, batch size, LoRA rank/alpha/dropout
│   ├── eval/
│   │   └── json_eval.yaml              # Inference params (max_new_tokens, temperature)
│   ├── sweep/
│   │   └── wandb_sweep.yaml            # Bayesian sweep — 10 parameters, objective metric
│   └── robustness/
│       └── robustness.yaml             # Perturbation parameter reference
│
├── scripts/
│   ├── run_split.py                    # Create stratified train/val/test splits
│   ├── run_train.py                    # Train (auto-detects CUDA/MPS/CPU)
│   ├── run_eval.py                     # Evaluate on validation set (all phases)
│   ├── run_test.py                     # Final evaluation on held-out test set
│   ├── build_benchmark.py              # Phase 2.5 — build locked gold benchmark
│   ├── run_benchmark.py                # Phase 2.5 — score checkpoint vs. benchmark
│   ├── run_sweep.py                    # Phase 2.75 — single W&B sweep trial
│   └── run_robustness.py               # Phase 3 — robustness + slice evaluation
│
├── src/
│   ├── utils/
│   │   ├── device.py                   # ★ Device detection & per-device config routing
│   │   ├── config.py                   # YAML config loader
│   │   ├── seed.py                     # Reproducible seeding (Python, NumPy, PyTorch)
│   │   ├── io.py                       # Atomic file I/O helpers
│   │   └── paths.py                    # Path constants
│   │
│   ├── data/
│   │   ├── dataset_contract.py         # ChatML schema validation (contract enforcement)
│   │   ├── load_jsonl.py               # JSONL loader with configurable malformed-row handling
│   │   ├── split_data.py               # Stratified train/val/test splitter (seed=42)
│   │   ├── dataset_builder.py          # ChatMLSupervisedDataset — PyTorch Dataset
│   │   └── collators.py                # AssistantOnlyCollator — masks system/user tokens to -100
│   │
│   ├── model/
│   │   ├── load_tokenizer.py           # AutoTokenizer loader with chat template
│   │   ├── load_qwen.py                # Qwen model loader — 4-bit NF4 (CUDA) or fp32 (CPU)
│   │   └── lora_factory.py             # PEFT LoRA attachment (CUDA/CPU only)
│   │
│   ├── train/
│   │   ├── train_loop.py               # ★ Orchestrator — dispatches to HF Trainer or MLX
│   │   ├── trainer_factory.py          # HuggingFace TrainingArguments + Trainer builder
│   │   ├── mlx_train.py                # ★ MLX backend: data prep, LoRA config, training, inference
│   │   └── callbacks.py                # Custom HF Trainer callbacks
│   │
│   ├── schemas/
│   │   └── workflow_schema.py          # Pydantic v2 models — WorkflowStep, Workflow,
│   │                                   #   ToolWorkflowResponse, MessageWorkflowResponse
│   │
│   ├── eval/
│   │   ├── generator.py                # ★ Device-aware generation (HF or MLX backend)
│   │   ├── metrics_json.py             # Phase 1 — 9 JSON structural quality metrics
│   │   ├── metrics_schema.py           # Phase 2A — Pydantic schema validation metrics
│   │   ├── metrics_fields.py           # Phase 2B — Field-level correctness metrics
│   │   ├── context_tool_parser.py      # Phase 2C — Extract tool schemas from system messages
│   │   ├── metrics_params.py           # Phase 2C — Parameter validation metrics
│   │   ├── metrics_aggregator.py       # Phase 2D — Unified pipeline, runs all phases
│   │   ├── error_taxonomy.py           # Phase 2.5 — 9-category failure labeling
│   │   ├── save_reports.py             # Write metrics.json, markdown, error samples
│   │   └── evaluator.py                # Full eval pipeline orchestrator
│   │
│   ├── robustness/
│   │   ├── perturbations.py            # 5 perturbation functions + perturb_dataset()
│   │   ├── slice_eval.py               # Slice grouping + per-slice metrics computation
│   │   └── robustness_runner.py        # Phase 3 orchestrator — loads model once, runs all passes
│   │
│   ├── test/
│   │   └── run_testset.py              # Test set evaluation wrapper
│   │
│   └── tracking/
│       ├── wandb_logger.py             # W&B integration with graceful degradation
│       └── artifact_logger.py          # Local run artifact archiving
│
├── data/
│   ├── splits/                         # Generated by run_split.py
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   ├── test.jsonl
│   │   └── split_summary.json
│   └── benchmarks/                     # Generated once by build_benchmark.py
│       ├── gold_benchmark.jsonl        # 30 locked evaluation questions
│       └── gold_benchmark_manifest.json
│
└── outputs/                            # All generated outputs (gitignored)
    ├── checkpoints/
    └── reports/
```

Files marked ★ contain device dispatch logic.

---

## 5. Data

### Source

This module consumes the output of the AutoMend data combiner pipeline:

```
AutoMend/data/processed/track_B_combined.jsonl   ← required (5,118 samples)
```

Produced by combining datasets ds3 + ds4 + ds5 + ds6. If the file is missing:

```bash
python -c "from src.combiner_track_b.combine import combine_track_b; combine_track_b()"
```

### Sample Format

Every sample in `track_B_combined.jsonl` uses ChatML format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant. Available Tools: {\"restart_pod\": {\"parameters\": [\"namespace\", \"pod\"], \"required\": [\"namespace\", \"pod\"]}}"
    },
    {
      "role": "user",
      "content": "The inference-worker pod is stuck in CrashLoopBackOff after the latest deploy."
    },
    {
      "role": "assistant",
      "content": "{\"workflow\": {\"steps\": [{\"tool\": \"restart_pod\", \"params\": {\"namespace\": \"prod\", \"pod\": \"inference-worker-7\"}}]}}"
    }
  ],
  "metadata": {
    "source_dataset": "ds5",
    "task_type": "tool_call"
  }
}
```

Samples failing the contract are handled via `malformed_row_strategy`:

- `"skip"` (default) — logged and dropped silently
- `"raise"` — raises `ContractViolation` immediately

### Split Sizes (full dataset)

| Split | Samples | Fraction |
|-------|---------|----------|
| Train | 4,094 | 80% |
| Validation | 511 | 10% |
| Test | 513 | 10% |
| **Total** | **5,118** | |

---

## 6. Phase 1 — Fine-Tuning

Fine-tunes Qwen2.5-1.5B-Instruct with QLoRA on the Track B dataset. Device detection routes automatically to the appropriate backend — the same four commands work on CUDA, MPS, and CPU.

### Step 1 — Create Splits

```bash
python scripts/run_split.py --config configs/data/track_b_chatml.yaml
```

Creates `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl`, and `split_summary.json`.

**Smoke test configuration** (set in `configs/data/track_b_chatml.yaml`):

```yaml
max_train_samples: 200
max_val_samples:   50
max_test_samples:  50
```

---

### Step 2 — Train

```bash
python scripts/run_train.py \
  --data-config  configs/data/track_b_chatml.yaml \
  --model-config configs/model/qwen_baseline.yaml \
  --train-config configs/train/qlora_sft.yaml
```

**Optional CLI overrides** (all platforms):

```bash
python scripts/run_train.py \
  --data-config  configs/data/track_b_chatml.yaml \
  --model-config configs/model/qwen_baseline.yaml \
  --train-config configs/train/qlora_sft.yaml \
  --epochs 3 \
  --lr 2e-4 \
  --batch-size 2
```

**What happens automatically by device:**

| Step | CUDA | MPS | CPU |
|------|------|-----|-----|
| Seed | ✓ | ✓ | ✓ |
| Config snapshot | `device: "cuda"` | `device: "mps"` | `device: "cpu"` |
| Tokenizer | HF `AutoTokenizer` | HF `AutoTokenizer` | HF `AutoTokenizer` |
| Data preparation | `ChatMLSupervisedDataset` | MLX chat JSONL conversion | `ChatMLSupervisedDataset` |
| Model load | Qwen + 4-bit NF4 | mlx-lm loads base model | Qwen fp32 |
| LoRA attachment | PEFT `get_peft_model` | mlx-lm built-in | PEFT `get_peft_model` |
| Training engine | HuggingFace Trainer | `mlx_lm.lora` subprocess | HuggingFace Trainer |
| Mixed precision | bf16 (Ampere+) / fp16 | Metal — managed by MLX | fp32 |
| Adapter saved | PEFT `.safetensors` | MLX `.npz` | PEFT `.safetensors` |

---

### Step 3 — Validate

```bash
python scripts/run_eval.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

Runs all Phase 1 + Phase 2 metrics automatically. Reports saved to `outputs/reports/val/`.

---

### Step 4 — Test (run once only)

```bash
python scripts/run_test.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

> Run **once only**, after all hyperparameter decisions are final. Never use test results to guide tuning — this set must remain a held-out final benchmark.

---

### Model & LoRA Architecture

#### Base Model

| Property | Value |
|----------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| HuggingFace ID | `Qwen/Qwen2.5-1.5B-Instruct` |
| Total parameters | ~1.55B |
| Context window | 32,768 tokens (capped at 2,048 during training) |

#### LoRA Configuration

| Property | CUDA / CPU (PEFT) | MPS (mlx-lm) |
|----------|-------------------|--------------|
| Rank (r) | 16 | 16 |
| Alpha (α) | 32 | 32 |
| Dropout | 0.05 | 0.05 |
| Layers trained | Last 16 of 28 (layers 12–27) | Last 16 of 28 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | All projection layers |
| Trainable parameters | ~42M (~2.7% of 1.55B) | ~42M (approx.) |

#### Loss Masking

`AssistantOnlyCollator` scans each tokenized sequence and sets `labels = -100` for all system and user tokens. Only assistant-role tokens contribute to the cross-entropy loss. On MPS, `mlx_lm.lora` handles loss masking internally using the chat format structure.

---

### Training Details by Backend

#### CUDA / CPU — HuggingFace Trainer

1. Tokenize with `apply_chat_template()`, truncate and pad to 2,048 tokens
2. `AssistantOnlyCollator` masks non-assistant tokens to `-100`
3. AdamW optimizer + cosine LR schedule + 5% warmup steps
4. bitsandbytes 4-bit NF4 quantization (CUDA only)
5. Best checkpoint restored by `eval_loss` at end of training

**Smoke test results — 200 samples, 1 epoch, RTX 5070:**

| Step | Training Loss | Learning Rate |
|------|--------------|---------------|
| 10 | 0.578 | 1.58e-4 |
| 20 | 0.428 | 3.17e-5 |

Loss dropped 26% over 25 steps — model is learning the JSON schema.

#### MPS — mlx-lm

1. `prepare_mlx_data()` converts splits to MLX chat format in `outputs/checkpoints/mlx_data/`
2. `build_mlx_lora_config()` writes `mlx_lora_config.yaml` for the `mlx_lm.lora` CLI
3. `python -m mlx_lm lora --config mlx_lora_config.yaml` runs on the Metal GPU
4. LoRA adapter saved to `outputs/checkpoints/best_model/` in MLX format
5. Validation loss reported every `eval_steps` iterations

**Iteration count formula:**

```
iters = ceil(n_samples / batch_size) × num_epochs
# Smoke test example: ceil(200 / 2) × 1 = 100 iterations
```

---

### Phase 1 Metrics

All metrics are computed on `outputs/reports/val/` after `run_eval.py`. The 9 Phase 1 structural metrics:

| Metric | Smoke Test (50 val samples) | Interpretation |
|--------|-----------------------------|----------------|
| JSON parse rate | **96%** | 48/50 outputs parse as valid JSON |
| Non-empty output rate | **100%** | Model always produces output |
| Starts with `{` or `[` | **100%** | JSON structure always initiated |
| Ends with `}` or `]` | **98%** | 1 sample slightly truncated |
| Quote balance rate | **98%** | Strings properly closed |
| Brace balance rate | **94%** | Nested braces mostly balanced |
| Truncation rate | 2% | 1 sample hit the max token limit |
| Malformed JSON rate | 4% | 2 samples with unclosed strings |
| Average output length | 392 chars | Appropriate verbosity |

---

## 7. Phase 2 — Structured Evaluation

Phase 1 confirmed the model produces valid JSON (96% parse rate). Phase 2 checks whether that JSON is **correct** — right structure, right field values, right parameter names and types.

```
Phase 1  → Did the JSON parse?
Phase 2A → Is the JSON the correct shape/structure?
Phase 2B → Are the field values meaningful?
Phase 2C → Are tool parameters valid against the tool's schema?
Phase 2D → Aggregate and report all phases together
```

Phase 2 runs **automatically** as part of `run_eval.py` — no separate commands needed.

---

### Phase 2A — Pydantic Schema Validation

#### Purpose

Validates every parsed output against the expected output contract using Pydantic v2 with `extra="forbid"`. Catches structural failures that `json.loads()` silently accepts: missing `workflow` key, `steps` that is not a list, steps missing `tool` or `params`, wrong field types, and unexpected extra keys at any nesting level.

#### Output Shapes Validated

| Shape | When | Required fields |
|-------|------|-----------------|
| `ToolWorkflowResponse` | Model calls one or more tools | `workflow.steps` non-empty list; no `message` key |
| `MessageWorkflowResponse` | No applicable tool available | `workflow.steps` empty list; `message` non-empty string |

Both shapes are validated in order. `correct_shape_rate` compares the generated shape against the reference — a model that refuses when it should act (or vice versa) fails this check even if the JSON structure is otherwise valid.

#### Key Files

| File | Role |
|------|------|
| `src/schemas/workflow_schema.py` | Pydantic models: `WorkflowStep`, `Workflow`, `ToolWorkflowResponse`, `MessageWorkflowResponse`; `parse_response()`; `infer_shape()` |
| `src/eval/metrics_schema.py` | `compute_schema_metrics()` — validation loop, metrics dict; `summarize_schema_errors()` — failure samples |

#### Metrics

| Metric | Smoke Test | Description |
|--------|-----------|-------------|
| `schema_valid_rate` | **94%** | Fraction passing full Pydantic validation |
| `correct_shape_rate` | **90%** | Fraction where generated shape matches reference shape |
| `extra_fields_rate` | 0% | Fraction with unexpected extra keys at any level |
| `wrong_type_rate` | 0% | Fraction with any field of incorrect Python type |

> **Key finding:** 1 step had a `tool` key but no `params` key at all — schema validation catches this; `json.loads()` does not.

---

### Phase 2B — Field-Level Correctness

#### Purpose

For each field in a schema-valid output, verifies the **value** is meaningful — not just present and correctly typed. Operates on raw JSON-parsed dicts so it catches value issues even in outputs that failed Phase 2A.

Each metric is computed only over the subset of predictions where that field is applicable:

#### Key Files

| File | Role |
|------|------|
| `src/eval/metrics_fields.py` | `compute_field_metrics()` — per-field rates; `compute_per_field_report()` — presence/non-empty/type breakdown |

#### Metrics

| Metric | Smoke Test | Denominator | Description |
|--------|-----------|-------------|-------------|
| `tool_name_nonempty_rate` | **100%** | 57 steps | Steps with a non-empty tool name string |
| `params_nonempty_rate` | **93%** | 57 steps | Steps where the params dict is populated |
| `steps_count_match_rate` | **87.5%** | 48 pairs | Generated step count equals reference step count |
| `message_nonempty_rate` | **100%** | 18 responses | Refusal responses with actual message content |
| `steps_is_list_rate` | **100%** | 48 parseable | Outputs where `workflow.steps` is a list |

> **Key finding:** 87.5% step count match. The remaining ~12.5% over-call — generate 2 steps when 1 is correct. This resolves significantly with full dataset training (200 → 5,118 samples).

---

### Phase 2C — Parameter Validation

#### Purpose

For each generated tool step, validates the parameters against the tool's own schema. Catches missing required parameters, unexpected extra parameters, and type mismatches when type information is available.

#### Design: No Hardcoded Tools

Tools are **never hardcoded**. They are provided at runtime via the system message in each sample. `context_tool_parser.py` extracts schemas per-sample so validation uses exactly the tool context the model saw during generation.

**System message formats handled:**

| Format | Example |
|--------|---------|
| Names only | `Available Tools: scale_service, restart_pod` |
| Simple list schema | `{"calc_tip": {"parameters": ["amount", "percent"], "required": ["amount"]}}` |
| JSON Schema (typed) | `{"get_weather": {"parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}}` |
| Empty | `Available Tools: {}` |

Steps without an extractable schema are **skipped** (not penalized) and tracked via `param_schema_coverage_rate`.

#### Type Checking Map

| JSON Schema type | Python type checked |
|------------------|---------------------|
| `"string"` | `str` |
| `"number"` | `int` or `float` |
| `"integer"` | `int` (not `bool`) |
| `"boolean"` | `bool` |
| `"array"` | `list` |
| `"object"` | `dict` |

#### Key Files

| File | Role |
|------|------|
| `src/eval/context_tool_parser.py` | `extract_tools_from_system_message()` — parses all 4 formats; `get_sample_tool_schemas()` — per-sample entry point |
| `src/eval/metrics_params.py` | `compute_param_metrics()` — validation rates; `validate_step_params()` — single-step validator; `summarize_param_errors()` — failure samples |

#### Metrics

| Metric | Smoke Test | Denominator | Description |
|--------|-----------|-------------|-------------|
| `param_schema_coverage_rate` | **89.5%** | 57 steps | Steps with an extractable tool schema |
| `param_completeness_rate` | **100%** | 51 validated steps | All required params present |
| `param_no_extras_rate` | **100%** | 51 validated steps | No unexpected params |
| `param_type_correctness_rate` | N/A | 0 | Activates when JSON Schema-format tool definitions are injected |
| `full_param_validity_rate` | **100%** | 51 validated steps | All available checks passed (composite) |

> `param_type_correctness_rate` will activate automatically once richer tool schemas (with `properties` + `type`) are injected via RAG at inference time. Current training data uses flat list schemas with no type information.

---

### Phase 2D — Aggregation & Reporting

`metrics_aggregator.py` wires all phases into a single call. Every `run_eval.py` invocation runs the full stack automatically.

- `run_all_metrics(predictions)` — runs Phase 1 → 2A → 2B → 2C in sequence, returns one flat dict of **54 phase-namespaced metrics**. Each phase is isolated; a failure in one does not stop the others.
- `save_reports.py` writes per-phase Markdown tables, schema error breakdown, per-field breakdown, and a multi-line interpretation section.
- `wandb_logger.log_eval_metrics()` logs all scalar metrics under `val/` or `test/` prefixes in W&B.

#### Modified Files (Phase 2D additions)

| File | Change |
|------|--------|
| `src/eval/metrics_aggregator.py` | New — `run_all_metrics()`, `get_per_field_report()`, `get_schema_errors()`, `get_param_errors()` |
| `src/eval/save_reports.py` | Per-phase markdown tables, `save_param_errors()`, updated `save_all_reports()` |
| `src/eval/evaluator.py` | Wired `run_all_metrics()` and new report helpers |
| `src/tracking/wandb_logger.py` | Added `log_eval_metrics()` |
| `configs/eval/json_eval.yaml` | Added `validation` section |
| `scripts/run_eval.py` / `run_test.py` | Updated summary log to show one key metric per phase |

---

### Running Phase 2

Phase 2 runs as part of `run_eval.py`. Commands are the same as Phase 1 evaluation.

**macOS / Linux:**

```bash
cd /path/to/AutoMend/model_2_training

python scripts/run_eval.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

**Windows:**

```powershell
cd C:\path\to\AutoMend\model_2_training

python scripts\run_eval.py `
  --config     configs\eval\json_eval.yaml `
  --checkpoint outputs\checkpoints\best_model
```

**Expected terminal output (one key metric per phase):**

```
INFO  | json_parse_rate        : 0.96
INFO  | schema_valid_rate      : 0.94
INFO  | steps_count_match_rate : 0.875
INFO  | full_param_validity    : 1.0
```

---

### Verifying Phase 2 Without a GPU (Smoke Test)

To verify all Phase 2 modules against previously saved predictions — no model or GPU needed:

**macOS / Linux:**

```bash
cd /path/to/AutoMend

python3 -c "
import sys, json
sys.path.insert(0, '.')

with open('model_2_training/data/splits/val.jsonl') as f:
    val_samples = {i: json.loads(l) for i, l in enumerate(f)}
with open('model_2_training/outputs/reports/val/sample_outputs.json') as f:
    saved = json.load(f)

preds = [
    {
        'index':     s['index'],
        'generated': s['generated'],
        'reference': s['reference'],
        'sample':    val_samples.get(s['index'], {}),
    }
    for s in saved
]

from model_2_training.src.eval.metrics_aggregator import run_all_metrics
metrics = run_all_metrics(preds)

for k, v in sorted(metrics.items()):
    if isinstance(v, float):
        print(f'  {k:<55} {v*100:.1f}%')
"
```

**Windows:**

```powershell
cd C:\path\to\AutoMend

python -c "
import sys, json
sys.path.insert(0, '.')
with open('model_2_training/data/splits/val.jsonl') as f:
    val_samples = {i: json.loads(l) for i, l in enumerate(f)}
with open('model_2_training/outputs/reports/val/sample_outputs.json') as f:
    saved = json.load(f)
preds = [{'index': s['index'], 'generated': s['generated'], 'reference': s['reference'], 'sample': val_samples.get(s['index'], {})} for s in saved]
from model_2_training.src.eval.metrics_aggregator import run_all_metrics
metrics = run_all_metrics(preds)
[print(f'  {k:<55} {v*100:.1f}%') for k, v in sorted(metrics.items()) if isinstance(v, float)]
"
```

---

### Running Individual Phases in Isolation

**macOS / Linux:**

```bash
cd /path/to/AutoMend

# Phase 2A — Schema validation only
python3 -c "
import sys, json; sys.path.insert(0, '.')
with open('model_2_training/outputs/reports/val/sample_outputs.json') as f:
    preds = [{'generated': s['generated'], 'reference': s['reference']} for s in json.load(f)]
from model_2_training.src.eval.metrics_schema import compute_schema_metrics
print(compute_schema_metrics(preds))
"

# Phase 2B — Field correctness only
python3 -c "
import sys, json; sys.path.insert(0, '.')
with open('model_2_training/outputs/reports/val/sample_outputs.json') as f:
    preds = [{'generated': s['generated'], 'reference': s['reference']} for s in json.load(f)]
from model_2_training.src.eval.metrics_fields import compute_field_metrics
print(compute_field_metrics(preds))
"

# Phase 2C — Parameter validation only (needs sample key for tool context)
python3 -c "
import sys, json; sys.path.insert(0, '.')
with open('model_2_training/data/splits/val.jsonl') as f:
    val = {i: json.loads(l) for i, l in enumerate(f)}
with open('model_2_training/outputs/reports/val/sample_outputs.json') as f:
    preds = [{'generated': s['generated'], 'reference': s['reference'], 'sample': val.get(s['index'], {})} for s in json.load(f)]
from model_2_training.src.eval.metrics_params import compute_param_metrics
print(compute_param_metrics(preds))
"
```

---

### Phase 2 Full Metrics Reference

All metrics are floats in `[0.0, 1.0]` unless noted. Keys match the `metrics.json` output.

| Phase | Metric Key | Description |
|-------|-----------|-------------|
| 1 | `phase1_structural/json_parse_rate` | Fraction of outputs that parse as valid JSON |
| 1 | `phase1_structural/non_empty_rate` | Fraction producing non-empty output |
| 1 | `phase1_structural/starts_with_brace_rate` | Fraction starting with `{` or `[` |
| 1 | `phase1_structural/ends_with_brace_rate` | Fraction ending with `}` or `]` |
| 1 | `phase1_structural/quote_balance_rate` | Fraction with even double-quote count |
| 1 | `phase1_structural/brace_balance_rate` | Fraction with balanced `{}`/`[]` |
| 1 | `phase1_structural/truncation_rate` | Fraction hitting the max token limit |
| 1 | `phase1_structural/malformed_json_rate` | Fraction that are non-empty but fail `json.loads()` |
| 1 | `phase1_structural/avg_output_length` | Mean character length of output (not a rate) |
| 1 | `phase1_structural/tax_valid_rate` | Fraction labeled `VALID` by the Phase 2.5 taxonomy |
| 2A | `phase2a_schema/schema_valid_rate` | Fraction passing full Pydantic v2 validation |
| 2A | `phase2a_schema/correct_shape_rate` | Fraction where generated shape matches reference shape |
| 2A | `phase2a_schema/extra_fields_rate` | Fraction with unexpected extra keys at any level |
| 2A | `phase2a_schema/wrong_type_rate` | Fraction with any field of wrong Python type |
| 2A | `phase2a_schema/schema_error_distribution` | `{error_type: count}` dict (not a rate) |
| 2B | `phase2b_fields/tool_name_nonempty_rate` | Fraction of steps with a non-empty tool name |
| 2B | `phase2b_fields/params_nonempty_rate` | Fraction of steps with populated params dict |
| 2B | `phase2b_fields/steps_count_match_rate` | Fraction where step count matches reference |
| 2B | `phase2b_fields/message_nonempty_rate` | Fraction of refusal responses with actual content |
| 2B | `phase2b_fields/steps_is_list_rate` | Fraction where `workflow.steps` is a list |
| 2C | `phase2c_params/param_schema_coverage_rate` | Fraction of steps with an extractable tool schema |
| 2C | `phase2c_params/param_completeness_rate` | Fraction of validated steps with all required params |
| 2C | `phase2c_params/param_no_extras_rate` | Fraction of validated steps with no extra params |
| 2C | `phase2c_params/param_type_correctness_rate` | Fraction with correct param types (when type info available) |
| 2C | `phase2c_params/full_param_validity_rate` | All available param checks passed (composite metric) |

---

## 8. Phase 2.5 — Gold Benchmark & Error Taxonomy

Before Phase 2.5, each evaluation run used different questions, making cross-model or cross-run comparisons meaningless. When predictions failed, the system provided only binary pass/fail with no diagnostic information. Phase 2.5 addresses both gaps.

### Building the Benchmark

Creates 30 **fixed, locked** test questions. Every model is evaluated on the same questions, enabling fair comparison across training runs, hyperparameter variants, and future model versions.

> **Run once only.** Rebuilding invalidates all historical comparisons. A manifest file records the lock with a warning.

**macOS / Linux:**

```bash
python scripts/build_benchmark.py
```

**Windows:**

```powershell
python scripts\build_benchmark.py
```

**Output files:**

| File | Contents |
|------|---------|
| `data/benchmarks/gold_benchmark.jsonl` | 30 locked evaluation samples |
| `data/benchmarks/gold_benchmark_manifest.json` | Seed, archetype counts, lock timestamp and warning |

**Manifest format:**

```json
{
  "seed": 42,
  "total_selected": 30,
  "archetype_counts": { "multi_step": 10, "refusal": 10, "single_step": 10 },
  "locked": true,
  "warning": "DO NOT rebuild without incrementing version. Rebuilding invalidates historical comparisons."
}
```

**Stratification — 10 samples per archetype:**

| Archetype | Expected model output |
|-----------|----------------------|
| `multi_step` | `workflow.steps` with 2 or more tool calls |
| `single_step` | `workflow.steps` with exactly 1 tool call |
| `refusal` | `workflow.steps: []` with a `message` field |

> A model scoring 96.7% overall but 70% on refusals is not production-safe — it will hallucinate tool calls for invalid requests. Stratification ensures all three output types are measured separately.

---

### Error Taxonomy

After scoring, every prediction receives exactly one label from the 9-category taxonomy. Labels are evaluated in priority order — the first condition that matches wins.

| Priority | Label | Condition |
|---------|-------|-----------|
| 1 | `VALID` | Non-empty output, parses as JSON, `workflow.steps` is correct format |
| 2 | `EMPTY` | Output is empty or whitespace only |
| 3 | `TRUNCATED` | Starts with `{` or `[` but missing the closing brace — hit the token limit |
| 4 | `MISSING_WORKFLOW` | Valid JSON but missing `workflow` key or `steps` key |
| 5 | `WRONG_STEPS_TYPE` | `workflow.steps` is present but is not a list |
| 6 | `EMPTY_STEPS` | `workflow.steps: []` when the reference expected non-empty steps |
| 7 | `MALFORMED_JSON` | Non-empty, not truncated, but `json.loads()` raises an exception |
| 8 | `UNBALANCED_BRACES` | Count of `{`/`}` or `[`/`]` does not match |
| 9 | `UNBALANCED_QUOTES` | Double-quote count is odd (broken string literal) |

> **Reference-awareness:** `EMPTY_STEPS` is only a failure when the reference itself expects non-empty steps. If the reference is a refusal (`steps: []`), an empty-steps output is **correct** and receives `VALID`. Without this distinction, the sweep would incorrectly penalize models for correctly refusing invalid requests.

---

### Scoring a Checkpoint

**macOS / Linux:**

```bash
python scripts/run_benchmark.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

**Windows:**

```powershell
python scripts\run_benchmark.py `
  --config     configs\eval\json_eval.yaml `
  --checkpoint outputs\checkpoints\best_model
```

To score a specific sweep trial checkpoint:

```bash
python scripts/run_benchmark.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/sweeps/Qwen2.5-1.5B-Instruct_4bit_lora-r16_200samples_20260323-0052
```

Results saved to `outputs/reports/benchmark/<checkpoint_name>/`.

---

## 9. Phase 2.75 — Hyperparameter Sweep

A Bayesian optimizer automatically explores the training hyperparameter space. Each trial trains a full model, benchmarks it against the 30 locked questions, and reports `tax_valid_rate` back to the optimizer. The optimizer uses all prior results to select the next configuration.

### Step 1 — Initialize the Sweep

```bash
wandb sweep configs/sweep/wandb_sweep.yaml
# Prints a sweep ID — copy it for the next command
```

**Sweep configuration:**

| Setting | Value |
|---------|-------|
| Project | `automend-model2` |
| Search method | Bayesian optimization |
| Objective | Maximize `benchmark/tax_valid_rate` |
| Early stopping | Hyperband (min_iter=1, eta=2) |

---

### Step 2 — Run Sweep Trials

Each agent runs one trial: trains a model, benchmarks it, and exits. Run multiple agents (locally or on separate machines) to search in parallel.

```bash
wandb agent mlops-team-northeastern-university/automend-model2/<sweep_id> --count 5
```

**Parameters searched (10 total):**

| Parameter | Search Space | Controls |
|-----------|-------------|---------|
| `lora_r` | {8, 16, 32, 64} | LoRA rank — model capacity vs. truncation risk |
| `learning_rate` | log-uniform [1e-5, 5e-4] | Update step size |
| `lora_alpha` | {16, 32, 64} | LoRA scaling factor |
| `lora_dropout` | {0.0, 0.05, 0.1} | Regularization inside LoRA layers |
| `per_device_train_batch_size` | {1, 2, 4} | Samples per gradient step |
| `gradient_accumulation_steps` | {4, 8, 16} | Effective batch multiplier |
| `lr_scheduler_type` | {cosine, linear, constant_with_warmup} | LR decay pattern |
| `warmup_ratio` | {0.03, 0.05, 0.10} | Fraction of steps used for warmup |
| `num_train_epochs` | {1, 2, 3} | Full training passes over the dataset |
| `weight_decay` | {0.0, 0.01, 0.1} | L2 regularization coefficient |

---

### Step 3 — Rank Trials

**macOS / Linux:**

```bash
for dir in outputs/reports/sweeps/Qwen2.5-1.5B-Instruct_*/; do
  rate=$(python3 -c "import json; d=json.load(open('$dir/metrics.json')); print(d.get('phase1_structural/tax_valid_rate','N/A'))")
  echo "$rate  $(basename $dir)"
done | sort -rn
```

**Windows:**

```powershell
Get-ChildItem -Directory "outputs/reports/sweeps/Qwen2.5-1.5B-Instruct_*" | ForEach-Object {
  $metrics = "$($_.FullName)/metrics.json"
  $rate = python -c "import json; d=json.load(open(r'$metrics')); print(d.get('phase1_structural/tax_valid_rate','N/A'))"
  [PSCustomObject]@{ Rate = $rate; Name = $_.Name }
} | Sort-Object Rate -Descending | Format-Table
```

---

### Sweep Results (21 trials completed)

| Rank | Config | Valid Rate | Schema Valid | Step Count Match |
|------|--------|-----------|-------------|-----------------|
| **1** | **r16 (0052)** | **96.7%** | **96.7%** | **75.9%** |
| 2 | r8 (0037) | 93.3% | 96.7% | 69.0% |
| 3 | r8 (1122) | 93.3% | 93.3% | 67.9% |
| 4 | r64 (0058) | 93.3% | 93.3% | 64.3% |

**LoRA rank analysis:**

| Rank | Trainable Params | Outcome |
|------|-----------------|---------|
| r8 | ~21M (1.4% of 1.55B) | Slight underfitting |
| **r16** | **~42M (2.7% of 1.55B)** | **Best score — sweep winner** |
| r64 | ~168M (10.9% of 1.55B) | Overfits — truncates output on 2/30 benchmark questions |

**Winning hyperparameters:**

| Parameter | Default | Sweep Winner | Why It Helped |
|-----------|---------|-------------|---------------|
| `lora_r` | 16 | **16** | Sweet spot — r8 underfits, r64 truncates |
| `learning_rate` | 2e-4 | **1.19e-4** | Conservative updates → stable structure learning |
| `per_device_train_batch_size` | 2 | **4** | More samples per gradient update |
| `gradient_accumulation_steps` | 8 | **16** | Effective batch of 64 — 32× larger than default |
| `lr_scheduler_type` | cosine | **constant_with_warmup** | Stable LR after warmup vs. decaying LR |
| `weight_decay` | 0.01 | **0.1** | Stronger regularization reduces overfitting |
| `num_train_epochs` | 3 | **2** | 2 epochs sufficient with larger effective batch |

> **Honest caveat:** The difference between r16 (96.7%) and r8 (93.3%) is one prediction out of 30 — within the margin of statistical noise for a 30-sample benchmark. A larger benchmark would be needed to confirm significance.

> **What 96.7% measures:** Structural validity — correct JSON with proper `workflow.steps` format. It does not confirm whether the model chose the right tool or the right parameter values. That is Phase 2C's job.

---

## 10. Phase 3 — Robustness & Slice Evaluation

Phase 3 stress-tests the best checkpoint against adversarial input variations and identifies where the model fails on clean inputs through slice-based analysis.

**Two mechanisms:**

1. **Perturbation testing** — Modify the 30 benchmark inputs in 5 ways, re-run the model, and measure the performance delta vs. the clean baseline.
2. **Slice-based evaluation** — Group clean predictions into meaningful subsets (by archetype, input length, complexity) and compute metrics per group to surface failure modes hidden by aggregate numbers.

The model is loaded **once** and reused for all 6 inference passes (1 clean + 5 perturbed), avoiding redundant model loading overhead.

---

### Step 1 — Run the Full Robustness Test

**macOS / Linux:**

```bash
python scripts/run_robustness.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

**Windows:**

```powershell
python scripts\run_robustness.py `
  --config     configs\eval\json_eval.yaml `
  --checkpoint outputs\checkpoints\best_model
```

**Run specific perturbation types only:**

```bash
python scripts/run_robustness.py \
  --config         configs/eval/json_eval.yaml \
  --checkpoint     outputs/checkpoints/best_model \
  --perturbations  typo noise truncation
```

**Use a custom benchmark (e.g., full test set):**

```bash
python scripts/run_robustness.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model \
  --benchmark  data/splits/test.jsonl
```

**What the runner does internally:**

1. Load 30 samples from `gold_benchmark.jsonl`
2. Load model and tokenizer once
3. Run generation on clean samples → compute 4-phase baseline metrics
4. For each of 5 perturbation types:
   - Apply perturbation to all samples
   - Run generation on perturbed samples
   - Compute all metrics
   - Identify regressions (samples that were VALID on clean but not VALID on perturbed)
5. Build delta summary table (clean vs. each perturbation type)
6. Run slice-based evaluation on clean predictions across 4 dimensions
7. Write all output artifacts to `outputs/reports/robustness/<checkpoint_name>/`

---

### Perturbation Types

All perturbations modify **only the user message content**. The reference answer and system message are unchanged, so the existing metrics can score perturbed outputs against the same ground truth.

#### `typo` — Character-Level Noise

For each alphabetic character, with 5% probability: swap it with the adjacent character (transposition), delete it, or replace it with a QWERTY-adjacent key.

```
Original : "The pod is stuck in CrashLoopBackOff after the latest deploy"
Perturbed: "The pdo is stuxk in CrashLoopBackOf aftdr the latset deploy"
```

**What it reveals:** Whether the model degrades on realistic typing errors. Models that learned exact token patterns from training data will be more sensitive than models that learned semantic patterns.

---

#### `noise` — Irrelevant Context Injection

Prepends one randomly selected boilerplate sentence from a fixed list of 10 support-ticket phrases.

```
Original : "Latency on the inference endpoint has spiked to 8s P99."
Perturbed: "High-priority incident — SLA breach imminent. Latency on the inference endpoint has spiked to 8s P99."
```

**What it reveals:** Whether the model correctly ignores task-irrelevant preamble. Real support tickets frequently contain boilerplate added by ticketing systems, escalation annotations, or on-call runbook headers.

---

#### `truncation` — Incomplete Input

Drops the last 30% of words from the user message. Always preserves at least 3 words.

```
Original : "The Spark job is failing with OutOfMemoryError. Executor logs show heap exhaustion. Increase executor memory."
Perturbed: "The Spark job is failing with OutOfMemoryError. Executor logs show heap"
```

**What it reveals:** Whether the model produces a valid response when the incident description is cut off. Simulates front-end bugs, mobile keyboard dismissal mid-submission, or copy-paste truncation.

---

#### `case_lower` — Lowercased Input

Converts the entire user message to lowercase. Fully deterministic, no random seed needed.

```
Original : "Pod OOMKilled — increase memory limit on the inference-worker container"
Perturbed: "pod oomkilled — increase memory limit on the inference-worker container"
```

**What it reveals:** Whether the model relies on capitalization as a structural signal. If `case_lower` degrades step count match but not JSON validity, the model understood the task type but misidentified specific tools.

---

#### `paraphrase` — Sentence Order Shuffle

Splits the user message on sentence boundaries and shuffles the order. Deterministic — no LLM required.

```
Original : "Executor logs show heap exhaustion. The Spark job is failing with OOMError. Increase executor memory."
Perturbed: "Increase executor memory. Executor logs show heap exhaustion. The Spark job is failing with OOMError."
```

**What it reveals:** Whether the model relies on sentence position rather than semantic content. This is the most realistic perturbation — users write incident descriptions in any order, and the model must derive meaning from content, not structure.

---

### Slice Dimensions

Slice evaluation runs on **clean predictions only** to get a pure signal on input-type sensitivity, decoupled from perturbation effects.

| Dimension | Groups | Purpose |
|-----------|--------|---------|
| `archetype` | `multi_step` / `single_step` / `refusal` | Failure rate by output type — a model that fails refusals is not production-safe |
| `dataset` | From `sample.metadata.source` | Per-dataset performance gaps — each source dataset has a distinct prompt style |
| `input_length` | `short` (<100 chars) / `medium` (100–300) / `long` (>300) | Length sensitivity — long inputs may exceed model comprehension |
| `complexity` | `refusal` (0 steps) / `simple` (1) / `moderate` (2–3) / `complex` (4+) | Step-count sensitivity — complex workflows may exceed learned capacity |

> **`dataset` dimension note:** All 30 current gold benchmark samples resolve to `"unknown"` because the benchmark JSONL was built without copying `metadata.source_dataset` into each sample. Add source labels when rebuilding the benchmark to activate this dimension.

> **Param validity for refusal / long-input rows:** `steps: []` means there are no parameters to validate — the denominator is zero. The validator returns `0.0` in this case. Read `0.0%` param validity in these rows as **not applicable**, not a failure.

---

### Phase 3 Output Files

All written to `outputs/reports/robustness/<checkpoint_name>/`:

| File | Contents | When to use |
|------|---------|-------------|
| `clean_predictions.jsonl` | Raw generations on unperturbed benchmark | Baseline inspection, manual review |
| `clean_metrics.json` | Full 54-metric baseline dict | Programmatic comparison |
| `{pert}_predictions.jsonl` | Raw generations per perturbation type (×5) | Manual inspection of perturbed outputs |
| `{pert}_metrics.json` | Full 54-metric dict per perturbation (×5) | Detailed metric drill-down per perturbation |
| `robustness_summary.json` | Delta table + worst perturbation identifier | Primary robustness result |
| `slice_report.json` | Per-slice metrics across all 4 dimensions | Input-type sensitivity analysis |
| `failure_log.json` | Every VALID→fail regression with original and perturbed I/O | Root cause debugging |
| `robustness_report.md` | Human-readable Markdown summary of all results | Presentations, team reviews |

---

### Phase 3 Results (2026-03-24)

**Checkpoint:** `Qwen2.5-1.5B-Instruct_4bit_lora-r16_200samples_20260323-0052`

#### Clean Baseline (30 samples)

| Metric | Value |
|--------|-------|
| JSON parse rate | 96.7% |
| Valid output rate (`tax_valid_rate`) | **96.7%** |
| Schema valid rate | 96.7% |
| Step count match rate | 75.9% |
| Param validity rate | 100.0% |

#### Robustness Deltas (positive = degradation vs. clean baseline)

| Perturbation | Valid Rate | JSON Parse | Schema Valid | Step Count | Verdict |
|-------------|-----------|-----------|-------------|-----------|---------|
| `typo` | +0.0% | −3.3% | −3.3% | −0.8% | Robust — negatives are 1-sample noise at n=30 |
| `noise` | **+0.0%** | **+0.0%** | **+0.0%** | **+0.0%** | **Fully robust** |
| `truncation` | +3.3% | +0.0% | +0.0% | +6.9% | Mild — step accuracy drops on incomplete inputs |
| `case_lower` | +0.0% | +0.0% | +0.0% | +3.5% | Mild — JSON valid but step count degrades |
| `paraphrase` | **+3.3%** | **+3.3%** | **+3.3%** | **+4.4%** | **Worst — model learned sentence-order patterns** |

**Total regressions (VALID → fail):** 4 across all 5 perturbation types out of 150 total (2.7%)

#### Slice Results (clean predictions)

**By archetype:**

| Slice | N | Valid Rate | Schema Valid | Param Valid |
|-------|---|-----------|-------------|------------|
| `multi_step` | 10 | 100.0% | 100.0% | 100.0% |
| `single_step` | 10 | 100.0% | 100.0% | 100.0% |
| `refusal` | 10 | 90.0% | 90.0% | N/A* |

**By input length:**

| Slice | N | Valid Rate | Schema Valid | Param Valid |
|-------|---|-----------|-------------|------------|
| `short` (<100 chars) | 22 | 100.0% | 100.0% | 100.0% |
| `medium` (100–300 chars) | 3 | 100.0% | 100.0% | 100.0% |
| `long` (>300 chars) | 5 | 80.0% | 80.0% | N/A* |

*N/A = no parameters to validate (empty steps or no extractable schema)

**Key findings:**

- `noise` shows zero degradation — the model correctly ignores irrelevant preamble text in the input.
- `paraphrase` is the worst perturbation — the model learned position-dependent patterns from training data; sentence order matters to it.
- `truncation` hurts step count match (+6.9%) but not JSON validity — model still outputs valid structure even on incomplete inputs.
- **Refusal slice is the production risk area** — 90% vs. 100% on tool-call archetypes. 1 of 10 refusal cases the model failed to refuse correctly; this is the highest-risk failure mode in production.
- **Long inputs degrade** — 80% vs. 100% on short inputs. Inspect `failure_log.json` for the failing long samples; may require increasing `max_new_tokens`.

---

## 11. GCP Cloud Deployment

> Production deployment of the AutoMend training pipeline on Google Cloud Platform using Vertex AI Pipelines for GPU compute, Cloud Run for orchestration webhooks, and Cloud Scheduler for automated trigger timing.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LOCAL MACHINE                                │
│                                                                     │
│  gcp/build_and_push.sh ───► Artifact Registry (Docker image)       │
│  gcp/jobs/submit_sweep_agent.py ───► Vertex AI Custom Training Jobs │
│  gcp/pipelines/submit_training_pipeline.py ───► Vertex AI Pipeline │
└─────────────────────────────────────────────────────────────────────┘
                │                        │
                ▼                        ▼
┌──────────────────────┐    ┌────────────────────────────────────────┐
│  W&B Sweep (W1)      │    │  Vertex AI Pipeline (W2/W3/W4)         │
│  N × L4 GPU trials   │    │  1. Split      (CPU)                   │
│  Bayesian optimizer  │    │  2. Train      (L4 GPU)                │
│  picks best config   │    │  3. Eval       (L4 GPU)                │
└──────────┬───────────┘    │  4. Test       (L4 GPU)                │
           │                │  5. Benchmark  (L4 GPU)                │
           ▼                │  6. Robustness (L4 GPU)                │
┌──────────────────────┐    └────────────────────────────────────────┘
│  Cloud Scheduler     │───►┌────────────────────────────────────────┐
│  fires after N hours │    │  Cloud Run (automend-webhook)          │
│  (auto-created by    │    │  /trigger endpoint:                    │
│   submit_sweep.py)   │    │  1. Check sweep finished               │
└──────────────────────┘    │  2. Fetch best config from W&B         │
                            │  3. Upload config to GCS               │
                            │  4. Submit Vertex AI Pipeline          │
                            └────────────────────────────────────────┘
                                          │
                                          ▼
                            ┌────────────────────────────────────────┐
                            │  GCS Bucket: gs://automend-model2      │
                            │  ├── data/                             │
                            │  │   ├── track_B_combined.jsonl        │
                            │  │   ├── splits/                       │
                            │  │   └── benchmarks/                   │
                            │  │       └── gold_benchmark.jsonl      │
                            │  ├── configs/train/                    │
                            │  │   └── best_sweep_config.yaml        │
                            │  └── outputs/runs/<run_id>/            │
                            │      ├── checkpoints/best_model/       │
                            │      └── reports/                      │
                            └────────────────────────────────────────┘
```

---

### GCP Infrastructure

| Resource | Name / Value |
|----------|-------------|
| Project ID | `automend` |
| Region | `us-central1` |
| GCS Bucket | `gs://automend-model2` |
| Artifact Registry | `us-central1-docker.pkg.dev/automend/automend-images` |
| Docker Image | `automend-train:latest` (PyTorch 2.5.1 + CUDA 12.4) |
| Training Machine | `g2-standard-8` + NVIDIA L4 24 GB |
| Eval / Benchmark / Robustness | `g2-standard-8` + NVIDIA L4 24 GB (GPU inference) |
| Split Machine | CPU only (`n1-standard-4`) |
| Trainer Service Account | `automend-trainer@automend.iam.gserviceaccount.com` |
| W&B API Key Secret | `WANDB_API_KEY` (in Secret Manager) |
| Cloud Run Service | `automend-webhook` |
| W&B Project | `automend-model2` (entity: `mlops-team-northeastern-university`) |

---

### GCP Repository Structure

```
gcp/
├── Dockerfile                        # Training container (PyTorch 2.5.1 + CUDA 12.4 + gcsfuse)
├── .dockerignore                     # Excludes .env, outputs/, data/splits/, wandb/
├── requirements-gcp.txt              # GCP Python dependencies for container
├── config.py                         # Central GCP config (project, region, SA, machine types)
├── build_and_push.sh                 # Build Docker image and push to Artifact Registry
│
├── jobs/
│   └── submit_sweep_agent.py         # Workflow 1: launch W&B sweep trials on Vertex AI or CE
│
├── pipelines/
│   └── submit_training_pipeline.py   # Workflow 2/3/4: full training pipeline (KFP v2)
│
└── cloud_run/
    ├── main.py                       # Flask webhook: /trigger + /health endpoints
    ├── Dockerfile                    # Lightweight python:3.11-slim image
    └── requirements.txt              # Flask + W&B + GCP SDK dependencies
```

---

### Prerequisites

**Windows (use Git Bash for all commands):**

1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install-sdk#windows)
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (enable WSL2 backend)
3. Install Python 3.11+ (Conda recommended)
4. Authenticate:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project automend
```

**macOS:**

```bash
brew install --cask google-cloud-sdk
gcloud auth login
gcloud auth application-default login
gcloud config set project automend
```

**Python dependencies (both platforms):**

```bash
pip install google-cloud-aiplatform kfp wandb pyyaml python-dotenv loguru \
            google-cloud-storage google-cloud-secret-manager
```

---

### One-Time GCP Setup

These steps only need to be done once per project.

#### Step 1 — Enable APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  compute.googleapis.com
```

#### Step 2 — Create Artifact Registry Repository

```bash
gcloud artifacts repositories create automend-images \
  --repository-format=docker \
  --location=us-central1
```

#### Step 3 — Create GCS Bucket

```bash
gsutil mb -l us-central1 gs://automend-model2
```

#### Step 4 — Create Service Account and Grant Roles

```bash
gcloud iam service-accounts create automend-trainer \
  --display-name="AutoMend Trainer"

PROJECT_ID=automend
SA="automend-trainer@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/run.invoker"
```

#### Step 5 — Store W&B API Key in Secret Manager

```bash
echo -n "YOUR_WANDB_API_KEY" | \
  gcloud secrets create WANDB_API_KEY \
    --data-file=- \
    --replication-policy=automatic
```

> The secret name must be exactly `WANDB_API_KEY`. The training container fetches it at runtime via the Python Secret Manager SDK — no gcloud CLI required inside the container.

#### Step 6 — Upload Required Data Files to GCS

The pipeline reads training data and benchmark files from the GCS FUSE mount. Upload once before running any pipeline:

```bash
gsutil cp model_2_training/data/processed/track_B_combined.jsonl \
  gs://automend-model2/data/track_B_combined.jsonl

gsutil cp model_2_training/data/benchmarks/gold_benchmark.jsonl \
  gs://automend-model2/data/benchmarks/gold_benchmark.jsonl
```

#### Step 7 — Request GPU Quota

Go to **IAM & Admin → Quotas** → search for `NVIDIA_L4` in `us-central1` → request an increase to at least 1 under **Vertex AI API**.

---

### Build & Push Docker Image

Run from the **repo root** (`AutoMend/`). Only needed once, or after changing code inside `model_2_training/` or `gcp/scripts/`:

```bash
# macOS / Linux / Git Bash on Windows
bash gcp/build_and_push.sh

# Optional: specify a custom tag
bash gcp/build_and_push.sh v1.2
```

> The Docker image includes: PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9, gcsfuse (GCS FUSE mount), and all training dependencies. Scripts that run locally (like `submit_training_pipeline.py`) do **not** require a rebuild.

---

### Workflow 1 — Hyperparameter Sweep

Uses W&B Bayesian optimization to find the best hyperparameters. Runs N parallel trials on Vertex AI or Compute Engine GPU VMs.

```bash
# Auto-create sweep + launch 10 trials on Vertex AI
python gcp/jobs/submit_sweep_agent.py --trials 10

# Resume an existing sweep
python gcp/jobs/submit_sweep_agent.py \
  --sweep-id mlops-team-northeastern-university/automend-model2/<sweep_id> \
  --trials 5

# Dry-run (verify config, no jobs submitted)
python gcp/jobs/submit_sweep_agent.py --trials 3 --dry-run
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `vertex` | `vertex` (managed) or `compute-engine` (self-deleting GPU VMs) |
| `--trials` | 10 | Number of parallel sweep trials |
| `--sweep-id` | (auto-created) | Resume an existing W&B sweep |
| `--image-tag` | `latest` | Docker image tag |
| `--cloud-run-url` | None | Cloud Run URL for automated post-sweep pipeline trigger |
| `--delay-hours` | 6.0 | Hours until Cloud Scheduler fires |
| `--dry-run` | False | Print config only, do not submit |

**Monitor:**
- Vertex AI Jobs: `https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=automend`
- W&B Sweep: `https://wandb.ai/mlops-team-northeastern-university/automend-model2`

---

### Workflow 2 — Full Training Pipeline (after sweep)

Run after the sweep completes to train the final model with the best hyperparameters.

#### Step 1 — Fetch best config from W&B

```bash
python model_2_training/scripts/fetch_best_config.py \
  --sweep-id mlops-team-northeastern-university/automend-model2/<sweep_id>
```

This script finds the best-scoring trial (by `benchmark/tax_valid_rate`), merges its hyperparameters into the base config, saves locally, and **uploads to GCS automatically** — no image rebuild needed when config changes.

#### Step 2 — Submit full training pipeline

```bash
# Windows (Git Bash) — MSYS_NO_PATHCONV=1 prevents path mangling
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --train-config /gcs/automend-model2/configs/train/best_sweep_config.yaml

# macOS / Linux
python gcp/pipelines/submit_training_pipeline.py \
  --train-config /gcs/automend-model2/configs/train/best_sweep_config.yaml
```

**Pipeline stages (executed in order on Vertex AI):**

| Stage | Machine | Description |
|-------|---------|-------------|
| **1 · Split** | CPU | train/val/test split (90/5/5) — writes splits to GCS |
| **2 · Train** | L4 GPU | QLoRA fine-tuning — checkpoint saved to `gs://automend-model2/outputs/runs/<run_id>/` |
| **3 · Eval** | L4 GPU | Validation set evaluation |
| **4 · Test** | L4 GPU | Test set evaluation |
| **5 · Benchmark** | L4 GPU | Gold benchmark scoring (30 samples) |
| **6 · Robustness** | L4 GPU | Robustness + slice evaluation |

Each run gets a unique timestamped `run_id` (`YYYYMMDD-HHMMSS`) — outputs never overwrite each other.

**Monitor:** `https://console.cloud.google.com/vertex-ai/pipelines?project=automend`

---

### Workflow 3 — Retrain on New Data

Use when new data is available and the sweep does not need to be re-run.

```bash
# Windows
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py --retrain-only

# macOS / Linux
python gcp/pipelines/submit_training_pipeline.py --retrain-only
```

**Pipeline stages:** Split → Train → Eval → Test (Benchmark and Robustness skipped).

---

### Workflow 4 — Resume from Checkpoint

Use when the pipeline failed on step 5 or 6 but training already completed. Skips split/train/eval/test and runs only Benchmark + Robustness against the existing checkpoint on GCS.

```bash
# Find your run_id
gsutil ls gs://automend-model2/outputs/runs/

# Windows
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --resume-from-checkpoint /gcs/automend-model2/outputs/runs/<run_id>/checkpoints/best_model

# macOS / Linux
python gcp/pipelines/submit_training_pipeline.py \
  --resume-from-checkpoint /gcs/automend-model2/outputs/runs/<run_id>/checkpoints/best_model
```

---

### Pipeline Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--train-config` | `qlora_sft.yaml` | Config filename (baked in image) or full GCS FUSE path |
| `--retrain-only` | `False` | Workflow 3: split→train→eval→test only |
| `--resume-from-checkpoint` | `""` | Workflow 4: skip to benchmark+robustness using existing checkpoint |
| `--image-tag` | `latest` | Docker image tag |
| `--dry-run` | `False` | Compile pipeline YAML only, do not submit |

---

### Cloud Run Webhook Setup

Required only for Workflow 1 fully automated mode (sweep → auto-trigger training). Deploy once:

```bash
# Build and push webhook image
gcloud builds submit gcp/cloud_run/ \
  --tag us-central1-docker.pkg.dev/automend/automend-images/automend-webhook:latest

# Deploy
gcloud run deploy automend-webhook \
  --image us-central1-docker.pkg.dev/automend/automend-images/automend-webhook:latest \
  --platform managed \
  --region us-central1 \
  --no-allow-unauthenticated \
  --service-account automend-trainer@automend.iam.gserviceaccount.com \
  --memory 512Mi \
  --timeout 600

# Get URL
gcloud run services describe automend-webhook \
  --region us-central1 \
  --format "value(status.url)"
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/trigger` | POST | Fetch best config + submit Vertex AI pipeline |
| `/health` | GET | Returns `{"status": "ok"}` |

---

### GCP Configuration Reference

**`gcp/config.py`** — central config for all GCP resources:

```python
PROJECT_ID         = "automend"
REGION             = "us-central1"
GCS_BUCKET         = "automend-model2"
TRAIN_MACHINE_TYPE = "g2-standard-8"   # 1× L4 24 GB VRAM, 8 vCPU, 32 GB RAM
TRAIN_ACCELERATOR  = "NVIDIA_L4"
TRAIN_ACCEL_COUNT  = 1
WANDB_PROJECT      = "automend-model2"
WANDB_ENTITY       = "mlops-team-northeastern-university"
```

**Data configs used by Vertex AI pipeline:**

| Config | Max Samples | Purpose |
|--------|-------------|---------|
| `configs/data/track_b_chatml.yaml` | null (all) | Full training pipeline (W2/W3) |
| `configs/data/track_b_chatml_sweep.yaml` | 2000 train / 200 val | Fast sweep trials (W1) |

Both use a 90/5/5 train/val/test split.

---

### GCP Monitoring

| Resource | URL |
|----------|-----|
| Vertex AI Custom Jobs | `https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=automend` |
| Vertex AI Pipelines | `https://console.cloud.google.com/vertex-ai/pipelines?project=automend` |
| Cloud Run Logs | `https://console.cloud.google.com/run/detail/us-central1/automend-webhook/logs?project=automend` |
| Cloud Scheduler | `https://console.cloud.google.com/cloudscheduler?project=automend` |
| W&B Dashboard | `https://wandb.ai/mlops-team-northeastern-university/automend-model2` |
| GCS Bucket | `https://console.cloud.google.com/storage/browser/automend-model2` |

---

### Cost Estimates

| Resource | Unit Cost | Typical Usage | Estimate |
|----------|-----------|--------------|---------|
| L4 GPU (`g2-standard-8`) Vertex AI | ~$0.70/hr | 10 trials × 30 min | ~$3.50/sweep |
| L4 GPU (full training pipeline) | ~$0.70/hr | ~3–5 hours | ~$2.10–$3.50 |
| Cloud Run | ~$0 | Very low traffic | <$0.01 |
| GCS storage | ~$0.02/GB/mo | ~10–50 GB | <$1/mo |
| Cloud Scheduler | Free tier | 1 job/sweep | $0 |

---

### GCP Troubleshooting

**Git Bash converts `/gcs/...` paths to `C:/Program Files/Git/gcs/...`**

Prefix pipeline submission commands with `MSYS_NO_PATHCONV=1`:
```bash
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --train-config /gcs/automend-model2/configs/train/best_sweep_config.yaml
```

---

**`WANDB_API_KEY not set` inside Vertex AI job**

The training container fetches the key at runtime via the Python Secret Manager SDK. Verify:
1. The secret exists: `gcloud secrets list --project=automend` — you should see `WANDB_API_KEY`
2. The trainer SA has `roles/secretmanager.secretAccessor`
3. `PROJECT_ID` env var is passed to the container (it is — set in `submit_training_pipeline.py`)

Test the secret:
```bash
gcloud secrets versions access latest --secret=WANDB_API_KEY --project=automend
```

---

**`FileNotFoundError` for training data or benchmark inside container**

The container reads data via the GCS FUSE mount. Check the files exist:
```bash
gsutil ls gs://automend-model2/data/track_B_combined.jsonl
gsutil ls gs://automend-model2/data/benchmarks/gold_benchmark.jsonl
```

If missing, upload them (see Step 6 in One-Time GCP Setup above).

---

**Benchmark or robustness step failed but training completed**

Use Workflow 4 (`--resume-from-checkpoint`) to rerun only steps 5+6:
```bash
gsutil ls gs://automend-model2/outputs/runs/   # find run_id
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --resume-from-checkpoint /gcs/automend-model2/outputs/runs/<run_id>/checkpoints/best_model
```

---

**Vertex AI pipeline stuck in PENDING**

GPU quota issue. Request an increase at: **IAM & Admin → Quotas** → `NVIDIA_L4_GPUS` under **Vertex AI API** in `us-central1`.

---

**`Permission denied` on Vertex AI submission**

```bash
gcloud projects add-iam-policy-binding automend \
  --member="serviceAccount:automend-trainer@automend.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

---

**Docker build fails on Windows**

Use Git Bash or WSL2. If unavailable, build manually:
```bash
docker build --file gcp/Dockerfile \
  --tag us-central1-docker.pkg.dev/automend/automend-images/automend-train:latest \
  --platform linux/amd64 .
docker push us-central1-docker.pkg.dev/automend/automend-images/automend-train:latest
```

---

**`Billing account not associated`**

```bash
gcloud billing accounts list
gcloud billing projects link automend --billing-account=XXXXXX-XXXXXX-XXXXXX
```

---

### GCP Quick Reference

```bash
# ── One-time setup ──────────────────────────────────────────────────
gsutil cp model_2_training/data/processed/track_B_combined.jsonl \
  gs://automend-model2/data/track_B_combined.jsonl
gsutil cp model_2_training/data/benchmarks/gold_benchmark.jsonl \
  gs://automend-model2/data/benchmarks/gold_benchmark.jsonl
bash gcp/build_and_push.sh

# ── Workflow 1: sweep ────────────────────────────────────────────────
python gcp/jobs/submit_sweep_agent.py --trials 10

# ── Workflow 2: full training after sweep (Windows) ──────────────────
python model_2_training/scripts/fetch_best_config.py \
  --sweep-id mlops-team-northeastern-university/automend-model2/<sweep_id>
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --train-config /gcs/automend-model2/configs/train/best_sweep_config.yaml

# ── Workflow 3: retrain on new data (Windows) ────────────────────────
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py --retrain-only

# ── Workflow 4: resume from checkpoint (Windows) ─────────────────────
gsutil ls gs://automend-model2/outputs/runs/
MSYS_NO_PATHCONV=1 python gcp/pipelines/submit_training_pipeline.py \
  --resume-from-checkpoint /gcs/automend-model2/outputs/runs/<run_id>/checkpoints/best_model
```

---

## 12. Configuration Reference

### `configs/data/track_b_chatml.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `artifact_path` | `data/processed/track_B_combined.jsonl` | Source JSONL path (relative to AutoMend root) |
| `splits_dir` | `data/splits` | Output directory for splits |
| `train_ratio` | `0.80` | Training fraction |
| `val_ratio` | `0.10` | Validation fraction |
| `test_ratio` | `0.10` | Test fraction |
| `shuffle_seed` | `42` | Reproducibility seed for splitting |
| `max_seq_length` | `2048` | Maximum token length — sequences are truncated to this |
| `malformed_row_strategy` | `"skip"` | `"skip"` (log and drop) or `"raise"` (stop immediately) |
| `stratify_by` | `[source_dataset]` | Metadata field for stratified splitting |
| `max_train_samples` | `null` | Cap training samples — set to `200` for smoke test |
| `max_val_samples` | `null` | Cap validation samples — set to `50` for smoke test |
| `max_test_samples` | `null` | Cap test samples — set to `50` for smoke test |

### `configs/model/qwen_baseline.yaml`

| Key | Value | Description |
|-----|-------|-------------|
| `model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model identifier |
| `tokenizer_name` | `Qwen/Qwen2.5-1.5B-Instruct` | Tokenizer (usually same as model) |
| `quantization` | `"4bit"` | CUDA: `"4bit"` / `"8bit"` / `null`. MPS/CPU: ignored |
| `device_map` | `"auto"` | CUDA: used as-is. MPS/CPU: overridden by `device.py` |
| `lora_target_modules` | 7 projection layers | CUDA/CPU only — MLX targets layers by index count |

### `configs/train/qlora_sft.yaml`

| Key | Default | Sweep Winner | Description |
|-----|---------|-------------|-------------|
| `num_train_epochs` | `3` | `2` | Full passes over the training data |
| `per_device_train_batch_size` | `2` | `4` | Samples per gradient step per device |
| `gradient_accumulation_steps` | `8` | `16` | Effective batch = batch × accum (winner: 64) |
| `learning_rate` | `2e-4` | `1.19e-4` | Initial learning rate |
| `warmup_ratio` | `0.05` | `0.05` | Fraction of steps used for LR warmup |
| `lr_scheduler_type` | `"cosine"` | `"constant_with_warmup"` | LR decay pattern after warmup |
| `weight_decay` | `0.01` | `0.1` | L2 regularization coefficient |
| `lora_r` | `16` | `16` | LoRA rank |
| `lora_alpha` | `32` | `32` | LoRA scaling factor |
| `lora_dropout` | `0.05` | `0.05` | Dropout on LoRA layers |
| `bf16` | `true` | — | Overridden by `device.py` (ignored on MPS/CPU) |
| `eval_steps` | `100` | — | Evaluate every N steps (CUDA/CPU) |
| `save_steps` | `100` | — | Save checkpoint every N steps (CUDA/CPU) |
| `report_to` | `"wandb"` | — | Set to `"none"` to disable W&B logging |

### `configs/eval/json_eval.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `max_new_tokens` | `512` | Maximum tokens to generate per sample |
| `temperature` | `0.1` | Sampling temperature (ignored when `do_sample: false`) |
| `do_sample` | `false` | Greedy decoding — deterministic, reproducible |
| `top_p` | `0.9` | Nucleus sampling threshold |
| `num_samples_to_save` | `50` | Maximum error/output samples to write to reports |

---

## 13. Experiment Tracking

W&B is integrated for all training and evaluation scripts. Each run is auto-named:

```
{model}_{quant}_lora-r{r}_{n_samples}samples_{datetime}

# Examples:
# CUDA:  Qwen2.5-1.5B-Instruct_4bit_lora-r16_200samples_20260319-2008
# MPS:   Qwen2.5-1.5B-Instruct_mlx_lora-r16_200samples_20260319-2008
# CPU:   Qwen2.5-1.5B-Instruct_fp32_lora-r16_200samples_20260319-2008
```

### W&B Coverage by Script

| Script | Logged Metrics | W&B Split Prefix |
|--------|---------------|-----------------|
| `run_train.py` | Training loss, LR curve, eval loss | (native HF Trainer logging) |
| `run_eval.py` | All 54 Phase 1–2C metrics | `val/` |
| `run_test.py` | All 54 Phase 1–2C metrics | `test/` |
| `run_benchmark.py` | Benchmark metrics + taxonomy labels | `benchmark/` |
| `run_sweep.py` | Per-trial metrics fed to Bayesian optimizer | `benchmark/` |
| `run_robustness.py` | Not yet integrated | Planned |

### Viewing Results in the W&B Dashboard

Evaluation and benchmark scripts log **one snapshot**, not a continuous series. Default W&B line charts will show a single dot at step 0 and appear empty. To view results:

1. Open the specific evaluation run in W&B
2. Navigate to the **Summary** tab
3. All 54+ metrics are listed as a flat table

### Disabling W&B

```yaml
# configs/train/qlora_sft.yaml
report_to: "none"
```

### MPS / MLX W&B Integration

`mlx_train.py` reads `WANDB_PROJECT` and `WANDB_RUN_NAME` from the environment and injects `report_to: wandb` into the `mlx_lm.lora` YAML config automatically. No extra steps needed — set credentials in `.env` and W&B logging works on MPS.

---

## 14. Output Artifacts

```
outputs/
│
├── checkpoints/
│   ├── best_model/                          # Final trained adapter
│   │   ├── adapter_config.json              # LoRA hyperparameters
│   │   ├── adapter_model.safetensors        # PEFT adapter weights (CUDA / CPU)
│   │   │     — OR —
│   │   ├── adapters.npz                     # MLX adapter weights (MPS)
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── chat_template.jinja
│   │
│   ├── checkpoint-N/                        # Intermediate checkpoints (CUDA/CPU)
│   ├── mlx_data/                            # MLX-format training data (MPS only)
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   ├── mlx_lora_config.yaml                 # mlx-lm training config (MPS only)
│   └── run_config_snapshot.json             # All configs + detected device at run time
│
└── reports/
    ├── val/                                 # Validation set reports
    │   ├── metrics.json                     # All 54 Phase 1–2C metrics (flat dict)
    │   ├── metrics_summary.md               # Human-readable per-phase tables
    │   ├── sample_outputs.json              # Generated vs. reference pairs
    │   ├── error_samples.json               # JSON parse failures (Phase 1)
    │   ├── param_errors.json                # Parameter validation failures (Phase 2C)
    │   └── validation_predictions.jsonl     # Full generation log
    │
    ├── test/                                # Test set reports (same structure as val/)
    │
    ├── benchmark/
    │   └── <checkpoint_name>/               # Phase 2.5 benchmark results
    │       ├── metrics.json
    │       ├── metrics_summary.md
    │       └── benchmark_predictions.jsonl
    │
    ├── sweeps/
    │   └── <run_name>/                      # Phase 2.75 — one directory per sweep trial
    │       └── metrics.json
    │
    └── robustness/
        └── <checkpoint_name>/               # Phase 3 robustness results
            ├── clean_predictions.jsonl
            ├── clean_metrics.json
            ├── typo_predictions.jsonl
            ├── typo_metrics.json
            ├── noise_predictions.jsonl
            ├── noise_metrics.json
            ├── truncation_predictions.jsonl
            ├── truncation_metrics.json
            ├── case_lower_predictions.jsonl
            ├── case_lower_metrics.json
            ├── paraphrase_predictions.jsonl
            ├── paraphrase_metrics.json
            ├── robustness_summary.json
            ├── slice_report.json
            ├── failure_log.json
            └── robustness_report.md
```

---

## 15. Inference

### CUDA / CPU — PEFT Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
model     = PeftModel.from_pretrained(base_model, "outputs/checkpoints/best_model")
tokenizer = AutoTokenizer.from_pretrained("outputs/checkpoints/best_model")
```

### MPS — MLX Adapter

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path="outputs/checkpoints/best_model"
)

response = generate(model, tokenizer, prompt="...", max_tokens=512)
```

### Switching to a Different Base Model

Only the model config needs to change. Both PEFT and mlx-lm support Llama 3, Mistral, and other architectures:

```yaml
# configs/model/llama_baseline.yaml
model_name:  "meta-llama/Llama-3.1-8B-Instruct"
quantization: "4bit"
lora_target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"
```

No code changes are required. Pass `--model-config configs/model/llama_baseline.yaml` to `run_train.py`.

---

## 16. Sample Predictions

### Exact Match — Multi-Step Tool Call (index 21)

| | Content |
|--|---------|
| **Generated** | `{"workflow": {"steps": [{"tool": "get_stock_price", "params": {"symbol": "AAPL"}}, {"tool": "get_stock_price", "params": {"symbol": "MSFT"}}]}}` |
| **Reference** | *(identical)* |
| **Taxonomy label** | `VALID` — exact match |

### Correct Refusal (index 1)

| | Content |
|--|---------|
| **Generated** | `{"workflow": {"steps": []}, "message": "I'm sorry, but I can't assist with that. My current capabilities allow me to translate words and phrases between languages."}` |
| **Reference** | `{"workflow": {"steps": []}, "message": "I'm sorry, but I'm unable to perform external tasks like ordering a pizza..."}` |
| **Taxonomy label** | `VALID` — correct shape, different wording (acceptable) |

### Over-Calling — Primary Failure Mode with 200 Training Samples (index 7)

| | Steps |
|--|-------|
| **Generated** | 2 calls: `calculate_tip(50, 15%)` and `calculate_tip(75, 20%)` |
| **Reference** | 1 call: `calculate_tip(50, 15%)` |
| **Taxonomy label** | `VALID` (JSON is correct) — but `steps_count_match_rate` fails |

Over-calling resolves significantly with full dataset training (200 → 5,118 samples provide more single-step examples).

---

## 17. Reproducing Results

### Smoke Test — CUDA (200 samples, 1 epoch, RTX 5070)

**Step 1 — Verify the environment:**

```bash
conda activate mlops_project_model
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
# Expected: 2.x.x+cu128  |  NVIDIA GeForce RTX 5070 Laptop GPU
```

**Step 2 — Configure smoke test caps:**

`configs/data/track_b_chatml.yaml`:
```yaml
max_train_samples: 200
max_val_samples:   50
max_test_samples:  50
```

`configs/train/qlora_sft.yaml`:
```yaml
num_train_epochs: 1
per_device_train_batch_size: 1
```

**Step 3 — Verify the source artifact:**

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

| Metric | Value |
|--------|-------|
| Step 10 training loss | ~0.578 |
| Step 20 training loss | ~0.428 |
| JSON parse rate | ~96% |
| Non-empty output rate | 100% |
| Training time | ~5 minutes |

---

### Smoke Test — Apple Silicon (MPS)

**Step 1 — Verify the environment:**

```bash
conda activate mlops_project_model

python -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0)

python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True
```

**Step 2 — Apply the same smoke test caps as above.**

**Step 3 — Run the same commands:**

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

**What happens differently on MPS:**

- `device.py` returns `"mps"` → `train_loop.py` routes to `_run_mlx_training()`
- Data is converted to MLX chat format in `outputs/checkpoints/mlx_data/`
- `mlx_lora_config.yaml` is written; `python -m mlx_lm lora` runs on the Metal GPU
- Adapter saved in MLX `.npz` format to `best_model/`
- Evaluation inference uses `mlx_lm.load()` + `mlx_lm.generate()` instead of HF

> Memory requirement: 1.5B model in fp32 ≈ 6 GB unified memory — works on M1 Pro 16 GB and above.

---

### Full Production Run (Sweep Winner Settings)

**Step 1 — Remove sample caps** (`configs/data/track_b_chatml.yaml`):

```yaml
max_train_samples: null
max_val_samples:   null
max_test_samples:  null
```

**Step 2 — Apply winning hyperparameters** (`configs/train/qlora_sft.yaml`):

```yaml
num_train_epochs:              2
per_device_train_batch_size:   4
gradient_accumulation_steps:   16
learning_rate:                 1.19e-4
lr_scheduler_type:             constant_with_warmup
weight_decay:                  0.1
lora_r:                        16
lora_alpha:                    32
lora_dropout:                  0.05
```

**Step 3 — Run the full pipeline:**

```bash
# Phase 1 — Train
python scripts/run_split.py --config configs/data/track_b_chatml.yaml
python scripts/run_train.py \
  --data-config  configs/data/track_b_chatml.yaml \
  --model-config configs/model/qwen_baseline.yaml \
  --train-config configs/train/qlora_sft.yaml

# Phase 2 — Evaluate (all phases run automatically)
python scripts/run_eval.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model

# Phase 2 — Final test (run once only)
python scripts/run_test.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model

# Phase 2.5 — Benchmark (build once; score every checkpoint)
python scripts/build_benchmark.py
python scripts/run_benchmark.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model

# Phase 3 — Robustness
python scripts/run_robustness.py \
  --config     configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

---

## 18. Troubleshooting

### `CUDA error: no kernel image is available for execution on the device`

**Cause:** RTX 50xx (Blackwell) uses sm_120 — unsupported by stable PyTorch cu126.
**Fix:** Install PyTorch nightly cu128 before all other packages. See [Installation — Blackwell GPUs](#blackwell-gpus-rtx-50xx-series).

---

### `TypeError: TrainingArguments.__init__() got unexpected keyword argument 'evaluation_strategy'`

**Cause:** Transformers 5.x renamed this argument to `eval_strategy`.
**Fix:** `trainer_factory.py` maps the YAML key `evaluation_strategy` to `eval_strategy=` internally. Ensure you are on the current version of `trainer_factory.py`.

---

### `TypeError: Trainer.__init__() got unexpected keyword argument 'tokenizer'`

**Cause:** Transformers 5.x renamed this argument to `processing_class`.
**Fix:** `trainer_factory.py` passes `processing_class=tokenizer`. Ensure you are on the current version.

---

### `HFValidationError: Repo id must be in the form 'repo_name'...` on MPS

**Cause:** `mlx_lm.load()` interprets non-absolute paths as HuggingFace Hub repo IDs.
**Fix:** All scripts resolve `--checkpoint` to an absolute path automatically. Ensure you are on the current version of `run_eval.py` and `run_test.py`.

---

### `AttributeError: 'dict' object has no attribute 'size'` during MLX inference

**Cause:** Older versions of `generator.py` called `.size` on nested parameter dicts returned by `model.parameters()`.
**Fix:** Replaced with recursive `_count_mlx_params()`. Already applied in current code — this note covers older checkouts.

---

### W&B prompts for login / run does not appear on wandb.ai

**Cause:** `AutoMend/.env` is missing or does not contain `WANDB_API_KEY`.
**Fix:** Copy `.env.example` to `.env` and set `WANDB_API_KEY`. All scripts call `load_dotenv()` at startup and pick up credentials automatically.

---

### W&B line charts appear empty (single dot at step 0)

**Cause:** Evaluation scripts log one snapshot, not a continuous time series. W&B cannot draw a line with one data point.
**Fix:** Open the run in W&B and check the **Summary** tab — all 54+ metrics are listed as a flat table, which is the correct view for a point-in-time evaluation.

---

### `ModuleNotFoundError: No module named 'model_2_training'`

**Cause:** Commands run from inside `model_2_training/` instead of from `AutoMend/`.
**Fix:** Run all commands from the `AutoMend/` root directory. The import path requires `AutoMend/` to be on `sys.path`.

---

### `ModuleNotFoundError: No module named 'pydantic'`

**Fix:** `pip install "pydantic>=2.0"`

---

### `dataset` slice shows all `"unknown"` in Phase 3 slice report

**Cause:** The gold benchmark JSONL was built without copying `metadata.source_dataset` from the source samples. All 30 samples fall back to `"unknown"`.
**Fix:** When rebuilding the benchmark, copy `source_dataset` into each sample's metadata under the key `source`. All benchmark comparisons must be invalidated when the benchmark is rebuilt — increment the version in the manifest.

---

### `0.0%` param validity in refusal or long-input slice rows

**Cause:** `steps: []` means no parameters exist to validate. The denominator is zero.
**Interpretation:** Not a failure — read as **not applicable** for these rows.

---

### `param_type_correctness_rate` is always `0.0 (n=0)`

**Cause:** Current training data uses flat list schemas (`"parameters": ["param1", "param2"]`) with no type information.
**Fix:** No action needed now. This metric activates automatically when JSON Schema-format tool definitions with `properties` and `type` fields are injected via RAG at inference time.

---

### `correct_shape_rate` is below `schema_valid_rate`

**Cause:** The model is choosing the wrong response shape — calling a tool when it should refuse, or refusing when it should call a tool. This is a training data distribution imbalance.
**Fix:** Inspect `error_samples.json` to identify which archetype is confused. Add more examples of that archetype to the training data.

---

## Appendix A — Data Contract

Every sample in `track_B_combined.jsonl` and all split files must conform to this structure:

```python
{
  "messages": [                              # Required — list of message dicts
    {
      "role": "system",                      # Required
      "content": "..."                       # Tool context injected here
    },
    {
      "role": "user",                        # Required
      "content": "..."                       # Incident description
    },
    {
      "role": "assistant",                   # Required — must be last message
      "content": "..."                       # JSON workflow or refusal
    }
  ],
  "metadata": {                              # Optional — source tracking
    "source_dataset": "ds5",                 # Used for stratification and slice analysis
    "task_type": "tool_call"                 # "tool_call" or "refusal"
  }
}
```

Samples that fail this contract are handled by `malformed_row_strategy` in the data config:

- `"skip"` (default): Logged at WARNING level and silently dropped. Training continues.
- `"raise"`: Raises `ContractViolation` immediately. Use during data pipeline development.

---

## Appendix B — Switching Base Models

To switch from Qwen2.5-1.5B to a different model (e.g., Llama 3.1-8B), create a new model config:

```yaml
# configs/model/llama_3_8b.yaml
model_name:      "meta-llama/Llama-3.1-8B-Instruct"
tokenizer_name:  "meta-llama/Llama-3.1-8B-Instruct"
quantization:    "4bit"
device_map:      "auto"
lora_target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"
```

Then pass it to the training script:

```bash
python scripts/run_train.py \
  --data-config  configs/data/track_b_chatml.yaml \
  --model-config configs/model/llama_3_8b.yaml \
  --train-config configs/train/qlora_sft.yaml
```

Both the CUDA/CPU (bitsandbytes + PEFT) and MPS (mlx-lm) backends support Llama 3, Mistral, and other standard architectures natively. No code changes are required.

> For the 8B variant, increase `max_seq_length` to match available VRAM. At 4-bit quantization, 8B requires approximately 8 GB VRAM.

---

*AutoMend — Track B Training Pipeline*
*Model: Qwen2.5-1.5B-Instruct · Method: QLoRA (PEFT) · Backends: HuggingFace Transformers (CUDA/CPU) · MLX + mlx-lm (Apple Silicon)*
*Phases: Fine-Tuning → Structured Evaluation → Gold Benchmark → Hyperparameter Sweep → Robustness Testing*
*Deployment: Vertex AI (T4 GPU) · Cloud Run · Cloud Scheduler · GCS · W&B · Artifact Registry*
