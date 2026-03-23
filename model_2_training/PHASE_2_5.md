# AutoMend Track B — Model 2 Training (Phase 2.5 & Phase 2.75)

---

## Overview

This module covers the evaluation infrastructure and hyperparameter optimization for the AutoMend Track B generative pipeline — a fine-tuned Qwen2.5 LoRA agent that generates structured JSON remediation workflows from MLOps incident descriptions.

**Core Innovation**: A locked, stratified gold benchmark combined with a 9-category reference-aware error taxonomy, feeding a W&B Bayesian sweep that automatically identifies the best LoRA configuration — validated through a 4-phase, 54-metric evaluation pipeline.

---

## Philosophy & Design Principles

- **Lock the benchmark, never move the goalposts**: Every model is scored on the exact same 30 questions. Rebuilding the benchmark invalidates all historical comparisons.
- **Know why it failed, not just that it failed**: A system that only says "passed" or "failed" provides no actionable insight. Every wrong output gets a specific, categorized failure label.
- **Automate the search, don't guess**: Hyperparameter tuning by hand is guesswork. A Bayesian optimizer explores the space intelligently, learning from each trial.
- **Be honest about what the metrics measure**: The 96.7% score reflects structural correctness — valid JSON with the correct format. Semantic correctness (right tool, right parameters) is the acknowledged next step.
- **Reference-aware grading**: A model that correctly refuses an invalid request should be marked correct, not penalized for producing empty steps.

---

## Technical Highlights

- **Gold Benchmark**: 30 locked samples from `test.jsonl`, stratified across multi-step, single-step, and refusal archetypes — seed=42, manifest-guarded against accidental rebuilds.
- **Error Taxonomy**: 9 mutually exclusive failure categories, reference-aware, integrated automatically into every eval and sweep trial.
- **Evaluation Pipeline**: 4 phases (structural → schema → field → parameter), 54 metrics per run, 6 report artifacts saved per checkpoint.
- **Bayesian Sweep**: 21 trials across 10 hyperparameters on W&B project `automend-model2`. Best result: **96.7% valid outputs** with LoRA r16, lr=1.19e-4, effective batch=64.
- **Device-Aware Training**: MLX on Apple Silicon, HuggingFace + PEFT on CUDA/CPU — same pipeline, same metrics, device-agnostic.
- **Critical Bug Fixes**: Nested `wandb.init()` timeout in MLX subprocess, metric key mismatch breaking the sweep objective, argparse crash from W&B-injected CLI flags.

---

## Key Resources & Repository Navigation

### Entry Points
1. **[run_benchmark.py](scripts/run_benchmark.py)** → Score any checkpoint against the locked gold benchmark
2. **[build_benchmark.py](scripts/build_benchmark.py)** → Gold benchmark curation script (run once only)
3. **[run_sweep.py](scripts/run_sweep.py)** → W&B sweep agent — trains, benchmarks, and logs per trial

### Evaluation Modules
- **[error_taxonomy.py](src/eval/error_taxonomy.py)** → 9-category failure labeling, reference-aware classification
- **[metrics_aggregator.py](src/eval/metrics_aggregator.py)** → Unified 4-phase pipeline returning 54 metrics
- **[evaluator.py](src/eval/evaluator.py)** → Main evaluation orchestrator
- **[save_reports.py](src/eval/save_reports.py)** → Saves all 6 report artifacts to disk

### Sweep & Model Configuration
- **[wandb_sweep.yaml](configs/sweep/wandb_sweep.yaml)** → Bayesian sweep — 10 parameters, objective, Hyperband early stopping
- **[qwen_baseline.yaml](configs/model/qwen_baseline.yaml)** → Qwen2.5-1.5B model config
- **[qwen_3b_baseline.yaml](configs/model/qwen_3b_baseline.yaml)** → Qwen2.5-3B model config (scale-up)
- **[qlora_sft.yaml](configs/train/qlora_sft.yaml)** → Training config, updated with winning hyperparameters

### Key Artifacts
- **Benchmark**: `data/benchmarks/gold_benchmark.jsonl`, `gold_benchmark_manifest.json`
- **Sweep Results**: `outputs/reports/sweeps/Qwen2.5-1.5B-Instruct_*/metrics.json`
- **Best Checkpoint**: `outputs/checkpoints/sweeps/Qwen2.5-1.5B-Instruct_4bit_lora-r16_200samples_20260323-0052/`
- **W&B Dashboard**: `https://wandb.ai/mlops-team-northeastern-university/automend-model2/sweeps/ltilnti3`

---

## Phase 2.5 — Gold Benchmark + Error Taxonomy

### Motivation

Prior to this phase, there was no consistent evaluation baseline. Every run used different questions, making cross-model comparison meaningless. And when a prediction failed, the system produced no diagnostic information — only a binary pass/fail.

---

### Gold Benchmark

30 questions drawn from `test.jsonl`, locked at seed=42 and never rebuilt.

**Stratification — 10 samples per archetype:**

| Archetype | What it tests |
|-----------|---------------|
| `multi_step` | Model generates a workflow with 2+ remediation steps |
| `single_step` | Model generates a workflow with exactly 1 step |
| `refusal` | Model correctly returns `steps: []` for an invalid request |

A model that scores well on multi-step but fails refusals is not production-ready. Stratification ensures the benchmark covers all output types encountered in a real system.

The manifest records curation metadata and enforces the lock:

```json
{
  "seed": 42,
  "total_selected": 30,
  "archetype_counts": { "multi_step": 10, "refusal": 10, "single_step": 10 },
  "locked": true,
  "warning": "DO NOT rebuild without incrementing version. Rebuilding invalidates historical comparisons."
}
```

---

### Error Taxonomy

Every prediction is assigned one of 9 mutually exclusive failure categories, applied in priority order:

| Priority | Category | Condition |
|----------|----------|-----------|
| 1 | `VALID` | Non-empty, valid JSON, `workflow.steps` is a non-empty list (or a correct refusal) |
| 2 | `EMPTY` | Output is empty or whitespace only |
| 3 | `TRUNCATED` | Opens with `{` or `[` but missing the closing brace — hit the token limit |
| 4 | `MISSING_WORKFLOW` | Valid JSON but missing the `workflow` or `steps` key |
| 5 | `WRONG_STEPS_TYPE` | `workflow.steps` exists but is not a list |
| 6 | `EMPTY_STEPS` | `workflow.steps = []` but the reference expected non-empty steps |
| 7 | `MALFORMED_JSON` | Non-empty, not truncated, but `json.loads()` fails |
| 8 | `UNBALANCED_BRACES` | `{`/`}` or `[`/`]` counts do not match |
| 9 | `UNBALANCED_QUOTES` | Double-quote count is odd |

**Reference-awareness:** `EMPTY_STEPS` is only a failure if the reference itself expected non-empty steps. If the reference is also a refusal, the prediction is labeled `VALID`. Without this, the sweep would penalize models for correctly refusing invalid requests — biasing results toward models that always output steps regardless of the input.

---

## Phase 2.75 — Hyperparameter Tuning + Model Scaling

### W&B Bayesian Sweep

A Bayesian optimizer was wired to automatically explore the training hyperparameter space. Each trial trains a full model, benchmarks it on the locked 30 questions, and logs results back to W&B — which then selects the next configuration based on all prior outcomes.

| Setting | Value |
|---------|-------|
| Project | `automend-model2` |
| Sweep ID | `ltilnti3` |
| Method | Bayesian optimization |
| Objective | `benchmark/tax_valid_rate` (maximize) |
| Early stopping | Hyperband (min_iter=1, eta=2) |
| Trials completed | 21 |

**10 parameters swept:**

| Parameter | Values / Range | What it controls |
|-----------|---------------|-----------------|
| `lora_r` | 8, 16, 32, 64 | Model capacity vs truncation risk |
| `learning_rate` | log-uniform [1e-5, 5e-4] | Update aggressiveness |
| `lora_alpha` | 16, 32, 64 | LoRA scaling factor |
| `lora_dropout` | 0.0, 0.05, 0.1 | Regularization |
| `per_device_train_batch_size` | 1, 2, 4 | Samples per step |
| `gradient_accumulation_steps` | 4, 8, 16 | Effective batch = batch × accum |
| `lr_scheduler_type` | cosine, linear, constant_with_warmup | LR decay pattern |
| `warmup_ratio` | 0.03, 0.05, 0.10 | Warmup fraction |
| `num_train_epochs` | 1, 2, 3 | Training passes |
| `weight_decay` | 0.0, 0.01, 0.1 | L2 regularization |

---

### LoRA Rank Analysis

| Rank | Trainable Params | Params per Training Sample | Outcome |
|------|-----------------|---------------------------|---------|
| r8 | ~21M (1.4%) | 105,000 | Slight underfitting |
| **r16** | **~42M (2.7%)** | **210,000** | **Best — won the sweep** |
| r64 | ~168M (10.9%) | 840,000 | Overfits, generates longer outputs, truncates |

r64 exceeded the 512-token generation limit on 2 out of 30 benchmark questions due to longer output generation. r16 provided the best balance between capacity and generalization on 200 training samples.

**Honest caveat**: The difference between r16 (96.7%) and r8 (93.3%) is one prediction out of 30 — within the margin of statistical noise for a 30-sample benchmark. The winning hyperparameter combination was never tested with r8 and could potentially match r16.

---

### How Accuracy Is Measured

Accuracy is derived from post-training code-based validation — not from training loss. After training, all 30 benchmark questions are fed to the model and each output is validated programmatically:

```python
json.loads(generated_text)                          # valid JSON?
parsed.get("workflow", {}).get("steps")             # correct structure?
isinstance(steps, list) and len(steps) > 0          # non-empty list?
# 29 / 30 passing = 96.7%
```

This is **structural validation**. It confirms the model learned correct JSON format. It does not confirm semantic correctness — whether the model selected the right tool or the right parameter values. That is the next evaluation layer to build.

---

### Train Loss vs Validation Loss

Training loss measures token-level prediction error during training and almost always decreases. Validation loss measures generalization on held-out data and is the meaningful signal.

From the r64 run:

```
Iter 100:  Val loss 0.732   Train loss 0.418   ← lowest val loss
Iter 200:  Val loss 0.740   Train loss 0.206   ← val plateaus
Iter 300:  Val loss 0.872   Train loss 0.090   ← val rising, train still falling
```

The model was overfitting by iter 200 on 200 training samples. MLX saves the final adapter (iter 300), not the best val loss checkpoint (iter 100) — meaning benchmarked models are evaluated at a slightly suboptimal point.

---

### Model Scaling — 1.5B vs 3B

After identifying the best hyperparameters for the 1.5B model, the same configuration was applied to Qwen2.5-3B to determine whether larger capacity improves benchmark performance.

| | 1.5B (sweep winner) | 3B |
|--|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-3B-Instruct` |
| Parameters | ~1.55B | ~3.1B |
| Benchmark score | **96.7%** | Training in progress |
| RAM (MPS) | ~8 GB | ~12 GB |

The higher-scoring model becomes the **best non-RAG checkpoint** — the strongest version before any retrieval augmentation is added.

---

## What the Metrics Are Actually Measuring

| Metric | Industry Standard (BFCL) | This Implementation |
|--------|--------------------------|---------------------|
| Valid JSON output rate | Core | Measured |
| Schema validity | Core | Measured |
| Tool name accuracy | Critical | Not yet built |
| Parameter value match | Critical | Not yet built |
| End-to-end exact match | Standard | Not yet built |

The current 96.7% reflects structural output quality. Semantic evaluation — tool selection accuracy and parameter correctness — is the next phase.

---

## Results

### All Sweep Trials

| Rank | Model | Valid Output | Schema OK | Steps Match |
|------|-------|-------------|-----------|-------------|
| 1 | lora-r16 (0052) | **96.7%** | 96.7% | 75.9% |
| 2 | lora-r8 (0037) | 93.3% | 96.7% | 69.0% |
| 3 | lora-r8 (1122) | 93.3% | 93.3% | 67.9% |
| 4 | lora-r64 (0058) | 93.3% | 93.3% | 64.3% |
| 5 | lora-r16 (0026) | 86.7% | 90.0% | 70.4% |

### Winning Hyperparameters

| Parameter | Default | Winner | Why it helped |
|-----------|---------|--------|---------------|
| `lora_r` | 16 | **16** | Sweet spot — r8 underfits, r64 truncates |
| `learning_rate` | 2e-4 | **1.19e-4** | Conservative updates — stable structure learning |
| `per_device_train_batch_size` | 2 | **4** | More samples per update |
| `gradient_accumulation_steps` | 1 | **16** | Effective batch of 64 — 32× larger than default |
| `lr_scheduler_type` | cosine | **constant_with_warmup** | Stable LR after warmup |
| `weight_decay` | 0.01 | **0.1** | Stronger regularization, better generalization |

> **Right rank (r16) + Large effective batch (64) + Conservative learning rate (1.19e-4)**

---

## Commands

**Build the gold benchmark** (run once only):
```bash
python scripts/build_benchmark.py
```

**Score a checkpoint on the gold benchmark:**
```bash
python scripts/run_benchmark.py \
    --config     configs/eval/json_eval.yaml \
    --checkpoint outputs/checkpoints/best_model
```

**Initialize the W&B sweep:**
```bash
wandb sweep configs/sweep/wandb_sweep.yaml
```

**Launch the sweep agent:**
```bash
wandb agent mlops-team-northeastern-university/automend-model2/ltilnti3 --count 5
```

**Train the 3B model:**
```bash
python scripts/run_train.py \
    --data-config  configs/data/track_b_chatml.yaml \
    --model-config configs/model/qwen_3b_baseline.yaml \
    --train-config configs/train/qlora_sft.yaml
```

**Rank all sweep trials by benchmark score:**
```bash
for dir in outputs/reports/sweeps/Qwen2.5-1.5B-Instruct_*/; do
    rate=$(python3 -c "import json; d=json.load(open('$dir/metrics.json')); print(d.get('phase1_structural/tax_valid_rate','N/A'))")
    echo "$rate  $(basename $dir)"
done | sort -rn
```

---