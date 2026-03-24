# Phase 2: Validation — Structured + Functional Correctness

> **Status:** Complete
> **Depends on:** Phase 1 (baseline pipeline, Qwen fine-tune, structural JSON metrics)
> **Location:** `model_2_training/`

---

## Overview

Phase 1 established that the model can produce syntactically valid JSON (96% parse rate).
Phase 2 goes deeper — it checks whether that JSON is **correct**, not just parseable.

The model outputs one of two shapes:

```json
// Shape A — Tool workflow
{
  "workflow": {
    "steps": [
      { "tool": "calculate_age", "params": { "birth_date": "1990-05-15" } }
    ]
  }
}

// Shape B — Natural language response (no applicable tool)
{
  "workflow": { "steps": [] },
  "message": "I'm sorry, I cannot assist with that using my current tools."
}
```

Phase 2 validates these outputs across four layers, each catching failures the previous layer misses:

```
Phase 1  → Did the JSON parse?
Phase 2A → Is the JSON the right shape/structure?
Phase 2B → Are the field values meaningful?
Phase 2C → Are the tool parameters valid against the tool's schema?
Phase 2D → Aggregate + report everything
```

---

## Dependencies

All Phase 2 code uses libraries already in `requirements.txt`. Before running,
ensure these are installed:

```
pydantic>=2.0
jsonschema>=4.21.0
loguru>=0.7.0
pyyaml>=6.0
```

### Install (Mac / Linux)

```bash
cd model_2_training
pip install -r requirements.txt
```

### Install (Windows)

```powershell
cd model_2_training
pip install -r requirements.txt
```

> **Note:** `pydantic` and `jsonschema` are the only new runtime additions for Phase 2.
> If you already had the base requirements installed, run:
>
> **Mac/Linux:** `pip install "pydantic>=2.0" "jsonschema>=4.21.0"`
> **Windows:** `pip install "pydantic>=2.0" "jsonschema>=4.21.0"`

---

## Phase 2A — Pydantic Schema Validation

### Purpose

Validates that every parsed JSON output conforms to the expected **output contract**.
Catches structural failures that `json.loads()` silently accepts:
missing `workflow` key, `steps` that isn't a list, steps missing `tool` or `params`,
wrong field types, and unexpected extra keys at any level.

### What It Does

Defines two Pydantic v2 models representing the valid output shapes:

| Shape | When | Required fields |
|---|---|---|
| `ToolWorkflowResponse` | Model decides to call a tool | `workflow.steps` non-empty, no `message` |
| `MessageWorkflowResponse` | No applicable tool | `workflow.steps` empty, `message` non-empty string |

For each prediction, tries to validate against both shapes in order.
Records the Pydantic error type (`missing`, `extra_forbidden`, `string_type`, etc.)
when validation fails.

Also computes `correct_shape_rate` by comparing the generated shape against the
reference output shape — without requiring full schema validity — so shape mismatches
are visible even in outputs that fail schema validation.

### Files Created

| File | Role |
|---|---|
| `src/schemas/workflow_schema.py` | Pydantic models (`WorkflowStep`, `Workflow`, `ToolWorkflowResponse`, `MessageWorkflowResponse`), `parse_response()`, `infer_shape()` |
| `src/eval/metrics_schema.py` | `compute_schema_metrics()` — runs validation, returns metrics dict; `summarize_schema_errors()` — samples of failures for inspection |

### Metrics Produced

| Metric | Description |
|---|---|
| `schema_valid_rate` | Fraction of outputs passing full Pydantic validation |
| `correct_shape_rate` | Fraction where generated shape (tool/message) matches reference |
| `extra_fields_rate` | Fraction with unexpected extra keys at any level |
| `wrong_type_rate` | Fraction where any field has the wrong Python type |

### Baseline Results (50 val samples)

| Metric | Value |
|---|---|
| `schema_valid_rate` | 94% |
| `correct_shape_rate` | 90% |
| `extra_fields_rate` | 0% |
| `wrong_type_rate` | 0% |

Key finding: 1 step had `tool` but no `params` key at all — invisible to Phase 1.

---

## Phase 2B — Field-Level Correctness

### Purpose

For each field in a schema-valid output, checks that the **value** is meaningful —
not just present and correctly typed. Goes one level deeper than Phase 2A.

### What It Does

Operates directly on raw JSON-parsed dicts (not Pydantic models) so it catches
value issues even in outputs that failed Phase 2A.

Each check is only counted over the subset of predictions where that field is
applicable:

| Check | Applicable to |
|---|---|
| `tool_name_nonempty_rate` | All steps in tool-shape predictions |
| `params_nonempty_rate` | All steps in tool-shape predictions |
| `steps_count_match_rate` | All predictions where both generated and reference parse |
| `message_nonempty_rate` | Message-shape predictions only |
| `steps_is_list_rate` | All parseable predictions |

Also provides `compute_per_field_report()` which gives a per-field breakdown of
`present`, `non_empty`, and `correct_type` rates for `tool`, `params`, `message`,
and `steps`.

### Files Created

| File | Role |
|---|---|
| `src/eval/metrics_fields.py` | `compute_field_metrics()` — field-level rates; `compute_per_field_report()` — per-field breakdown table |

### Metrics Produced

| Metric | Description |
|---|---|
| `tool_name_nonempty_rate` | Fraction of steps with a real non-empty tool name |
| `params_nonempty_rate` | Fraction of steps where params dict is populated |
| `steps_count_match_rate` | Fraction where generated step count == reference step count |
| `message_nonempty_rate` | Fraction of message responses with actual content |
| `steps_is_list_rate` | Fraction where `steps` is actually a list |

### Baseline Results (50 val samples)

| Metric | Value | Denominator |
|---|---|---|
| `tool_name_nonempty_rate` | 100% | 57 steps |
| `params_nonempty_rate` | 93% | 57 steps |
| `steps_count_match_rate` | **87.5%** | 48 pairs |
| `message_nonempty_rate` | 100% | 18 responses |
| `steps_is_list_rate` | 100% | 48 parseable |

Key finding: 87.5% step count match. The remaining ~12.5% of samples have a mismatch
between the generated and reference step count — a clear training signal for further
fine-tuning iterations.

---

## Phase 2C — Parameter Validation (Context-Based)

### Purpose

For each step, validates that the generated parameters are correct against the
tool's own schema. Catches missing required params, unexpected extra params, and
(when type info is available) type mismatches like `"complexity": 3` instead of
`"complexity": "high"`.

### Design: No Hardcoded Tools

Tools are **never hardcoded**. They are provided at runtime via RAG — each sample
carries its own tool context in its system message. The `ContextToolParser` extracts
those schemas per-sample so validation uses exactly the same tool context the model
saw during generation.

System message formats handled:

| Format | Example |
|---|---|
| Names only | `Available Tools: scale_service, restart_pod` |
| Simple schema (list params) | `Available Tools: {"calc_tip": {"parameters": ["amount"], "required": ["amount"]}}` |
| JSON Schema (typed params) | `Available Tools: {"get_weather": {"parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}}` |
| Empty | `Available Tools: {}` |

Steps without an extractable schema are **skipped** (not penalised) and tracked via
`param_schema_coverage_rate`.

### What It Does

For each step in a tool-shape prediction:
1. Reads its system message from `sample["messages"]`
2. Extracts tool schemas via `ContextToolParser`
3. Validates params against the extracted schema:
   - **required_params_present** — all required params are in `step.params`
   - **no_extra_params** — no params outside the known param list
   - **param_types_correct** — types match JSON Schema (only when type info available)

### Files Created

| File | Role |
|---|---|
| `src/eval/context_tool_parser.py` | `extract_tools_from_system_message()` — parses all 4 formats; `get_sample_tool_schemas()` — per-sample entry point |
| `src/eval/metrics_params.py` | `compute_param_metrics()` — validation rates; `validate_step_params()` — single-step validator; `summarize_param_errors()` — failure samples |

### Type Checking Map

| JSON Schema type | Python type checked |
|---|---|
| `"string"` | `str` |
| `"number"` | `int` or `float` |
| `"integer"` | `int` (not bool) |
| `"boolean"` | `bool` |
| `"array"` | `list` |
| `"object"` | `dict` |

### Metrics Produced

| Metric | Description |
|---|---|
| `param_schema_coverage_rate` | Fraction of steps with an extractable schema |
| `param_completeness_rate` | Fraction of steps with all required params present |
| `param_no_extras_rate` | Fraction of steps with no unexpected params |
| `param_type_correctness_rate` | Fraction of steps with correct param types (when type info available) |
| `full_param_validity_rate` | Fraction passing all available checks (composite) |

### Baseline Results (50 val samples)

| Metric | Value | Denominator |
|---|---|---|
| `param_schema_coverage_rate` | 89.5% | 57 steps |
| `param_completeness_rate` | 100% | 51 validated steps |
| `param_no_extras_rate` | 100% | 51 validated steps |
| `param_type_correctness_rate` | N/A | 0 (no type info in current training data) |
| `full_param_validity_rate` | 100% | 51 validated steps |

> **Note on type checking:** Current training data stores params as a flat list of
> names with no type info. `param_type_correctness_rate` will activate automatically
> once richer tool schemas (with `properties` + `type`) are injected via RAG.

---

## Phase 2D — Aggregation & Reporting

### Purpose

Wires all phases into a single unified pipeline so every `run_eval.py` and
`run_test.py` call automatically runs all four layers and produces a consolidated
report.

### What It Does

- **`metrics_aggregator.py`** — `run_all_metrics()` runs Phases 1 → 2A → 2B → 2C
  in order and returns one flat dict. All metric keys are namespaced by phase
  (e.g. `phase1_structural/json_parse_rate`, `phase2a_schema/schema_valid_rate`).
  Each phase is isolated — a failure in one does not stop the others.

- **`save_reports.py`** — `save_markdown_report()` now writes a per-phase Markdown
  table for each phase, plus a schema error breakdown sub-table, a per-field
  breakdown sub-table, and a multi-line interpretation section.
  `save_all_reports()` now also writes `param_errors.json`.

- **`evaluator.py`** — replaced the single `compute_metrics()` call with
  `run_all_metrics()`, wires per-field and param error reports into `save_all_reports`.

- **`wandb_logger.py`** — new `log_eval_metrics()` logs all phase-namespaced metrics
  under `val/` or `test/` in W&B, skipping non-scalar values automatically.

- **`json_eval.yaml`** — new `validation` section with config flags.

- **`run_eval.py` / `run_test.py`** — summary log now shows one key metric from
  each phase.

### Files Created

| File | Role |
|---|---|
| `src/eval/metrics_aggregator.py` | `run_all_metrics()`, `get_per_field_report()`, `get_schema_errors()`, `get_param_errors()` |

### Files Modified

| File | Change |
|---|---|
| `configs/eval/json_eval.yaml` | Added `validation` section |
| `src/eval/save_reports.py` | Per-phase markdown tables, `save_param_errors()`, updated `save_all_reports()` |
| `src/eval/evaluator.py` | Wired `run_all_metrics()` and new report helpers |
| `src/tracking/wandb_logger.py` | Added `log_eval_metrics()` |
| `scripts/run_eval.py` | Updated summary log lines |
| `scripts/run_test.py` | Updated summary log lines |

### Report Outputs

After running eval, the following files are written to `outputs/reports/val/`
(or `outputs/reports/test/`):

| File | Contents |
|---|---|
| `metrics.json` | All metrics from all phases as a flat JSON dict |
| `metrics_summary.md` | Human-readable Markdown with per-phase tables and interpretation |
| `error_samples.json` | Predictions that failed JSON parsing (Phase 1) |
| `sample_outputs.json` | Random sample of predictions for manual review |
| `param_errors.json` | Steps that failed parameter validation (Phase 2C) |

---

## Full Phase 2 File Map

```
model_2_training/
├── src/
│   ├── schemas/
│   │   └── workflow_schema.py          ← NEW  (Phase 2A)
│   │
│   └── eval/
│       ├── metrics_json.py             ← Phase 1 (unchanged)
│       ├── metrics_schema.py           ← NEW  (Phase 2A)
│       ├── metrics_fields.py           ← NEW  (Phase 2B)
│       ├── context_tool_parser.py      ← NEW  (Phase 2C)
│       ├── metrics_params.py           ← NEW  (Phase 2C)
│       ├── metrics_aggregator.py       ← NEW  (Phase 2D)
│       ├── evaluator.py                ← MODIFIED (Phase 2D)
│       └── save_reports.py             ← MODIFIED (Phase 2D)
│
├── src/tracking/
│   └── wandb_logger.py                 ← MODIFIED (Phase 2D)
│
├── configs/eval/
│   └── json_eval.yaml                  ← MODIFIED (Phase 2D)
│
└── scripts/
    ├── run_eval.py                     ← MODIFIED (Phase 2D)
    └── run_test.py                     ← MODIFIED (Phase 2D)
```

---

## Running Phase 2

Phase 2 runs automatically as part of the normal eval pipeline. No separate
commands needed — every `run_eval.py` call now runs all phases.

### Prerequisites

You need:
1. A trained checkpoint at `outputs/checkpoints/best_model/`
2. Data splits at `data/splits/` (created by `run_split.py`)
3. Dependencies installed (see top of this document)

---

### Step 1 — Create Data Splits (if not already done)

**Mac/Linux:**
```bash
cd /path/to/AutoMend/model_2_training
python3 scripts/run_split.py \
  --config configs/data/track_b_chatml.yaml
```

**Windows:**
```powershell
cd C:\path\to\AutoMend\model_2_training
python scripts\run_split.py `
  --config configs\data\track_b_chatml.yaml
```

---

### Step 2 — Run Validation Evaluation (Phase 2 included)

**Mac/Linux:**
```bash
cd /path/to/AutoMend/model_2_training
python3 scripts/run_eval.py \
  --config configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

**Windows:**
```powershell
cd C:\path\to\AutoMend\model_2_training
python scripts\run_eval.py `
  --config configs\eval\json_eval.yaml `
  --checkpoint outputs\checkpoints\best_model
```

Expected terminal output (one key metric per phase):
```
INFO  | json_parse_rate       : 0.96
INFO  | schema_valid_rate     : 0.94
INFO  | steps_count_match_rate: 0.875
INFO  | full_param_validity   : 1.0
```

---

### Step 3 — Run Test Evaluation (final benchmark only)

**Mac/Linux:**
```bash
cd /path/to/AutoMend/model_2_training
python3 scripts/run_test.py \
  --config configs/eval/json_eval.yaml \
  --checkpoint outputs/checkpoints/best_model
```

**Windows:**
```powershell
cd C:\path\to\AutoMend\model_2_training
python scripts\run_test.py `
  --config configs\eval\json_eval.yaml `
  --checkpoint outputs\checkpoints\best_model
```
---

### Viewing Results in Weights & Biases(WANDB)

When viewing your evaluation runs (e.g., `eval-val-best_model`) in the W&B dashboard, you might notice that the default line charts appear empty, showing only a single dot on the far left at "Step 0".

**Why this happens:** Unlike training scripts that log data continuously over hundreds of steps to draw a line, evaluation scripts calculate the final scores for the entire dataset and send a single snapshot of numbers to W&B exactly once. Because there is no "Step 1" or "Step 2" to connect to, W&B cannot draw a line.

**How to view your metrics:**
**The Summary Tab:** Open your specific evaluation run in W&B and look for the **Summary** table (usually located on the left side or within the Overview tab). Because evaluation is a one-time final result, this flat list is the most accurate and readable way to view all 34+ metrics.

---

### Quick Smoke Test (No Model Required) [OPTIONAL]

To verify all Phase 2 modules are importable and working correctly against
the existing saved val predictions — without needing a GPU or a model loaded:

**Mac/Linux:**
```bash
cd /path/to/AutoMend

python3 -c "
import sys, json
sys.path.insert(0, '.')

# Load saved val predictions (already generated, no model needed)
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

print()
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

print()
[print(f'  {k:<55} {v*100:.1f}%') for k, v in sorted(metrics.items()) if isinstance(v, float)]
"
```

---

### Run Individual Phase Tests

To verify a single phase in isolation:

**Mac/Linux:**
```bash
cd /path/to/AutoMend

# Phase 2A only
python3 -c "
import sys, json; sys.path.insert(0, '.')
with open('model_2_training/outputs/reports/val/sample_outputs.json') as f:
    preds = [{'generated': s['generated'], 'reference': s['reference']} for s in json.load(f)]
from model_2_training.src.eval.metrics_schema import compute_schema_metrics
print(compute_schema_metrics(preds))
"

# Phase 2B only
python3 -c "
import sys, json; sys.path.insert(0, '.')
with open('model_2_training/outputs/reports/val/sample_outputs.json') as f:
    preds = [{'generated': s['generated'], 'reference': s['reference']} for s in json.load(f)]
from model_2_training.src.eval.metrics_fields import compute_field_metrics
print(compute_field_metrics(preds))
"

# Phase 2C only (needs sample key for tool context)
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

**Windows:**
```powershell
cd C:\path\to\AutoMend

# Phase 2A only
python -c "import sys, json; sys.path.insert(0, '.'); [exec(open('model_2_training/outputs/reports/val/sample_outputs.json').read())]"

# (Use the same python3 commands above, replacing python3 with python)
```

---

## Metrics Reference

All metrics are floats in `[0.0, 1.0]` unless noted.

| Phase | Metric key (in metrics.json) | Description |
|---|---|---|
| 1 | `phase1_structural/json_parse_rate` | Fraction of outputs that parse as valid JSON |
| 1 | `phase1_structural/truncation_rate` | Fraction of outputs hitting the token limit |
| 1 | `phase1_structural/brace_balance_rate` | Fraction with balanced `{}`/`[]` |
| 1 | `phase1_structural/avg_output_length` | Mean character length (not a rate) |
| 2A | `phase2a_schema/schema_valid_rate` | Fraction passing full Pydantic validation |
| 2A | `phase2a_schema/correct_shape_rate` | Fraction with correct tool/message shape vs reference |
| 2A | `phase2a_schema/extra_fields_rate` | Fraction with unexpected extra keys |
| 2A | `phase2a_schema/wrong_type_rate` | Fraction with wrong field types |
| 2A | `phase2a_schema/schema_error_distribution` | `{error_type: count}` dict (not a rate) |
| 2B | `phase2b_fields/tool_name_nonempty_rate` | Fraction of steps with a real tool name |
| 2B | `phase2b_fields/params_nonempty_rate` | Fraction of steps with populated params |
| 2B | `phase2b_fields/steps_count_match_rate` | Fraction where step count matches reference |
| 2B | `phase2b_fields/message_nonempty_rate` | Fraction of message responses with content |
| 2B | `phase2b_fields/steps_is_list_rate` | Fraction where `steps` is a list |
| 2C | `phase2c_params/param_schema_coverage_rate` | Fraction of steps with an extractable schema |
| 2C | `phase2c_params/param_completeness_rate` | Fraction of steps with all required params |
| 2C | `phase2c_params/param_no_extras_rate` | Fraction of steps with no extra params |
| 2C | `phase2c_params/param_type_correctness_rate` | Fraction with correct param types |
| 2C | `phase2c_params/full_param_validity_rate` | All available param checks passed |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pydantic'`**
Run: `pip install "pydantic>=2.0"`

**`ModuleNotFoundError: No module named 'model_2_training'`**
Make sure you run commands from the `AutoMend/` root directory (not from inside `model_2_training/`), and that `sys.path` includes `.`.

**`FileNotFoundError: outputs/reports/val/sample_outputs.json`**
The quick smoke test requires existing saved predictions. Run `run_eval.py` with
a trained checkpoint first, or use predictions from a previous run.

**`param_type_correctness_rate` is always 0.0 (n=0)**
This is expected. The current training data uses simple list-format tool schemas
(`"parameters": ["param1", "param2"]`) which have no type information.
Type checking will activate automatically when JSON Schema-format tool definitions
(with `properties` + `type` fields) are injected via RAG at inference time.

**`correct_shape_rate` is below `schema_valid_rate`**
The model is choosing the wrong response type for some inputs — calling a tool
when it should give a natural language response, or vice versa. This is a
training data distribution issue and indicates the model needs more diverse
examples of when to use each shape.

**`AttributeError: 'dict' object has no attribute 'size'` during MLX inference**
Pre-existing bug in `generator.py` — the original MLX parameter counting code called
`.size` on nested dicts returned by `model.parameters()`. Fixed by replacing the one-liner
with a recursive `_count_mlx_params()` function that traverses dict/list/array nodes.
This fix is already applied; this note is here in case you see it in an older checkout.

**`HFValidationError: Repo id must be in the form 'repo_name'...` on MPS/MLX**
`mlx_lm.load()` interprets non-absolute paths as HuggingFace repo IDs. The scripts now
resolve `--checkpoint` to an absolute path automatically. If you see this error,
ensure you are on the latest version of `run_eval.py` / `run_test.py`.
