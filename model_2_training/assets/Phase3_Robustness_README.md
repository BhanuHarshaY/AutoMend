---

## Phase 3 — Robustness Testing

---

### Step 1 — Run the Full Robustness Test

Tests the model against 5 types of bad inputs. The model is loaded once and reused for all runs.

```bash
# Mac / Linux
python scripts/run_robustness.py \
    --config     configs/eval/json_eval.yaml \
    --checkpoint outputs/checkpoints/best_model
```

```powershell
# Windows
python scripts/run_robustness.py `
    --config     configs/eval/json_eval.yaml `
    --checkpoint outputs/checkpoints/best_model
```

**What gets tested:**

| Type         | Simulates              | How                                |
| ------------ | ---------------------- | ---------------------------------- |
| `typo`       | Typing mistakes        | ~5% character errors               |
| `noise`      | Extra boilerplate text | Random sentence added at the start |
| `truncation` | Message cut off        | Last 30% of words removed          |
| `case_lower` | No capital letters     | Entire input lowercased            |
| `paraphrase` | Shuffled sentences     | Sentence order randomized          |

---

### Step 2 — Run Specific Perturbations Only (Optional)

```bash
# Mac / Linux
python scripts/run_robustness.py \
    --config         configs/eval/json_eval.yaml \
    --checkpoint     outputs/checkpoints/best_model \
    --perturbations  typo noise truncation
```

```powershell
# Windows
python scripts/run_robustness.py `
    --config         configs/eval/json_eval.yaml `
    --checkpoint     outputs/checkpoints/best_model `
    --perturbations  typo noise truncation
```

---

### Step 3 — Check the Results

**Clean baseline:** 96.7% valid · 65.5% step count match

**Drop in valid output per perturbation:**

| Perturbation | Drop  | Worst area          |
| ------------ | ----- | ------------------- |
| `typo`       | −6.7% | Valid output        |
| `noise`      | −3.3% | JSON parse + schema |
| `case_lower` | −3.3% | JSON parse + schema |
| `truncation` | −3.3% | Valid output        |
| `paraphrase` | 0%    | None                |

**Also broken down by sub-group:**

| Group            | What it reveals                     |
| ---------------- | ----------------------------------- |
| `archetype`      | How each output type performs       |
| `dataset`        | Which dataset source struggles most |
| `input_length`   | Short vs medium vs long inputs      |
| `complexity`     | Simple vs complex workflows         |
| `error_category` | Which failure type is most common   |

**Key finding:** Long inputs (>300 chars) dropped to 80% valid vs. 100% for shorter inputs. Worst perturbation: `case_lower`.

**Output files** — saved to `outputs/reports/robustness/best_model/`:

| File                          | Contents                            |
| ----------------------------- | ----------------------------------- |
| `clean_metrics.json`          | Baseline scores                     |
| `{pert}_metrics.json`         | Scores per perturbation             |
| `robustness_summary.json`     | Drop table + worst perturbation     |
| `slice_report.json`           | Scores per sub-group                |
| `cross_slice_robustness.json` | Archetype × perturbation            |
| `paraphrase_consistency.json` | Consistency across 3 seeds          |
| `failure_log.json`            | Samples that went from pass to fail |
| `robustness_report.md`        | Summary in plain text               |

---

## Key Files

| File                                                                       | What it does                     |
| -------------------------------------------------------------------------- | -------------------------------- |
| [scripts/build_benchmark.py](scripts/build_benchmark.py)                   | Builds the 30-question benchmark |
| [scripts/run_benchmark.py](scripts/run_benchmark.py)                       | Scores a checkpoint              |
| [scripts/run_sweep.py](scripts/run_sweep.py)                               | Runs one sweep trial             |
| [scripts/run_robustness.py](scripts/run_robustness.py)                     | Runs robustness + slice eval     |
| [src/eval/error_taxonomy.py](src/eval/error_taxonomy.py)                   | Error label logic                |
| [src/eval/metrics_aggregator.py](src/eval/metrics_aggregator.py)           | Full metrics pipeline            |
| [src/robustness/perturbations.py](src/robustness/perturbations.py)         | Perturbation functions           |
| [src/robustness/slice_eval.py](src/robustness/slice_eval.py)               | Slice grouping + metrics         |
| [src/robustness/robustness_runner.py](src/robustness/robustness_runner.py) | Robustness orchestrator          |
| [configs/sweep/wandb_sweep.yaml](configs/sweep/wandb_sweep.yaml)           | Sweep config                     |
| [configs/model/qwen_baseline.yaml](configs/model/qwen_baseline.yaml)       | 1.5B model config                |
| [configs/model/qwen_3b_baseline.yaml](configs/model/qwen_3b_baseline.yaml) | 3B model config                  |

---
