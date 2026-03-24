# AutoMend Track B — Phase 2.5, 2.75 & 3

The Track B model reads an MLOps incident description and outputs a JSON workflow with fix steps. These three phases cover how we test it, tune it, and check how it handles messy inputs.

| Phase    | What it does                   | Why                                      |
| -------- | ------------------------------ | ---------------------------------------- |
| **2.5**  | Fixed benchmark + error labels | Models were compared inconsistently      |
| **2.75** | Auto hyperparameter search     | Settings were picked by hand             |
| **3**    | Stress test with noisy inputs  | No idea how it handles real-world inputs |

---

## Phase 2.5 — Benchmark + Error Labels

---

### Step 1 — Build the Benchmark

This creates 30 fixed test questions. Every model gets scored on the same 30 questions, so comparisons are fair.

> Run this **once only**. Running it again will break historical comparisons.

```bash
python scripts/build_benchmark.py
```

**Saved to:**

| File                                           | What's in it              |
| ---------------------------------------------- | ------------------------- |
| `data/benchmarks/gold_benchmark.jsonl`         | The 30 questions          |
| `data/benchmarks/gold_benchmark_manifest.json` | Build info + lock warning |

**Questions are split evenly — 10 per type:**

| Type          | What the model should output           |
| ------------- | -------------------------------------- |
| `multi_step`  | A workflow with 2 or more steps        |
| `single_step` | A workflow with exactly 1 step         |
| `refusal`     | Empty steps — the request is not valid |

---

### Step 2 — Know the Error Labels

After scoring, every prediction gets one label that explains exactly what went wrong. Labels are checked in order — first match wins.

| #   | Label               | Meaning                                      |
| --- | ------------------- | -------------------------------------------- |
| 1   | `VALID`             | Correct output                               |
| 2   | `EMPTY`             | Model returned nothing                       |
| 3   | `TRUNCATED`         | Output got cut off (hit token limit)         |
| 4   | `MISSING_WORKFLOW`  | Valid JSON but missing `workflow` or `steps` |
| 5   | `WRONG_STEPS_TYPE`  | `steps` is not a list                        |
| 6   | `EMPTY_STEPS`       | `steps` is empty when it shouldn't be        |
| 7   | `MALFORMED_JSON`    | Can't be parsed by `json.loads()`            |
| 8   | `UNBALANCED_BRACES` | `{` and `}` counts don't match               |
| 9   | `UNBALANCED_QUOTES` | Odd number of `"` characters                 |

> `EMPTY_STEPS` is only a failure if the reference also expected steps. If the reference is a refusal, empty steps is the right answer.

---

### Step 3 — Score a Checkpoint

Run this to score any trained checkpoint against the 30 locked questions.

```bash
# Mac / Linux
python scripts/run_benchmark.py \
    --config     configs/eval/json_eval.yaml \
    --checkpoint outputs/checkpoints/best_model
```

```powershell
# Windows
python scripts/run_benchmark.py `
    --config     configs/eval/json_eval.yaml `
    --checkpoint outputs/checkpoints/best_model
```

Results saved to `outputs/reports/benchmark/best_model/`.

---

## Phase 2.75 — Hyperparameter Sweep

---

### Step 4 — Start the Sweep

This sets up the auto-search. It tries different training settings, scores each one, and keeps improving based on what works.

```bash
wandb sweep configs/sweep/wandb_sweep.yaml
```

This prints a sweep ID. Copy it — you'll need it in the next step.

**Sweep settings:**

| Setting    | Value                      |
| ---------- | -------------------------- |
| Project    | `automend-model2`          |
| Method     | Bayesian optimization      |
| Goal       | Maximize valid output rate |
| Early stop | Hyperband                  |

---

### Step 5 — Run Sweep Trials

Each trial trains a model, scores it, and reports back. Run more agents to search faster.

```bash
wandb agent mlops-team-northeastern-university/automend-model2/ltilnti3 --count 5
```

**What gets tuned (10 settings):**

| Setting                       | What it controls                               |
| ----------------------------- | ---------------------------------------------- |
| `lora_r`                      | How much the adapter can learn (8, 16, 32, 64) |
| `learning_rate`               | How fast weights update                        |
| `lora_alpha`                  | LoRA scaling                                   |
| `lora_dropout`                | Dropout inside LoRA layers                     |
| `per_device_train_batch_size` | Samples per step                               |
| `gradient_accumulation_steps` | Makes the effective batch bigger               |
| `lr_scheduler_type`           | How the learning rate changes over time        |
| `warmup_ratio`                | Warm-up period before full learning rate       |
| `num_train_epochs`            | How many times the model sees the data         |
| `weight_decay`                | Regularization to prevent overfitting          |

---

### Step 6 — Rank the Trials

Once trials finish, run this to see which config scored best.

```bash
# Mac / Linux
for dir in outputs/reports/sweeps/Qwen2.5-1.5B-Instruct_*/; do
    rate=$(python3 -c "import json; d=json.load(open('$dir/metrics.json')); print(d.get('phase1_structural/tax_valid_rate','N/A'))")
    echo "$rate  $(basename $dir)"
done | sort -rn
```

```powershell
# Windows
Get-ChildItem -Directory "outputs/reports/sweeps/Qwen2.5-1.5B-Instruct_*" | ForEach-Object {
    $metrics = "$($_.FullName)/metrics.json"
    $rate = python -c "import json; d=json.load(open(r'$metrics')); print(d.get('phase1_structural/tax_valid_rate','N/A'))"
    [PSCustomObject]@{ Rate = $rate; Name = $_.Name }
} | Sort-Object Rate -Descending | Format-Table
```

**Best results so far (21 trials):**

| Rank | Config     | Valid Output | Schema OK | Steps Match |
| ---- | ---------- | ------------ | --------- | ----------- |
| 1    | r16 (0052) | **96.7%**    | 96.7%     | 75.9%       |
| 2    | r8 (0037)  | 93.3%        | 96.7%     | 69.0%       |
| 3    | r8 (1122)  | 93.3%        | 93.3%     | 67.9%       |
| 4    | r64 (0058) | 93.3%        | 93.3%     | 64.3%       |

**Winner:** `lora_r=16` · `lr=1.19e-4` · `batch=4` · `accum=16` → effective batch of 64.

**Why r16 won:**

| Rank    | Params   | What happened                               |
| ------- | -------- | ------------------------------------------- |
| r8      | ~21M     | Slight underfitting                         |
| **r16** | **~42M** | **Best score**                              |
| r64     | ~168M    | Overfit — hit token limit on 2/30 questions |

> The 96.7% score means the output is **correctly formatted JSON**. It does not yet check if the right tool or parameters were chosen — that comes next.

