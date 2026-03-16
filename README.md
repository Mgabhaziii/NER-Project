# Named Entity Recognition — NER Project

A custom spaCy NER system trained to extract named entities from news articles.
Entities recognised: **PER** (people), **ORG** (organisations), **LOC** (locations), **DATE** (dates).

**Best overall F1: 0.853** (fine-tuned en_core_web_sm + LR scheduling)

---

## Results Summary

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| DATE | 0.888 | 0.905 | 0.896 |
| LOC | 0.907 | 0.923 | 0.915 |
| ORG | 0.743 | 0.742 | 0.743 |
| PER | 0.858 | 0.825 | 0.841 |
| **Overall** | **0.855** | **0.852** | **0.853** |

> **Note:** On the human-verified validation subset (44 examples), honest F1 = 0.823.
> Auto-labelled evaluation scores are slightly inflated due to a 64.3% annotation error rate
> in the silver-standard labels.

---

## Project Structure

```
NER_Project/
├── data/
│   ├── dataModel.txt                  <- raw text (one sentence per line)
│   ├── annotated_data.json            <- auto-generated on first run (2,114 examples)
│   ├── augmented_train.json           <- training data + active learning candidates
│   ├── verified_val.json              <- human-verified validation examples
│   ├── verification_stats.json        <- annotation error rate stats
│   └── active_learning_candidates.json <- 50 most uncertain training examples
│
├── src/                               <- modular pipeline components (LO4)
│   ├── annotate.py                    <- auto-label raw text using en_core_web_trf
│   ├── clean.py                       <- validate spans, split train/val
│   ├── train.py                       <- fine-tune with early stopping + LR scheduling
│   ├── evaluate.py                    <- precision, recall, F1 per entity type
│   ├── infer.py                       <- batch inference on new text
│   └── postprocess.py                 <- regex DATE post-processing rules
│
├── tools/                             <- improvement tools
│   ├── verify_labels.py               <- interactive human annotation reviewer
│   └── active_learning.py             <- uncertainty sampling for efficient labelling
│
├── experiments/                       <- experimental methodology (LO2, LO3)
│   ├── experiment_runner.py           <- compare 3 training strategies
│   └── error_analysis.py             <- linguistic error analysis (LO1)
│
├── results/                           <- auto-created, all outputs saved here
│   ├── finetuned_model/               <- best trained model checkpoint
│   ├── finetuned_scores.json          <- evaluation results
│   ├── training_log.json              <- per-iteration loss and F1
│   ├── bad_spans.json                 <- rejected spans during cleaning
│   ├── error_analysis.json            <- linguistic error breakdown
│   └── postprocess_stats.json         <- DATE rule match statistics
|
├── pipeline.py                        <- run everything end to end
├── README.md
└── report.md                          <- written analysis covering all LOs
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install spacy scikit-learn
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
```

### 2. Place your data

```
data/dataModel.txt
```

### 3. Run the full pipeline

```bash
python pipeline.py
```

On subsequent runs, skip re-annotation (already saved):

```bash
python pipeline.py --skip-annotate
```

With all improvements active:

```bash
python pipeline.py --skip-annotate --lr-schedule --with-postprocess --eval-verified
```

Using augmented training data (after active learning):

```bash
python pipeline.py --skip-annotate --lr-schedule --train-file data/augmented_train.json
```

### 4. Compare training strategies 

```bash
python experiments/experiment_runner.py
```

### 5. Analyse errors 

```bash
python experiments/error_analysis.py
```

---

## Pipeline Flags

| Flag | Description |
|---|---|
| `--skip-annotate` | Skip annotation step, load existing `annotated_data.json` |
| `--lr-schedule` | Enable warmup + decay learning rate schedule (+0.003 F1) |
| `--with-postprocess` | Apply regex DATE rules after model prediction |
| `--eval-verified` | Evaluate on `verified_val.json` instead of auto-labelled val set |
| `--train-file <path>` | Use a custom training file (e.g. augmented data) |

---

## Training Strategies Compared

| Strategy | Precision | Recall | F1 | Iterations |
|---|---|---|---|---|
| Baseline (no training) | 0.177 | 0.212 | 0.193 | — |
| Blank model (from scratch) | 0.841 | 0.815 | 0.828 | 25 |
| Fine-tuned en_core_web_sm | 0.856 | 0.843 | 0.850 | 15 |
| Fine-tuned + LR schedule | 0.855 | 0.852 | **0.853** | 17–24 |

Fine-tuning is the recommended strategy. It outperforms blank training because pre-trained
weights encode English grammar and entity patterns that would require millions of examples
to learn from scratch.

---

## Improvements Implemented

### #5 — Human Verification of Validation Labels (`tools/verify_labels.py`)

Auto-generated labels contain errors. Evaluating against them is circular — it measures how
well the small model mimics the large annotator, not how accurately it identifies real entities.

```bash
python tools/verify_labels.py
```

- Displays each example with colour-highlighted entities
- Supports accept / edit / skip per example
- Progress saved incrementally — can quit and resume
- **Finding: 64.3% of reviewed examples required corrections**

### #2 — Regex DATE Post-Processing (`src/postprocess.py`)

The neural model misses structured temporal expressions. Post-processing fills these gaps
with rule-based regex patterns applied after model prediction.

Patterns covered: fiscal quarters (`Q3 2024`), named quarters (`the third quarter of 2023`),
fiscal years (`fiscal year 2022`, `FY2023`), decade references (`the 1990s`, `early 2000s`),
date ranges, and anchored seasonal references.

### #6 — Learning Rate Scheduling (`src/train.py --lr-schedule`)

A two-phase schedule replaces the fixed learning rate:

- **Warmup** (iterations 1–3): ramps LR from 0 → peak (0.001)
- **Decay** (iteration 4+): multiplies LR by 0.98 each iteration

Result: consistent **+0.003 F1** gain over fixed LR across multiple runs.

### #8 — Active Learning (`tools/active_learning.py`)

Selects the 50 most uncertain training examples for priority annotation, based on entity
density, single-token ratio, context length, and ORG/LOC ambiguity signals.

```bash
# Select candidates
python tools/active_learning.py --select 50

# View stats
python tools/active_learning.py --stats

# Add verified candidates to training data
python tools/active_learning.py --add
```

All top candidates showed ORG/LOC ambiguity — consistent with error analysis findings.

---

## Evaluation Metrics

- **Precision** — of all predicted entities, how many were correct?
- **Recall** — of all true entities, how many did the model find?
- **F1** — harmonic mean of precision and recall (primary NER metric)
- **Per-entity breakdown** — separate scores for PER, ORG, LOC, DATE

Evaluation is run against:
- `annotated_data.json` (auto-labelled, 423 examples) — convenient but inflated
- `verified_val.json` (human-verified, 44 examples) — honest ground truth

---


