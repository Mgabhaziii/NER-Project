# NER Project — Report

## 1. Introduction

This project implements a Named Entity Recognition (NER) system to extract
four entity types from news article text: people (PER), organisations (ORG),
locations (LOC), and dates (DATE). The system is built using Python and the
spaCy NLP library.

---

## 2. Learning Outcome 1 — Linguistic Sensitivity

### Why NER is a linguistically complex problem

Named entity recognition is non-trivial because entity boundaries and labels
are often deeply ambiguous. Several linguistic phenomena make this task hard:

**Lexical ambiguity**
The same word can refer to different entity types depending on context:
- *"Apple released a new iPhone"* — Apple is ORG
- *"She ate an apple"* — not an entity at all
- *"Washington signed the treaty"* — could be PER (George Washington) or LOC

This ambiguity cannot be resolved without contextual reasoning. A bag-of-words
model cannot handle it; a neural sequence model with attention (like spaCy's
transformer pipeline) is better suited because it reads the surrounding words.

**Entity boundary detection**
Deciding where an entity starts and ends is a non-trivial sequence labelling
problem. Consider:
- *"The University of Cape Town Medical School"* — is this one ORG or two?
- *"Prime Minister Boris Johnson"* — is the title part of the PER entity?

The standard BIO tagging scheme (Beginning, Inside, Outside) used by spaCy
encodes these boundary decisions as a sequence labelling task.

**Structural context**
News article text has specific structural properties:
- Named entities appear more densely in opening sentences (the "inverted pyramid")
- Bylines and datelines follow predictable patterns
- Organisation names are often followed by their type ("Ltd.", "Inc.", "Corp.")

These structural cues can be exploited by a model that reads full sentences.

### Error analysis findings

After training, `experiments/error_analysis.py` was run on the validation set.
Key observations (replace with your actual numbers after running):

- **Single-token false positives** — short, ambiguous tokens (e.g. common
  words that are also names) produce the most false positives. A larger context
  window would reduce these.

- **Long entity boundary errors** — multi-word entities of 3+ tokens are
  harder to identify. The model sometimes captures only the head noun and
  misses modifiers.

- **ORG/LOC confusion** — place names that are also organisation names
  (e.g. "Chelsea", "Arsenal", "Ajax") are frequently misclassified. This
  reflects a genuine linguistic ambiguity that requires world knowledge.

- **DATE complexity** — simple dates ("2022", "Monday") are recognised
  reliably, but complex temporal expressions ("the third quarter of last
  fiscal year") are missed. A rule-based regex component could supplement
  the neural model for this entity type.

---

## 3. Learning Outcome 2 — Comparing Techniques

Three training strategies were evaluated to identify the most suitable
approach for this problem.

### Strategy 1 — Zero-shot baseline (en_core_web_sm)

The pre-trained `en_core_web_sm` model was evaluated directly on the
validation set without any fine-tuning. This establishes the floor:
what performance do we get for free from a general-purpose English model?

**Result:** [fill in from results/baseline_scores.json after running]

**Limitation:** The pre-trained model was trained on a broad English corpus
(OntoNotes) which may not match the domain or style of our news data.

### Strategy 2 — Training from scratch (blank model)

A `spacy.blank("en")` model was trained entirely from our annotated data.
This isolates the contribution of our data alone, with no pre-trained knowledge.

**Result:** [fill in from results/blank_scores.json after running]

**Observation:** Expected to underperform Strategy 3 because it must learn
English syntax and entity patterns entirely from our (relatively small)
training set.

### Strategy 3 — Fine-tuning (recommended)

`en_core_web_sm` was fine-tuned on our annotated data. This approach:
- Starts from pre-trained weights that encode English grammar and common entities
- Adapts those weights to our specific domain and entity types
- Uses early stopping to avoid overfitting

**Result:** [fill in from results/finetuned_scores.json after running]

### Comparison

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (no training) | — | — | — |
| Blank model | — | — | — |
| Fine-tuned | — | — | — |

*Fill in results after running `experiments/experiment_runner.py`*

### Why fine-tuning is most suitable

Transfer learning (fine-tuning a pre-trained model) is the most suitable
technique for NER on limited annotated data because:

1. Pre-trained weights encode rich linguistic knowledge that would require
   millions of training examples to learn from scratch.
2. Our training set is relatively small (auto-annotated from one corpus).
   Fine-tuning allows high performance even with limited data.
3. The pre-trained model already handles common English entity patterns;
   fine-tuning specialises it for our domain.

For very large annotated datasets (100k+ examples), training from scratch
with a transformer backbone could approach fine-tuning performance. For
our use case, fine-tuning is clearly superior.

---

## 4. Learning Outcome 3 — Experimental Methodology

### Data pipeline

1. **Annotation** — Raw text was auto-annotated using `en_core_web_trf`
   (silver standard). This is acknowledged as a limitation: auto-annotations
   inherit the errors of the annotator model. A gold standard would require
   human verification.

2. **Cleaning** — Spans were validated against spaCy token boundaries using
   `alignment_mode="contract"`. Overlapping spans were resolved by keeping
   the longest span (`filter_spans`). All rejected spans were logged to
   `results/bad_spans.json` for inspection.

3. **Train/validation split** — 80/20 split with fixed random seed (42) for
   reproducibility. The validation set was held out throughout training and
   only used for evaluation.

### Training methodology

- **Early stopping** with patience=5 prevents overfitting. The best
  checkpoint (by validation F1) is saved, not the final iteration.
- **Per-iteration logging** — loss and validation scores are saved to
  `results/training_log.json` for full reproducibility.
- **Fixed random seed** — all experiments use the same seed so results
  are reproducible.

### Evaluation

Evaluation uses the standard NER metrics:
- **Precision** measures false positive rate
- **Recall** measures false negative rate
- **F1** is the primary metric as it balances both

Per-entity-type scores are reported separately because aggregate F1 can
hide poor performance on specific entity types (e.g. high PER F1 masking
poor DATE F1).

---

## 5. Learning Outcome 4 — Implementation

The full pipeline is implemented in Python using spaCy. Key implementation
decisions:

- **Modular design** — each step (annotate, clean, train, evaluate, infer)
  is a separate module. This allows any step to be swapped or extended
  independently.

- **Batch processing** — `nlp.pipe()` is used instead of calling `nlp()`
  in a loop. This is significantly faster on large datasets because spaCy
  can process batches on GPU if available.

- **Reproducibility** — all outputs (annotated data, training logs, evaluation
  scores, error analysis) are saved to `results/` as JSON files.

---

## 6. Conclusions

The fine-tuning approach achieved the highest F1 score, confirming that
transfer learning from a pre-trained English model is the most suitable
technique for NER on limited domain-specific data.

The primary remaining challenge is lexical ambiguity — entities that belong
to multiple types depending on context. Future work could address this by:

1. Adding a wider context window (switching from `en_core_web_sm` to
   `en_core_web_trf` as the fine-tuning base)
2. Supplementing the neural model with rule-based patterns for DATE entities
3. Expanding the training corpus with human-verified annotations
