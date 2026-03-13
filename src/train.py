"""

Two training strategies are implemented so they can be compared:

    Strategy A — "blank"
        Train from spacy.blank("en") — random weight initialisation.
        Useful as a lower-bound baseline to show how much pre-training helps.

    Strategy B — "finetune" (recommended)
        Fine-tune en_core_web_sm — starts from pre-trained weights on a large
        English corpus. Reaches higher F1 faster because the model already
        understands English syntax and common entity patterns.


"""

import json
import random
import spacy
import numpy as np
from pathlib import Path
from spacy.training.example import Example
from spacy.util import minibatch


class _JsonEncoder(json.JSONEncoder):
    """Convert numpy floats/ints to plain Python types for JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return super().default(obj)


# ── Defaults (overridable per experiment) ─────────────────────────────────────

DEFAULT_CONFIG = {
    "iterations":   50,
    "batch_size":   8,
    "dropout":      0.3,
    "patience":     5,
    "learn_rate":   0.001,
    "seed":         42,
    # ── Learning rate schedule (Improvement #6) ───────────────────────────
    # Set lr_schedule: True to enable warmup + decay instead of fixed LR.
    "lr_schedule":  False,
    "warmup_iters": 3,      # ramp LR from 0 to learn_rate over this many iters
    "lr_decay":     0.98,   # multiply LR by this factor each iter after warmup
}


def train(
    train_data: list,
    val_data:   list,
    output_dir: str,
    strategy:   str  = "finetune",
    config:     dict = None,
    log_file:   str  = None,
) -> spacy.Language:
    """
    Train a spaCy NER model.

    Args:
        train_data: List of [text, {"entities": [...]}] training examples.
        val_data:   Validation set in the same format.
        output_dir: Where to save the best model checkpoint.
        strategy:   "finetune" (recommended) or "blank" (baseline).
        config:     Training hyperparameters (see DEFAULT_CONFIG).
        log_file:   Path to save per-iteration metrics as JSON.

    Returns:
        The best spaCy model loaded from output_dir.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    random.seed(cfg["seed"])

    print(f"\n[train] Strategy: {strategy}")
    print(f"[train] Config:   {cfg}")

    # ── Load base model ────────────────────────────────────────────────────
    if strategy == "finetune":
        try:
            nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer"])
            print("[train] Base model: en_core_web_sm (pre-trained weights)")
        except OSError:
            raise OSError(
                "Model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    elif strategy == "blank":
        nlp = spacy.blank("en")
        print("[train] Base model: blank English (random initialisation)")
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'finetune' or 'blank'.")

    # ── Set up NER pipe ────────────────────────────────────────────────────
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annot in train_data:
        for _, _, label in annot["entities"]:
            ner.add_label(label)

    # ── Training loop ──────────────────────────────────────────────────────
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    best_f1     = 0.0
    no_improve  = 0
    history     = []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with nlp.disable_pipes(*other_pipes):
        if strategy == "finetune":
            optimizer = nlp.resume_training()
        else:
            optimizer = nlp.begin_training()
        optimizer.learn_rate = cfg["learn_rate"]

        # ── LR schedule setup (Improvement #6) ────────────────────────────
        use_schedule  = cfg.get("lr_schedule", False)
        peak_lr       = cfg["learn_rate"]
        warmup_iters  = cfg.get("warmup_iters", 3)
        lr_decay      = cfg.get("lr_decay", 0.98)
        if use_schedule:
            print(f"[train] LR schedule: warmup {warmup_iters} iters → "
                  f"peak {peak_lr} → decay ×{lr_decay}/iter")

        for itn in range(cfg["iterations"]):
            random.shuffle(train_data)
            losses = {}

            # ── Apply LR schedule ──────────────────────────────────────────
            if use_schedule:
                if itn < warmup_iters:
                    # Linear warmup: 0 → peak_lr
                    current_lr = peak_lr * (itn + 1) / warmup_iters
                else:
                    # Exponential decay after warmup
                    current_lr = peak_lr * (lr_decay ** (itn - warmup_iters + 1))
                optimizer.learn_rate = current_lr

            for batch in minibatch(train_data, size=cfg["batch_size"]):
                examples = [
                    Example.from_dict(nlp.make_doc(text), ann)
                    for text, ann in batch
                ]
                nlp.update(examples, drop=cfg["dropout"], losses=losses, sgd=optimizer)

            # Evaluate this iteration
            scores = _evaluate(nlp, val_data)
            f1     = scores["ents_f"]
            is_best = f1 > best_f1

            row = {
                "iteration": itn + 1,
                "loss":      round(losses.get("ner", 0), 4),
                "precision": round(scores["ents_p"], 4),
                "recall":    round(scores["ents_r"], 4),
                "f1":        round(f1, 4),
                "best":      is_best,
                "learn_rate": round(float(optimizer.learn_rate), 6),
            }
            history.append(row)

            marker = " <-- best" if is_best else ""
            print(
                f"  [{itn+1:>3}]  loss={row['loss']:.4f}  "
                f"P={row['precision']:.3f}  R={row['recall']:.3f}  "
                f"F1={row['f1']:.3f}{marker}"
            )

            if is_best:
                best_f1    = f1
                no_improve = 0
                nlp.to_disk(output_dir)
            else:
                no_improve += 1

            if no_improve >= cfg["patience"]:
                print(f"\n[train] Early stopping at iteration {itn+1}")
                break

    print(f"[train] Best F1: {best_f1:.3f} — saved to '{output_dir}'")

    # Save training log
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).write_text(
            json.dumps({"strategy": strategy, "config": cfg, "history": history},
                       indent=2, cls=_JsonEncoder),
            encoding="utf-8",
        )
        print(f"[train] Training log saved to '{log_file}'")

    return spacy.load(output_dir)


def _evaluate(nlp: spacy.Language, data: list) -> dict:
    """Run spaCy's built-in evaluator on a dataset."""
    examples = []
    for text, annot in data:
        example = Example.from_dict(nlp.make_doc(text), annot)
        example.predicted = nlp(text)
        examples.append(example)
    return nlp.evaluate(examples)
