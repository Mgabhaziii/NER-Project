"""

Master pipeline — runs all steps end to end.

Usage
-----
    python pipeline.py                      # full run
    python pipeline.py --skip-annotate      # skip annotation if already done
    python pipeline.py --infer-only         # run inference on best model only
    python pipeline.py --lr-schedule        # enable LR warmup + decay (#6)
    python pipeline.py --with-postprocess   # apply DATE rules after inference (#2)
    python pipeline.py --train-file <path>  # use augmented training data (#8)
    python pipeline.py --eval-verified      # evaluate on verified_val.json (#5)

Steps
-----
    1. Annotate  — label dataModel.txt with en_core_web_trf
    2. Clean     — validate spans, remove overlaps
    3. Split     — 80/20 train/val
    4. Train     — fine-tune en_core_web_sm with early stopping
                   (+ LR scheduling if --lr-schedule is set)
    5. Evaluate  — precision/recall/F1 overall and per entity
                   (on verified_val.json if --eval-verified is set)
    6. Post-proc — apply regex DATE rules (if --with-postprocess)
    7. Infer     — run best model on sample sentences
"""

import sys
import json
import argparse
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

INPUT_FILE      = "data/dataModel.txt"
ANNOTATED_FILE  = "data/annotated_data.json"
VERIFIED_VAL    = "data/verified_val.json"
AUGMENTED_TRAIN = "data/augmented_train.json"
MODEL_OUTPUT    = "results/finetuned_model"
RESULTS_DIR     = "results"

TRAIN_CONFIG = {
    "iterations":   50,
    "batch_size":   8,
    "dropout":      0.3,
    "patience":     5,
    "learn_rate":   0.001,
    "seed":         42,
    # Improvement #6 — set via --lr-schedule flag
    "lr_schedule":  False,
    "warmup_iters": 3,
    "lr_decay":     0.98,
}

SAMPLE_SENTENCES = [
    "President John Smith met with Apple executives in Cupertino on Monday.",
    "The United Nations held a summit in Geneva on climate change in 2023.",
    "Dr. Rachel Green, a top surgeon at St. Mary's Hospital, spoke at the conference.",
    "Elon Musk announced that Tesla will open a new factory in Berlin next year.",
    "The FIFA World Cup was held in Qatar in November 2022.",
    "Chancellor Angela Merkel met with EU leaders in Brussels on Thursday.",
    "Amazon and Microsoft are competing for a Pentagon cloud computing contract.",
]


# ── Steps ──────────────────────────────────────────────────────────────────────

def step_annotate(skip: bool = False) -> list:
    from src.annotate import annotate, load_annotations
    if skip and Path(ANNOTATED_FILE).exists():
        print("[pipeline] Skipping annotation — loading existing data")
        return load_annotations(ANNOTATED_FILE)
    if not Path(INPUT_FILE).exists():
        print(f"[pipeline] ERROR: Input file '{INPUT_FILE}' not found.")
        print(f"           Place your text file at: {INPUT_FILE}")
        sys.exit(1)
    return annotate(INPUT_FILE, ANNOTATED_FILE)


def step_clean(data: list, train_file: str = None):
    from src.clean import clean, split

    # Improvement #8: use augmented training data if provided
    if train_file and Path(train_file).exists():
        print(f"[pipeline] Using augmented training data from {train_file}")
        augmented = json.loads(Path(train_file).read_text(encoding="utf-8"))
        # Still use standard val split from annotated_data
        _, val_data = split(clean(data, log_file=f"{RESULTS_DIR}/bad_spans.json"))
        train_data = augmented
        print(f"[pipeline] Train: {len(train_data)} (augmented)  Val: {len(val_data)}")
        return train_data, val_data

    data = clean(data, log_file=f"{RESULTS_DIR}/bad_spans.json")
    return split(data)


def step_train(train_data: list, val_data: list, config: dict = None):
    from src.train import train
    return train(
        train_data, val_data,
        output_dir=MODEL_OUTPUT,
        strategy="finetune",
        config=config or TRAIN_CONFIG,
        log_file=f"{RESULTS_DIR}/training_log.json",
    )


def step_evaluate(nlp, val_data: list, use_verified: bool = False):
    from src.evaluate import evaluate

    # Improvement #5: evaluate on human-verified labels if available
    if use_verified and Path(VERIFIED_VAL).exists():
        verified = json.loads(Path(VERIFIED_VAL).read_text(encoding="utf-8"))
        eval_data = [[ex["text"], {"entities": ex["entities"]}] for ex in verified]
        print(f"[pipeline] Evaluating on verified_val.json ({len(eval_data)} examples)")
        label = "fine-tuned en_core_web_sm [verified labels]"
    else:
        eval_data = val_data
        label = "fine-tuned en_core_web_sm"

    return evaluate(
        nlp, eval_data,
        output_file=f"{RESULTS_DIR}/finetuned_scores.json",
        label=label,
    )


def step_postprocess(nlp, val_data: list):
    """Improvement #2: evaluate DATE rule gain and show demo."""
    from src.postprocess import evaluate_postprocess_gain

    print("\n[pipeline] Applying DATE post-processing rules ...")
    stats = evaluate_postprocess_gain(nlp, val_data)
    print(f"[pipeline] DATE rule results:")
    print(f"  Overall F1 : {stats['before']['overall_f1']:.3f} → "
          f"{stats['after']['overall_f1']:.3f} "
          f"(+{stats['gain']['overall_f1']:.3f})")
    print(f"  DATE F1    : {stats['before']['date_f1']:.3f} → "
          f"{stats['after']['date_f1']:.3f} "
          f"(+{stats['gain']['date_f1']:.3f})")
    print(f"  DATE spans added : {stats['date_additions']}")
    Path(f"{RESULTS_DIR}/postprocess_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


def step_infer(nlp, with_postprocess: bool = False):
    from src.infer import predict, print_predictions
    from src.postprocess import apply_date_rules
    import spacy

    print("\n[pipeline] Running inference on sample sentences ...")

    if with_postprocess:
        # Apply DATE rules after model prediction
        results = []
        for text in SAMPLE_SENTENCES:
            doc = apply_date_rules(nlp(text))
            results.append((text, [(e.text, e.label_) for e in doc.ents]))
        print("\n  (DATE post-processing active)\n")
        for text, ents in results:
            print(f"  Text: {text}")
            for entity, label in ents:
                print(f"    [{label}]  '{entity}'")
            print()
    else:
        results = predict(nlp, SAMPLE_SENTENCES)
        print_predictions(results)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="NER Pipeline")
    parser.add_argument("--skip-annotate",   action="store_true",
                        help="Skip annotation if annotated_data.json already exists")
    parser.add_argument("--infer-only",      action="store_true",
                        help="Load existing model and run inference only")
    parser.add_argument("--lr-schedule",     action="store_true",
                        help="Enable LR warmup + decay schedule (Improvement #6)")
    parser.add_argument("--with-postprocess",action="store_true",
                        help="Apply regex DATE rules after model (Improvement #2)")
    parser.add_argument("--train-file",      type=str, default=None,
                        help="Path to augmented training data (Improvement #8)")
    parser.add_argument("--eval-verified",   action="store_true",
                        help="Evaluate on verified_val.json if available (Improvement #5)")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Apply CLI flags to config
    config = dict(TRAIN_CONFIG)
    if args.lr_schedule:
        config["lr_schedule"] = True
        print("[pipeline] LR scheduling enabled (warmup + decay)")

    print("=" * 55)
    print("  Named Entity Recognition Pipeline")
    print("  Entities: PER | ORG | LOC | DATE")
    print("=" * 55)

    if args.infer_only:
        if not Path(MODEL_OUTPUT).exists():
            print(f"No model found at '{MODEL_OUTPUT}'. Run pipeline.py first.")
            sys.exit(1)
        import spacy
        nlp = spacy.load(MODEL_OUTPUT)
        step_infer(nlp, with_postprocess=args.with_postprocess)
        return

    # Full pipeline
    data                   = step_annotate(skip=args.skip_annotate)
    train_data, val_data   = step_clean(data, train_file=args.train_file)
    nlp                    = step_train(train_data, val_data, config=config)
    step_evaluate(nlp, val_data, use_verified=args.eval_verified)

    if args.with_postprocess:
        step_postprocess(nlp, val_data)

    step_infer(nlp, with_postprocess=args.with_postprocess)

    print("\n" + "=" * 55)
    print("  Pipeline complete.")
    print(f"  Model saved to : {MODEL_OUTPUT}")
    print(f"  Results saved  : {RESULTS_DIR}/")
    print("\n  Improvement commands:")
    print("  - Verify labels  : python tools/verify_labels.py")
    print("  - DATE rules     : python pipeline.py --with-postprocess")
    print("  - LR schedule    : python pipeline.py --lr-schedule")
    print("  - Active learning: python tools/active_learning.py --select 50")
    print("=" * 55)


if __name__ == "__main__":
    main()
