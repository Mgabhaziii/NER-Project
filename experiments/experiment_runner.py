"""

Runs three experiments and compares them:

    Experiment 1 — Baseline (en_core_web_sm, no fine-tuning)
        The pre-trained model evaluated directly on our data.

    Experiment 2 — Blank model trained from scratch
        spacy.blank("en") trained on our annotated data.
        Shows whether our data alone is sufficient.

    Experiment 3 — Fine-tuned model (Ideal approach)
        en_core_web_sm fine-tuned on our annotated data.
        Expected to outperform both baselines.

Results are saved to results/ and printed as a comparison table.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.annotate import load_annotations
from src.clean    import clean, split
from src.train    import train
from src.evaluate import evaluate, compare
import spacy


ANNOTATED_FILE = "data/annotated_data.json"
RESULTS_DIR    = "results"

EXPERIMENT_CONFIG = {
    "iterations": 30,
    "batch_size": 8,
    "dropout":    0.3,
    "patience":   5,
    "learn_rate": 0.001,
    "seed":       42,
}


def run_baseline(val_data: list) -> dict:
    """
    Experiment 1: Evaluate en_core_web_sm with no fine-tuning.
    This is our zero-shot baseline — measures what the pre-trained
    model already knows about our entity types.
    """
    print("\n" + "=" * 55)
    print("  EXPERIMENT 1 — Baseline (no fine-tuning)")
    print("=" * 55)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[experiment] en_core_web_sm not found — skipping baseline")
        return {}

    return evaluate(
        nlp, val_data,
        output_file=f"{RESULTS_DIR}/baseline_scores.json",
        label="baseline (en_core_web_sm)",
    )


def run_blank_training(train_data: list, val_data: list) -> dict:
    """
    Experiment 2: Train a blank model from scratch on our data.
    Isolates the contribution of our training data alone.
    """
    print("\n" + "=" * 55)
    print("  EXPERIMENT 2 — Blank model trained from scratch")
    print("=" * 55)

    nlp = train(
        train_data, val_data,
        output_dir="results/blank_model",
        strategy="blank",
        config=EXPERIMENT_CONFIG,
        log_file=f"{RESULTS_DIR}/blank_training_log.json",
    )
    return evaluate(
        nlp, val_data,
        output_file=f"{RESULTS_DIR}/blank_scores.json",
        label="blank model (trained from scratch)",
    )


def run_finetuned_training(train_data: list, val_data: list) -> dict:
    """
    Experiment 3: Fine-tune en_core_web_sm on our data.
    Expected to give the best results by combining pre-trained
    English knowledge with our domain-specific annotations.
    """
    print("\n" + "=" * 55)
    print("  EXPERIMENT 3 — Fine-tuned en_core_web_sm")
    print("=" * 55)

    nlp = train(
        train_data, val_data,
        output_dir="results/finetuned_model",
        strategy="finetune",
        config=EXPERIMENT_CONFIG,
        log_file=f"{RESULTS_DIR}/finetuned_training_log.json",
    )
    return evaluate(
        nlp, val_data,
        output_file=f"{RESULTS_DIR}/finetuned_scores.json",
        label="fine-tuned en_core_web_sm",
    )


def save_experiment_summary(results: list[dict]) -> None:
    """Save a combined summary of all experiments."""
    summary = {
        "experiments": results,
        "winner": max(
            (r for r in results if r),
            key=lambda r: r.get("overall", {}).get("f1", 0),
        ).get("model", "unknown"),
    }
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{RESULTS_DIR}/experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n[experiments] Summary saved to '{RESULTS_DIR}/experiment_summary.json'")


def main():
    print("=" * 55)
    print("  NER Experiment Runner")
    print("  Comparing 3 training strategies")
    print("=" * 55)

    # Load and prepare data
    if not Path(ANNOTATED_FILE).exists():
        print(f"\nAnnotated data not found at '{ANNOTATED_FILE}'.")
        print("Run pipeline.py first to annotate and clean your data.")
        sys.exit(1)

    data       = load_annotations(ANNOTATED_FILE)
    data       = clean(data, log_file=f"{RESULTS_DIR}/bad_spans.json")
    train_data, val_data = split(data)

    # Run all three experiments
    results = []
    results.append(run_baseline(val_data))
    results.append(run_blank_training(train_data, val_data))
    results.append(run_finetuned_training(train_data, val_data))

    # Print comparison table
    compare([
        f"{RESULTS_DIR}/baseline_scores.json",
        f"{RESULTS_DIR}/blank_scores.json",
        f"{RESULTS_DIR}/finetuned_scores.json",
    ])

    # Save summary
    save_experiment_summary([r for r in results if r])
    print("\n[experiments] All experiments complete.")


if __name__ == "__main__":
    main()
