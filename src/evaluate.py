"""

Provides a full evaluation suite:
    - Overall precision, recall, F1
    - Per-entity-type breakdown (essential for NER — models often perform
      very differently across PER/ORG/LOC/DATE)
    - Saves results to JSON for later comparison across experiments
"""

import json
import spacy
from pathlib import Path
from spacy.training.example import Example


def evaluate(
    nlp:        spacy.Language,
    data:       list,
    output_file: str = None,
    label:      str  = "model",
) -> dict:
    """
    Evaluate a spaCy NER model on a labelled dataset.

    Metrics reported:
        - Precision: of all entities the model predicted, how many were correct?
        - Recall:    of all true entities in the data, how many did the model find?
        - F1:        harmonic mean of precision and recall (primary metric for NER)

    Args:
        nlp:         Trained spaCy model.
        data:        List of [text, {"entities": [...]}] examples.
        output_file: If provided, saves results as JSON.
        label:       Name for this model in the output (e.g. "baseline", "finetuned").

    Returns:
        Dict with overall and per-entity scores.
    """
    print(f"\n[evaluate] Evaluating '{label}' on {len(data)} examples ...")

    examples = []
    for text, annot in data:
        example = Example.from_dict(nlp.make_doc(text), annot)
        example.predicted = nlp(text)
        examples.append(example)

    raw = nlp.evaluate(examples)

    results = {
        "model":     label,
        "n_examples": len(data),
        "overall": {
            "precision": round(raw["ents_p"], 4),
            "recall":    round(raw["ents_r"], 4),
            "f1":        round(raw["ents_f"], 4),
        },
        "per_entity": {
            entity: {
                "precision": round(scores["p"], 4),
                "recall":    round(scores["r"], 4),
                "f1":        round(scores["f"], 4),
            }
            for entity, scores in raw.get("ents_per_type", {}).items()
        },
    }

    _print_results(results)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(f"[evaluate] Results saved to '{output_file}'")

    return results


def compare(results_files: list[str]) -> None:
    """
    Load multiple results files and print a side-by-side comparison table.
    
    Args:
        results_files: List of paths to JSON results files.
    """
    all_results = []
    for f in results_files:
        p = Path(f)
        if p.exists():
            all_results.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            print(f"[evaluate] WARNING: '{f}' not found — skipping")

    if not all_results:
        print("[evaluate] No results to compare.")
        return

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-' * 54}")
    for r in all_results:
        o = r["overall"]
        print(f"  {r['model']:<20} {o['precision']:>10.3f} {o['recall']:>10.3f} {o['f1']:>10.3f}")

    # Per-entity comparison
    all_labels = sorted({
        label
        for r in all_results
        for label in r.get("per_entity", {}).keys()
    })
    if all_labels:
        print(f"\n  Per-entity F1:")
        header = f"  {'Label':<10}" + "".join(f"{r['model']:>14}" for r in all_results)
        print(header)
        print(f"  {'-' * (10 + 14 * len(all_results))}")
        for label in all_labels:
            row = f"  {label:<10}"
            for r in all_results:
                f1 = r.get("per_entity", {}).get(label, {}).get("f1", "-")
                row += f"{f1:>14}" if isinstance(f1, str) else f"{f1:>14.3f}"
            print(row)
    print("=" * 60)


def _print_results(results: dict) -> None:
    """Pretty-print evaluation results to the terminal."""
    o = results["overall"]
    print(f"\n  Model: {results['model']}  ({results['n_examples']} examples)")
    print(f"\n  {'Metric':<12} {'Score':>8}")
    print(f"  {'-' * 22}")
    print(f"  {'Precision':<12} {o['precision']:>8.3f}")
    print(f"  {'Recall':<12} {o['recall']:>8.3f}")
    print(f"  {'F1':<12} {o['f1']:>8.3f}")

    per = results.get("per_entity", {})
    if per:
        print(f"\n  Per-entity breakdown:")
        print(f"  {'Label':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-' * 44}")
        for entity, scores in sorted(per.items()):
            print(
                f"  {entity:<10} {scores['precision']:>10.3f} "
                f"{scores['recall']:>10.3f} {scores['f1']:>10.3f}"
            )
