"""

Analyses the types of errors the model makes and links them to
specific linguistic phenomena. 

Error categories investigated:
    1. False Positives (FP) — model predicts an entity that isn't one
    2. False Negatives (FN) — model misses a true entity
    3. Label confusion    — model finds the right span but wrong label
       e.g. predicting ORG for "Apple" when context implies LOC
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
from spacy.training.example import Example


def analyse_errors(
    nlp:        spacy.Language,
    val_data:   list,
    output_file: str = "results/error_analysis.json",
) -> dict:
    """
    Collect and categorise all prediction errors on the validation set.

    Returns a dict with false positives, false negatives, and
    label confusions, plus linguistic statistics.
    """
    print("[error_analysis] Analysing prediction errors ...")

    false_positives = []   # predicted but not in gold
    false_negatives = []   # in gold but not predicted
    label_confusions = []  # right span, wrong label
    correct = []

    for text, annot in val_data:
        doc = nlp(text)

        gold_ents = {
            (start, end): label
            for start, end, label in annot["entities"]
        }
        pred_ents = {
            (ent.start_char, ent.end_char): ent.label_
            for ent in doc.ents
        }

        # False positives — predicted, not in gold
        for (start, end), pred_label in pred_ents.items():
            span_text = text[start:end]
            if (start, end) not in gold_ents:
                false_positives.append({
                    "text":       text,
                    "span":       span_text,
                    "pred_label": pred_label,
                    "start":      start,
                    "end":        end,
                    "span_length": len(span_text.split()),
                })
            elif gold_ents[(start, end)] != pred_label:
                # Right span, wrong label
                label_confusions.append({
                    "text":       text,
                    "span":       span_text,
                    "gold_label": gold_ents[(start, end)],
                    "pred_label": pred_label,
                })
            else:
                correct.append({"span": span_text, "label": pred_label})

        # False negatives — in gold, not predicted
        for (start, end), gold_label in gold_ents.items():
            span_text = text[start:end]
            if (start, end) not in pred_ents:
                false_negatives.append({
                    "text":       text,
                    "span":       span_text,
                    "gold_label": gold_label,
                    "start":      start,
                    "end":        end,
                    "span_length": len(span_text.split()),
                })

    analysis = {
        "summary": {
            "correct":          len(correct),
            "false_positives":  len(false_positives),
            "false_negatives":  len(false_negatives),
            "label_confusions": len(label_confusions),
        },
        "false_positives_by_label":  _group_by_label(false_positives, "pred_label"),
        "false_negatives_by_label":  _group_by_label(false_negatives, "gold_label"),
        "label_confusion_matrix":    _confusion_matrix(label_confusions),
        "top_false_positives":       false_positives[:20],
        "top_false_negatives":       false_negatives[:20],
        "label_confusions":          label_confusions[:20],
        "linguistic_observations":   _linguistic_observations(
            false_positives, false_negatives, label_confusions
        ),
    }

    _print_analysis(analysis)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[error_analysis] Full report saved to '{output_file}'")
    return analysis


def _group_by_label(errors: list, label_key: str) -> dict:
    counts = defaultdict(int)
    for e in errors:
        counts[e[label_key]] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _confusion_matrix(confusions: list) -> dict:
    """Build a label confusion matrix: {gold_label: {pred_label: count}}."""
    matrix = defaultdict(lambda: defaultdict(int))
    for c in confusions:
        matrix[c["gold_label"]][c["pred_label"]] += 1
    return {k: dict(v) for k, v in matrix.items()}


def _linguistic_observations(fps: list, fns: list, confusions: list) -> list:
    """
    Generate human-readable observations about the linguistic patterns
    behind the errors. This directly addresses LO1.
    """
    observations = []

    # Ambiguous short tokens (1 word) tend to cause more FPs
    short_fps = [e for e in fps if e.get("span_length", 0) == 1]
    if short_fps:
        examples = list({e["span"] for e in short_fps[:5]})
        observations.append({
            "phenomenon": "Single-token ambiguity",
            "description": (
                f"{len(short_fps)} false positives were single tokens "
                f"(e.g. {examples}). Single-word entities are harder to "
                "classify because context is the only disambiguating signal."
            ),
            "implication": (
                "A model with a wider context window (e.g. en_core_web_trf) "
                "would likely reduce these errors by attending to more context."
            ),
        })

    # Long entity spans (3+ words) tend to cause more FNs
    long_fns = [e for e in fns if e.get("span_length", 0) >= 3]
    if long_fns:
        examples = list({e["span"] for e in long_fns[:5]})
        observations.append({
            "phenomenon": "Long entity boundary detection",
            "description": (
                f"{len(long_fns)} false negatives were multi-word entities "
                f"of 3+ tokens (e.g. {examples}). The model struggles to "
                "identify the correct extent of long named entities."
            ),
            "implication": (
                "More training examples containing long entity spans, "
                "or a CRF output layer, may improve boundary detection."
            ),
        })

    # Label confusions between ORG and LOC
    org_loc = [
        c for c in confusions
        if set([c["gold_label"], c["pred_label"]]) == {"ORG", "LOC"}
    ]
    if org_loc:
        examples = list({c["span"] for c in org_loc[:5]})
        observations.append({
            "phenomenon": "ORG / LOC ambiguity",
            "description": (
                f"{len(org_loc)} entities were confused between ORG and LOC "
                f"(e.g. {examples}). Place names frequently appear as both "
                "entity types depending on context (e.g. 'Chelsea' as a "
                "football club vs a neighbourhood in London)."
            ),
            "implication": (
                "This is a well-known challenge in NER. Wider context and "
                "coreference resolution can help disambiguate these cases."
            ),
        })

    # DATE vs other confusions
    date_conf = [c for c in confusions if "DATE" in [c["gold_label"], c["pred_label"]]]
    if date_conf:
        observations.append({
            "phenomenon": "Temporal expression complexity",
            "description": (
                f"{len(date_conf)} DATE-related confusions. Temporal expressions "
                "range from simple ('2023') to complex ('the third quarter of last "
                "fiscal year'), making consistent annotation and recognition difficult."
            ),
            "implication": (
                "Rule-based post-processing for date patterns (regex) could "
                "supplement the neural model for this entity type."
            ),
        })

    return observations


def _print_analysis(analysis: dict) -> None:
    s = analysis["summary"]
    print(f"\n  Error Summary")
    print(f"  {'Correct':.<30} {s['correct']}")
    print(f"  {'False Positives':.<30} {s['false_positives']}")
    print(f"  {'False Negatives':.<30} {s['false_negatives']}")
    print(f"  {'Label Confusions':.<30} {s['label_confusions']}")

    print(f"\n  False Positives by label: {analysis['false_positives_by_label']}")
    print(f"  False Negatives by label: {analysis['false_negatives_by_label']}")

    print(f"\n  Linguistic Observations:")
    for obs in analysis.get("linguistic_observations", []):
        print(f"\n  [{obs['phenomenon']}]")
        print(f"    {obs['description']}")
        print(f"    --> {obs['implication']}")


def main():
    model_dir = "results/finetuned_model"
    if not Path(model_dir).exists():
        print(f"Model not found at '{model_dir}'. Run pipeline.py or experiment_runner.py first.")
        sys.exit(1)

    from src.annotate import load_annotations
    from src.clean    import clean, split

    nlp        = spacy.load(model_dir)
    data       = load_annotations("data/annotated_data.json")
    data       = clean(data, log_file="results/bad_spans.json")
    _, val_data = split(data)

    analyse_errors(nlp, val_data, output_file="results/error_analysis.json")


if __name__ == "__main__":
    main()
