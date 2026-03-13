"""

Batch inference module. Loads a trained model and runs it on
new, unseen text — either a single string, a list of strings,
or a plain text file.
"""

import json
import spacy
from pathlib import Path


def load_model(model_dir: str) -> spacy.Language:
    """Load a trained spaCy model from disk."""
    try:
        nlp = spacy.load(model_dir)
        print(f"[infer] Model loaded from '{model_dir}'")
        return nlp
    except OSError:
        raise OSError(f"No model found at '{model_dir}'. Run pipeline.py first.")


def predict(nlp: spacy.Language, texts: list[str]) -> list[dict]:
    """
    Run the NER model on a list of texts.

    Args:
        nlp:   Trained spaCy model.
        texts: List of raw text strings.

    Returns:
        List of dicts: {"text": ..., "entities": [{"text", "label", "start", "end"}]}
    """
    results = []
    for doc in nlp.pipe(texts, batch_size=16):
        results.append({
            "text": doc.text,
            "entities": [
                {
                    "text":  ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end":   ent.end_char,
                }
                for ent in doc.ents
            ],
        })
    return results


def predict_file(nlp: spacy.Language, input_file: str, output_file: str) -> list[dict]:
    """
    Run inference on every line of a text file and save results to JSON.

    Args:
        nlp:         Trained model.
        input_file:  Path to plain text file (one sentence per line).
        output_file: Path to save predictions as JSON.

    Returns:
        List of prediction dicts.
    """
    lines = [
        l.strip()
        for l in Path(input_file).read_text(encoding="utf-8", errors="replace").splitlines()
        if l.strip()
    ]
    print(f"[infer] Running inference on {len(lines)} lines from '{input_file}' ...")

    results = predict(nlp, lines)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[infer] Predictions saved to '{output_file}'")
    return results


def print_predictions(results: list[dict]) -> None:
    """Print predictions in a readable format."""
    for r in results:
        print(f"\n  Text: {r['text']}")
        if r["entities"]:
            for ent in r["entities"]:
                print(f"    [{ent['label']}]  {ent['text']!r}")
        else:
            print("    (no entities found)")
