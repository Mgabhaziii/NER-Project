"""

Auto-annotates raw text using spaCy's transformer model (en_core_web_trf),
which provides the highest out-of-the-box accuracy for named entity recognition.

"""

import json
import spacy
from pathlib import Path

# Map spaCy's built-in labels to our project labels
LABEL_MAP = {
    "PERSON": "PER",
    "ORG":    "ORG",
    "GPE":    "LOC",
    "LOC":    "LOC",
    "DATE":   "DATE",
}


def load_text(filepath: str) -> list[str]:
    """Read a text file and return non-empty lines."""
    lines = Path(filepath).read_text(encoding="utf-8", errors="replace").splitlines()
    return [l.strip() for l in lines if l.strip()]


def annotate(input_file: str, output_file: str, batch_size: int = 16) -> list:
    """
    Annotate raw text using en_core_web_trf.

    Processes lines in batches for efficiency (nlp.pipe is significantly
    faster than calling nlp() in a loop on large datasets).

    Args:
        input_file:  Path to raw text file (one sentence/paragraph per line).
        output_file: Path to save annotations as JSON.
        batch_size:  Lines processed per batch.

    Returns:
        List of [text, {"entities": [[start, end, label], ...]}] entries.
.
    """
    print("[annotate] Loading en_core_web_trf ...")
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        raise OSError(
            "Model 'en_core_web_trf' not found. "
            "Run: python -m spacy download en_core_web_trf"
        )

    lines = load_text(input_file)
    print(f"[annotate] Processing {len(lines)} lines from '{input_file}' ...")

    annotated = []
    for doc in nlp.pipe(lines, batch_size=batch_size):
        entities = []
        for ent in doc.ents:
            if ent.label_ in LABEL_MAP:
                entities.append([ent.start_char, ent.end_char, LABEL_MAP[ent.label_]])
        if entities:
            annotated.append([doc.text, {"entities": entities}])

    Path(output_file).write_text(
        json.dumps(annotated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[annotate] Saved {len(annotated)} annotated examples to '{output_file}'")
    return annotated


def load_annotations(filepath: str) -> list:
    """Load previously saved annotations from a JSON file."""
    data = json.loads(Path(filepath).read_text(encoding="utf-8"))
    print(f"[annotate] Loaded {len(data)} examples from '{filepath}'")
    return data
