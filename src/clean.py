"""
Validates and cleans annotated data before training.

Key linguistic issues handled:
    1. Misaligned spans — entity boundaries that do not align with spaCy
       token boundaries (e.g. mid-word matches caused by punctuation or
       hyphenated compounds like "New York-based"). alignment_mode="contract"
       shrinks the span to the nearest valid token boundary.

    2. Overlapping spans — when two entities share characters (e.g. a PER
       span nested inside an ORG span). filter_spans() keeps the longest
       span, which is the standard resolution strategy.

    3. Empty or whitespace-only spans — these arise from tokenisation
       differences between the annotator model and the training model.

"""

import json
import spacy
from pathlib import Path
from spacy.util import filter_spans


def clean(
    data: list,
    log_file: str = "results/bad_spans.json",
) -> list:
    """
    Validate and clean annotated data.

    Args:
        data:     List of [text, {"entities": [...]}] entries.
        log_file: Path to save rejected spans for review.

    Returns:
        Cleaned list with the same structure as input.
    """
    print("[clean] Validating spans ...")
    nlp_blank = spacy.blank("en")
    cleaned   = []
    bad_spans = []

    for text, annot in data:
        doc  = nlp_blank.make_doc(text)
        ents = []

        for start, end, label in annot["entities"]:
            # Guard against out-of-range indices
            if start < 0 or end > len(text) or start >= end:
                bad_spans.append({
                    "text": text,
                    "span": text[max(0, start):end],
                    "reason": "invalid offsets",
                    "label": label,
                })
                continue

            # Skip whitespace-only spans
            span_text = text[start:end].strip()
            if not span_text:
                bad_spans.append({
                    "text": text,
                    "span": repr(text[start:end]),
                    "reason": "whitespace only",
                    "label": label,
                })
                continue

            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                bad_spans.append({
                    "text": text,
                    "span": text[start:end],
                    "reason": "misaligned token boundary",
                    "label": label,
                })
                continue

            ents.append(span)

        # Resolve overlapping spans — keep longest
        filtered = filter_spans(ents)
        removed  = len(ents) - len(filtered)
        if removed > 0:
            bad_spans.append({
                "text": text,
                "span": "",
                "reason": f"{removed} overlapping span(s) removed",
                "label": "multiple",
            })

        if filtered:
            cleaned.append([
                text,
                {"entities": [[s.start_char, s.end_char, s.label_] for s in filtered]}
            ])

    # Save bad spans log
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(log_file).write_text(
        json.dumps(bad_spans, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[clean] {len(data) - len(cleaned)} examples removed")
    print(f"[clean] {len(bad_spans)} bad spans logged to '{log_file}'")
    print(f"[clean] {len(cleaned)} clean examples remaining")
    return cleaned


def split(data: list, test_size: float = 0.2, seed: int = 42):
    """
    Reproducible train/validation split.

    Args:
        data:      Cleaned annotated data.
        test_size: Fraction to use as validation set.
        seed:      Random seed for reproducibility.

    Returns:
        (train_data, val_data) tuple.
    """
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(data, test_size=test_size, random_state=seed)
    print(f"[clean] Split: {len(train)} train / {len(val)} validation")
    _label_distribution("  Train", train)
    _label_distribution("  Val  ", val)
    return train, val


def _label_distribution(name: str, data: list):
    """Print entity label counts for a dataset split."""
    counts: dict = {}
    for _, annot in data:
        for _, _, label in annot["entities"]:
            counts[label] = counts.get(label, 0) + 1
    dist = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    print(f"[clean] {name} — {dist}")
