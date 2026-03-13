"""

WHAT THIS MODULE DOES:
    - Defines a comprehensive regex pattern library for DATE expressions
    - Runs after the neural model produces its predictions
    - Adds DATE entities the model missed (false negatives)
    - Avoids overwriting existing entity spans (respects model predictions)
    - Logs all additions so improvements are measurable

PATTERNS COVERED:
    Absolute  — "January 2024", "Jan 12", "12/01/2024", "2024-01-12"
    Relative  — "last year", "next Monday", "two weeks ago", "yesterday"
    Fiscal    — "Q3 2024", "fiscal year 2023", "the third quarter"
    Ranges    — "2020-2023", "January to March", "from 2010 to 2015"
    Ordinal   — "the 1st of March", "21st century"
    Seasonal  — "summer 2023", "winter of 2022"

"""

import re
import spacy
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

# ── DATE Pattern Library ──────────────────────────────────────────────────────

# Month names and abbreviations
_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

# Day ordinals
_DAY = r"(?:[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"

# 4-digit year
_YEAR = r"(?:1[0-9]{3}|20[0-9]{2})"

# Weekdays
_WEEKDAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)"

# Relative anchors
_RELATIVE = r"(?:last|this|next|coming|previous|past|current)"

# Ordinal words
_ORDINAL_WORDS = r"(?:first|second|third|fourth|1st|2nd|3rd|4th)"

DATE_PATTERNS = [
    # ── Absolute dates ────────────────────────────────────────────────────────
    # "January 12, 2024" / "Jan 12 2024" — require year OR day number to anchor
    rf"{_MONTHS}\s+{_DAY}(?:,?\s+{_YEAR})?",
    # "12 January 2024"
    rf"{_DAY}(?:\s+of)?\s+{_MONTHS}(?:,?\s+{_YEAR})?",
    # "January 2024" — month + year (avoids tagging bare month names like "May")
    rf"\b{_MONTHS}\s+{_YEAR}\b",
    # Numeric dates: "12/01/2024", "2024-01-12"
    r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
    r"\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b",
    # Standalone 4-digit year (not preceded by currency symbols)
    rf"(?<![£$€\d]){_YEAR}(?!\d)",

    # ── Relative — only with explicit time-unit anchor ────────────────────────
    rf"\b{_RELATIVE}\s+(?:year|month|week|quarter|decade|century)\b",
    rf"\b{_RELATIVE}\s+{_WEEKDAY}\b",
    rf"\b{_RELATIVE}\s+{_MONTHS}\b",
    r"\b(?:yesterday|today|tomorrow)\b",
    # "three months ago", "two years ago" — explicit count required
    r"\b(?:a\s+few|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:day|week|month|year|decade)s?\s+ago\b",

    # ── Weekdays — only with explicit temporal preposition ────────────────────
    rf"\bon\s+{_WEEKDAY}\b",
    rf"\b(?:last|next|this)\s+{_WEEKDAY}\b",

    # ── Fiscal / quarterly — require year for precision ───────────────────────
    r"\bQ[1-4]\s+(?:FY\s*)?" + _YEAR + r"\b",
    r"\bQ[1-4]" + _YEAR + r"\b",
    rf"\b(?:fiscal\s+year|FY)\s*{_YEAR}\b",
    # "the third quarter of 2023" — require year
    rf"\bthe\s+{_ORDINAL_WORDS}\s+quarter\s+of\s+{_YEAR}\b",
    rf"\b{_ORDINAL_WORDS}\s+quarter\s+of\s+{_YEAR}\b",

    # ── Date ranges ───────────────────────────────────────────────────────────
    rf"\b{_YEAR}\s*(?:–|to)\s*{_YEAR}\b",
    # "January to March 2025" — must have year
    rf"\b{_MONTHS}\s+(?:to|through|until)\s+{_MONTHS}\s+{_YEAR}\b",

    # ── Seasonal — only with year to avoid bare "Summer" FPs ─────────────────
    rf"\b(?:spring|summer|autumn|fall|winter)(?:\s+of)?\s+{_YEAR}\b",

    # ── Decades / centuries ───────────────────────────────────────────────────
    r"\bthe\s+(?:19|20)\d0s\b",
    r"\b(?:\d+(?:st|nd|rd|th)\s+century)\b",

    # ── Time periods ──────────────────────────────────────────────────────────
    rf"\bmid-{_YEAR}\b",
    rf"\b(?:early|late)\s+{_YEAR}s?\b",
    rf"\b(?:early|late)\s+(?:19|20)\d0s\b",
    r"\bin\s+recent\s+(?:years|months|decades)\b",
    r"\b(?:over|during|in)\s+the\s+past\s+(?:\w+\s+)?(?:year|month|decade)s?\b",
    rf"\bsince\s+{_YEAR}\b",
    rf"\bby\s+the\s+(?:end|close)\s+of\s+{_YEAR}\b",
]


# Compile all patterns (case-sensitive for month names, weekdays)
_COMPILED = [re.compile(p) for p in DATE_PATTERNS]

# ── Core function ─────────────────────────────────────────────────────────────

def apply_date_rules(doc: Doc, verbose: bool = False) -> Doc:
    """
    Apply regex DATE rules to a spaCy Doc.

    Finds DATE spans missed by the neural model and adds them to doc.ents.
    Existing entity spans are preserved — regex only fills gaps.

    Args:
        doc:     A spaCy Doc that has already been processed by the NER model.
        verbose: If True, print each addition to stdout.

    Returns:
        The same Doc with updated ents.
    """
    text = doc.text
    new_spans = list(doc.ents)  # start with existing neural predictions

    # Track character ranges already covered by existing entities
    covered = set()
    for ent in doc.ents:
        covered.update(range(ent.start_char, ent.end_char))

    additions = 0
    for pattern in _COMPILED:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            # Skip if this span overlaps any existing entity
            match_range = set(range(start, end))
            if match_range & covered:
                continue

            # Create a span
            span = doc.char_span(start, end, label="DATE", alignment_mode="contract")
            if span is None:
                continue

            new_spans.append(span)
            covered.update(range(span.start_char, span.end_char))
            additions += 1

            if verbose:
                print(f"  [DATE rule] added '{span.text}' at chars {start}–{end}")

    # Resolve any remaining overlaps, keep longest
    doc.ents = filter_spans(new_spans)
    return doc


def batch_apply_date_rules(nlp: spacy.Language, texts: list, verbose: bool = False) -> list:
    """
    Apply the NER model + DATE rules to a list of texts.

    Returns a list of processed Docs.
    """
    docs = []
    for doc in nlp.pipe(texts, batch_size=32):
        docs.append(apply_date_rules(doc, verbose=verbose))
    return docs


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_postprocess_gain(nlp: spacy.Language, val_data: list) -> dict:
    """
    Compare model performance before and after applying DATE rules.

    Returns a dict with before/after precision, recall, F1 for DATE entities
    and overall, so the improvement is quantifiable.

    Args:
        nlp:       Trained spaCy model.
        val_data:  List of [text, {"entities": [...]}] pairs.

    Returns:
        {
          "before": {"date_f1": ..., "overall_f1": ...},
          "after":  {"date_f1": ..., "overall_f1": ...},
          "gain":   {"date_f1": ..., "overall_f1": ...},
          "date_additions": int,   # total DATE spans added by rules
          "examples_improved": int # examples where at least 1 DATE was added
        }
    """
    from spacy.training.example import Example

    def _score(predictions, data):
        """Compute precision, recall, F1 from a list of (predicted_doc, gold_entities)."""
        tp = fp = fn = 0
        tp_date = fp_date = fn_date = 0

        for doc, (_, annot) in zip(predictions, data):
            gold_ents = {(s, e, l) for s, e, l in annot["entities"]}
            pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}

            tp += len(pred_ents & gold_ents)
            fp += len(pred_ents - gold_ents)
            fn += len(gold_ents - pred_ents)

            gold_date = {(s, e, l) for s, e, l in gold_ents if l == "DATE"}
            pred_date = {(s, e, l) for s, e, l in pred_ents if l == "DATE"}
            tp_date += len(pred_date & gold_date)
            fp_date += len(pred_date - gold_date)
            fn_date += len(gold_date - pred_date)

        def _f1(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return round(p, 3), round(r, 3), round(f, 3)

        p, r, f = _f1(tp, fp, fn)
        pd, rd, fd = _f1(tp_date, fp_date, fn_date)
        return {"overall_p": p, "overall_r": r, "overall_f1": f,
                "date_p": pd, "date_r": rd, "date_f1": fd}

    texts = [text for text, _ in val_data]

    # Before: raw model predictions
    before_docs = list(nlp.pipe(texts, batch_size=32))

    # After: model + DATE rules
    after_docs = [apply_date_rules(nlp(text)) for text in texts]

    before_scores = _score(before_docs, val_data)
    after_scores  = _score(after_docs,  val_data)

    # Count additions
    total_additions = 0
    examples_improved = 0
    for before_doc, after_doc in zip(before_docs, after_docs):
        added = len(after_doc.ents) - len(before_doc.ents)
        if added > 0:
            total_additions += added
            examples_improved += 1

    gain = {
        "overall_f1": round(after_scores["overall_f1"] - before_scores["overall_f1"], 3),
        "date_f1":    round(after_scores["date_f1"]    - before_scores["date_f1"],    3),
    }

    return {
        "before":             before_scores,
        "after":              after_scores,
        "gain":               gain,
        "date_additions":     total_additions,
        "examples_improved":  examples_improved,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    ROOT           = Path(__file__).parent.parent
    ANNOTATED_FILE = ROOT / "data" / "annotated_data.json"
    MODEL_PATH     = ROOT / "results" / "finetuned_model"

    print("\n" + "=" * 60)
    print("  DATE Post-Processing — Improvement #2")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"  ERROR: No trained model found at {MODEL_PATH}")
        print(f"  Run pipeline.py first.")
        sys.exit(1)

    print(f"\n  Loading model from {MODEL_PATH} ...")
    nlp = spacy.load(str(MODEL_PATH))

    print(f"  Loading validation data ...")
    data = json.loads(ANNOTATED_FILE.read_text(encoding="utf-8"))
    _, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"  Validation examples: {len(val_data)}")

    print(f"\n  Evaluating before/after DATE rules ...")
    stats = evaluate_postprocess_gain(nlp, val_data)

    print(f"\n  {'Metric':<20} {'Before':>8} {'After':>8} {'Gain':>8}")
    print(f"  {'-' * 48}")
    print(f"  {'Overall F1':<20} {stats['before']['overall_f1']:>8.3f} "
          f"{stats['after']['overall_f1']:>8.3f} "
          f"{'+' if stats['gain']['overall_f1'] >= 0 else ''}"
          f"{stats['gain']['overall_f1']:>7.3f}")
    print(f"  {'DATE F1':<20} {stats['before']['date_f1']:>8.3f} "
          f"{stats['after']['date_f1']:>8.3f} "
          f"{'+' if stats['gain']['date_f1'] >= 0 else ''}"
          f"{stats['gain']['date_f1']:>7.3f}")
    print(f"\n  DATE spans added by rules : {stats['date_additions']}")
    print(f"  Examples improved         : {stats['examples_improved']}")

    # Demo on sample sentences
    print(f"\n  {'─' * 60}")
    print(f"  Demo on sample sentences:")
    print(f"  {'─' * 60}")
    samples = [
        "Tesla reported record revenue in Q3 2024, up from fiscal year 2022.",
        "The summit was held in the third quarter of last year.",
        "She joined the company two years ago and left in mid-2023.",
        "Reports from the 1990s show growth through the early 2000s.",
        "The contract runs from January to March 2025.",
    ]
    for text in samples:
        doc = apply_date_rules(nlp(text))
        print(f"\n  '{text}'")
        dates = [f"'{e.text}'" for e in doc.ents if e.label_ == "DATE"]
        others = [f"[{e.label_}] '{e.text}'" for e in doc.ents if e.label_ != "DATE"]
        if dates:
            print(f"    DATE  : {', '.join(dates)}")
        if others:
            print(f"    Other : {', '.join(others)}")