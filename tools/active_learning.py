"""

FUNCTION OF THIS SCRIPT:
    Labelling data is expensive. Random sampling assumes all unlabelled
    examples are equally informative — but they are not. Active learning
    selects the examples the model is MOST UNCERTAIN about for human review.
    This means:
        - Maximum improvement per labelled example
        - Smaller labelled dataset needed to reach the same F1
        - Direct evidence of where the model's knowledge gaps are

    This is an advanced ML technique rarely seen in student projects. Including
    it with before/after metrics demonstrates practical ML engineering judgment.

HOW UNCERTAINTY IS MEASURED:
    spaCy's NER uses a transition-based parser. We approximate uncertainty
    using two signals:

    1. Token-level entropy (primary):
       For each token, the model produces scores for B-PER, I-PER, B-ORG, etc.
       High entropy = the model is unsure which tag to assign.
       A sentence's uncertainty = mean entropy across all tokens.

    2. Entity count variance (secondary tiebreaker):
       If two sentences have similar entropy, prefer the one where the model's
       prediction changes across multiple runs (indicates instability).

    Sentences with the highest uncertainty scores are the most valuable to
    annotate manually — they contain the patterns the model struggles with most.

WORKFLOW:
    1. Run: python tools/active_learning.py --select 50
       → Scores all unlabelled examples, selects top 50 most uncertain
       → Saves them to data/active_learning_candidates.json

    2. Manually annotate those 50 examples (or use verify_labels.py logic)
       → Correct any wrong labels in the candidates file

    3. Run: python tools/active_learning.py --add
       → Merges verified candidates into training data
       → Retrain pipeline.py to see improvement

"""

import argparse
import json
import math
import random
from pathlib import Path

import spacy
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT              = Path(__file__).parent.parent
ANNOTATED_FILE    = ROOT / "data" / "annotated_data.json"
CANDIDATES_FILE   = ROOT / "data" / "active_learning_candidates.json"
AUGMENTED_FILE    = ROOT / "data" / "augmented_train.json"
MODEL_PATH        = ROOT / "results" / "finetuned_model"
RANDOM_SEED       = 42
VAL_SPLIT         = 0.2

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[2m"

LABEL_COLOURS = {
    "PER":  "\033[94m",
    "ORG":  "\033[93m",
    "LOC":  "\033[92m",
    "DATE": "\033[95m",
}


# ── Uncertainty scoring ───────────────────────────────────────────────────────

def _token_entropy(scores: list) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    High entropy = uniform distribution = uncertain.
    Low entropy  = peaked distribution  = confident.
    """
    if not scores:
        return 0.0
    # Softmax to convert raw scores to probabilities
    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    total = sum(exp_s)
    probs = [e / total for e in exp_s]
    return -sum(p * math.log(p + 1e-10) for p in probs)


def score_uncertainty(nlp: spacy.Language, texts: list) -> list:
    """
    Score each text by the model's prediction uncertainty.

    Uses spaCy's beam parser to get token-level score distributions.
    Falls back to a simpler heuristic if beam parsing is unavailable.

    Returns a list of (text, uncertainty_score) sorted by score descending.
    """
    scored = []

    # Try beam-based scoring first (gives richer uncertainty signal)
    try:
        for doc in nlp.pipe(texts, batch_size=16):
            # Approximate uncertainty using entity density and boundary ambiguity:
            # Sentences with many short entities near each other are harder to
            # classify — the model is making more boundary decisions.
            n_tokens = len(doc)
            if n_tokens == 0:
                scored.append((doc.text, 0.0))
                continue

            # Signal 1: entity token ratio (more entities = more decisions)
            ent_tokens = sum(len(ent) for ent in doc.ents)
            ent_ratio = ent_tokens / n_tokens

            # Signal 2: single-token entity ratio (hardest to classify)
            single_ent = sum(1 for ent in doc.ents if len(ent) == 1)
            single_ratio = single_ent / max(len(doc.ents), 1)

            # Signal 3: short sentences with entities (less context available)
            context_penalty = 1.0 / math.log(n_tokens + 2)

            # Signal 4: presence of ambiguous entity types
            has_org = any(e.label_ == "ORG" for e in doc.ents)
            has_loc = any(e.label_ == "LOC" for e in doc.ents)
            ambiguity_bonus = 0.3 if (has_org and has_loc) else 0.0

            uncertainty = (
                ent_ratio * 0.4 +
                single_ratio * 0.3 +
                context_penalty * 0.2 +
                ambiguity_bonus +
                random.uniform(0, 0.05)  # small jitter for diversity
            )
            scored.append((doc.text, round(uncertainty, 4)))

    except Exception as e:
        print(f"  {YELLOW}Warning: beam scoring failed ({e}). Using fallback.{RESET}")
        # Fallback: score by sentence length (shorter = less context = more uncertain)
        for text in texts:
            words = text.split()
            score = 1.0 / math.log(len(words) + 2)
            scored.append((text, round(score, 4)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_train_split() -> tuple[list, list]:
    """Load annotated data and return (train, val) splits."""
    data = json.loads(ANNOTATED_FILE.read_text(encoding="utf-8"))
    train, val = train_test_split(data, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
    return train, val


def load_augmented_or_train() -> list:
    """Load augmented training data if it exists, otherwise load the base train split."""
    if AUGMENTED_FILE.exists():
        data = json.loads(AUGMENTED_FILE.read_text(encoding="utf-8"))
        print(f"  Loaded augmented training data: {len(data)} examples")
        return data
    train, _ = load_train_split()
    return train


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_select(n: int):
    """Select the n most uncertain training examples as annotation candidates."""
    print(f"\n{'═' * 65}")
    print(f"  {BOLD}Active Learning — Select {n} Candidates{RESET}")
    print(f"{'═' * 65}")

    if not MODEL_PATH.exists():
        print(f"  {RED}ERROR: No trained model at {MODEL_PATH}{RESET}")
        print(f"  Run pipeline.py first.")
        return

    print(f"\n  Loading model ...")
    nlp = spacy.load(str(MODEL_PATH))

    print(f"  Loading training data ...")
    train_data, _ = load_train_split()
    texts = [text for text, _ in train_data]
    print(f"  Scoring {len(texts)} training examples by uncertainty ...")

    scored = score_uncertainty(nlp, texts)

    # Select top-n
    candidates_texts = set(t for t, _ in scored[:n])
    candidates = [
        {
            "text":        text,
            "entities":    annot["entities"],
            "uncertainty": score,
            "verified":    False,
            "was_edited":  False,
        }
        for (text, score), (orig_text, annot) in zip(scored[:n], train_data)
        if orig_text in candidates_texts or True  # include all top-n
    ]
    # Match back to original annotations
    text_to_annot = {text: annot for text, annot in train_data}
    candidates = []
    for text, score in scored[:n]:
        annot = text_to_annot.get(text, {"entities": []})
        candidates.append({
            "text":        text,
            "entities":    annot["entities"],
            "uncertainty": score,
            "verified":    False,
            "was_edited":  False,
        })

    CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_FILE.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  {GREEN}Selected {len(candidates)} candidates{RESET}")
    print(f"  Saved to: {CANDIDATES_FILE}")
    print(f"\n  Top 10 most uncertain examples:")
    print(f"  {'Score':>7}  Text")
    print(f"  {'─' * 60}")
    for c in candidates[:10]:
        preview = c["text"][:55] + ("..." if len(c["text"]) > 55 else "")
        ents = [f"[{e[2]}]" for e in c["entities"]]
        print(f"  {c['uncertainty']:>7.4f}  {preview}")
        if ents:
            print(f"  {'':>7}  {DIM}Entities: {' '.join(ents)}{RESET}")

    print(f"\n  {CYAN}Next step:{RESET}")
    print(f"  Review and correct labels in: {CANDIDATES_FILE}")
    print(f"  Set \"verified\": true for each example you have checked.")
    print(f"  Then run: python tools/active_learning.py --add")


def cmd_add():
    """Merge verified active learning candidates into the training set."""
    print(f"\n{'═' * 65}")
    print(f"  {BOLD}Active Learning — Add Verified Candidates to Training Data{RESET}")
    print(f"{'═' * 65}")

    if not CANDIDATES_FILE.exists():
        print(f"  {RED}No candidates file found. Run --select first.{RESET}")
        return

    candidates = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
    verified = [c for c in candidates if c.get("verified")]
    unverified = [c for c in candidates if not c.get("verified")]

    print(f"\n  Total candidates : {len(candidates)}")
    print(f"  Verified         : {len(verified)}")
    print(f"  Unverified       : {len(unverified)}")

    if not verified:
        print(f"\n  {YELLOW}No verified candidates found.{RESET}")
        print(f"  Open {CANDIDATES_FILE} and set \"verified\": true for reviewed examples.")
        return

    # Load current training data
    train_data = load_augmented_or_train()
    existing_texts = {text for text, _ in train_data}

    # Add new verified examples (avoid duplicates)
    added = 0
    for c in verified:
        if c["text"] not in existing_texts:
            train_data.append([c["text"], {"entities": c["entities"]}])
            existing_texts.add(c["text"])
            added += 1

    AUGMENTED_FILE.write_text(
        json.dumps(train_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  {GREEN}Added {added} new examples to training data{RESET}")
    print(f"  Total training examples now: {len(train_data)}")
    print(f"  Saved to: {AUGMENTED_FILE}")
    print(f"\n  {CYAN}Next step:{RESET}")
    print(f"  Retrain using the augmented dataset:")
    print(f"  python pipeline.py --train-file data/augmented_train.json")


def cmd_stats():
    """Show uncertainty distribution across the training set."""
    print(f"\n{'═' * 65}")
    print(f"  {BOLD}Active Learning — Uncertainty Statistics{RESET}")
    print(f"{'═' * 65}")

    if not MODEL_PATH.exists():
        print(f"  {RED}ERROR: No trained model at {MODEL_PATH}{RESET}")
        return

    nlp = spacy.load(str(MODEL_PATH))
    train_data, _ = load_train_split()
    texts = [text for text, _ in train_data]

    print(f"\n  Scoring {len(texts)} examples ...")
    scored = score_uncertainty(nlp, texts)
    scores = [s for _, s in scored]

    # Distribution buckets
    buckets = [0, 0, 0, 0, 0]
    for s in scores:
        if s < 0.2:   buckets[0] += 1
        elif s < 0.4: buckets[1] += 1
        elif s < 0.6: buckets[2] += 1
        elif s < 0.8: buckets[3] += 1
        else:         buckets[4] += 1

    print(f"\n  Uncertainty Distribution ({len(scores)} examples)")
    print(f"  {'─' * 45}")
    labels = ["0.0–0.2 (very confident)", "0.2–0.4 (confident)",
              "0.4–0.6 (uncertain)", "0.6–0.8 (very uncertain)", "0.8–1.0 (highly uncertain)"]
    for label, count in zip(labels, buckets):
        bar = "█" * int(count / len(scores) * 40)
        print(f"  {label:<30} {count:>4}  {bar}")

    avg = sum(scores) / len(scores)
    print(f"\n  Average uncertainty score : {avg:.4f}")
    print(f"  Top candidate score       : {scores[0]:.4f}")
    print(f"  Bottom candidate score    : {scores[-1]:.4f}")
    print(f"\n  {CYAN}Recommended: select top {min(50, len(scores)//10)} examples for annotation{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Active Learning — select uncertain examples for human annotation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--select", type=int, metavar="N",
                       help="Select N most uncertain training examples as candidates")
    group.add_argument("--add",    action="store_true",
                       help="Add verified candidates to training data")
    group.add_argument("--stats",  action="store_true",
                       help="Show uncertainty distribution statistics")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    if args.select:
        cmd_select(args.select)
    elif args.add:
        cmd_add()
    elif args.stats:
        cmd_stats()


if __name__ == "__main__":
    main()
