"""

WHAT THIS SCRIPT DOES:
    - Loads the validation split
    - Shows each example with its auto-annotated entities highlighted
    - Lets you ACCEPT, EDIT, or SKIP each example interactively
    - Saves verified examples to data/verified_val.json
    - Tracks correction statistics so you can document annotation quality

HOW TO RUN:
    python tools/verify_labels.py

CONTROLS:
    Press ENTER        — accept the example as-is
    Press 's'          — skip this example (exclude from verified set)
    Press 'e'          — edit entities manually
    Press 'q'          — quit and save progress (resume later)

RESUMING:
    Already-verified examples are saved incrementally. Re-running the script
    skips examples you have already reviewed.
"""

import json
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
ANNOTATED_FILE  = ROOT / "data" / "annotated_data.json"
VERIFIED_FILE   = ROOT / "data" / "verified_val.json"
STATS_FILE      = ROOT / "data" / "verification_stats.json"
VAL_SPLIT       = 0.2
RANDOM_SEED     = 42

VALID_LABELS = {"PER", "ORG", "LOC", "DATE"}

# ── Colours for terminal output ───────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

LABEL_COLOURS = {
    "PER":  "\033[94m",   # blue
    "ORG":  "\033[93m",   # yellow
    "LOC":  "\033[92m",   # green
    "DATE": "\033[95m",   # magenta
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def highlight(text: str, entities: list) -> str:
    """Return text with entity spans highlighted using ANSI colours."""
    if not entities:
        return text

    # Sort by start position
    sorted_ents = sorted(entities, key=lambda e: e[0])
    result = ""
    prev = 0
    for start, end, label in sorted_ents:
        colour = LABEL_COLOURS.get(label, CYAN)
        result += text[prev:start]
        result += f"{colour}{BOLD}[{label}]{RESET}{colour}{text[start:end]}{RESET}"
        prev = end
    result += text[prev:]
    return result


def display_example(idx: int, total: int, text: str, entities: list):
    """Print a single example with highlighted entities."""
    print(f"\n{'─' * 70}")
    print(f"  {DIM}Example {idx + 1} of {total}{RESET}")
    print(f"{'─' * 70}")
    print(f"\n  {highlight(text, entities)}\n")

    if entities:
        print(f"  {DIM}Entities:{RESET}")
        for start, end, label in sorted(entities, key=lambda e: e[0]):
            colour = LABEL_COLOURS.get(label, CYAN)
            print(f"    {colour}{label:<6}{RESET}  '{text[start:end]}'  "
                  f"{DIM}(chars {start}–{end}){RESET}")
    else:
        print(f"  {DIM}No entities.{RESET}")
    print()


def prompt_action() -> str:
    """Prompt the user for an action and return it."""
    while True:
        raw = input(
            f"  {GREEN}[ENTER]{RESET} accept  "
            f"{YELLOW}[e]{RESET} edit  "
            f"{RED}[s]{RESET} skip  "
            f"{DIM}[q]{RESET} quit  > "
        ).strip().lower()
        if raw in ("", "e", "s", "q"):
            return raw
        print(f"  {RED}Invalid input. Press ENTER, e, s, or q.{RESET}")


def edit_entities(text: str, current_entities: list) -> list:
    """
    Interactive entity editor.
    Displays current entities and lets the user add, remove, or replace them.
    Returns the updated entity list.
    """
    print(f"\n  {CYAN}Text:{RESET} {text}")
    print(f"\n  {CYAN}Current entities:{RESET}")

    entities = list(current_entities)

    if not entities:
        print(f"    {DIM}None{RESET}")
    else:
        for i, (start, end, label) in enumerate(entities):
            print(f"    [{i}]  {label:<6}  '{text[start:end]}'  "
                  f"{DIM}(chars {start}–{end}){RESET}")

    print(f"\n  {DIM}Commands:{RESET}")
    print(f"  {DIM}  add <start> <end> <LABEL>   — e.g.  add 10 20 ORG{RESET}")
    print(f"  {DIM}  remove <index>               — e.g.  remove 2{RESET}")
    print(f"  {DIM}  clear                        — remove all entities{RESET}")
    print(f"  {DIM}  done                         — finish editing{RESET}")
    print(f"  {DIM}  show                         — re-display text and entities{RESET}")
    print()

    while True:
        raw = input(f"  {CYAN}edit>{RESET} ").strip()
        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "done":
            break

        elif cmd == "show":
            print(f"\n  {CYAN}Text:{RESET} {highlight(text, entities)}\n")
            for i, (s, e, l) in enumerate(entities):
                print(f"    [{i}]  {l:<6}  '{text[s:e]}'")
            print()

        elif cmd == "clear":
            entities = []
            print(f"  {GREEN}All entities removed.{RESET}")

        elif cmd == "add":
            if len(parts) != 4:
                print(f"  {RED}Usage: add <start> <end> <LABEL>{RESET}")
                continue
            try:
                start, end = int(parts[1]), int(parts[2])
                label = parts[3].upper()
            except ValueError:
                print(f"  {RED}start and end must be integers.{RESET}")
                continue

            if label not in VALID_LABELS:
                print(f"  {RED}Label must be one of: {', '.join(VALID_LABELS)}{RESET}")
                continue
            if start < 0 or end > len(text) or start >= end:
                print(f"  {RED}Invalid span: text length is {len(text)}{RESET}")
                continue

            entities.append([start, end, label])
            entities.sort(key=lambda e: e[0])
            print(f"  {GREEN}Added [{label}] '{text[start:end]}'{RESET}")

        elif cmd == "remove":
            if len(parts) != 2:
                print(f"  {RED}Usage: remove <index>{RESET}")
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                print(f"  {RED}Index must be an integer.{RESET}")
                continue
            if idx < 0 or idx >= len(entities):
                print(f"  {RED}Index out of range (0–{len(entities)-1}){RESET}")
                continue
            removed = entities.pop(idx)
            print(f"  {GREEN}Removed [{removed[2]}] '{text[removed[0]:removed[1]]}'{RESET}")

        else:
            print(f"  {RED}Unknown command. Use: add, remove, clear, show, done{RESET}")

    return entities


def load_val_split() -> list:
    """Load annotated data and return the validation split."""
    from sklearn.model_selection import train_test_split
    import random
    random.seed(RANDOM_SEED)

    data = json.loads(ANNOTATED_FILE.read_text(encoding="utf-8"))
    _, val = train_test_split(data, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
    return val


def load_progress() -> tuple[list, set]:
    """Load already-verified examples and return (verified_list, verified_texts)."""
    if VERIFIED_FILE.exists():
        verified = json.loads(VERIFIED_FILE.read_text(encoding="utf-8"))
        verified_texts = {ex["text"] for ex in verified}
        return verified, verified_texts
    return [], set()


def save_progress(verified: list, stats: dict):
    """Save verified examples and statistics to disk."""
    VERIFIED_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERIFIED_FILE.write_text(
        json.dumps(verified, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    STATS_FILE.write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )


def print_summary(stats: dict, verified: list):
    """Print a summary of the verification session."""
    total_reviewed = stats["accepted"] + stats["edited"] + stats["skipped"]
    print(f"\n{'═' * 70}")
    print(f"  {BOLD}Verification Summary{RESET}")
    print(f"{'═' * 70}")
    print(f"  Examples reviewed this session : {total_reviewed}")
    print(f"  {GREEN}Accepted (no changes)          : {stats['accepted']}{RESET}")
    print(f"  {YELLOW}Edited                         : {stats['edited']}{RESET}")
    print(f"  {RED}Skipped                        : {stats['skipped']}{RESET}")
    print(f"  Total verified examples saved  : {len(verified)}")
    if stats["edited"] > 0:
        edit_rate = stats["edited"] / max(total_reviewed, 1) * 100
        print(f"\n  {CYAN}Annotation error rate          : {edit_rate:.1f}%{RESET}")
        print(f"  {DIM}(This is the % of auto-annotations that needed correction){RESET}")
    print(f"\n  Saved to: {VERIFIED_FILE}")
    print(f"  Stats  : {STATS_FILE}")
    print(f"{'═' * 70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═' * 70}")
    print(f"  {BOLD}NER Label Verification Tool{RESET}")
    print(f"  Phase 1 — Human Verification of Validation Set")
    print(f"{'═' * 70}")
    print(f"\n  This tool loads your validation split and lets you review")
    print(f"  each auto-annotated example. Corrections are saved to:")
    print(f"  {CYAN}data/verified_val.json{RESET}")
    print(f"\n  {DIM}Tip: You don't need to review all examples in one session.")
    print(f"  Press 'q' to quit and resume later.{RESET}\n")

    # Load data
    print("  Loading validation split ...")
    val_data = load_val_split()
    verified, verified_texts = load_progress()

    # Filter out already-reviewed examples
    remaining = [ex for ex in val_data if ex[0] not in verified_texts]

    print(f"  Validation set        : {len(val_data)} examples")
    print(f"  Already verified      : {len(verified)}")
    print(f"  Remaining to review   : {len(remaining)}")

    if not remaining:
        print(f"\n  {GREEN}All examples already verified!{RESET}")
        print(f"  Verified set saved at: {VERIFIED_FILE}")
        return

    input(f"\n  Press ENTER to start reviewing ...")

    # Session stats
    stats = {"accepted": 0, "edited": 0, "skipped": 0}

    for i, (text, annot) in enumerate(remaining):
        entities = annot.get("entities", [])

        display_example(
            idx=len(verified) + i,
            total=len(val_data),
            text=text,
            entities=entities
        )

        action = prompt_action()

        if action == "q":
            print(f"\n  {YELLOW}Quitting and saving progress ...{RESET}")
            save_progress(verified, stats)
            print_summary(stats, verified)
            sys.exit(0)

        elif action == "s":
            stats["skipped"] += 1
            print(f"  {RED}Skipped.{RESET}")

        elif action == "e":
            new_entities = edit_entities(text, entities)
            verified.append({
                "text": text,
                "entities": new_entities,
                "verified": True,
                "was_edited": True,
                "original_entities": entities
            })
            stats["edited"] += 1
            print(f"  {GREEN}Saved with edits.{RESET}")

        else:  # ENTER — accept
            verified.append({
                "text": text,
                "entities": entities,
                "verified": True,
                "was_edited": False,
                "original_entities": entities
            })
            stats["accepted"] += 1
            print(f"  {GREEN}Accepted.{RESET}")

        # Save after every example so progress is never lost
        save_progress(verified, stats)

    print(f"\n  {GREEN}All remaining examples reviewed!{RESET}")
    print_summary(stats, verified)


if __name__ == "__main__":
    main()