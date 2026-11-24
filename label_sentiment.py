"""Hand-label the sentiment of the 437 real test messages (blind, resumable, keyboard-driven).

Shows one message at a time - never the model's prediction - and saves after every keypress to
labels/real_test_sentiment_<annotator>.csv. Quit any time and run again to continue.
Read labels/GUIDELINES.md first.

Keys:  n = negative   u = neutral   p = positive   s = skip (unsure)
       b = back (redo the previous message)   g = show guidelines   q = save and quit

Usage (run in a normal terminal so single keypresses work):
    python label_sentiment.py --annotator alex --practice       # 15 made-up messages with feedback, first
    python label_sentiment.py --annotator alex
    python label_sentiment.py --annotator alex --show-intent    # also show the true intent label
    python label_sentiment.py --annotator alex --review-skipped  # go back over skipped messages

Messages are shown in a fixed shuffled order (same for every annotator), so the order doesn't
follow the intent grouping of the source file.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from human_labels import LABELS_DIR, labels_path, real_test_messages

KEYS = {"n": "negative", "u": "neutral", "p": "positive", "s": "skip"}
GUIDELINES = LABELS_DIR / "GUIDELINES.md"
REMINDER = ("Label the FEELING expressed, not the topic:  p = thanks/relief/satisfaction   "
            "n = complaint/worry/frustration   u = everything else (requests, questions, facts, symptoms)")

# Made-up calibration messages (not from the test set): (text, label, rule).
PRACTICE = [
    ("I'd like to schedule a check-up for next month.", "neutral", "a request - no feeling expressed"),
    ("Thank you so much, that really helped!", "positive", "explicit thanks"),
    ("I've had a sore throat for three days.", "neutral", "a factual symptom description"),
    ("I've called three times and still nobody has called me back.", "negative", "frustration/complaint"),
    ("Can you tell me if my plan covers physical therapy?", "neutral", "an information question"),
    ("I'm scared this lump might be something serious.", "negative", "expressed fear"),
    ("My policy number is BC4471920.", "neutral", "giving information"),
    ("Great, Thursday at 10 works perfectly for me.", "positive", "satisfaction ('works perfectly')"),
    ("I have a sharp pain in my lower back when I bend.", "neutral", "severity describes the symptom, not a feeling"),
    ("Why was I charged twice for the same visit?", "negative", "a complaint about a billing problem"),
    ("I need to refill my blood pressure medication.", "neutral", "a request"),
    ("Could you please email me the intake forms? Thanks.", "neutral", "politeness alone is not positive"),
    ("What a relief, the test results came back normal.", "positive", "expressed relief"),
    ("My claim for the MRI was rejected.", "negative", "an unwanted outcome reported as a problem"),
    ("Afternoons would be best, ideally after 2.", "neutral", "a scheduling preference"),
]


def read_key(prompt: str) -> str:
    """One keypress without Enter on a real terminal; falls back to input() otherwise."""
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        return (sys.stdin.readline().strip()[:1] or "").lower()
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print(ch)
    if ch == "\x03":  # Ctrl-C
        raise KeyboardInterrupt
    return ch.lower()


def save(labels: dict[str, dict], path: Path) -> None:
    df = pd.DataFrame(labels.values(), columns=["id", "text", "human_sentiment", "annotator", "labeled_at"])
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written file


def practice() -> None:
    """Made-up messages with immediate feedback; nothing is saved."""
    print(f"\nPractice round: {len(PRACTICE)} made-up messages (not from the test set). Nothing is saved.\n")
    correct = 0
    for i, (text, gold, rule) in enumerate(PRACTICE, 1):
        print("-" * 78)
        print(REMINDER)
        print(f"[practice {i}/{len(PRACTICE)}]\n\n    {text}\n")
        key = ""
        while key not in ("n", "u", "p", "q"):
            key = read_key("n / u / p  (q quit) > ")
        if key == "q":
            break
        ok = KEYS[key] == gold
        correct += ok
        print(f"  {'correct' if ok else 'expected ' + gold.upper()} - {rule}")
    print(f"\nPractice score: {correct}/{len(PRACTICE)}. "
          "Re-read labels/GUIDELINES.md for any misses, then run without --practice.")


def summary(labels: dict[str, dict], msgs: pd.DataFrame) -> None:
    """Distribution by intent - a quick check that labels follow tone, not topic."""
    d = pd.DataFrame(labels.values()).merge(msgs[["id", "intent_label"]], on="id")
    d = d[d["human_sentiment"].isin(["negative", "neutral", "positive"])]
    if d.empty:
        return
    tab = pd.crosstab(d["intent_label"], d["human_sentiment"])
    print("\nYour labels by intent (requests and symptom reports should be mostly neutral):")
    print(tab.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotator", required=True, help="Your name/initials (one file per annotator)")
    parser.add_argument("--show-intent", action="store_true", help="Also show the true intent label")
    parser.add_argument("--review-skipped", action="store_true", help="Only show messages you skipped")
    parser.add_argument("--practice", action="store_true",
                        help="Calibration round on made-up messages with feedback (nothing is saved)")
    args = parser.parse_args()
    if args.practice:
        practice()
        return

    msgs = real_test_messages()
    path = labels_path(args.annotator)
    labels: dict[str, dict] = {}
    if path.exists():
        prev = pd.read_csv(path, dtype=str, keep_default_na=False)
        labels = {r["id"]: r for r in prev.to_dict("records")}

    if args.review_skipped:
        queue = [m for m in msgs.itertuples() if labels.get(m.id, {}).get("human_sentiment") == "skip"]
    else:
        queue = [m for m in msgs.itertuples() if m.id not in labels]
    total = len(msgs)
    print(f"\nAnnotator: {args.annotator}   file: {path}")
    print(f"{total - len([m for m in msgs.itertuples() if m.id not in labels])}/{total} labelled; "
          f"{len(queue)} to go.  Keys: n=negative u=neutral p=positive s=skip b=back g=guidelines q=quit\n")

    history: list[str] = []
    i = 0
    while i < len(queue):
        m = queue[i]
        done = sum(1 for v in labels.values() if v["human_sentiment"] in ("negative", "neutral", "positive"))
        print("-" * 78)
        print(REMINDER)
        print(f"[{done}/{total} labelled]  {m.id}" + (f"   intent: {m.intent_label}" if args.show_intent else ""))
        print(f"\n    {m.text}\n")
        key = read_key("n / u / p / s  (b back, g guidelines, q quit) > ")
        if key == "q":
            break
        if key == "g":
            print(GUIDELINES.read_text() if GUIDELINES.exists() else "(labels/GUIDELINES.md not found)")
            continue
        if key == "b":
            if history:
                prev_id = history.pop()
                labels.pop(prev_id, None)
                save(labels, path)
                i -= 1
                print("  <- back")
            else:
                print("  (nothing to undo)")
            continue
        if key not in KEYS:
            print("  ? use n, u, p, s, b, g or q")
            continue
        labels[m.id] = {"id": m.id, "text": m.text, "human_sentiment": KEYS[key], "annotator": args.annotator,
                        "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        save(labels, path)
        history.append(m.id)
        i += 1

    counts = pd.Series([v["human_sentiment"] for v in labels.values()]).value_counts().to_dict()
    print(f"\nSaved {len(labels)} rows to {path}: {counts}")
    if labels:
        summary(labels, msgs)
    remaining = total - sum(1 for v in labels.values() if v["human_sentiment"] != "skip")
    if remaining:
        print(f"{remaining} still unlabelled or skipped - run again to continue "
              f"(--review-skipped to revisit skips).")
    else:
        print("All messages labelled. Next: python evaluation_report.py")


if __name__ == "__main__":
    main()
