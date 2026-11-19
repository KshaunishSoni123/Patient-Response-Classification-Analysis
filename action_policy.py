"""Action policy: (intent, sentiment) -> action, plus safety overrides.

The table below is the single source of truth for what the platform does with a
patient message. It is applied in two places:
  * Training data: `python action_policy.py` joins the cleaned intents with the
    sentiment-head output and writes data/processed/healthcare_actions.csv with
    `group` and `action` for every message (the reference action used in evaluation).
  * Inference: classify.py runs the intent + sentiment models and calls decide().

Safety rules applied at inference (on top of the table):
  1. Emergency override: if P(emergency) >= EMERGENCY_THRESHOLD, the action is
     route_to_human_emergency regardless of the top intent. A missed emergency is
     far costlier than a false alarm.
  2. Low-confidence fallback: if the top intent probability < LOW_CONFIDENCE_THRESHOLD
     and the table would pick an automated action, route_to_human instead.

Usage:
    python action_policy.py          # validate table, rebuild healthcare_actions.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTENTS_CSV = ROOT / "data" / "processed" / "healthcare_intents.csv"
SENTIMENT_CSV = ROOT / "data" / "processed" / "healthcare_sentiment.csv"
ACTIONS_CSV = ROOT / "data" / "processed" / "healthcare_actions.csv"

SENTIMENTS = ("negative", "neutral", "positive")
ALL = SENTIMENTS

EMERGENCY_INTENT = "emergency"
EMERGENCY_THRESHOLD = 0.2
LOW_CONFIDENCE_THRESHOLD = 0.5

# Actions
ROUTE_EMERGENCY = "route_to_human_emergency"
ROUTE_HUMAN = "route_to_human"
BOOK = "book_appt"
RESCHEDULE = "reschedule_appt"
THANK_YOU = "thank_you_message"
SEND_EOB = "send_explanation_of_benefits"
SEND_PAYMENT_LINK = "send_payment_link"
PRACTICE_INFO = "get_practice_info"
AUTOMATED_ACTIONS = {BOOK, RESCHEDULE, THANK_YOU, SEND_EOB, SEND_PAYMENT_LINK, PRACTICE_INFO}

POSITIVE_NEUTRAL = ("positive", "neutral")

# (group, intent, sentiments, action). Every intent x sentiment pair must be covered exactly once.
POLICY: list[tuple[str, str, tuple[str, ...], str]] = [
    # Schedule
    ("schedule", "emergency", ALL, ROUTE_EMERGENCY),
    ("schedule", "book_appointment", POSITIVE_NEUTRAL, BOOK),
    ("schedule", "book_appointment", ("negative",), ROUTE_HUMAN),
    ("schedule", "reschedule_appointment", POSITIVE_NEUTRAL, RESCHEDULE),
    ("schedule", "reschedule_appointment", ("negative",), ROUTE_HUMAN),
    # Cancel requests are offered a reschedule rather than a plain cancellation.
    ("schedule", "cancel_appointment", POSITIVE_NEUTRAL, RESCHEDULE),
    ("schedule", "cancel_appointment", ("negative",), ROUTE_HUMAN),
    # Collection
    ("collection", "insurance", ("positive",), THANK_YOU),
    ("collection", "insurance", ("neutral",), SEND_EOB),
    ("collection", "insurance", ("negative",), ROUTE_HUMAN),
    ("collection", "payment", ("positive",), THANK_YOU),
    ("collection", "payment", ("neutral",), SEND_PAYMENT_LINK),
    ("collection", "payment", ("negative",), ROUTE_HUMAN),
    # Practice
    ("practice", "clinic_info", POSITIVE_NEUTRAL, PRACTICE_INFO),
    ("practice", "clinic_info", ("negative",), ROUTE_HUMAN),
    ("practice", "find_provider", POSITIVE_NEUTRAL, PRACTICE_INFO),
    ("practice", "find_provider", ("negative",), ROUTE_HUMAN),
    ("practice", "provide_information", POSITIVE_NEUTRAL, PRACTICE_INFO),
    ("practice", "provide_information", ("negative",), ROUTE_HUMAN),
    # Patient: always a human. Emergencies inside symptom reports are caught by the override.
    ("patient", "symptom_check", ALL, ROUTE_HUMAN),
    ("patient", "referral", ALL, ROUTE_HUMAN),
    ("patient", "general_health_info", ALL, ROUTE_HUMAN),
    ("patient", "account_update", ALL, ROUTE_HUMAN),
    ("patient", "medication_info", ALL, ROUTE_HUMAN),
    ("patient", "prescription_refill", ALL, ROUTE_HUMAN),
    ("patient", "lab_results", ALL, ROUTE_HUMAN),
    ("patient", "other", ALL, ROUTE_HUMAN),  # includes medical_records (too rare for its own class)
]


def _build_tables() -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    table: dict[tuple[str, str], str] = {}
    groups: dict[str, str] = {}
    for group, intent, sentiments, action in POLICY:
        if groups.setdefault(intent, group) != group:
            raise ValueError(f"Intent {intent!r} assigned to two groups")
        for s in sentiments:
            if s not in SENTIMENTS:
                raise ValueError(f"Unknown sentiment {s!r} for {intent!r}")
            if (intent, s) in table:
                raise ValueError(f"Duplicate policy entry for ({intent!r}, {s!r})")
            table[(intent, s)] = action
    for intent in groups:
        missing = [s for s in SENTIMENTS if (intent, s) not in table]
        if missing:
            raise ValueError(f"Intent {intent!r} has no action for sentiment(s) {missing}")
    return table, groups


ACTION_TABLE, INTENT_GROUP = _build_tables()
ACTIONS = sorted(set(ACTION_TABLE.values()))


def check_intents(intents) -> None:
    """Raise if the models/data use an intent the policy doesn't cover (or vice versa)."""
    intents = set(intents)
    unmapped, unused = intents - set(INTENT_GROUP), set(INTENT_GROUP) - intents
    if unmapped:
        raise ValueError(f"Intents with no policy: {sorted(unmapped)}")
    if unused:
        logging.getLogger("action_policy").warning("Policy intents not present in data: %s", sorted(unused))


def table_action(intent: str, sentiment: str) -> str:
    return ACTION_TABLE[(intent, sentiment)]


# --------------------------------------------------------------------------- #
# Inference-time decision
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Decision:
    intent: str
    intent_confidence: float
    emergency_probability: float
    sentiment: str
    group: str
    action: str
    reason: str  # "table" | "emergency_override" | "low_confidence"


def decide(intent_probs: dict[str, float], sentiment: str) -> Decision:
    intent = max(intent_probs, key=intent_probs.get)
    confidence = float(intent_probs[intent])
    p_emergency = float(intent_probs.get(EMERGENCY_INTENT, 0.0))
    action, reason = table_action(intent, sentiment), "table"

    if intent != EMERGENCY_INTENT and p_emergency >= EMERGENCY_THRESHOLD:
        action, reason = ROUTE_EMERGENCY, "emergency_override"
    elif confidence < LOW_CONFIDENCE_THRESHOLD and action in AUTOMATED_ACTIONS:
        action, reason = ROUTE_HUMAN, "low_confidence"

    return Decision(intent, confidence, p_emergency, sentiment, INTENT_GROUP[intent], action, reason)


def decide_batch(probs: np.ndarray, classes, sentiments) -> pd.DataFrame:
    classes = list(classes)
    check_intents(classes)
    return pd.DataFrame([
        decide(dict(zip(classes, row)), s).__dict__ for row, s in zip(probs, sentiments)
    ])


# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #
def build_training_data() -> pd.DataFrame:
    """Cleaned intents + sentiment-head sentiment -> reference group and action per message."""
    intents = pd.read_csv(INTENTS_CSV)
    sentiment = pd.read_csv(SENTIMENT_CSV, usecols=["id", "intent_label", "sentiment",
                                                    "sent_negative", "sent_neutral", "sent_positive"])
    df = intents.merge(sentiment, on="id", how="left", suffixes=("", "_sent"), validate="one_to_one")
    if df["sentiment"].isna().any() or (df["intent_label"] != df["intent_label_sent"]).any():
        raise RuntimeError(f"{SENTIMENT_CSV.name} is stale; rerun train_sentiment_head.py")
    df = df.drop(columns="intent_label_sent")
    check_intents(df["intent_label"].unique())
    df["group"] = df["intent_label"].map(INTENT_GROUP)
    df["action"] = [table_action(i, s) for i, s in zip(df["intent_label"], df["sentiment"])]
    return df


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log = logging.getLogger("action_policy")
    log.info("Policy: %d intents, %d actions, %d (intent, sentiment) entries",
             len(INTENT_GROUP), len(ACTIONS), len(ACTION_TABLE))

    df = build_training_data()
    df.to_csv(ACTIONS_CSV, index=False)
    log.info("Actions by group:\n%s", pd.crosstab(df["action"], df["group"], margins=True).to_string())
    log.info("Wrote %d rows to %s", len(df), ACTIONS_CSV)


if __name__ == "__main__":
    main()
