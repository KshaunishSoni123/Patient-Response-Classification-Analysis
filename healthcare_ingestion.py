"""Ingest the "Healthcare Chatbot Intent" dataset (10k rows) for training.

This replaces the Kaggle customer-support dataset (see data_ingestion.py) as the
training source.

Raw file (added manually, not downloaded):
    data/raw/healthcare-chatbot-intent-dataset-10k-rows.csv

Raw quirks this pipeline handles:
  * Mixed speakers: ~2/3 of rows are patient ("user") messages, ~1/3 are bot
    replies. Only patient messages are kept by default (--include-bot to keep both).
  * Label explosion: ~1,000 free-form intent labels (~480 on user rows) that are
    mostly synonyms (insurance_inquiry / insurance_query / check_insurance ...).
    They are mapped onto a small canonical taxonomy (INTENT_RULES); the raw label
    is preserved in `intent_raw`. Canonical classes with fewer than
    MIN_CLASS_COUNT rows are folded into "other".
  * Inconsistent topics ("symptom reporting" vs "symptom_reporting" vs
    "symptoms") are normalized the same way into `topic`.
  * message_id is NOT unique (only 3,155 distinct); transcript_id is, so it is
    used as the row id.
  * Heavy text duplication (~5.7k unique texts / 10k rows), sometimes with
    conflicting labels. Rows are collapsed to one per unique text using the
    majority canonical intent, so identical texts can't leak across splits.
    `n_occurrences` records how many raw rows were merged.

Usage:
    python healthcare_ingestion.py                 # process user messages
    python healthcare_ingestion.py --include-bot   # keep bot messages too
    python healthcare_ingestion.py --keep-duplicates
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_CSV = RAW_DIR / "healthcare-chatbot-intent-dataset-10k-rows.csv"
PROCESSED_CSV = PROCESSED_DIR / "healthcare_intents.csv"

TEXT_COL = "text"
TARGET_COL = "intent_label"
MIN_CLASS_COUNT = 20
OTHER = "other"

EXPECTED_COLUMNS = [
    "transcript_id",
    "message_id",
    "message_order",
    "sender_type",
    "message_text",
    "timestamp",
    "intent_label",
    "confidence_score",
    "entities_extracted",
    "conversation_topic",
    "user_id",
]

# Ordered (canonical_intent, regex) rules applied to the normalized raw label.
# First match wins, so more specific rules come first (e.g. an insurance
# question about a medication is "insurance_billing", not "medication_info").
INTENT_RULES: list[tuple[str, str]] = [
    ("emergency", r"emergency|urgent|critical|check_urgency"),
    ("insurance_billing", r"insur|bill|claim|copay|co_pay|deductible|cost|price|payment|coverage|benefit|eligib|policy|plan"),
    ("cancel_appointment", r"cancel"),
    ("reschedule_appointment", r"reschedul"),
    ("prescription_refill", r"refill|renew|prescription|order_medication|medication_pickup|pickup"),
    ("lab_results", r"(^|_)labs?(_|$)|test_result|results"),
    ("referral", r"referral|second_opinion"),
    ("medical_records", r"record|medical_history"),
    ("medication_info", r"medicat|drug|side_effect|dosage|adverse_reaction|interaction"),
    ("symptom_check", r"symptom|pain|health_concern|confirm_symptom|inform_duration"),
    ("book_appointment", r"book|schedule|request_appointment|appointment_request|find_appointment"
                         r"|availability|consultation|telehealth|request_doctor|follow_up"),
    ("appointment_inquiry", r"appointment"),
    ("find_provider", r"find_|provider|specialist|doctor|pharmacy|locat|facility|optometrist"),
    ("clinic_info", r"clinic|hospital|hours|portal"),
    ("account_update", r"update|profile|account|contact_info|personal_info"),
    ("general_health_info", r"health|condition|disease|illness|vaccin|procedure|treatment|advice"
                            r"|recommend|guideline|sleep"),
    ("provide_information", r"provide|inform|verif|dob|specify|confirm|select|preference|information|info"),
]

# Same idea for conversation_topic (applied after lowercasing + snake_casing).
TOPIC_RULES: list[tuple[str, str]] = [
    ("emergency", r"emergen|urgent"),
    ("insurance_billing", r"insur|bill|claim|payment"),
    ("appointments", r"appoint|schedul|booking"),
    ("account_records", r"profile|account|record"),
    ("medication", r"medicat|drug|prescri|pharm|refill"),
    ("lab_results", r"(^|_)labs?(_|$)|result|test"),
    ("symptoms", r"symptom"),
    ("referrals_providers", r"referr|provider|doctor|specialist"),
    ("clinic_info", r"clinic|facility|location"),
    ("mental_health", r"mental"),
    ("general_health", r"health|vaccin|wellness|advice|condition|nutrition"),
]

log = logging.getLogger("healthcare_ingestion")


# --------------------------------------------------------------------------- #
# Load / validate
# --------------------------------------------------------------------------- #
def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place "
            f"'{RAW_CSV.name}' in {RAW_DIR} (data/ is git-ignored)."
        )
    df = pd.read_csv(path, dtype=str)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing)}")
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        log.warning("Unexpected extra columns will be dropped: %s", sorted(extra))
    return df[EXPECTED_COLUMNS]


# --------------------------------------------------------------------------- #
# Label normalization
# --------------------------------------------------------------------------- #
def _snake(s: pd.Series) -> pd.Series:
    return (
        s.astype("string").str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )


def _apply_rules(labels: pd.Series, rules: list[tuple[str, str]]) -> pd.Series:
    compiled = [(name, re.compile(pat)) for name, pat in rules]
    lookup = {}
    for label in labels.dropna().unique():
        lookup[label] = next((name for name, rx in compiled if rx.search(label)), OTHER)
    return labels.map(lookup).fillna(OTHER)


def canonicalize_intent(raw: pd.Series) -> pd.Series:
    return _apply_rules(_snake(raw), INTENT_RULES)


def canonicalize_topic(raw: pd.Series) -> pd.Series:
    return _apply_rules(_snake(raw), TOPIC_RULES)


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame, include_bot: bool = False, dedupe: bool = True) -> pd.DataFrame:
    df = df.copy()

    df["sender_type"] = _snake(df["sender_type"])
    if not include_bot:
        df = df[df["sender_type"] == "user"]
        log.info("Kept %d patient (user) messages", len(df))

    out = pd.DataFrame({
        "id": df["transcript_id"].astype("string").str.strip(),
        TEXT_COL: df["message_text"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True),
        TARGET_COL: canonicalize_intent(df["intent_label"]),
        "intent_raw": _snake(df["intent_label"]),
        "topic": canonicalize_topic(df["conversation_topic"]),
        "topic_raw": _snake(df["conversation_topic"]),
        "sender_type": df["sender_type"],
        "entities": df["entities_extracted"].astype("string").str.strip(),
        "confidence_score": pd.to_numeric(df["confidence_score"], errors="coerce"),
        "message_order": pd.to_numeric(df["message_order"], errors="coerce").astype("Int64"),
        "timestamp": pd.to_datetime(df["timestamp"], errors="coerce"),
        "user_id": df["user_id"].astype("string").str.strip(),
    })

    before = len(out)
    out = out[out[TEXT_COL].notna() & (out[TEXT_COL] != "") & out["intent_raw"].notna()]
    out = out.drop_duplicates(subset="id", keep="first")
    if len(out) < before:
        log.info("Dropped %d empty/duplicate-id rows", before - len(out))

    if dedupe:
        out = _collapse_duplicate_texts(out)

    # Fold rare canonical classes into "other" so every class is learnable.
    counts = out[TARGET_COL].value_counts()
    rare = counts[counts < MIN_CLASS_COUNT].index.difference([OTHER])
    if len(rare):
        log.info("Folding rare classes (<%d rows) into '%s': %s", MIN_CLASS_COUNT, OTHER, sorted(rare))
        out.loc[out[TARGET_COL].isin(rare), TARGET_COL] = OTHER

    out[TARGET_COL] = out[TARGET_COL].astype("category")
    out["topic"] = out["topic"].astype("category")
    out["sender_type"] = out["sender_type"].astype("category")
    out["word_count"] = out[TEXT_COL].str.split().str.len().astype("int64")

    return out.sort_values("id", key=lambda s: s.str.extract(r"(\d+)", expand=False).astype(float)).reset_index(drop=True)


def _collapse_duplicate_texts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique text, labelled with the majority canonical intent."""
    n_rows = len(df)
    # Majority label per text (ties broken alphabetically for determinism).
    votes = df.groupby([TEXT_COL, TARGET_COL]).size().rename("votes").reset_index()
    votes = votes.sort_values([TEXT_COL, "votes", TARGET_COL], ascending=[True, False, True])
    majority = votes.drop_duplicates(TEXT_COL).set_index(TEXT_COL)[TARGET_COL]
    conflicting = (votes.groupby(TEXT_COL).size() > 1).sum()

    df = df.assign(n_occurrences=df.groupby(TEXT_COL)[TEXT_COL].transform("size"))
    df = df[df[TARGET_COL] == df[TEXT_COL].map(majority)]
    df = df.drop_duplicates(TEXT_COL, keep="first")
    log.info(
        "Collapsed %d rows to %d unique texts (%d texts had conflicting canonical labels; majority kept)",
        n_rows, len(df), conflicting,
    )
    return df


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame) -> None:
    log.info("Rows: %d | Columns: %d", *df.shape)
    log.info("Target distribution (%s):\n%s", TARGET_COL, df[TARGET_COL].value_counts().to_string())
    log.info("Topic distribution:\n%s", df["topic"].value_counts().to_string())
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        log.info("Missing values:\n%s", nulls.to_string())

    dup_text = df[TEXT_COL].duplicated(keep=False)
    if dup_text.any():
        log.warning(
            "%d rows share their text with another row. Group by text when splitting to avoid leakage.",
            dup_text.sum(),
        )


def load_data(include_bot: bool = False, dedupe: bool = True) -> pd.DataFrame:
    """Full pipeline: load -> validate -> clean. Importable from other modules."""
    return clean(load_raw(), include_bot=include_bot, dedupe=dedupe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-bot", action="store_true", help="Keep bot messages as well as patient messages")
    parser.add_argument("--keep-duplicates", action="store_true", help="Don't collapse duplicate texts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df = load_data(include_bot=args.include_bot, dedupe=not args.keep_duplicates)
    summarize(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    log.info("Wrote cleaned dataset to %s", PROCESSED_CSV)


if __name__ == "__main__":
    main()
