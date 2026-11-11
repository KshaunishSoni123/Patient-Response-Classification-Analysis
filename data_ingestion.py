"""Ingest the Kaggle "NLP Customer Support Classification" dataset.

Source: https://www.kaggle.com/datasets/devtayyabsajjad/nlp-customer-support-classification

Steps:
  1. Download the dataset zip (kaggle CLI if available, else Kaggle's public
     download endpoint, which needs no credentials for this public dataset).
  2. Extract the CSV into data/raw/.
  3. Load, validate the schema, and clean/normalize the columns.
  4. Write the cleaned dataset to data/processed/customer_support.csv.

Usage:
    python data_ingestion.py            # download if missing, then process
    python data_ingestion.py --force    # re-download even if raw data exists
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET_SLUG = "devtayyabsajjad/nlp-customer-support-classification"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_CSV = RAW_DIR / "nlp_customer_support_classification.csv"
PROCESSED_CSV = PROCESSED_DIR / "customer_support.csv"

TEXT_COL = "text"
TARGET_COL = "intent_label"

EXPECTED_COLUMNS = {
    "message_id": "string",
    "text": "string",
    "intent_label": "category",
    "sentiment": "category",
    "urgency_level": "category",
    "classification_confidence": "float64",
    "channel": "category",
    "language": "category",
    "word_count": "int64",
    "response_time_hours": "float64",
    "is_resolved": "bool",
    "csat_score": "Int64",  # nullable: missing for unresolved tickets
}

URGENCY_ORDER = ["low", "medium", "high", "critical"]

log = logging.getLogger("data_ingestion")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_dataset(force: bool = False) -> Path:
    """Download and extract the dataset CSV into RAW_DIR. Returns the CSV path."""
    if RAW_CSV.exists() and not force:
        log.info("Raw data already present at %s (use --force to re-download)", RAW_CSV)
        return RAW_CSV

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "dataset.zip"

    if shutil.which("kaggle"):
        log.info("Downloading %s via kaggle CLI", DATASET_SLUG)
        subprocess.run(
            ["kaggle", "datasets", "download", DATASET_SLUG, "-p", str(RAW_DIR), "--force"],
            check=True,
        )
        zip_path = RAW_DIR / f"{DATASET_SLUG.split('/')[-1]}.zip"
    else:
        log.info("kaggle CLI not found; downloading from %s", DOWNLOAD_URL)
        # Kaggle rejects urllib's default User-Agent with a 404.
        req = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as dst:
            shutil.copyfileobj(resp, dst)

    with zipfile.ZipFile(zip_path) as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise RuntimeError(f"Expected exactly one CSV in archive, found: {csv_members}")
        # The archive's file name contains " (1)"; write it to a stable name instead.
        with zf.open(csv_members[0]) as src, open(RAW_CSV, "wb") as dst:
            shutil.copyfileobj(src, dst)

    zip_path.unlink(missing_ok=True)
    log.info("Saved raw CSV to %s", RAW_CSV)
    return RAW_CSV


# --------------------------------------------------------------------------- #
# Load / validate / clean
# --------------------------------------------------------------------------- #
def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing)}")
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        log.warning("Unexpected extra columns will be kept: %s", sorted(extra))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize text: strip and collapse internal whitespace.
    df[TEXT_COL] = df[TEXT_COL].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

    # Normalize categorical labels (lowercase, trimmed).
    for col, dtype in EXPECTED_COLUMNS.items():
        if dtype == "category":
            df[col] = df[col].astype("string").str.strip().str.lower()

    # Drop rows unusable for training: empty text or missing target.
    before = len(df)
    df = df[df[TEXT_COL].notna() & (df[TEXT_COL] != "") & df[TARGET_COL].notna()]
    df = df.drop_duplicates(subset="message_id", keep="first")
    if len(df) < before:
        log.info("Dropped %d invalid/duplicate rows", before - len(df))

    # Apply dtypes.
    for col, dtype in EXPECTED_COLUMNS.items():
        df[col] = df[col].astype(dtype)
    df["urgency_level"] = pd.Categorical(df["urgency_level"], categories=URGENCY_ORDER, ordered=True)

    # Recompute word_count from the cleaned text so it's consistent.
    df["word_count"] = df[TEXT_COL].str.split().str.len().astype("int64")

    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    log.info("Rows: %d | Columns: %d", *df.shape)
    log.info("Target distribution (%s):\n%s", TARGET_COL, df[TARGET_COL].value_counts().to_string())
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        log.info("Missing values:\n%s", nulls.to_string())

    # Identical texts can leak between train/test splits; flag them.
    dup_text = df[TEXT_COL].duplicated(keep=False)
    if dup_text.any():
        conflicting = df[dup_text].groupby(TEXT_COL)[TARGET_COL].nunique().gt(1).sum()
        log.warning(
            "%d rows share their text with another row (%d unique texts, %d with conflicting labels). "
            "Group by text when splitting to avoid leakage.",
            dup_text.sum(),
            df.loc[dup_text, TEXT_COL].nunique(),
            conflicting,
        )


def load_data(force_download: bool = False) -> pd.DataFrame:
    """Full pipeline: download (if needed) -> load -> clean. Importable from other modules."""
    raw_path = download_dataset(force=force_download)
    return clean(load_raw(raw_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-download the raw dataset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df = load_data(force_download=args.force)
    summarize(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    log.info("Wrote cleaned dataset to %s", PROCESSED_CSV)


if __name__ == "__main__":
    main()
