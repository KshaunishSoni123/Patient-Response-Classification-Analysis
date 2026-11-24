"""One shared train / val / test split of the synthetic messages, used by all three models.

The split follows the REAL split of the seed messages (the synthetic rows are rewrites of real
train + val messages; real test messages were never used as seeds):

    synthetic train  = rewrites of real TRAIN messages                     (~82%)
    synthetic val    = rewrites of half of the real VAL messages, by seed  (~9%)
    synthetic test   = rewrites of the other half of the real VAL messages (~9%)

Why not a plain 70/15/15: the intent model trains on the real train messages. If synthetic
val/test contained rewrites of those, the intent model would have seen their seed message, and
every score that uses its predictions (intent on synthetic test, the end-to-end action score)
would be optimistic. Splitting by the seed's real split keeps all three models leak-free:

    intent     trains on real train          -> has never seen a synthetic val/test seed
    sentiment  trains on synthetic train     -> val/test seeds are disjoint from train seeds
    action     trains on synthetic train     -> same; its val/test inputs come from the final models

Residual caveat: the intent model is early-stopped (and its seed chosen) on the real val messages,
which are the seeds of synthetic val/test. That is one scalar choice per model, so the effect
on synthetic-test scores is small; the real test split stays fully clean.

The val/test halves are seed-grouped (a seed's rewrites stay together). Because whole seeds move
together, a single random 2-way split balances rare classes poorly, so N_CANDIDATES seed-grouped
StratifiedGroupKFold(2) splits are tried and the one with the most even intent AND action counts
between val and test is kept. Only labels are used to choose, never model outputs. The assignment is written to data/processed/synthetic_splits.csv
and checked against the current synthetic data on every load.

Usage:
    python synthetic_splits.py           # (re)build and print the split summary
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
SPLITS_CSV = ROOT / "data" / "processed" / "synthetic_splits.csv"
SEED = 42
SPLITS = ("train", "val", "test")
SCHEME = "seed-follows-real-split/v2"
N_CANDIDATES = 200

log = logging.getLogger("synthetic_splits")


def _fingerprint(syn: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(syn["synthetic_id"].astype(str) + "|" + syn["seed_id"].astype(str))
                          .encode()).hexdigest()[:16]


def _imbalance(a: pd.Series, b: pd.Series) -> float:
    """Sum over classes of |a - b| / (a + b): 0 = identical counts, 1 per class entirely on one side."""
    ca, cb = a.value_counts(), b.value_counts()
    classes = ca.index.union(cb.index)
    ca, cb = ca.reindex(classes, fill_value=0), cb.reindex(classes, fill_value=0)
    return float((abs(ca - cb) / (ca + cb)).sum())


def _balanced_halves(held: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The most balanced of N_CANDIDATES seed-grouped 2-way splits (by intent + action counts)."""
    groups = held["seed_id"].astype(str)
    best, best_score = None, np.inf
    for rs in range(SEED, SEED + N_CANDIDATES):
        a, b = next(StratifiedGroupKFold(2, shuffle=True, random_state=rs).split(held, held["action_label"], groups))
        score = sum(_imbalance(held[col].iloc[a], held[col].iloc[b]) for col in ("intent", "action_label"))
        if score < best_score:
            best, best_score = (a, b), score
    log.info("Val/test halves: most balanced of %d candidates (imbalance %.2f)", N_CANDIDATES, best_score)
    return best


def build(syn: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assign every synthetic row to train / val / test and save the assignment."""
    import train_intent_lstm as intent_lstm  # heavy import (Keras backend setup)
    from generate_synthetic import load_labelled

    syn = load_labelled() if syn is None else syn
    real, _, y, _ = intent_lstm.prepare_data()
    r_train, r_val, r_test = intent_lstm.split_indices(y)
    ids = real["id"].astype(str).to_numpy()
    real_split = {**{i: "train" for i in ids[r_train]}, **{i: "val" for i in ids[r_val]},
                  **{i: "test" for i in ids[r_test]}}

    seed_split = syn["seed_id"].astype(str).map(real_split)
    if seed_split.isna().any():
        raise RuntimeError(f"{int(seed_split.isna().sum())} synthetic rows have a seed that is not a real message; "
                           "the synthetic data was generated from a different healthcare snapshot")
    if (seed_split == "test").any():
        raise RuntimeError("Some synthetic rows rewrite REAL TEST messages - they must never be seeds")

    split = pd.Series("train", index=syn.index, dtype=object)
    held = np.flatnonzero((seed_split == "val").to_numpy())
    val_part, test_part = _balanced_halves(syn.iloc[held].reset_index(drop=True))
    split.iloc[held[val_part]] = "val"
    split.iloc[held[test_part]] = "test"

    out = pd.DataFrame({"synthetic_id": syn["synthetic_id"].astype(str), "seed_id": syn["seed_id"].astype(str),
                        "split": split.to_numpy()})
    out.attrs["fingerprint"] = _fingerprint(syn)
    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SPLITS_CSV, index=False)
    (SPLITS_CSV.with_suffix(".meta")).write_text(f"{SCHEME}\n{_fingerprint(syn)}\n")
    return out


def split_indices(syn: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row positions (in `syn` order, i.e. generate_synthetic.load_labelled()) for train / val / test.

    Builds the split on first use; rebuilds if the synthetic data changed since it was saved.
    """
    meta = SPLITS_CSV.with_suffix(".meta")
    current = _fingerprint(syn)
    if not SPLITS_CSV.exists() or not meta.exists() or meta.read_text().split() != [SCHEME, current]:
        log.info("Building the shared synthetic split -> %s", SPLITS_CSV)
        table = build()  # needs the full labelled data (intent / action) to balance val vs test
    else:
        table = pd.read_csv(SPLITS_CSV, dtype=str)
    if list(table["synthetic_id"]) != list(syn["synthetic_id"].astype(str)):
        raise RuntimeError(f"{SPLITS_CSV.name} doesn't line up with the synthetic data; delete it to rebuild")
    return {s: np.flatnonzero(table["split"].to_numpy() == s) for s in SPLITS}


def fingerprint() -> str:
    """Identifier of the saved split (for cache keys / metrics)."""
    meta = SPLITS_CSV.with_suffix(".meta")
    return meta.read_text().split()[1] if meta.exists() else "none"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("healthcare_ingestion", "generate_synthetic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from generate_synthetic import load_labelled

    syn = load_labelled()
    table = build(syn)
    idx = split_indices(syn)
    log.info("Shared synthetic split (%s): %s", SCHEME, {k: len(v) for k, v in idx.items()})
    summary = pd.crosstab(syn["action_label"], table["split"])[list(SPLITS)]
    log.info("Actions per split:\n%s", summary.to_string())
    log.info("Seeds per split: %s", table.groupby("split")["seed_id"].nunique().to_dict())


if __name__ == "__main__":
    main()
