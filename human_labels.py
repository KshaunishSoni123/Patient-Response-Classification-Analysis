"""Human sentiment labels for the real test messages (written by label_sentiment.py).

labels/real_test_sentiment_<annotator>.csv - one file per annotator. The final label per message:
  * one annotator: their label (skips excluded);
  * several: the majority label; ties are excluded and counted as disputed.
Agreement between annotators is reported as mean pairwise Cohen's kappa (on messages both labelled).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent
LABELS_DIR = ROOT / "labels"
SENTIMENTS = ("negative", "neutral", "positive")
SHUFFLE_SEED = 2026


def labels_path(annotator: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in annotator.strip().lower())
    if not safe:
        raise ValueError("Annotator name is empty")
    LABELS_DIR.mkdir(exist_ok=True)
    return LABELS_DIR / f"real_test_sentiment_{safe}.csv"


def real_test_messages() -> pd.DataFrame:
    """The 437 real test messages (same split as every evaluation), in a fixed shuffled order."""
    import train_intent_lstm as intent_lstm  # heavy import; only needed here

    df, _, y, _ = intent_lstm.prepare_data()
    _, _, idx_test = intent_lstm.split_indices(y)
    test = df.iloc[idx_test]
    out = pd.DataFrame({"id": test["id"].astype(str).to_numpy(), "text": test[intent_lstm.TEXT_COL].to_numpy(),
                        "intent_label": test[intent_lstm.TARGET_COL].to_numpy()})
    return out.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)


def load_human_sentiment() -> tuple[pd.DataFrame, dict]:
    """(DataFrame id -> human_sentiment for usable messages, info dict). Empty if no labels exist."""
    files = sorted(LABELS_DIR.glob("real_test_sentiment_*.csv"))
    info = {"annotators": [], "n_files": len(files)}
    if not files:
        return pd.DataFrame(columns=["id", "human_sentiment"]), info

    per = {}
    for f in files:
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        d = d[d["human_sentiment"].isin(SENTIMENTS)]
        name = f.stem.removeprefix("real_test_sentiment_")
        per[name] = d.set_index("id")["human_sentiment"]
    info["annotators"] = list(per)
    wide = pd.DataFrame(per)

    if len(per) >= 2:
        kappas = []
        for a, b in combinations(per, 2):
            both = wide[[a, b]].dropna()
            if len(both) >= 10:
                kappas.append({"pair": f"{a}/{b}", "n": len(both),
                               "agreement": round(float((both[a] == both[b]).mean()), 4),
                               "cohen_kappa": round(float(cohen_kappa_score(both[a], both[b])), 4)})
        info["pairwise"] = kappas

    def final(row: pd.Series):
        votes = row.dropna().value_counts()
        if votes.empty or (len(votes) > 1 and votes.iloc[0] == votes.iloc[1]):
            return None  # nothing usable, or a tie
        return votes.index[0]

    final_labels = wide.apply(final, axis=1)
    info["n_labelled"] = int(wide.notna().any(axis=1).sum())
    info["n_disputed"] = int(final_labels.isna().sum() - (~wide.notna().any(axis=1)).sum())
    out = final_labels.dropna().rename("human_sentiment").rename_axis("id").reset_index()
    info["n_final"] = len(out)
    info["distribution"] = out["human_sentiment"].value_counts().to_dict()
    return out, info
