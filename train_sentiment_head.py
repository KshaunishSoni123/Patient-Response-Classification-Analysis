"""Sentiment classifier: frozen `brettclaus/Hospital_Reviews` BERT -> 2x64 ReLU head -> softmax.

The pretrained BERT encoder (12 layers, 110M params) is frozen; only a small Keras
head is trained, on the customer-support sentiment labels (negative / neutral /
positive).

Pipeline:
  1. Load customer-support messages, one row per unique text (702 of 800 rows;
     duplicates never disagree on sentiment). The dataset is synthetic: the 702
     texts come from ~150 sentence templates with product names / numbers swapped,
     and each template has a single sentiment. Random splits therefore put
     near-copies in train and test (even bag-of-words scores 100%), so ALL splits
     here are grouped by template: the model is always tested on unseen templates.
  2. Encode every text once with the frozen BERT. Features are the pooled [CLS]
     vector (768-d) that the model's own Negative/Mixed/Positive classifier reads
     (--pooling mean uses masked mean of the last hidden layer instead).
     Because the encoder is frozen, features are computed once and reused.
  3. 5-fold template-grouped CV: train the head (early stopping on an inner val split)
     and, as a baseline, logistic regression on the model's original 3 output
     probabilities. Reports accuracy / macro F1 for both.
  4. Final head: fixed template-grouped ~70/15/15 split, trained with several seeds; keep the seed
     with the best val macro F1, report test metrics, save to models/sentiment_head/.
  5. Score the healthcare patient messages -> data/processed/healthcare_sentiment.csv
     (joinable on `id`) for the final classifier.

Usage:
    python train_sentiment_head.py
    python train_sentiment_head.py --hidden-units 128 --dropout 0.3 --pooling mean
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
# Keras and transformers both probe for TensorFlow, which segfaults on import here.
sys.modules.setdefault("tensorflow", None)

import argparse
import json
import logging
import re
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
import torch
from huggingface_hub import try_to_load_from_cache
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import healthcare_ingestion

ROOT = Path(__file__).resolve().parent
MODEL_ID = "brettclaus/Hospital_Reviews"
SUPPORT_CSV = ROOT / "data" / "processed" / "customer_support.csv"
OUT_CSV = ROOT / "data" / "processed" / "healthcare_sentiment.csv"
MODEL_DIR = ROOT / "models" / "sentiment_head"
CLASSES = ["negative", "neutral", "positive"]
MAX_LENGTH = 128
SEED = 42

log = logging.getLogger("train_sentiment_head")


# --------------------------------------------------------------------------- #
# Template-grouped splits
# --------------------------------------------------------------------------- #
def template_of(text: str) -> str:
    """Mask numbers/prices and capitalized product/app names to recover the sentence template."""
    text = re.sub(r"\$?\d[\d,.:]*", "<N>", text)
    return re.sub(r"(?<!^)(?<![.!?] )\b[A-Z][A-Za-z0-9]*(?:[ -][A-Z0-9][A-Za-z0-9]*)*", "<P>", text)


def group_split(idx: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    """One stratified, group-disjoint split of idx: (larger part, ~1/n_splits part)."""
    a, b = next(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(idx, y[idx], groups[idx]))
    return idx[a], idx[b]


# --------------------------------------------------------------------------- #
# Frozen BERT features
# --------------------------------------------------------------------------- #
class FrozenEncoder:
    def __init__(self, model_id: str = MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        cached = try_to_load_from_cache(model_id, "config.json")
        self.revision = Path(cached).parent.name if isinstance(cached, str) else None

    @torch.no_grad()
    def encode(self, texts: list[str], pooling: str = "cls", batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Returns (features [n, 768], original model probabilities [n, 3])."""
        feats, probs = [], []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                 max_length=MAX_LENGTH, return_tensors="pt")
            out = self.model.base_model(**enc)
            if pooling == "cls":
                f = out.pooler_output  # what the model's own classifier head consumes
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                f = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            logits = self.model.classifier(self.model.dropout(out.pooler_output))
            feats.append(f.numpy())
            probs.append(torch.softmax(logits, -1).numpy())
        return np.concatenate(feats).astype(np.float32), np.concatenate(probs)


# --------------------------------------------------------------------------- #
# Head
# --------------------------------------------------------------------------- #
def build_head(input_dim: int, *, hidden_layers: int, hidden_units: int, dropout: float, lr: float) -> keras.Model:
    model = keras.Sequential(name="sentiment_head")
    model.add(keras.Input(shape=(input_dim,)))
    for i in range(hidden_layers):
        model.add(keras.layers.Dense(hidden_units, activation="relu", name=f"hidden_{i + 1}"))
        model.add(keras.layers.Dropout(dropout, name=f"dropout_{i + 1}"))
    model.add(keras.layers.Dense(len(CLASSES), activation="softmax", name="sentiment"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_head(X, y, idx_train, idx_val, args, seed: int) -> tuple[keras.Model, StandardScaler, int]:
    keras.utils.set_random_seed(seed)
    scaler = StandardScaler().fit(X[idx_train])
    Xs = scaler.transform(X).astype(np.float32)
    model = build_head(X.shape[1], hidden_layers=args.hidden_layers, hidden_units=args.hidden_units,
                       dropout=args.dropout, lr=args.lr)
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=y[idx_train])
    hist = model.fit(
        Xs[idx_train], y[idx_train], validation_data=(Xs[idx_val], y[idx_val]),
        epochs=args.epochs, batch_size=args.batch_size, class_weight=dict(enumerate(weights)),
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                                 restore_best_weights=True)],
        verbose=0,
    )
    return model, scaler, int(np.argmin(hist.history["val_loss"])) + 1


def predict(model, scaler, X) -> np.ndarray:
    return model.predict(scaler.transform(X).astype(np.float32), verbose=0)


def scores(y_true, y_pred) -> dict:
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("healthcare_ingestion", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 1. Data ------------------------------------------------------------- #
    df = pd.read_csv(SUPPORT_CSV).drop_duplicates("text").reset_index(drop=True)
    y = df["sentiment"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = df["text"].map(template_of).to_numpy()
    log.info("Customer-support texts: %d | templates: %d | %s",
             len(df), len(set(groups)), df["sentiment"].value_counts().to_dict())

    # 2. Frozen features ------------------------------------------------- #
    enc = FrozenEncoder()
    X, base_probs = enc.encode(df["text"].tolist(), pooling=args.pooling)
    log.info("Encoded with frozen %s (revision %s): features %s, pooling=%s",
             MODEL_ID, enc.revision, X.shape, args.pooling)

    # 3. 5-fold CV: head vs. logistic regression on the original 3 outputs -- #
    cv_rows = []
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups)
    for k, (tr, te) in enumerate(folds):
        tr, va = group_split(tr, y, groups, n_splits=6, seed=SEED)
        model, scaler, _ = fit_head(X, y, tr, va, args, SEED)
        head_pred = predict(model, scaler, X[te]).argmax(1)
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(base_probs[np.r_[tr, va]], y[np.r_[tr, va]])
        cv_rows.append({"fold": k, **{f"head_{m}": v for m, v in scores(y[te], head_pred).items()},
                        **{f"baseline_{m}": v for m, v in scores(y[te], lr.predict(base_probs[te])).items()}})
    cv = pd.DataFrame(cv_rows)
    cv_summary = cv.drop(columns="fold").agg(["mean", "std"]).round(4)
    log.info("5-fold template-grouped CV (frozen BERT + %dx%d head vs. LR on original 3 outputs):\n%s",
             args.hidden_layers, args.hidden_units, cv_summary.to_string())

    # 4. Final head on a fixed 70/15/15 split ----------------------------- #
    idx = np.arange(len(y))
    idx_rest, idx_test = group_split(idx, y, groups, n_splits=7, seed=SEED)   # ~15% test
    idx_train, idx_val = group_split(idx_rest, y, groups, n_splits=6, seed=SEED)  # ~15% val
    log.info("Final split (template-disjoint): train=%d val=%d test=%d", len(idx_train), len(idx_val), len(idx_test))
    runs = []
    for seed in args.seeds:
        model, scaler, best_epoch = fit_head(X, y, idx_train, idx_val, args, seed)
        val = scores(y[idx_val], predict(model, scaler, X[idx_val]).argmax(1))
        test = scores(y[idx_test], predict(model, scaler, X[idx_test]).argmax(1))
        log.info("Seed %d: best_epoch=%d val=%s test=%s", seed, best_epoch, val, test)
        runs.append(dict(seed=seed, model=model, scaler=scaler, best_epoch=best_epoch, val=val, test=test))
    best = max(runs, key=lambda r: r["val"]["macro_f1"])  # choose by val, never test
    test_pred = predict(best["model"], best["scaler"], X[idx_test]).argmax(1)
    report = classification_report(y[idx_test], test_pred, target_names=CLASSES, zero_division=0)
    log.info("Kept seed %d. Test report:\n%s\nconfusion (rows=true %s):\n%s", best["seed"], report, CLASSES,
             confusion_matrix(y[idx_test], test_pred))

    # 5. Score healthcare messages ----------------------------------------- #
    hc = healthcare_ingestion.load_data()
    hX, h_base = enc.encode(hc["text"].tolist(), pooling=args.pooling)
    hp = predict(best["model"], best["scaler"], hX)
    out = hc[["id", "text", "intent_label"]].copy()
    for i, c in enumerate(CLASSES):
        out[f"sent_{c}"] = hp[:, i]
    out["sentiment"] = np.array(CLASSES)[hp.argmax(1)]
    for i, lab in enumerate(enc.labels):  # original model outputs, kept for reference
        out[f"hosp_{lab.lower()}"] = h_base[:, i]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    log.info("Healthcare sentiment by intent:\n%s", pd.crosstab(out["intent_label"], out["sentiment"]).to_string())
    log.info("Wrote %d rows to %s", len(out), OUT_CSV)

    # 6. Save ---------------------------------------------------------------- #
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best["model"].save(out_dir / "head.keras")
    joblib.dump({"scaler": best["scaler"], "classes": CLASSES, "encoder_model": MODEL_ID,
                 "encoder_revision": enc.revision, "pooling": args.pooling, "max_length": MAX_LENGTH},
                out_dir / "preprocessing.joblib")
    seeds_summary = {f"{split}_{m}": {"mean": round(float(np.mean([r[split][m] for r in runs])), 4),
                                      "std": round(float(np.std([r[split][m] for r in runs])), 4)}
                     for split in ("val", "test") for m in ("accuracy", "macro_f1")}
    (out_dir / "metrics.json").write_text(json.dumps({
        "encoder": MODEL_ID, "encoder_revision": enc.revision, "args": vars(args),
        "cv": {c: cv_summary[c].to_dict() for c in cv_summary.columns},
        "final_seed": best["seed"], "final_val": best["val"], "final_test": best["test"],
        "seeds_summary": seeds_summary,
    }, indent=2, default=str))
    (out_dir / "test_report.txt").write_text(report)
    log.info("Saved head and artifacts to %s", out_dir)


if __name__ == "__main__":
    main()
