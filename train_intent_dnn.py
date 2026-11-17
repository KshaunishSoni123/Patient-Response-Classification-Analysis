"""Intent classifier: Word2Vec sentence embeddings -> 4x64 Keras DNN.

Pipeline:
  1. Load cleaned patient messages from healthcare_ingestion (one row per unique text).
  2. Stratified train / val / test split.
  3. Train Word2Vec on the *training* texts only (no vocabulary leakage).
  4. Embed each message as the mean of its word vectors, then standardize.
  5. Train a DNN: 4 hidden Dense(64, relu) layers with dropout -> softmax.
  6. Report test metrics and save the model + preprocessing artifacts to models/.

Keras runs on the PyTorch backend: TensorFlow 2.20 segfaults on import in this
Anaconda (Python 3.13, arm64) environment. Override with KERAS_BACKEND if needed.

Usage:
    python train_intent_dnn.py
    python train_intent_dnn.py --epochs 500 --dropout 0.2 --w2v-dim 200
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
if os.environ["KERAS_BACKEND"] != "tensorflow":
    # Keras probes for TensorFlow (seeding, saving) even on other backends, and
    # importing TF segfaults in this env. A None entry makes `import tensorflow`
    # raise ImportError, so Keras treats it as unavailable.
    sys.modules["tensorflow"] = None

import argparse
import json
import logging
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from healthcare_ingestion import TARGET_COL, TEXT_COL, load_data

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "intent_dnn"
SEED = 42
HIDDEN_LAYERS = 4
HIDDEN_UNITS = 64

log = logging.getLogger("train_intent_dnn")


# --------------------------------------------------------------------------- #
# Word2Vec embeddings
# --------------------------------------------------------------------------- #
def tokenize(texts: pd.Series) -> list[list[str]]:
    return [simple_preprocess(t, min_len=1) for t in texts]


def train_word2vec(tokens: list[list[str]], dim: int, epochs: int) -> Word2Vec:
    # workers=1 + seed makes training deterministic.
    return Word2Vec(
        sentences=tokens, vector_size=dim, window=5, min_count=1,
        sg=1, epochs=epochs, seed=SEED, workers=1,
    )


def embed(w2v: Word2Vec, tokens: list[list[str]]) -> np.ndarray:
    """Mean of in-vocabulary word vectors; zero vector if none are known."""
    out = np.zeros((len(tokens), w2v.vector_size), dtype=np.float32)
    for i, toks in enumerate(tokens):
        known = [t for t in toks if t in w2v.wv]
        if known:
            out[i] = w2v.wv[known].mean(axis=0)
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_dnn(input_dim: int, n_classes: int, dropout: float, lr: float) -> keras.Model:
    model = keras.Sequential(name="intent_dnn")
    model.add(keras.Input(shape=(input_dim,)))
    for i in range(HIDDEN_LAYERS):
        model.add(keras.layers.Dense(HIDDEN_UNITS, activation="relu", name=f"hidden_{i + 1}"))
        model.add(keras.layers.Dropout(dropout, name=f"dropout_{i + 1}"))
    model.add(keras.layers.Dense(n_classes, activation="softmax", name="intent"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w2v-dim", type=int, default=100)
    parser.add_argument("--w2v-epochs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("gensim").setLevel(logging.WARNING)
    keras.utils.set_random_seed(SEED)  # seeds python, numpy and the backend

    # 1. Data -------------------------------------------------------------- #
    df = load_data()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET_COL].astype(str))
    n_classes = len(encoder.classes_)

    # 2. Split 70 / 15 / 15, stratified ---------------------------------- #
    idx = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=SEED)
    log.info("Split sizes: train=%d val=%d test=%d | classes=%d",
             len(idx_train), len(idx_val), len(idx_test), n_classes)

    # 3. Word2Vec on training texts only --------------------------------- #
    tokens = tokenize(df[TEXT_COL])
    w2v = train_word2vec([tokens[i] for i in idx_train], args.w2v_dim, args.w2v_epochs)
    log.info("Word2Vec vocab: %d words, dim=%d", len(w2v.wv), w2v.vector_size)

    # 4. Sentence embeddings + scaling ----------------------------------- #
    X = embed(w2v, tokens)
    for name, ids in [("val", idx_val), ("test", idx_test)]:
        oov = np.mean([not any(t in w2v.wv for t in tokens[i]) for i in ids])
        log.info("%s messages with no known words: %.1f%%", name, 100 * oov)
    scaler = StandardScaler().fit(X[idx_train])
    X = scaler.transform(X).astype(np.float32)

    # 5. DNN -------------------------------------------------------------- #
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y[idx_train])
    model = build_dnn(X.shape[1], n_classes, args.dropout, args.lr)
    model.summary(print_fn=log.info)

    history = model.fit(
        X[idx_train], y[idx_train],
        validation_data=(X[idx_val], y[idx_val]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=dict(enumerate(weights)),
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                                 restore_best_weights=True)],
        verbose=2,
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    log.info("Best epoch: %d / %d", best_epoch, len(history.history["val_loss"]))

    # 6. Evaluate ---------------------------------------------------------- #
    metrics = {"best_epoch": best_epoch}
    for name, ids in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        pred = model.predict(X[ids], verbose=0).argmax(axis=1)
        metrics[f"{name}_accuracy"] = round(float(accuracy_score(y[ids], pred)), 4)
        metrics[f"{name}_macro_f1"] = round(float(f1_score(y[ids], pred, average="macro")), 4)
    log.info("Metrics: %s", metrics)

    test_pred = model.predict(X[idx_test], verbose=0).argmax(axis=1)
    report = classification_report(y[idx_test], test_pred, target_names=encoder.classes_,
                                   labels=np.arange(n_classes), zero_division=0)
    log.info("Test classification report:\n%s", report)

    # 7. Save -------------------------------------------------------------- #
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "model.keras")
    w2v.save(str(MODEL_DIR / "word2vec.model"))
    joblib.dump({"scaler": scaler, "label_encoder": encoder}, MODEL_DIR / "preprocessing.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps({**metrics, "args": vars(args)}, indent=2))
    (MODEL_DIR / "test_report.txt").write_text(report)
    log.info("Saved model and artifacts to %s", MODEL_DIR)


if __name__ == "__main__":
    main()
