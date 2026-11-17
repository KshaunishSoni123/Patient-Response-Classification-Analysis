"""Intent classifier: Word2Vec-initialized Embedding -> LSTM -> 4x64 ReLU -> softmax (Keras).

Pipeline:
  1. Load cleaned patient messages from healthcare_ingestion (one row per unique text).
  2. Stratified train / val / test split (same seed + ratios as train_intent_dnn.py).
  3. Build a vocabulary and train Word2Vec on the *training* texts only.
  4. Encode messages as padded token-id sequences (0 = padding, 1 = unknown word).
  5. Model: Embedding (initialized from Word2Vec, fine-tuned) -> LSTM -> 4x Dense(64, relu) -> softmax.
  6. Report test metrics and save the model + preprocessing artifacts to models/.

Keras runs on the PyTorch backend: TensorFlow 2.20 segfaults on import in this
Anaconda (Python 3.13, arm64) environment. Override with KERAS_BACKEND if needed.

Usage:
    python train_intent_lstm.py
    python train_intent_lstm.py --bidirectional --lstm-units 128
    python train_intent_lstm.py --no-w2v-init      # random embedding init
    python train_intent_lstm.py --freeze-embeddings
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
from collections import Counter
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from healthcare_ingestion import TARGET_COL, TEXT_COL, load_data

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "intent_lstm"
SEED = 42
PAD, OOV = "<pad>", "<unk>"  # ids 0 and 1

log = logging.getLogger("train_intent_lstm")


# --------------------------------------------------------------------------- #
# Text -> sequences
# --------------------------------------------------------------------------- #
def tokenize(texts) -> list[list[str]]:
    return [simple_preprocess(t, min_len=1) for t in texts]


def build_vocab(tokens: list[list[str]], min_count: int = 1) -> dict[str, int]:
    counts = Counter(t for toks in tokens for t in toks)
    words = sorted(w for w, c in counts.items() if c >= min_count)
    return {PAD: 0, OOV: 1, **{w: i + 2 for i, w in enumerate(words)}}


def encode(tokens: list[list[str]], vocab: dict[str, int], max_len: int) -> np.ndarray:
    """Token ids, truncated / post-padded with 0 to max_len."""
    out = np.zeros((len(tokens), max_len), dtype=np.int64)
    for i, toks in enumerate(tokens):
        ids = [vocab.get(t, 1) for t in toks][:max_len]
        out[i, : len(ids)] = ids
    return out


def embedding_matrix(vocab: dict[str, int], w2v: Word2Vec) -> np.ndarray:
    """Rows from Word2Vec; padding stays zero, unknown words get small random vectors."""
    rng = np.random.default_rng(SEED)
    mat = rng.normal(0, 0.1, (len(vocab), w2v.vector_size)).astype(np.float32)
    mat[0] = 0.0
    hits = 0
    for word, idx in vocab.items():
        if word in w2v.wv:
            mat[idx] = w2v.wv[word]
            hits += 1
    log.info("Embedding matrix: %d x %d (%d rows from Word2Vec)", *mat.shape, hits)
    return mat


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_lstm(
    vocab_size: int, embed_dim: int, n_classes: int, *, embed_init: np.ndarray | None,
    trainable_embed: bool, lstm_units: int, bidirectional: bool, dropout: float, lr: float,
    hidden_layers: int = 4, hidden_units: int = 64,
) -> keras.Model:
    init = keras.initializers.Constant(embed_init) if embed_init is not None else "uniform"
    lstm = keras.layers.LSTM(lstm_units, dropout=dropout, name="lstm")
    layers = [
        keras.Input(shape=(None,), dtype="int64"),
        keras.layers.Embedding(vocab_size, embed_dim, embeddings_initializer=init,
                               trainable=trainable_embed, mask_zero=True, name="embedding"),
        keras.layers.Bidirectional(lstm, name="bilstm") if bidirectional else lstm,
        keras.layers.Dropout(dropout, name="dropout_lstm"),
    ]
    # Hidden ReLU block between the LSTM and the softmax, dropout after each layer.
    for i in range(hidden_layers):
        layers += [
            keras.layers.Dense(hidden_units, activation="relu", name=f"hidden_{i + 1}"),
            keras.layers.Dropout(dropout, name=f"dropout_{i + 1}"),
        ]
    layers.append(keras.layers.Dense(n_classes, activation="softmax", name="intent"))
    model = keras.Sequential(layers, name="intent_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# Training (reused by tune_intent_lstm.py)
# --------------------------------------------------------------------------- #
DEFAULT_PARAMS = dict(
    w2v_dim=100, w2v_epochs=50, w2v_init=True, freeze_embeddings=False,
    lstm_units=64, bidirectional=False, hidden_layers=4, hidden_units=64,
    max_len=None, epochs=100, batch_size=32, dropout=0.3, lr=1e-3, patience=10,
)


def prepare_data() -> tuple[pd.DataFrame, list[list[str]], np.ndarray, LabelEncoder]:
    df = load_data()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET_COL].astype(str))
    return df, tokenize(df[TEXT_COL]), y, encoder


def split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed 70 / 15 / 15 stratified split (always SEED, independent of model seed)."""
    idx = np.arange(len(y))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=SEED)
    return idx_train, idx_val, idx_test


def fit(
    tokens: list[list[str]], y: np.ndarray, idx_train: np.ndarray, idx_val: np.ndarray,
    n_classes: int, *, seed: int = SEED, verbose: int = 2, **params,
) -> dict:
    """Build vocab + Word2Vec on idx_train, train the LSTM (early-stopped on idx_val).

    Returns the fitted model and everything needed to encode new text.
    """
    p = {**DEFAULT_PARAMS, **params}
    unknown = set(params) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"Unknown params: {sorted(unknown)}")
    keras.utils.set_random_seed(seed)  # seeds python, numpy and the backend

    train_tokens = [tokens[i] for i in idx_train]
    vocab = build_vocab(train_tokens)
    max_len = p["max_len"] or max(len(t) for t in train_tokens)

    w2v = embed_init = None
    if p["w2v_init"]:
        w2v = Word2Vec(sentences=train_tokens, vector_size=p["w2v_dim"], window=5, min_count=1,
                       sg=1, epochs=p["w2v_epochs"], seed=seed, workers=1)
        embed_init = embedding_matrix(vocab, w2v)

    X = encode(tokens, vocab, max_len)
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y[idx_train])
    model = build_lstm(
        len(vocab), p["w2v_dim"], n_classes, embed_init=embed_init,
        trainable_embed=not p["freeze_embeddings"], lstm_units=p["lstm_units"],
        bidirectional=p["bidirectional"], dropout=p["dropout"], lr=p["lr"],
        hidden_layers=p["hidden_layers"], hidden_units=p["hidden_units"],
    )
    history = model.fit(
        X[idx_train], y[idx_train],
        validation_data=(X[idx_val], y[idx_val]),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        class_weight=dict(enumerate(weights)),
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=p["patience"],
                                                 restore_best_weights=True)],
        verbose=verbose,
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return dict(model=model, w2v=w2v, vocab=vocab, max_len=max_len, X=X,
                best_epoch=best_epoch, params=p, seed=seed)


def evaluate(fitted: dict, y: np.ndarray, splits: dict[str, np.ndarray]) -> dict:
    metrics = {"best_epoch": fitted["best_epoch"]}
    for name, ids in splits.items():
        pred = fitted["model"].predict(fitted["X"][ids], verbose=0).argmax(axis=1)
        metrics[f"{name}_accuracy"] = round(float(accuracy_score(y[ids], pred)), 4)
        metrics[f"{name}_macro_f1"] = round(float(f1_score(y[ids], pred, average="macro")), 4)
    return metrics


def save(fitted: dict, encoder: LabelEncoder, y: np.ndarray, idx_test: np.ndarray,
         metrics: dict, out: Path, extra: dict | None = None) -> str:
    """Save model + preprocessing + metrics + test reports. Returns the text report."""
    n_classes = len(encoder.classes_)
    test_pred = fitted["model"].predict(fitted["X"][idx_test], verbose=0).argmax(axis=1)
    kw = dict(target_names=encoder.classes_, labels=np.arange(n_classes), zero_division=0)
    report = classification_report(y[idx_test], test_pred, **kw)
    per_class = classification_report(y[idx_test], test_pred, output_dict=True, **kw)

    out.mkdir(parents=True, exist_ok=True)
    fitted["model"].save(out / "model.keras")
    if fitted["w2v"] is not None:
        fitted["w2v"].save(str(out / "word2vec.model"))
    joblib.dump({"vocab": fitted["vocab"], "max_len": fitted["max_len"], "label_encoder": encoder},
                out / "preprocessing.joblib")
    payload = {**metrics, "seed": fitted["seed"], "params": fitted["params"], **(extra or {})}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    (out / "test_report.txt").write_text(report)
    (out / "test_per_class.json").write_text(json.dumps(per_class, indent=2))
    return report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    d = DEFAULT_PARAMS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w2v-dim", type=int, default=d["w2v_dim"])
    parser.add_argument("--w2v-epochs", type=int, default=d["w2v_epochs"])
    parser.add_argument("--no-w2v-init", action="store_true", help="Randomly initialize the embedding layer")
    parser.add_argument("--freeze-embeddings", action="store_true", help="Don't fine-tune the embedding layer")
    parser.add_argument("--lstm-units", type=int, default=d["lstm_units"])
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--hidden-layers", type=int, default=d["hidden_layers"], help="ReLU dense layers after the LSTM")
    parser.add_argument("--hidden-units", type=int, default=d["hidden_units"], help="Units per hidden layer")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Seed for weight init / Word2Vec / shuffling (data split always uses SEED)")
    parser.add_argument("--max-len", type=int, default=None, help="Default: longest training message")
    parser.add_argument("--epochs", type=int, default=d["epochs"])
    parser.add_argument("--batch-size", type=int, default=d["batch_size"])
    parser.add_argument("--dropout", type=float, default=d["dropout"])
    parser.add_argument("--lr", type=float, default=d["lr"])
    parser.add_argument("--patience", type=int, default=d["patience"])
    parser.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("gensim").setLevel(logging.WARNING)

    df, tokens, y, encoder = prepare_data()
    idx_train, idx_val, idx_test = split_indices(y)
    log.info("Split sizes: train=%d val=%d test=%d | classes=%d",
             len(idx_train), len(idx_val), len(idx_test), len(encoder.classes_))

    params = {k: v for k, v in vars(args).items() if k in DEFAULT_PARAMS}
    params["w2v_init"] = not args.no_w2v_init
    fitted = fit(tokens, y, idx_train, idx_val, len(encoder.classes_), seed=args.seed, **params)
    fitted["model"].summary(print_fn=log.info)
    log.info("Vocab: %d tokens (incl. pad/unk) | max_len=%d | best epoch: %d",
             len(fitted["vocab"]), fitted["max_len"], fitted["best_epoch"])

    metrics = evaluate(fitted, y, {"train": idx_train, "val": idx_val, "test": idx_test})
    log.info("Metrics: %s", metrics)
    report = save(fitted, encoder, y, idx_test, metrics, args.out_dir)
    log.info("Test classification report:\n%s", report)
    log.info("Saved model and artifacts to %s", args.out_dir)


if __name__ == "__main__":
    main()
