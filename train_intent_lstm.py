"""Intent classifier: Word2Vec-initialized Embedding -> LSTM -> 4x64 ReLU -> softmax (Keras).

Pipeline:
  1. Load the real patient messages (healthcare_ingestion, one row per unique text) and
     the synthetic messages (generate_synthetic.load_labelled, label = llm_intent unless
     a reviewer overrode it).
  2. Fixed stratified 70 / 15 / 15 split of the REAL messages (SEED, unchanged).
  3. Training data depends on --train-on:
       synthetic (default)  train on the synthetic messages only
       both                 synthetic + the real train split
       real                 the real train split only (the original setup)
     For synthetic/both, early stopping uses a 10% hold-out of the training pool grouped
     by seed message (the 3 tone rewrites of one seed stay together). For real, early
     stopping uses the real val split, as before.
  4. Build the vocabulary and train Word2Vec on the *training* texts only (so with the
     default, Word2Vec is trained on the synthetic text).
  5. Encode messages as padded token-id sequences (0 = padding, 1 = unknown word).
  6. Model: Embedding (initialized from Word2Vec, fine-tuned) -> LSTM -> 4x Dense(64, relu) -> softmax.
  7. Report metrics on the real val and real test splits; save model + artifacts to models/.

The real test split was never used as a generation seed, so it stays unseen. The real
val messages WERE seeds, so the synthetic training set contains rewrites of them and
val scores are optimistic when training on synthetic data; use them only to compare
seeds, and judge the model on test.

Keras runs on the PyTorch backend: TensorFlow 2.20 segfaults on import in this
Anaconda (Python 3.13, arm64) environment. Override with KERAS_BACKEND if needed.

Usage:
    python train_intent_lstm.py                    # train on synthetic data
    python train_intent_lstm.py --train-on real    # original real-data baseline
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
from dataclasses import dataclass
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from healthcare_ingestion import TARGET_COL, TEXT_COL, load_snapshot

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "intent_lstm"
SEED = 42
TRAIN_SOURCES = ("synthetic", "both", "real")
STOP_SPLITS = 10  # synthetic/both: 1/10 of the training pool (grouped by seed) for early stopping
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
    df = load_snapshot()  # frozen processed snapshot, not the raw CSV
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET_COL].astype(str))
    return df, tokenize(df[TEXT_COL]), y, encoder


def split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed 70 / 15 / 15 stratified split of the REAL messages (always SEED, independent of model seed)."""
    idx = np.arange(len(y))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=SEED)
    return idx_train, idx_val, idx_test


def group_split(idx: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    """One stratified, group-disjoint split of idx: (larger part, ~1/n_splits part)."""
    a, b = next(StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED).split(idx, y[idx], groups[idx]))
    return idx[a], idx[b]


@dataclass
class Dataset:
    """Real messages (rows 0..n_real-1, same order as prepare_data) followed by synthetic ones.

    groups: the real message a row is about - its own id for real rows, seed_id for
    synthetic rows - so a real message and its rewrites always land on the same side.
    """
    df: pd.DataFrame
    tokens: list[list[str]]
    y: np.ndarray
    encoder: LabelEncoder
    groups: np.ndarray
    n_real: int
    train_on: str

    @property
    def n_classes(self) -> int:
        return len(self.encoder.classes_)

    def real_splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return split_indices(self.y[: self.n_real])

    def training_pool(self, exclude_groups: set | None = None, real_rows: np.ndarray | None = None) -> np.ndarray:
        """Rows that may be trained on: synthetic (and real_rows for 'both'), minus excluded groups."""
        pool = np.arange(self.n_real, len(self.y)) if self.train_on != "real" else np.array([], dtype=int)
        if self.train_on != "synthetic" and real_rows is not None:
            pool = np.concatenate([pool, real_rows])
        if exclude_groups:
            pool = pool[~np.isin(self.groups[pool], list(exclude_groups))]
        return pool

    def splits(self) -> dict[str, np.ndarray]:
        """train / stop (early stopping) / val (real) / test (real).

        real:            train = real train, stop = real val.
        synthetic, both: the shared synthetic split (synthetic_splits.py) - train = synthetic train
                         (+ real train for `both`), stop = synthetic val. Synthetic val/test rewrite
                         real val messages only, so no seed straddles train and stop/test.
        """
        r_train, r_val, r_test = self.real_splits()
        if self.train_on == "real":
            return {"train": r_train, "stop": r_val, "val": r_val, "test": r_test}
        import synthetic_splits  # local import: avoids a cycle (it imports this module)

        syn_rows = np.arange(self.n_real, len(self.y))
        syn = pd.DataFrame({"synthetic_id": self.df["id"].iloc[syn_rows].astype(str).to_numpy(),
                            "seed_id": self.groups[syn_rows]})
        shared = synthetic_splits.split_indices(syn)
        train = syn_rows[shared["train"]]
        if self.train_on == "both":
            train = np.concatenate([train, r_train])
        return {"train": train, "stop": syn_rows[shared["val"]], "val": r_val, "test": r_test}

    def cv_folds(self, n_folds: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(train, stop, eval) folds for tuning, scored on REAL dev messages (train + val split).

        Real dev messages are split into n_folds stratified folds. For synthetic/both, each fold
        trains on rows whose seed is NOT in the eval fold, so the model never trains on a rewrite
        of a message it is scored on. The real test split is never used.
        """
        r_train, r_val, r_test = self.real_splits()
        dev = np.concatenate([r_train, r_val])
        test_groups = set(self.groups[r_test])
        folds = []
        for tr, va in StratifiedKFold(n_folds, shuffle=True, random_state=SEED).split(dev, self.y[dev]):
            eval_idx = dev[va]
            if self.train_on == "real":
                folds.append((dev[tr], eval_idx, eval_idx))  # original behaviour: stop + score on the fold
                continue
            pool = self.training_pool(exclude_groups=set(self.groups[eval_idx]) | test_groups, real_rows=dev[tr])
            train, stop = group_split(pool, self.y, self.groups, STOP_SPLITS)
            folds.append((train, stop, eval_idx))
        return folds


def prepare_dataset(train_on: str = "synthetic") -> Dataset:
    """Real + synthetic messages with one label encoding (fit on the real intent labels)."""
    if train_on not in TRAIN_SOURCES:
        raise ValueError(f"train_on must be one of {TRAIN_SOURCES}")
    real, _, y_real, encoder = prepare_data()
    real = real.assign(source="real", group=real["id"])
    parts = [real[["id", TEXT_COL, TARGET_COL, "source", "group"]]]
    if train_on != "real":
        from generate_synthetic import load_labelled  # local import: only needed for synthetic data

        syn = load_labelled()
        unknown = set(syn["intent"]) - set(encoder.classes_)
        if unknown:
            raise ValueError(f"Synthetic intents not in the real label set: {sorted(unknown)} - "
                             "were INTENT_RULES changed after generation?")
        parts.append(pd.DataFrame({"id": syn["synthetic_id"], TEXT_COL: syn["text"], TARGET_COL: syn["intent"],
                                   "source": "synthetic", "group": syn["seed_id"]}))
    df = pd.concat(parts, ignore_index=True)
    y = encoder.transform(df[TARGET_COL].astype(str))
    assert (y[: len(real)] == y_real).all()
    return Dataset(df=df, tokens=tokenize(df[TEXT_COL]), y=y, encoder=encoder,
                   groups=df["group"].astype(str).to_numpy(), n_real=len(real), train_on=train_on)


def fit(
    tokens: list[list[str]], y: np.ndarray, idx_train: np.ndarray, idx_val: np.ndarray,
    n_classes: int, *, seed: int = SEED, verbose: int = 2, **params,
) -> dict:
    """Build vocab + Word2Vec on idx_train, train the LSTM (early-stopped on idx_val - the
    'stop' set, which is a synthetic hold-out when training on synthetic data).

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
    parser.add_argument("--train-on", choices=TRAIN_SOURCES, default="synthetic",
                        help="Training data (val/test are always the real splits)")
    parser.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("gensim").setLevel(logging.WARNING)

    ds = prepare_dataset(args.train_on)
    splits = ds.splits()
    log.info("Training on %s | split sizes: %s | classes=%d", args.train_on,
             {k: len(v) for k, v in splits.items()}, ds.n_classes)

    params = {k: v for k, v in vars(args).items() if k in DEFAULT_PARAMS}
    params["w2v_init"] = not args.no_w2v_init
    fitted = fit(ds.tokens, ds.y, splits["train"], splits["stop"], ds.n_classes, seed=args.seed, **params)
    fitted["model"].summary(print_fn=log.info)
    log.info("Vocab: %d tokens (incl. pad/unk) | max_len=%d | best epoch: %d",
             len(fitted["vocab"]), fitted["max_len"], fitted["best_epoch"])

    metrics = evaluate(fitted, ds.y, splits)
    log.info("Metrics: %s", metrics)
    report = save(fitted, ds.encoder, ds.y, splits["test"], metrics, args.out_dir,
                  extra={"train_on": args.train_on, "split_sizes": {k: len(v) for k, v in splits.items()}})
    log.info("Test classification report:\n%s", report)
    log.info("Saved model and artifacts to %s", args.out_dir)


if __name__ == "__main__":
    main()
