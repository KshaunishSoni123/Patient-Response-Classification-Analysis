r"""End-to-end message classifier: intent LSTM + sentiment head -> action DNN.

    text --> intent LSTM (models/intent_lstm_tuned) --+--> intent probs (17) ----------------------+
         |                                            +--> LSTM final hidden state (64) = text ----+
         +-> frozen Hospital_Reviews BERT + head (models/sentiment_head) -----> sentiment probs (3) --+
                                                                                                       |
            concat [text vector | intent probs | sentiment probs] -> 1x Dense(64, relu) -> softmax -> action

Action DNN (models/action_dnn), default --text-input lstm_state (chosen in experiments.py):
  * Text vector: the intent LSTM's final hidden state (frozen; the intent model's own reading of
    the message). --text-input avg_pool restores the v1 design: a trainable Embedding layer (same
    vocabulary, initialised from the intent model's Word2Vec) with masked average pooling.
    The LSTM state for the synthetic training rows comes from the final intent model (hidden
    states of different fold models aren't in a common basis, so they can't be stacked out-of-fold).
  * 1 hidden ReLU layer of 64 units (dropout after it), softmax over the 8 actions.
    v1 used 5 x 64 with avg_pool; experiments.py showed 1 x 64 + lstm_state is better on real
    messages (real-test macro F1 0.736 -> 0.806, mean of 3 seeds) with ~30x fewer parameters.
  * Trained on the synthetic messages (label = action_label: reviewer_action, else the
    action policy on the final intent/sentiment labels, with the LLM emergency flag).
  * Data: the shared synthetic split (synthetic_splits.py) - train = rewrites of real train
    messages, val / test = rewrites of real val messages (split by seed). The intent model trains
    on real train messages and the sentiment head on synthetic train, so neither has seen a
    synthetic val/test row or its seed message.
  * Inputs per split (action_features): val / test rows get the FINAL models' predictions, so
    synthetic-val/test scores are true end-to-end scores. Train rows get OUT-OF-FOLD predictions
    (5 seed-grouped folds over the train rows; fresh intent LSTM and sentiment head fit without
    each fold - and without the real messages it rewrites), so the DNN learns from probabilities
    as noisy as the ones it gets on new messages. Cached in <action dir>/oof_features.npz;
    --oof-folds 0 uses the final models' in-sample predictions (faster, over-confident).
  * Early stopping and seed choice on synthetic val; several seeds, best kept by val.

Usage:
    python classify.py --train                         # train the action DNN (needs 3a + 3b)
    python classify.py "I need to book a check-up" "I have chest pain"
    python classify.py --evaluate                      # action F1 + confusion matrix on real test messages

Evaluation compares predicted actions on the 437 real test messages against the reference
`action` in data/processed/healthcare_actions.csv (the policy applied to the true intent and
the sentiment head's sentiment; evaluation_report.py adds the hand-labelled reference). The
rule-based routing table (action_policy.decide, with its emergency / low-confidence
overrides) is reported alongside as a baseline.
"""

from __future__ import annotations

# Import first: sets the Keras backend and blocks the TensorFlow import that segfaults here.
import train_intent_lstm as intent_lstm  # isort: skip

import argparse
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

import action_policy as policy
import synthetic_splits
import train_sentiment_head as sentiment_head
from train_sentiment_head import FrozenEncoder

ROOT = Path(__file__).resolve().parent
INTENT_DIR = ROOT / "models" / "intent_lstm_tuned"
SENTIMENT_DIR = ROOT / "models" / "sentiment_head"
ACTION_DIR = ROOT / "models" / "action_dnn"
EVAL_DIR = ROOT / "models" / "action_eval"
SEED = 42

ACTION_PARAMS = dict(hidden_layers=1, hidden_units=64, dropout=0.1, lr=1e-3, batch_size=32,
                     epochs=200, patience=15, trainable_embedding=True, text_input="lstm_state")
TEXT_INPUTS = ("lstm_state", "avg_pool")

log = logging.getLogger("classify")


# --------------------------------------------------------------------------- #
# Action DNN
# --------------------------------------------------------------------------- #
def build_action_dnn(embedding: np.ndarray, n_intents: int, n_sentiments: int, n_actions: int, *,
                     hidden_layers: int, hidden_units: int, dropout: float, lr: float,
                     trainable_embedding: bool, **_) -> keras.Model:
    tokens = keras.Input(shape=(None,), dtype="int64", name="tokens")
    intent_probs = keras.Input(shape=(n_intents,), name="intent_probs")
    sentiment_probs = keras.Input(shape=(n_sentiments,), name="sentiment_probs")

    emb_layer = keras.layers.Embedding(embedding.shape[0], embedding.shape[1], mask_zero=True,
                                       trainable=trainable_embedding, name="embedding")
    text = keras.layers.GlobalAveragePooling1D(name="text_embedding")(emb_layer(tokens))  # masked mean
    x = keras.layers.Concatenate(name="features")([text, intent_probs, sentiment_probs])
    for i in range(hidden_layers):
        x = keras.layers.Dense(hidden_units, activation="relu", name=f"hidden_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)
    out = keras.layers.Dense(n_actions, activation="softmax", name="action")(x)

    model = keras.Model([tokens, intent_probs, sentiment_probs], out, name="action_dnn")
    emb_layer.set_weights([embedding])  # set after building (a Constant initializer would bloat the saved config)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_state_action_dnn(state_dim: int, n_intents: int, n_sentiments: int, n_actions: int, *,
                           hidden_layers: int, hidden_units: int, dropout: float, lr: float, **_) -> keras.Model:
    """Action DNN on the intent LSTM's hidden state (instead of an averaged embedding)."""
    state = keras.Input(shape=(state_dim,), name="lstm_state")
    intent_probs = keras.Input(shape=(n_intents,), name="intent_probs")
    sentiment_probs = keras.Input(shape=(n_sentiments,), name="sentiment_probs")
    x = keras.layers.Concatenate(name="features")([state, intent_probs, sentiment_probs])
    for i in range(hidden_layers):
        x = keras.layers.Dense(hidden_units, activation="relu", name=f"hidden_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)
    out = keras.layers.Dense(n_actions, activation="softmax", name="action")(x)
    model = keras.Model([state, intent_probs, sentiment_probs], out, name="action_state_dnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def lstm_layer_name(model: keras.Model) -> str:
    return "bilstm" if any(layer.name == "bilstm" for layer in model.layers) else "lstm"


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class MessageClassifier:
    def __init__(self, intent_dir: Path = INTENT_DIR, sentiment_dir: Path = SENTIMENT_DIR,
                 action_dir: Path | None = ACTION_DIR):
        self.intent_model = keras.models.load_model(intent_dir / "model.keras")
        pre = joblib.load(intent_dir / "preprocessing.joblib")
        self.vocab, self.max_len = pre["vocab"], pre["max_len"]
        self.intent_classes = list(pre["label_encoder"].classes_)
        policy.check_intents(self.intent_classes)
        # One pass of the intent model returns both its LSTM state and its probabilities.
        self.intent_features_model = keras.Model(
            self.intent_model.inputs,
            [self.intent_model.get_layer(lstm_layer_name(self.intent_model)).output,
             self.intent_model.get_layer("intent").output])

        self.sentiment_head = keras.models.load_model(sentiment_dir / "head.keras")
        spre = joblib.load(sentiment_dir / "preprocessing.joblib")
        self.sentiment_scaler, self.sentiment_classes = spre["scaler"], list(spre["classes"])
        self.pooling = spre["pooling"]
        self.encoder = FrozenEncoder(spre["encoder_model"])

        self.action_model = None
        self.text_input = None
        if action_dir is not None:
            if not (action_dir / "model.keras").exists():
                raise FileNotFoundError(f"No action model in {action_dir}; run `python classify.py --train` first")
            self.action_model = keras.models.load_model(action_dir / "model.keras")
            apre = joblib.load(action_dir / "preprocessing.joblib")
            self.actions = apre["actions"]
            self.text_input = apre.get("text_input", "avg_pool")  # v1 models predate the option
            if (apre["intent_classes"] != self.intent_classes or apre["sentiment_classes"] != self.sentiment_classes
                    or apre["vocab"] != self.vocab):
                raise RuntimeError("The action DNN was trained against different intent/sentiment models; "
                                   "rerun `python classify.py --train`")

    def tokens(self, texts: list[str]) -> np.ndarray:
        return intent_lstm.encode(intent_lstm.tokenize(texts), self.vocab, self.max_len)

    def action_tokens(self, texts: list[str]) -> np.ndarray:
        """Token ids for the action DNN; an empty message becomes one unknown-word token
        (an all-padding sequence would make the masked average divide by zero -> NaN)."""
        X = self.tokens(texts)
        X[(X == 0).all(axis=1), 0] = 1
        return X

    def intent_probs(self, texts: list[str]) -> np.ndarray:
        return self.intent_model.predict(self.tokens(texts), verbose=0)

    def intent_state_and_probs(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(intent LSTM final hidden state, intent probabilities) in one pass."""
        state, probs = self.intent_features_model.predict(self.tokens(texts), batch_size=256, verbose=0)
        return np.asarray(state, dtype=np.float32), np.asarray(probs, dtype=np.float32)

    def action_inputs(self, texts: list[str], intent_probs: np.ndarray, sentiment_probs: np.ndarray,
                      state: np.ndarray | None = None, text_input: str | None = None) -> list[np.ndarray]:
        """Inputs for the action DNN: [text vector, intent probs, sentiment probs]."""
        text_input = text_input or self.text_input
        if text_input == "lstm_state":
            text = state if state is not None else self.intent_state_and_probs(texts)[0]
        elif text_input == "avg_pool":
            text = self.action_tokens(texts)
        else:
            raise ValueError(f"Unknown text input {text_input!r}")
        return [text, intent_probs, sentiment_probs]

    def sentiment_features(self, texts: list[str], cache: bool = False) -> np.ndarray:
        """Frozen BERT features; cache=True stores them on disk (for the fixed training/eval sets)."""
        if cache:
            return sentiment_head.encode_cached(self.encoder, texts, self.pooling)[0]
        return self.encoder.encode(texts, pooling=self.pooling)[0]

    def sentiment_probs(self, texts: list[str], feats: np.ndarray | None = None, cache: bool = False) -> np.ndarray:
        feats = self.sentiment_features(texts, cache) if feats is None else feats
        return self.sentiment_head.predict(self.sentiment_scaler.transform(feats).astype(np.float32), verbose=0)

    def predict(self, texts: list[str], cache: bool = False) -> pd.DataFrame:
        texts = list(texts)
        state, ip = self.intent_state_and_probs(texts)
        sp = self.sentiment_probs(texts, cache=cache)
        sentiments = np.array(self.sentiment_classes)[sp.argmax(1)]
        # Rule-based routing table, kept as a baseline / explanation.
        rules = policy.decide_batch(ip, self.intent_classes, sentiments)
        out = pd.DataFrame({
            "text": texts,
            "intent": rules["intent"], "intent_confidence": rules["intent_confidence"],
            "emergency_probability": rules["emergency_probability"],
            "sentiment": sentiments, "sentiment_confidence": sp.max(1),
        })
        if self.action_model is not None:
            ap = self.action_model.predict(self.action_inputs(texts, ip, sp, state=state), verbose=0)
            out["action"] = np.array(self.actions)[ap.argmax(1)]
            out["action_confidence"] = ap.max(1)
        out["policy_action"], out["policy_reason"] = rules["action"], rules["reason"]
        return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _fingerprint(*parts) -> str:
    return hashlib.sha256("\n".join(map(str, parts)).encode()).hexdigest()[:16]


def action_features(clf: MessageClassifier, syn: pd.DataFrame, split: dict[str, np.ndarray], n_folds: int,
                    cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Intent and sentiment probabilities for every synthetic row, as the action DNN should see them.

    * synthetic val / test rows: the FINAL intent and sentiment models' predictions. Neither model has
      seen these rows or their seed message (see synthetic_splits.py), so this is exactly what the
      pipeline produces for a new message - val/test scores are true end-to-end scores.
    * synthetic train rows: OUT-OF-FOLD predictions (n_folds seed-grouped folds over the train rows).
      For each fold a fresh intent LSTM and sentiment head are fit without that fold's rows - and, for
      the intent model, without the real messages those rows rewrite - so the action DNN trains on
      probabilities as noisy as the ones it will get on new messages. n_folds=0 uses the final models.
    """
    texts = syn["text"].tolist()
    held = np.concatenate([split["val"], split["test"]])
    ip = np.zeros((len(syn), len(clf.intent_classes)), dtype=np.float32)
    sp = np.zeros((len(syn), len(clf.sentiment_classes)), dtype=np.float32)
    ip[held] = clf.intent_probs([texts[i] for i in held])
    sp[held] = clf.sentiment_probs([texts[i] for i in held], cache=True)
    tr_rows = split["train"]
    if n_folds == 0:
        log.warning("--oof-folds 0: train-row inputs are in-sample predictions of the final models (over-confident)")
        ip[tr_rows] = clf.intent_probs([texts[i] for i in tr_rows])
        sp[tr_rows] = clf.sentiment_probs([texts[i] for i in tr_rows], cache=True)
        return ip, sp

    intent_meta = json.loads((INTENT_DIR / "metrics.json").read_text())
    sent_meta = json.loads((SENTIMENT_DIR / "metrics.json").read_text())
    key = _fingerprint(n_folds, synthetic_splits.fingerprint(), *syn["synthetic_id"], *syn["intent"], *syn["sentiment"],
                       json.dumps(intent_meta, sort_keys=True), json.dumps(sent_meta, sort_keys=True))
    cache = cache_dir / "oof_features.npz"
    if cache.exists():
        c = np.load(cache)
        if str(c["key"]) == key:
            log.info("Using cached out-of-fold train features from %s", cache)
            ip[tr_rows], sp[tr_rows] = c["intent"], c["sentiment"]
            return ip, sp

    groups = syn["seed_id"].astype(str).to_numpy()
    folds = list(StratifiedGroupKFold(n_folds, shuffle=True, random_state=SEED)
                 .split(tr_rows, syn["action_label"].iloc[tr_rows], groups[tr_rows]))

    # Intent: refit the LSTM (saved model's hyperparameters and training source) without each fold.
    train_on = intent_meta.get("train_on", "synthetic")
    ds = intent_lstm.prepare_dataset(train_on)
    if list(ds.encoder.classes_) != clf.intent_classes:
        raise RuntimeError("Intent classes don't line up with the intent model; retrain step 3a")
    params = {k: v for k, v in intent_meta["params"].items() if k in intent_lstm.DEFAULT_PARAMS}
    base = ds.splits()  # the final model's train / stop rows (real and/or synthetic-train)
    for k, (tr, te) in enumerate(folds):
        rows_out = tr_rows[te]
        seeds_out = set(groups[rows_out])
        ids_out = set(syn["synthetic_id"].iloc[rows_out])
        keep = ~np.isin(ds.groups[base["train"]], list(seeds_out)) & ~ds.df["id"].iloc[base["train"]].isin(ids_out).to_numpy()
        fitted = intent_lstm.fit(ds.tokens, ds.y, base["train"][keep], base["stop"], ds.n_classes,
                                 seed=SEED, verbose=0, **params)
        X_out = intent_lstm.encode(intent_lstm.tokenize([texts[i] for i in rows_out]), fitted["vocab"], fitted["max_len"])
        ip[rows_out] = fitted["model"].predict(X_out, verbose=0)
        y_out = ds.encoder.transform(syn["intent"].iloc[rows_out])
        log.info("OOF intent fold %d/%d: accuracy %.3f on %d train rows (%d of %d intent training rows kept)",
                 k + 1, n_folds, accuracy_score(y_out, ip[rows_out].argmax(1)), len(rows_out), keep.sum(), len(keep))

    # Sentiment: refit the head (saved model's settings) on the other folds, early-stopped on synthetic val.
    X = clf.sentiment_features(texts, cache=True)
    ys = syn["sentiment"].map({c: i for i, c in enumerate(clf.sentiment_classes)}).to_numpy()
    head_args = SimpleNamespace(**sent_meta["args"])
    for k, (tr, te) in enumerate(folds):
        model, scaler, _ = sentiment_head.fit_head(X, ys, tr_rows[tr], split["val"], head_args, SEED)
        sp[tr_rows[te]] = sentiment_head.predict(model, scaler, X[tr_rows[te]])
        log.info("OOF sentiment fold %d/%d: accuracy %.3f", k + 1, n_folds,
                 accuracy_score(ys[tr_rows[te]], sp[tr_rows[te]].argmax(1)))

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=key, intent=ip[tr_rows], sentiment=sp[tr_rows])
    return ip, sp


def train(out_dir: Path, n_folds: int, seeds: list[int], params: dict) -> None:
    from generate_synthetic import load_labelled

    sent_meta = json.loads((SENTIMENT_DIR / "metrics.json").read_text())
    if sent_meta.get("split") != synthetic_splits.SCHEME:
        raise RuntimeError("The sentiment head was not trained on the shared synthetic split; "
                           "rerun train_sentiment_head.py first")

    clf = MessageClassifier(action_dir=None)
    syn = load_labelled()
    actions = list(policy.ACTIONS)
    y = syn["action_label"].map({a: i for i, a in enumerate(actions)}).to_numpy()
    log.info("Action labels (synthetic):\n%s", syn["action_label"].value_counts().to_string())
    split = synthetic_splits.split_indices(syn)
    idx_train, idx_val, idx_test = split["train"], split["val"], split["test"]
    log.info("Shared synthetic split (%s): train=%d val=%d test=%d", synthetic_splits.SCHEME,
             len(idx_train), len(idx_val), len(idx_test))

    ip, sp = action_features(clf, syn, split, n_folds, out_dir)
    texts = syn["text"].tolist()
    text_input = params["text_input"]
    if text_input == "lstm_state":
        text = clf.intent_state_and_probs(texts)[0]  # final intent model (see module docstring)
    else:
        text = clf.action_tokens(texts)
    embedding = clf.intent_model.get_layer("embedding").get_weights()[0]
    inputs = lambda idx: [text[idx], ip[idx], sp[idx]]  # noqa: E731
    log.info("Action DNN: text input %s, %d x %d hidden", text_input, params["hidden_layers"], params["hidden_units"])

    weights = compute_class_weight("balanced", classes=np.arange(len(actions)), y=y[idx_train])

    def score(model, ids):
        pred = model.predict(inputs(ids), verbose=0).argmax(1)
        return {"accuracy": round(float(accuracy_score(y[ids], pred)), 4),
                "macro_f1": round(float(f1_score(y[ids], pred, average="macro")), 4)}

    runs = []
    for seed in seeds:
        keras.utils.set_random_seed(seed)
        if text_input == "lstm_state":
            model = build_state_action_dnn(text.shape[1], len(clf.intent_classes), len(clf.sentiment_classes),
                                           len(actions), **params)
        else:
            model = build_action_dnn(embedding, len(clf.intent_classes), len(clf.sentiment_classes),
                                     len(actions), **params)
        hist = model.fit(inputs(idx_train), y[idx_train], validation_data=(inputs(idx_val), y[idx_val]),
                         epochs=params["epochs"], batch_size=params["batch_size"],
                         class_weight=dict(enumerate(weights)), verbose=0,
                         callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"],
                                                                  restore_best_weights=True)])
        run = {"seed": seed, "model": model, "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
               "val": score(model, idx_val), "test": score(model, idx_test)}
        log.info("Seed %d: best_epoch=%d val=%s synthetic test=%s", seed, run["best_epoch"], run["val"], run["test"])
        runs.append(run)
    best = max(runs, key=lambda r: r["val"]["macro_f1"])  # choose by val, never test
    best["model"].summary(print_fn=log.info)

    test_pred = best["model"].predict(inputs(idx_test), verbose=0).argmax(1)
    report = classification_report(y[idx_test], test_pred, labels=np.arange(len(actions)),
                                   target_names=actions, zero_division=0)
    log.info("Kept seed %d. Synthetic test report:\n%s", best["seed"], report)

    out_dir.mkdir(parents=True, exist_ok=True)
    best["model"].save(out_dir / "model.keras")
    joblib.dump({"actions": actions, "text_input": text_input, "intent_classes": clf.intent_classes,
                 "sentiment_classes": clf.sentiment_classes, "vocab": clf.vocab, "max_len": clf.max_len},
                out_dir / "preprocessing.joblib")
    (out_dir / "synthetic_test_report.txt").write_text(report)
    (out_dir / "metrics.json").write_text(json.dumps({
        "params": params, "oof_folds": n_folds, "split": synthetic_splits.SCHEME,
        "split_fingerprint": synthetic_splits.fingerprint(), "seed": best["seed"], "best_epoch": best["best_epoch"],
        "split_sizes": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)},
        "val": best["val"], "synthetic_test": best["test"],
        "seeds": [{k: r[k] for k in ("seed", "best_epoch", "val", "test")} for r in runs],
    }, indent=2))
    np.savez(out_dir / "action_inputs.npz", intent=ip, sentiment=sp)  # what evaluation scores the DNN on
    log.info("Saved action DNN to %s", out_dir)


# --------------------------------------------------------------------------- #
# Evaluation on real messages
# --------------------------------------------------------------------------- #
def evaluate(clf: MessageClassifier, out_dir: Path) -> None:
    ref = pd.read_csv(policy.ACTIONS_CSV)
    # Same rows, order and split as intent-model training.
    df, _, y, _ = intent_lstm.prepare_data()
    if not (df["id"].to_numpy() == ref["id"].to_numpy()).all():
        raise RuntimeError(f"{policy.ACTIONS_CSV.name} is stale; rerun action_policy.py")
    _, _, idx_test = intent_lstm.split_indices(y)
    test = ref.iloc[idx_test].reset_index(drop=True)

    pred = clf.predict(test["text"].tolist(), cache=True)
    sentiment_mismatch = (pred["sentiment"] != test["sentiment"]).sum()
    if sentiment_mismatch:
        log.warning("%d test messages got a different sentiment than %s; rerun action_policy.py",
                    sentiment_mismatch, policy.ACTIONS_CSV.name)

    labels = policy.ACTIONS
    y_true = test["action"]
    results = {}
    for name, col in (("action_dnn", "action"), ("routing_table", "policy_action")):
        y_pred = pred[col]
        results[name] = {
            "action_macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
            "action_weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted",
                                                       zero_division=0)), 4),
            "action_accuracy": round(float((y_true == y_pred).mean()), 4),
        }
    log.info("Real test messages: %d\n%s", len(test), pd.DataFrame(results).T.to_string())

    y_pred = pred["action"]
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                      index=[f"true: {a}" for a in labels], columns=labels)
    baseline_cm = pd.DataFrame(confusion_matrix(y_true, pred["policy_action"], labels=labels),
                               index=[f"true: {a}" for a in labels], columns=labels)
    log.info("Action DNN per-action report:\n%s", report)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        log.info("Action DNN confusion matrix (rows = reference action, columns = predicted action):\n%s",
                 cm.to_string())

    errors = pd.concat([test[["id", "text", "intent_label", "sentiment", "action"]],
                        pred[["intent", "intent_confidence", "emergency_probability", "action_confidence",
                              "policy_action"]].rename(columns={"intent": "pred_intent"}),
                        y_pred.rename("pred_action")], axis=1)
    errors = errors[errors["action"] != errors["pred_action"]]
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        log.info("Action DNN errors (%d):\n%s", len(errors), errors[
            ["text", "intent_label", "pred_intent", "action", "pred_action", "policy_action"]].to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    cm.to_csv(out_dir / "confusion_matrix.csv")
    baseline_cm.to_csv(out_dir / "confusion_matrix_routing_table.csv")
    (out_dir / "classification_report.txt").write_text(report)
    (out_dir / "classification_report_routing_table.txt").write_text(
        classification_report(y_true, pred["policy_action"], labels=labels, zero_division=0))
    errors.to_csv(out_dir / "errors.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_test": len(test), **results["action_dnn"], "routing_table_baseline": results["routing_table"],
        "routing_table_reasons": pred["policy_reason"].value_counts().to_dict(),
        "emergency_threshold": policy.EMERGENCY_THRESHOLD,
        "low_confidence_threshold": policy.LOW_CONFIDENCE_THRESHOLD,
    }, indent=2))
    log.info("Saved evaluation to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="*", help="Messages to classify")
    parser.add_argument("--train", action="store_true", help="Train the action DNN on the synthetic data")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate actions on the real test split")
    parser.add_argument("--oof-folds", type=int, default=5,
                        help="Folds for out-of-fold intent/sentiment inputs (0 = in-sample, faster)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--hidden-layers", type=int, default=ACTION_PARAMS["hidden_layers"])
    parser.add_argument("--hidden-units", type=int, default=ACTION_PARAMS["hidden_units"])
    parser.add_argument("--dropout", type=float, default=ACTION_PARAMS["dropout"])
    parser.add_argument("--lr", type=float, default=ACTION_PARAMS["lr"])
    parser.add_argument("--epochs", type=int, default=ACTION_PARAMS["epochs"])
    parser.add_argument("--freeze-embedding", action="store_true", help="Don't train the embedding layer (avg_pool)")
    parser.add_argument("--text-input", choices=TEXT_INPUTS, default=ACTION_PARAMS["text_input"],
                        help="Text vector for the action DNN: intent LSTM hidden state (default) or "
                             "averaged trainable embedding (v1)")
    parser.add_argument("--action-dir", type=Path, default=ACTION_DIR)
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR, help="Evaluation output folder")
    args = parser.parse_args()
    if not (args.texts or args.train or args.evaluate):
        parser.error("give one or more texts, --train, or --evaluate")
    if args.oof_folds == 1 or args.oof_folds < 0:
        parser.error("--oof-folds must be 0 or >= 2")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("healthcare_ingestion", "httpx", "huggingface_hub", "generate_synthetic", "gensim",
                  "train_intent_lstm", "train_sentiment_head"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.train:
        params = {**ACTION_PARAMS, "hidden_layers": args.hidden_layers, "hidden_units": args.hidden_units,
                  "dropout": args.dropout, "lr": args.lr, "epochs": args.epochs,
                  "trainable_embedding": not args.freeze_embedding, "text_input": args.text_input}
        train(args.action_dir, args.oof_folds, args.seeds, params)
    if args.texts or args.evaluate:
        clf = MessageClassifier(action_dir=args.action_dir)
        if args.texts:
            with pd.option_context("display.width", 250, "display.max_colwidth", 60):
                print(clf.predict(args.texts).round(3).to_string(index=False))
        if args.evaluate:
            evaluate(clf, args.out_dir)


if __name__ == "__main__":
    main()
