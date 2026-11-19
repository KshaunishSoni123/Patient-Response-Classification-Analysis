r"""End-to-end message classifier: intent LSTM + sentiment head -> action policy.

    text --> intent LSTM (models/intent_lstm_tuned)          --> intent probabilities --+
         \-> frozen Hospital_Reviews BERT + head (models/sentiment_head) --> sentiment --+--> action_policy.decide()

Usage:
    python classify.py "I need to book a check-up" "I have chest pain"
    python classify.py --evaluate            # action F1 + confusion matrix on the test split

Evaluation compares the predicted action against the reference `action` in
data/processed/healthcare_actions.csv (the policy applied to the true intent).
The reference sentiment is the sentiment head's own output (there are no human
sentiment labels for the healthcare data), so the evaluation measures how intent
errors and the safety overrides change the action, not sentiment errors.
"""

from __future__ import annotations

# Import first: sets the Keras backend and blocks the TensorFlow import that segfaults here.
import train_intent_lstm as intent_lstm  # isort: skip

import argparse
import json
import logging
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import action_policy as policy
from train_sentiment_head import FrozenEncoder

ROOT = Path(__file__).resolve().parent
INTENT_DIR = ROOT / "models" / "intent_lstm_tuned"
SENTIMENT_DIR = ROOT / "models" / "sentiment_head"
EVAL_DIR = ROOT / "models" / "action_eval"

log = logging.getLogger("classify")


class MessageClassifier:
    def __init__(self, intent_dir: Path = INTENT_DIR, sentiment_dir: Path = SENTIMENT_DIR):
        self.intent_model = keras.models.load_model(intent_dir / "model.keras")
        pre = joblib.load(intent_dir / "preprocessing.joblib")
        self.vocab, self.max_len = pre["vocab"], pre["max_len"]
        self.intent_classes = list(pre["label_encoder"].classes_)
        policy.check_intents(self.intent_classes)

        self.sentiment_head = keras.models.load_model(sentiment_dir / "head.keras")
        spre = joblib.load(sentiment_dir / "preprocessing.joblib")
        self.sentiment_scaler, self.sentiment_classes = spre["scaler"], spre["classes"]
        self.pooling = spre["pooling"]
        self.encoder = FrozenEncoder(spre["encoder_model"])

    def intent_probs(self, texts: list[str]) -> np.ndarray:
        X = intent_lstm.encode(intent_lstm.tokenize(texts), self.vocab, self.max_len)
        return self.intent_model.predict(X, verbose=0)

    def sentiment_probs(self, texts: list[str]) -> np.ndarray:
        feats, _ = self.encoder.encode(texts, pooling=self.pooling)
        return self.sentiment_head.predict(self.sentiment_scaler.transform(feats).astype(np.float32), verbose=0)

    def predict(self, texts: list[str]) -> pd.DataFrame:
        ip, sp = self.intent_probs(texts), self.sentiment_probs(texts)
        sentiments = np.array(self.sentiment_classes)[sp.argmax(1)]
        out = policy.decide_batch(ip, self.intent_classes, sentiments)
        out.insert(0, "text", texts)
        out["sentiment_confidence"] = sp.max(1)
        return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(clf: MessageClassifier, out_dir: Path) -> None:
    ref = pd.read_csv(policy.ACTIONS_CSV)
    # Same rows, order and split as intent-model training.
    df, _, y, _ = intent_lstm.prepare_data()
    if not (df["id"].to_numpy() == ref["id"].to_numpy()).all():
        raise RuntimeError(f"{policy.ACTIONS_CSV.name} is stale; rerun action_policy.py")
    _, _, idx_test = intent_lstm.split_indices(y)
    test = ref.iloc[idx_test].reset_index(drop=True)

    pred = clf.predict(test["text"].tolist())
    sentiment_mismatch = (pred["sentiment"] != test["sentiment"]).sum()
    if sentiment_mismatch:
        log.warning("%d test messages got a different sentiment than the training data", sentiment_mismatch)

    y_true, y_pred = test["action"], pred["action"]
    labels = policy.ACTIONS
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                      index=[f"true: {a}" for a in labels], columns=labels)

    log.info("Test messages: %d | action macro F1 = %.4f | weighted F1 = %.4f", len(test), macro, weighted)
    log.info("Per-action report:\n%s", report)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        log.info("Confusion matrix (rows = reference action, columns = predicted action):\n%s", cm.to_string())
    log.info("Decision reasons:\n%s", pred["reason"].value_counts().to_string())

    errors = pd.concat([test[["id", "text", "intent_label", "sentiment", "action"]],
                        pred[["intent", "intent_confidence", "emergency_probability", "reason"]]
                        .rename(columns={"intent": "pred_intent"}),
                        y_pred.rename("pred_action")], axis=1)
    errors = errors[errors["action"] != errors["pred_action"]]
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        log.info("Action errors (%d):\n%s", len(errors), errors[
            ["text", "intent_label", "pred_intent", "action", "pred_action", "reason"]].to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    cm.to_csv(out_dir / "confusion_matrix.csv")
    (out_dir / "classification_report.txt").write_text(report)
    errors.to_csv(out_dir / "errors.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_test": len(test), "action_macro_f1": round(float(macro), 4),
        "action_weighted_f1": round(float(weighted), 4),
        "action_accuracy": round(float((y_true == y_pred).mean()), 4),
        "reasons": pred["reason"].value_counts().to_dict(),
        "emergency_threshold": policy.EMERGENCY_THRESHOLD,
        "low_confidence_threshold": policy.LOW_CONFIDENCE_THRESHOLD,
    }, indent=2))
    log.info("Saved evaluation to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="*", help="Messages to classify")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate actions on the test split")
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    args = parser.parse_args()
    if not args.texts and not args.evaluate:
        parser.error("give one or more texts, or --evaluate")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("healthcare_ingestion", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    clf = MessageClassifier()
    if args.texts:
        with pd.option_context("display.width", 250, "display.max_colwidth", 60):
            print(clf.predict(args.texts).round(3).to_string(index=False))
    if args.evaluate:
        evaluate(clf, args.out_dir)


if __name__ == "__main__":
    main()
