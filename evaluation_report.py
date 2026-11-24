"""Evaluation report for the three saved models: summary metrics with 95% CIs, per-class
precision / recall, and confusion matrices for the train, validation and test splits.

Every split is rebuilt with the training code's own split functions. All synthetic rows use the
SHARED synthetic split (synthetic_splits.py: train = rewrites of real train messages, val / test =
rewrites of real val messages, split by seed), so the three models are scored on the same
synthetic test messages:

  intent LSTM     train (its training source) | validation (real) | test (real, 437 messages)
                  | test (synthetic: rewrites of messages it never trained on)
  sentiment head  train / validation / test (synthetic) | test (customer support, out-of-domain)
  action DNN      train / validation / test (synthetic). Val/test inputs come from the FINAL
                  intent and sentiment models, so these are end-to-end pipeline scores; train
                  inputs are the out-of-fold probabilities it was trained on.
                  | test (real; reference = policy(true intent, sentiment-head sentiment))
  routing table   test (synthetic; reference = LLM-labelled action - not circular) | test (real)

With hand labels (labels/real_test_sentiment_*.csv, from label_sentiment.py) three more rows:
  sentiment head  test (real, human labels)
  action DNN / routing table  test (real, human-sentiment reference): reference action =
                  policy(true intent, HUMAN sentiment) - no longer derived from the sentiment head.

95% confidence intervals: percentile bootstrap (resampling messages with replacement) for
accuracy / F1 / macro averages; Wilson score intervals for per-class precision and recall
(the bootstrap degenerates on classes with only a handful of examples).
For the real test split the reference actions depend on the sentiment head (no human labels),
so treat action scores there as agreement with the policy, not ground truth.

Usage:
    python evaluation_report.py                 # writes reports/
    python evaluation_report.py --n-boot 5000 --out-dir reports
"""

from __future__ import annotations

# Import first: sets the Keras backend and blocks the TensorFlow import that segfaults here.
import train_intent_lstm as intent_lstm  # isort: skip

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import action_policy as policy  # noqa: E402
import train_sentiment_head as sentiment_head  # noqa: E402
from classify import ACTION_DIR, INTENT_DIR, SEED, MessageClassifier  # noqa: E402
from generate_synthetic import load_labelled  # noqa: E402
from human_labels import load_human_sentiment  # noqa: E402
import synthetic_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("evaluation_report")

# Sequential blue ramp (light -> dark) for the confusion-matrix heatmaps.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_MUTED, SURFACE = "#1f1f1e", "#6b6a66", "#ffffff"
MODEL_NAMES = {"intent_lstm": "Intent LSTM", "sentiment_head": "Sentiment head",
               "action_dnn": "Action DNN", "routing_table": "Routing table (baseline)"}


# --------------------------------------------------------------------------- #
# Metrics from confusion matrices (fast enough to bootstrap thousands of times)
# --------------------------------------------------------------------------- #
def confusion(t: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(t * k + p, minlength=k * k).reshape(k, k)


def per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall, F1 per class (NaN where undefined)."""
    tp = np.diag(cm).astype(float)
    pred, true = cm.sum(0), cm.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred > 0, tp / pred, np.nan)
        recall = np.where(true > 0, tp / true, np.nan)
        f1 = np.where(pred + true > 0, 2 * tp / (pred + true), np.nan)
    return precision, recall, f1


def summary(cm: np.ndarray) -> dict[str, float]:
    """sklearn-equivalent metrics: macro averages over classes present in y_true or y_pred
    (labels=None, zero_division=0); weighted averages by true support."""
    precision, recall, f1 = per_class(cm)
    true, present = cm.sum(1), (cm.sum(0) + cm.sum(1)) > 0
    p0, r0, f0 = (np.nan_to_num(x) for x in (precision, recall, f1))
    w = true / true.sum()
    return {
        "accuracy": np.trace(cm) / cm.sum(),
        "macro_f1": f0[present].mean(),
        "weighted_f1": (f0 * w).sum(),
        "macro_precision": p0[present].mean(),
        "macro_recall": r0[present].mean(),
    }


def bootstrap_cms(t: np.ndarray, p: np.ndarray, k: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = len(t)
    idx = rng.integers(0, n, size=(n_boot, n))
    codes = t[idx] * k + p[idx]
    offsets = np.arange(n_boot)[:, None] * (k * k)
    flat = np.bincount((codes + offsets).ravel(), minlength=n_boot * k * k)
    return flat.reshape(n_boot, k, k)


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion. Used for per-class precision / recall,
    where the bootstrap collapses on small classes (e.g. 3/3 -> [1, 1])."""
    if n == 0:
        return np.nan, np.nan
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# --------------------------------------------------------------------------- #
# Predictions per model and split
# --------------------------------------------------------------------------- #
SOURCE_NAMES = {"synthetic": "synthetic", "both": "synthetic + real train", "real": "real"}


def shared_split() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    syn = load_labelled()
    return syn, synthetic_splits.split_indices(syn)


def intent_runs(clf: MessageClassifier) -> list[dict]:
    train_on = json.loads((INTENT_DIR / "metrics.json").read_text()).get("train_on", "synthetic")
    ds = intent_lstm.prepare_dataset(train_on)
    if list(ds.encoder.classes_) != clf.intent_classes:
        raise RuntimeError("Intent classes don't match the saved intent model; retrain step 3a")
    splits = ds.splits()
    texts = ds.df[intent_lstm.TEXT_COL].tolist()
    source = SOURCE_NAMES[train_on]
    rows = [("train", source, "train")]
    if not np.array_equal(splits["stop"], splits["val"]):  # real training early-stops on real val itself
        rows.append(("validation", f"{source} (early-stop hold-out)", "stop"))
    rows += [("validation", "real", "val"), ("test", "real", "test")]
    runs = []
    for split, data, key in rows:
        ids = splits[key]
        pred = clf.intent_probs([texts[i] for i in ids]).argmax(1)
        runs.append(dict(model="intent_lstm", split=split, data=data, classes=clf.intent_classes,
                         y_true=ds.y[ids], y_pred=pred))
    # Shared synthetic test split: rewrites of real val messages the intent model never trained on.
    syn, shared = shared_split()
    ids = shared["test"]
    pred = clf.intent_probs(syn["text"].iloc[ids].tolist()).argmax(1)
    runs.append(dict(model="intent_lstm", split="test", data="synthetic", classes=clf.intent_classes,
                     y_true=ds.encoder.transform(syn["intent"].iloc[ids]), y_pred=pred))
    return runs


def sentiment_runs(clf: MessageClassifier) -> list[dict]:
    classes = clf.sentiment_classes
    label_id = {c: i for i, c in enumerate(classes)}
    syn, shared = shared_split()
    y = syn["sentiment"].map(label_id).to_numpy()
    X = clf.sentiment_features(syn["text"].tolist(), cache=True)
    runs = []
    for split, key in (("train", "train"), ("validation", "val"), ("test", "test")):
        ids = shared[key]
        pred = clf.sentiment_probs([], feats=X[ids]).argmax(1)
        runs.append(dict(model="sentiment_head", split=split, data="synthetic", classes=classes,
                         y_true=y[ids], y_pred=pred))
    try:
        sup, _ = sentiment_head.load_support()
        pred = clf.sentiment_probs(sup["text"].tolist(), cache=True).argmax(1)
        runs.append(dict(model="sentiment_head", split="test", data="customer support (out-of-domain)",
                         classes=classes, y_true=sup["sentiment"].map(label_id).to_numpy(), y_pred=pred))
    except FileNotFoundError as e:
        log.warning("Skipping out-of-domain sentiment test: %s", e)
    return runs


def action_runs(clf: MessageClassifier) -> list[dict]:
    """Action DNN on the shared synthetic split (val/test = end-to-end: inputs from the final
    intent and sentiment models), the routing table on synthetic test, and both on real test."""
    actions = list(clf.actions)
    a_id = {a: i for i, a in enumerate(actions)}
    syn, shared = shared_split()
    y = syn["action_label"].map(a_id).to_numpy()

    saved = ACTION_DIR / "action_inputs.npz"
    if not saved.exists():
        raise FileNotFoundError(f"{saved} missing; rerun `python classify.py --train`")
    inp = np.load(saved)
    ip, sp = inp["intent"], inp["sentiment"]
    if len(ip) != len(syn):
        raise RuntimeError(f"{saved.name} doesn't match the synthetic data; rerun `python classify.py --train`")
    texts = syn["text"].tolist()
    text = clf.intent_state_and_probs(texts)[0] if clf.text_input == "lstm_state" else clf.action_tokens(texts)

    runs = []
    for split, key in (("train", "train"), ("validation", "val"), ("test", "test")):
        ids = shared[key]
        pred = clf.action_model.predict([text[ids], ip[ids], sp[ids]], verbose=0).argmax(1)
        runs.append(dict(model="action_dnn", split=split, data="synthetic", classes=actions,
                         y_true=y[ids], y_pred=pred))
    # Routing table on the same synthetic test messages. The reference here is the LLM-labelled
    # action, not the sentiment head's, so this comparison is not circular.
    ids = shared["test"]
    sentiments = np.array(clf.sentiment_classes)[sp[ids].argmax(1)]
    rules = policy.decide_batch(ip[ids], clf.intent_classes, sentiments)
    runs.append(dict(model="routing_table", split="test", data="synthetic", classes=actions,
                     y_true=y[ids], y_pred=rules["action"].map(a_id).to_numpy()))

    # Real test messages (same rows as classify.py --evaluate).
    ref = pd.read_csv(policy.ACTIONS_CSV)
    df, _, yi, _ = intent_lstm.prepare_data()
    if not (df["id"].to_numpy() == ref["id"].to_numpy()).all():
        raise RuntimeError(f"{policy.ACTIONS_CSV.name} is stale; rerun action_policy.py")
    _, _, idx_real_test = intent_lstm.split_indices(yi)
    test = ref.iloc[idx_real_test].reset_index(drop=True)
    pred = clf.predict(test["text"].tolist(), cache=True)
    y_true = test["action"].map(a_id).to_numpy()
    for model, col in (("action_dnn", "action"), ("routing_table", "policy_action")):
        runs.append(dict(model=model, split="test", data="real", classes=actions,
                         y_true=y_true, y_pred=pred[col].map(a_id).to_numpy()))
    return runs + human_label_runs(clf, test, pred)


def human_label_runs(clf: MessageClassifier, test: pd.DataFrame, pred: pd.DataFrame) -> list[dict]:
    """Rows scored against the hand-labelled sentiment (if labels exist)."""
    human, info = load_human_sentiment()
    if human.empty:
        log.info("No hand labels in labels/ (run label_sentiment.py) - skipping human-label rows")
        return []
    log.info("Hand labels: %s", {k: v for k, v in info.items() if k != "annotators"} | {"annotators": info["annotators"]})
    merged = test.assign(id=test["id"].astype(str)).reset_index().merge(human, on="id", how="inner")
    if len(merged) < len(human):
        log.warning("%d labelled ids are not in the real test split", len(human) - len(merged))
    rows = merged["index"].to_numpy()  # positions in `test` / `pred`
    s_id = {c: i for i, c in enumerate(clf.sentiment_classes)}
    actions = list(clf.actions)
    a_id = {a: i for i, a in enumerate(actions)}
    ref = [policy.table_action(i, s) for i, s in zip(merged["intent_label"], merged["human_sentiment"])]
    data_s, data_a = "real (human labels)", "real (human-sentiment reference)"
    runs = [dict(model="sentiment_head", split="test", data=data_s, classes=clf.sentiment_classes,
                 y_true=merged["human_sentiment"].map(s_id).to_numpy(),
                 y_pred=pred["sentiment"].iloc[rows].map(s_id).to_numpy())]
    for model, col in (("action_dnn", "action"), ("routing_table", "policy_action")):
        runs.append(dict(model=model, split="test", data=data_a, classes=actions,
                         y_true=np.array([a_id[a] for a in ref]), y_pred=pred[col].iloc[rows].map(a_id).to_numpy()))
    changed = int((np.array(ref) != test["action"].iloc[rows].to_numpy()).sum())
    info["reference_actions_changed_vs_model_sentiment"] = changed
    model_ref = test["action"].iloc[rows]
    info["reference_action_counts"] = {
        "human_sentiment": pd.Series(ref).value_counts().to_dict(),
        "model_sentiment": model_ref.value_counts().to_dict(),
    }
    info["sentiment_agreement_with_model"] = round(float((merged["human_sentiment"].to_numpy()
                                                          == pred["sentiment"].iloc[rows].to_numpy()).mean()), 4)
    info["labeling_scheme"] = "see labels/GUIDELINES.md"
    runs[0]["meta"] = info  # written to human_labels_info.json by main()
    log.info("Reference action changes for %d / %d messages when the human sentiment replaces the model's",
             changed, len(rows))
    return runs


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def slug(run: dict) -> str:
    data = re.sub(r"[^a-z0-9]+", "_", run["data"].lower()).strip("_")
    return f"{run['model']}__{run['split']}__{data}"


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, subtitle: str, path: Path) -> None:
    """Counts in each cell; colour = share of the true class (row-normalised recall)."""
    k = len(classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(cm.sum(1, keepdims=True) > 0, cm / cm.sum(1, keepdims=True), 0.0)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    cell = 0.62 if k > 8 else 0.9
    fig, ax = plt.subplots(figsize=(2.6 + cell * k, 1.6 + cell * k), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    shown = np.ma.masked_where(cm == 0, share)  # zero cells stay surface-white
    ax.set_facecolor(SURFACE)
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8 if k > 8 else 10,
                        color=SURFACE if share[i, j] > 0.55 else INK)
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(k + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(k + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(k), classes, rotation=45, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(k), classes, fontsize=8, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Predicted", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Reference (true)", color=INK_MUTED, fontsize=9)
    ax.set_title(f"{title}\n", loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Share of true class", color=INK_MUTED, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for the 95%% CIs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("healthcare_ingestion", "httpx", "huggingface_hub", "generate_synthetic", "gensim",
                  "train_intent_lstm", "train_sentiment_head", "action_policy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out = args.out_dir
    shutil.rmtree(out / "confusion_matrices", ignore_errors=True)  # no stale plots from an older model
    (out / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    clf = MessageClassifier()
    runs = intent_runs(clf) + sentiment_runs(clf) + action_runs(clf)
    for run in runs:
        if "meta" in run:
            (out / "human_labels_info.json").write_text(json.dumps(run["meta"], indent=2))
    rng = np.random.default_rng(SEED)
    summary_rows, class_rows = [], []
    for run in runs:
        classes, k = run["classes"], len(run["classes"])
        t, p = np.asarray(run["y_true"]), np.asarray(run["y_pred"])
        cm = confusion(t, p, k)
        boots = bootstrap_cms(t, p, k, args.n_boot, rng)
        boot_summ = pd.DataFrame([summary(b) for b in boots])

        row = {"model": run["model"], "split": run["split"], "data": run["data"], "n": len(t)}
        for metric, value in summary(cm).items():
            lo, hi = ci(boot_summ[metric].to_numpy())
            row.update({metric: round(value, 4), f"{metric}_ci_low": round(lo, 4), f"{metric}_ci_high": round(hi, 4)})
        summary_rows.append(row)

        prec, rec, f1 = per_class(cm)
        bf = np.array([per_class(b)[2] for b in boots])
        for c, name in enumerate(classes):
            tp, support, predicted = int(cm[c, c]), int(cm[c].sum()), int(cm[:, c].sum())
            r = {"model": run["model"], "split": run["split"], "data": run["data"], "class": name,
                 "support": support, "predicted": predicted, "true_positives": tp}
            f1_ci = ci(bf[:, c]) if not np.isnan(bf[:, c]).all() else (np.nan, np.nan)
            for metric, val, (lo, hi) in (("precision", prec, wilson(tp, predicted)),
                                          ("recall", rec, wilson(tp, support)),
                                          ("f1", f1, f1_ci)):
                r.update({metric: round(float(val[c]), 4), f"{metric}_ci_low": round(lo, 4),
                          f"{metric}_ci_high": round(hi, 4)})
            class_rows.append(r)

        name = slug(run)
        pd.DataFrame(cm, index=[f"true: {c}" for c in classes], columns=classes).to_csv(
            out / "confusion_matrices" / f"{name}.csv")
        pretty = MODEL_NAMES.get(run["model"], run["model"])
        s = row
        plot_confusion(cm, classes, f"{pretty} - {run['split']} ({run['data']})",
                       f"n = {s['n']}  |  accuracy {s['accuracy']:.3f} "
                       f"[{s['accuracy_ci_low']:.3f}, {s['accuracy_ci_high']:.3f}]  |  macro F1 {s['macro_f1']:.3f} "
                       f"[{s['macro_f1_ci_low']:.3f}, {s['macro_f1_ci_high']:.3f}]",
                       out / "confusion_matrices" / f"{name}.png")

    summary_df = pd.DataFrame(summary_rows)
    class_df = pd.DataFrame(class_rows)
    summary_df.to_csv(out / "summary_metrics.csv", index=False)
    class_df.to_csv(out / "per_class_metrics.csv", index=False)

    # Emergency handling: the class a miss is costliest for.
    emergency = class_df[class_df["class"].isin([policy.ROUTE_EMERGENCY, policy.EMERGENCY_INTENT])]
    emergency.to_csv(out / "emergency_metrics.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        cols = ["model", "split", "data", "n", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
                "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "weighted_f1"]
        log.info("Summary (95%% bootstrap CIs, %d resamples):\n%s", args.n_boot, summary_df[cols].to_string(index=False))
        log.info("Emergency class:\n%s", emergency[["model", "split", "data", "class", "support", "predicted",
                                                     "precision", "precision_ci_low", "precision_ci_high",
                                                     "recall", "recall_ci_low", "recall_ci_high"]].to_string(index=False))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
