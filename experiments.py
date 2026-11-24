r"""Action-model architecture experiments (results saved for the write-up).

Three factors, fully crossed:

  intent source   synthetic  - the current intent LSTM (models/intent_lstm_tuned), trained on synthetic data
                  both       - same hyperparameters retrained on synthetic + the REAL train split
  text input      avg_pool   - current action DNN: token Embedding (init from the intent model) -> masked mean
                  lstm_state - the intent LSTM's final hidden state (64-d) instead of the averaged embedding
  hidden head     layers {1, 2, 3} x units {32, 50, 64}, plus 5 x 64 (the current model)

Every configuration is trained with 3 seeds. All other settings match classify.py (dropout 0.1, Adam 1e-3,
batch 32, balanced class weights, early stopping on the synthetic val split, seed-grouped 70/15/15 synthetic
split). The intent / sentiment probability inputs for the synthetic training rows are out-of-fold
(5 seed-grouped folds) exactly as in classify.py; for the `both` source they are recomputed with fold models
trained on synthetic + real train (real messages whose rewrites are held out are excluded from that fold).

Leakage note for lstm_state: hidden states from different fold models live in different bases, so they can't
be stacked out-of-fold. The final intent model's states are used for every row; they are in-sample (more
confident) for the synthetic training rows. The real val/test messages are unseen either way, so the real
scores are the honest comparison.

Model selection: mean real-VAL macro F1 over seeds (436 messages; reference = policy on the true intent and the
sentiment head's sentiment). Real-val messages were generation seeds, so their scores are optimistic in absolute
terms but comparable across configurations. The real TEST split is reported for every run but must not be used
to pick a configuration; the final comparison (best-by-val vs the current model) is a paired bootstrap on test.

Usage:
    python experiments.py                                   # full grid (resumes; finished runs are skipped)
    python experiments.py --sources synthetic --grid 1x32 --seeds 42     # smoke test
    python experiments.py --report-only                     # rebuild summary / plots from runs.csv

Outputs (reports/experiments/action_architecture/):
    runs.csv              one row per (config, seed): synthetic val/test, real val/test metrics
    summary.csv           per config: mean / std over seeds, ranked by real-val macro F1
    intent_models.csv     the two intent models on the real val / test splits
    final_comparison.csv  best config vs current model on real test (paired bootstrap 95% CI)
    real_val_macro_f1.png, config.json
Caches (git-ignored): models/experiments/ (intent `both` model, features, per-run predictions).
"""

from __future__ import annotations

# Import first: sets the Keras backend and blocks the TensorFlow import that segfaults here.
import train_intent_lstm as intent_lstm  # isort: skip

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

import action_policy as policy  # noqa: E402
import train_sentiment_head as sentiment_head  # noqa: E402
from classify import (ACTION_DIR, ACTION_PARAMS, INTENT_DIR, SEED, SENTIMENT_DIR,  # noqa: E402
                      MessageClassifier, _fingerprint, build_action_dnn)

# The experiments compare against the v1 pipeline (synthetic-trained intent model, 5x64 avg-pool action DNN).
# Once the main pipeline is retrained with the winning config, v1 lives in these archived copies.
def _archived(current: Path, name: str) -> Path:
    archive = current.parent / name
    return archive if archive.exists() else current


V1_INTENT_DIR = _archived(INTENT_DIR, "intent_lstm_tuned_v1_synthetic")
V1_ACTION_DIR = _archived(ACTION_DIR, "action_dnn_v1_avgpool")
from generate_synthetic import load_labelled  # noqa: E402

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "models" / "experiments" / "action_architecture"
OUT = ROOT / "reports" / "experiments" / "action_architecture"
SOURCES = ("synthetic", "both")
TEXT_INPUTS = ("avg_pool", "lstm_state")
GRID = [(lay, u) for lay in (1, 2, 3) for u in (32, 50, 64)] + [(5, 64)]
BASELINE = ("synthetic", "avg_pool", 5, 64)  # the current action DNN
SEEDS = [42, 7, 123]
N_OOF_FOLDS = 5
N_BOOT = 2000
EMERGENCY = policy.ROUTE_EMERGENCY
INK, INK_MUTED, SURFACE, GRID_C = "#1f1f1e", "#6b6a66", "#ffffff", "#e4e3df"
SERIES = {"avg_pool": "#2a78d6", "lstm_state": "#eb6834"}

log = logging.getLogger("experiments")


# --------------------------------------------------------------------------- #
# Intent models
# --------------------------------------------------------------------------- #
def intent_params() -> dict:
    meta = json.loads((V1_INTENT_DIR / "metrics.json").read_text())
    return {k: v for k, v in meta["params"].items() if k in intent_lstm.DEFAULT_PARAMS}


def intent_dir(source: str) -> Path:
    return V1_INTENT_DIR if source == "synthetic" else CACHE / "intent_both"


def ensure_intent_both() -> None:
    """Train the intent LSTM on synthetic + real train with the tuned hyperparameters (seed 42)."""
    out = intent_dir("both")
    if (out / "model.keras").exists():
        return
    log.info("Training intent LSTM on synthetic + real train (tuned params, seed %d)...", SEED)
    ds = intent_lstm.prepare_dataset("both")
    splits = ds.splits()
    fitted = intent_lstm.fit(ds.tokens, ds.y, splits["train"], splits["stop"], ds.n_classes,
                             seed=SEED, verbose=0, **intent_params())
    metrics = intent_lstm.evaluate(fitted, ds.y, splits)
    intent_lstm.save(fitted, ds.encoder, ds.y, splits["test"], metrics, out,
                     extra={"train_on": "both", "split_sizes": {k: len(v) for k, v in splits.items()}})
    log.info("Intent (both): %s", metrics)


def load_intent(source: str) -> tuple[keras.Model, dict]:
    d = intent_dir(source)
    return keras.models.load_model(d / "model.keras"), joblib.load(d / "preprocessing.joblib")


def intent_outputs(model: keras.Model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(LSTM final hidden state, intent probabilities)."""
    sub = keras.Model(model.inputs, [model.get_layer("lstm").output, model.get_layer("intent").output])
    state, probs = sub.predict(X, batch_size=256, verbose=0)
    return np.asarray(state, dtype=np.float32), np.asarray(probs, dtype=np.float32)


def action_tokens(texts, pre: dict) -> np.ndarray:
    X = intent_lstm.encode(intent_lstm.tokenize(texts), pre["vocab"], pre["max_len"])
    X[(X == 0).all(axis=1), 0] = 1  # empty message -> one unknown token (avoids a NaN masked mean)
    return X


# --------------------------------------------------------------------------- #
# Out-of-fold intent probabilities for the synthetic rows
# --------------------------------------------------------------------------- #
def action_folds(syn: pd.DataFrame) -> list:
    groups = syn["seed_id"].astype(str).to_numpy()
    return list(StratifiedGroupKFold(N_OOF_FOLDS, shuffle=True, random_state=SEED)
                .split(np.zeros(len(syn)), syn["action_label"], groups))


def cached_oof(syn: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The OOF intent (synthetic source) and sentiment probabilities classify.py --train cached."""
    c = np.load(V1_ACTION_DIR / "oof_features.npz")
    intent_meta = json.loads((V1_INTENT_DIR / "metrics.json").read_text())
    sent_meta = json.loads((SENTIMENT_DIR / "metrics.json").read_text())
    key = _fingerprint(N_OOF_FOLDS, *syn["synthetic_id"], *syn["intent"], *syn["sentiment"],
                       json.dumps(intent_meta, sort_keys=True), json.dumps(sent_meta, sort_keys=True))
    if str(c["key"]) != key:
        raise RuntimeError("models/action_dnn/oof_features.npz is stale; rerun `python classify.py --train`")
    return c["intent"], c["sentiment"]


def oof_intent_both(syn: pd.DataFrame) -> np.ndarray:
    """OOF intent probabilities from fold LSTMs trained on synthetic (other folds) + real train."""
    path = CACHE / "oof_intent_both.npy"
    if path.exists():
        return np.load(path)
    ds = intent_lstm.prepare_dataset("both")
    if list(ds.df["id"].iloc[ds.n_real:]) != list(syn["synthetic_id"]):
        raise RuntimeError("Synthetic rows don't line up with the intent dataset")
    r_train, _, _ = ds.real_splits()
    syn_rows = np.arange(ds.n_real, len(ds.y))
    params = intent_params()
    ip = np.zeros((len(syn), ds.n_classes), dtype=np.float32)
    for k, (tr, te) in enumerate(action_folds(syn)):
        held_out_groups = set(ds.groups[syn_rows[te]])
        real_rows = r_train[~np.isin(ds.groups[r_train], list(held_out_groups))]
        pool = np.concatenate([syn_rows[tr], real_rows])
        train, stop = intent_lstm.group_split(pool, ds.y, ds.groups, intent_lstm.STOP_SPLITS)
        t0 = time.time()
        fitted = intent_lstm.fit(ds.tokens, ds.y, train, stop, ds.n_classes, seed=SEED, verbose=0, **params)
        ip[te] = fitted["model"].predict(fitted["X"][syn_rows[te]], verbose=0)
        log.info("OOF intent (both) fold %d/%d: accuracy %.3f (%d real rows in pool, %.0fs)", k + 1, N_OOF_FOLDS,
                 accuracy_score(ds.y[syn_rows[te]], ip[te].argmax(1)), len(real_rows), time.time() - t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ip)
    return ip


# --------------------------------------------------------------------------- #
# Feature sets per intent source
# --------------------------------------------------------------------------- #
def build_features(source: str, clf: MessageClassifier, syn: pd.DataFrame, real: pd.DataFrame) -> dict:
    path = CACHE / f"features_{source}.npz"
    if path.exists():
        return dict(np.load(path))
    model, pre = load_intent(source)
    if list(pre["label_encoder"].classes_) != clf.intent_classes:
        raise RuntimeError(f"Intent classes of the {source} model differ from the policy's")
    syn_oof_intent, syn_oof_sent = cached_oof(syn)
    if source == "both":
        syn_oof_intent = oof_intent_both(syn)

    syn_tokens, real_tokens = action_tokens(syn["text"], pre), action_tokens(real["text"], pre)
    syn_state, _ = intent_outputs(model, syn_tokens)
    real_state, real_intent = intent_outputs(model, real_tokens)
    feats = dict(
        syn_tokens=syn_tokens, syn_state=syn_state, syn_intent=syn_oof_intent, syn_sent=syn_oof_sent,
        real_tokens=real_tokens, real_state=real_state, real_intent=real_intent,
        real_sent=clf.sentiment_probs(real["text"].tolist(), cache=True).astype(np.float32),
        embedding=model.get_layer("embedding").get_weights()[0],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **feats)
    return feats


# --------------------------------------------------------------------------- #
# Action models
# --------------------------------------------------------------------------- #
def build_state_dnn(state_dim: int, n_intents: int, n_sentiments: int, n_actions: int, *, hidden_layers: int,
                    hidden_units: int, dropout: float, lr: float, **_) -> keras.Model:
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


def inputs(feats: dict, text_input: str, part: str, idx: np.ndarray) -> list:
    text = feats[f"{part}_tokens"] if text_input == "avg_pool" else feats[f"{part}_state"]
    return [text[idx], feats[f"{part}_intent"][idx], feats[f"{part}_sent"][idx]]


def scores(y: np.ndarray, pred: np.ndarray, prefix: str) -> dict:
    e = policy.ACTIONS.index(EMERGENCY)
    kw = dict(labels=[e], average="macro", zero_division=0)
    return {f"{prefix}_accuracy": accuracy_score(y, pred),
            f"{prefix}_macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            f"{prefix}_weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            f"{prefix}_emergency_recall": recall_score(y, pred, **kw) if (y == e).any() else np.nan,
            f"{prefix}_emergency_precision": precision_score(y, pred, **kw) if (pred == e).any() else np.nan}


def run_one(cfg: dict, feats: dict, y_syn: np.ndarray, syn_split: dict, y_real: np.ndarray, real_split: dict,
            n_intents: int, n_sents: int) -> tuple[dict, dict]:
    params = {**ACTION_PARAMS, "hidden_layers": cfg["layers"], "hidden_units": cfg["units"],
              "text_input": cfg["text_input"]}
    n_actions = len(policy.ACTIONS)
    keras.utils.set_random_seed(cfg["seed"])
    if cfg["text_input"] == "avg_pool":
        model = build_action_dnn(feats["embedding"], n_intents, n_sents, n_actions, **params)
    else:
        model = build_state_dnn(feats["syn_state"].shape[1], n_intents, n_sents, n_actions, **params)
    tr, va = syn_split["train"], syn_split["val"]
    weights = compute_class_weight("balanced", classes=np.arange(n_actions), y=y_syn[tr])
    t0 = time.time()
    hist = model.fit(inputs(feats, cfg["text_input"], "syn", tr), y_syn[tr],
                     validation_data=(inputs(feats, cfg["text_input"], "syn", va), y_syn[va]),
                     epochs=params["epochs"], batch_size=params["batch_size"], class_weight=dict(enumerate(weights)),
                     verbose=0, callbacks=[keras.callbacks.EarlyStopping(
                         monitor="val_loss", patience=params["patience"], restore_best_weights=True)])
    row = {**cfg, "params": model.count_params(), "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
           "fit_seconds": round(time.time() - t0, 1)}
    preds = {}
    for part, split, y in (("syn", syn_split, y_syn), ("real", real_split, y_real)):
        for name in ("val", "test"):
            ids = split[name]
            pred = model.predict(inputs(feats, cfg["text_input"], part, ids), verbose=0).argmax(1)
            label = f"{'synthetic' if part == 'syn' else 'real'}_{name}"
            row.update(scores(y[ids], pred, label))
            preds[label] = pred
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, preds


def run_id(cfg: dict) -> str:
    return f"{cfg['source']}|{cfg['text_input']}|{cfg['layers']}x{cfg['units']}|s{cfg['seed']}"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
METRICS = ["real_val_macro_f1", "real_val_accuracy", "real_test_macro_f1", "real_test_accuracy",
           "synthetic_val_macro_f1", "synthetic_test_macro_f1", "synthetic_test_emergency_recall",
           "real_test_emergency_recall", "real_test_emergency_precision", "params", "best_epoch"]
KEYS = ["source", "text_input", "layers", "units"]


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.astype({"layers": int, "units": int})
    g = runs.groupby(KEYS)
    summ = g[METRICS].mean().round(4).add_suffix("_mean")
    summ = summ.join(g[["real_val_macro_f1", "real_test_macro_f1", "synthetic_test_macro_f1"]].std(ddof=0)
                     .round(4).add_suffix("_std"))
    summ["n_seeds"] = g.size()
    summ = summ.reset_index().sort_values("real_val_macro_f1_mean", ascending=False).reset_index(drop=True)
    summ.insert(0, "rank_by_real_val", np.arange(1, len(summ) + 1))
    summ["is_current_model"] = [tuple(r) == BASELINE for r in summ[KEYS].itertuples(index=False)]
    return summ


def plot_summary(runs: pd.DataFrame, routing: pd.DataFrame, path: Path) -> None:
    runs = runs.astype({"layers": int, "units": int})
    heads = [f"{lay}x{u}" for lay, u in GRID]
    sources = [s for s in SOURCES if s in set(runs["source"])]
    fig, axes = plt.subplots(2, len(sources), figsize=(6.2 * len(sources), 7), dpi=150, squeeze=False, sharey="row")
    fig.patch.set_facecolor(SURFACE)
    for col, source in enumerate(sources):
        for row, (metric, title) in enumerate((("real_val_macro_f1", "Real validation macro F1 (selection metric)"),
                                               ("synthetic_test_macro_f1", "Synthetic test macro F1"))):
            ax = axes[row, col]
            for j, (text_input, color) in enumerate(SERIES.items()):
                d = runs[(runs["source"] == source) & (runs["text_input"] == text_input)].copy()
                if d.empty:
                    continue
                d["head"] = d["layers"].astype(str) + "x" + d["units"].astype(str)
                agg = d.groupby("head")[metric].agg(["mean", "min", "max"]).reindex(heads)
                x = np.arange(len(heads)) + (j - 0.5) * 0.22
                ax.vlines(x, agg["min"], agg["max"], color=color, linewidth=2, alpha=0.45)
                ax.plot(x, agg["mean"], "o", color=color, markersize=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=text_input.replace("_", " "))
            if metric == "real_val_macro_f1" and source in set(routing["source"]):
                rt = routing.loc[routing["source"] == source, "real_val_macro_f1"].iloc[0]
                ax.axhline(rt, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
                ax.annotate(f"routing table {rt:.3f}", (len(heads) - 0.5, rt), xytext=(0, 4),
                            textcoords="offset points", ha="right", fontsize=7, color=INK_MUTED)
            ax.axvline(len(heads) - 1.5, color=GRID_C, linewidth=1)
            ax.set_xticks(range(len(heads)), heads, fontsize=8, color=INK)
            ax.grid(axis="y", color=GRID_C, linewidth=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(GRID_C)
            ax.tick_params(length=0, labelsize=8, colors=INK_MUTED)
            ax.set_title(f"{title}  -  intent trained on {source}", loc="left", fontsize=9, color=INK)
            if row == 1:
                ax.set_xlabel("Hidden layers x units  (5x64 = current model)", fontsize=8, color=INK_MUTED)
            if col == 0:
                ax.legend(fontsize=8, frameon=False, loc="lower left", title="text input", title_fontsize=8)
    fig.suptitle("Action model architecture experiments (dot = mean of 3 seeds, bar = seed range)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """macro F1(a) - macro F1(b) on the same messages, 95% percentile CI."""
    f = lambda p, i: f1_score(y[i], p[i], average="macro", zero_division=0)  # noqa: E731
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f(a, i) - f(b, i))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    full = np.arange(len(y))
    return {"macro_f1_a": round(f(a, full), 4), "macro_f1_b": round(f(b, full), 4),
            "difference": round(f(a, full) - f(b, full), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "p_a_not_better": round(float(np.mean(np.array(diffs) <= 0)), 4)}


def report(runs: pd.DataFrame, routing: pd.DataFrame) -> None:
    summ = summarise(runs)
    summ.to_csv(OUT / "summary.csv", index=False)
    plot_summary(runs, routing, OUT / "real_val_macro_f1.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        cols = ["rank_by_real_val", *KEYS, "real_val_macro_f1_mean", "real_val_macro_f1_std",
                "synthetic_test_macro_f1_mean", "real_test_macro_f1_mean", "params_mean", "is_current_model"]
        log.info("Configurations ranked by real-val macro F1 (mean of seeds):\n%s", summ[cols].to_string(index=False))

    # Final comparison on real test: best-by-val config vs the current model. Within each config the seed
    # kept is the one with the best synthetic-val macro F1 (as classify.py does).
    preds_path = CACHE / "predictions.joblib"
    if not preds_path.exists() or not tuple(BASELINE) in {tuple(r) for r in summ[KEYS].itertuples(index=False)}:
        return
    preds = joblib.load(preds_path)
    real = pd.read_csv(CACHE / "real_test_labels.csv")
    y = real["y"].to_numpy()

    def kept_run(cfg_row) -> str:
        d = runs[(runs[KEYS] == pd.Series(cfg_row, index=KEYS)).all(axis=1)]
        return d.sort_values("synthetic_val_macro_f1", ascending=False)["run_id"].iloc[0]

    best = tuple(summ.loc[0, KEYS])
    rng = np.random.default_rng(SEED)
    rows = []
    for label, cfg in (("best_by_real_val", best),):
        a_id, b_id = kept_run(cfg), kept_run(BASELINE)
        res = paired_bootstrap(y, preds[a_id]["real_test"], preds[b_id]["real_test"], rng)
        rows.append({"comparison": f"{label} vs current model", "a": a_id, "b": b_id, **res})
    for _, r in routing.iterrows():  # the routing table on the same intent source, for context
        a_id = kept_run(best)
        if r["source"] == best[0]:
            res = paired_bootstrap(y, preds[a_id]["real_test"], np.load(CACHE / f"routing_{r['source']}.npy"), rng)
            rows.append({"comparison": "best_by_real_val vs routing table", "a": a_id,
                         "b": f"routing_table|{r['source']}", **res})
    final = pd.DataFrame(rows)
    final.to_csv(OUT / "final_comparison.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        log.info("Real test, paired bootstrap (%d resamples):\n%s", N_BOOT, final.to_string(index=False))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--text-inputs", nargs="+", choices=TEXT_INPUTS, default=list(TEXT_INPUTS))
    parser.add_argument("--grid", nargs="+", default=[f"{lay}x{u}" for lay, u in GRID],
                        help="Hidden heads as LAYERSxUNITS (default: 1-3 x 32/50/64 plus 5x64)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--report-only", action="store_true", help="Only rebuild summary / plots from runs.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    for noisy in ("healthcare_ingestion", "httpx", "huggingface_hub", "generate_synthetic", "gensim",
                  "train_intent_lstm", "train_sentiment_head", "action_policy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    runs_csv = OUT / "runs.csv"

    if not args.report_only:
        grid = [tuple(int(v) for v in g.lower().split("x")) for g in args.grid]
        (OUT / "config.json").write_text(json.dumps({
            "sources": args.sources, "text_inputs": args.text_inputs, "grid": grid, "seeds": args.seeds,
            "fixed_action_params": {k: v for k, v in ACTION_PARAMS.items()
                                    if k not in ("hidden_layers", "hidden_units", "text_input")},
            "intent_params": intent_params(), "oof_folds": N_OOF_FOLDS, "selection_metric": "mean real_val_macro_f1",
        }, indent=2))

        clf = MessageClassifier(intent_dir=V1_INTENT_DIR, action_dir=None)
        syn = load_labelled()
        actions = list(policy.ACTIONS)
        a_id = {a: i for i, a in enumerate(actions)}
        y_syn = syn["action_label"].map(a_id).to_numpy()
        groups = syn["seed_id"].astype(str).to_numpy()
        idx_rest, idx_test = sentiment_head.group_split(np.arange(len(y_syn)), y_syn, groups, n_splits=7, seed=SEED)
        idx_train, idx_val = sentiment_head.group_split(idx_rest, y_syn, groups, n_splits=6, seed=SEED)
        syn_split = {"train": idx_train, "val": idx_val, "test": idx_test}

        real, _, y_intent, _ = intent_lstm.prepare_data()
        ref = pd.read_csv(policy.ACTIONS_CSV)
        if not (real["id"].to_numpy() == ref["id"].to_numpy()).all():
            raise RuntimeError(f"{policy.ACTIONS_CSV.name} is stale; rerun action_policy.py")
        real = real.assign(text=real[intent_lstm.TEXT_COL])
        y_real = ref["action"].map(a_id).to_numpy()
        _, r_val, r_test = intent_lstm.split_indices(y_intent)
        real_split = {"val": r_val, "test": r_test}
        pd.DataFrame({"id": real["id"].iloc[r_test].to_numpy(), "y": y_real[r_test]}).to_csv(
            CACHE / "real_test_labels.csv", index=False)

        # Intent models and the routing table per source (context rows).
        intent_rows, routing_rows = [], []
        feats_by_source = {}
        for source in args.sources:
            if source == "both":
                ensure_intent_both()
            feats = feats_by_source[source] = build_features(source, clf, syn, real)
            sentiments = np.array(clf.sentiment_classes)[feats["real_sent"].argmax(1)]
            rules = policy.decide_batch(feats["real_intent"], clf.intent_classes, sentiments)
            rt_pred = rules["action"].map(a_id).to_numpy()
            np.save(CACHE / f"routing_{source}.npy", rt_pred[r_test])
            routing_rows.append({"source": source, **{k: round(v, 4) for k, v in {
                **scores(y_real[r_val], rt_pred[r_val], "real_val"),
                **scores(y_real[r_test], rt_pred[r_test], "real_test")}.items()}})
            ip = feats["real_intent"].argmax(1)
            e = clf.intent_classes.index(policy.EMERGENCY_INTENT)
            for name, ids in (("real_val", r_val), ("real_test", r_test)):
                intent_rows.append({"intent_source": source, "split": name, "n": len(ids),
                                    "accuracy": round(accuracy_score(y_intent[ids], ip[ids]), 4),
                                    "macro_f1": round(f1_score(y_intent[ids], ip[ids], average="macro"), 4),
                                    "emergency_recall": round(recall_score(y_intent[ids], ip[ids], labels=[e],
                                                                           average="macro", zero_division=0), 4)})
        pd.DataFrame(intent_rows).to_csv(OUT / "intent_models.csv", index=False)
        pd.DataFrame(routing_rows).to_csv(OUT / "routing_table.csv", index=False)
        log.info("Intent models on real data:\n%s", pd.DataFrame(intent_rows).to_string(index=False))

        # Grid of action models (resumable).
        done = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame(columns=["run_id"])
        preds_path = CACHE / "predictions.joblib"
        preds = joblib.load(preds_path) if preds_path.exists() else {}
        cfgs = [dict(source=s, text_input=t, layers=lay, units=u, seed=seed)
                for s in args.sources for t in args.text_inputs for lay, u in grid for seed in args.seeds]
        todo = [c for c in cfgs if run_id(c) not in set(done["run_id"])]
        log.info("%d runs requested, %d already done, %d to run", len(cfgs), len(cfgs) - len(todo), len(todo))
        for i, cfg in enumerate(todo, 1):
            row, p = run_one(cfg, feats_by_source[cfg["source"]], y_syn, syn_split, y_real, real_split,
                             len(clf.intent_classes), len(clf.sentiment_classes))
            row = {"run_id": run_id(cfg), **row}
            preds[row["run_id"]] = p
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(runs_csv, index=False)
            joblib.dump(preds, preds_path)
            log.info("[%d/%d] %s  real-val F1 %.4f  synthetic-test F1 %.4f  (%ss, %d epochs)", i, len(todo),
                     row["run_id"], row["real_val_macro_f1"], row["synthetic_test_macro_f1"],
                     row["fit_seconds"], row["best_epoch"])

    runs = pd.read_csv(runs_csv)
    routing = pd.read_csv(OUT / "routing_table.csv")
    report(runs, routing)
    log.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
