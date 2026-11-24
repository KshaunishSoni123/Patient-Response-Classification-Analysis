"""Hidden-layer analysis for the intent LSTM, sentiment head and action DNN.

For every hidden layer of each model, an intermediate Keras model returns the layer's
activations. Then:

  1. Class-mean activations, z-scored per node across classes (true labels), drawn as a
     heatmap per layer. Nodes are sorted by their preferred class (highest z); nodes that
     never vary on the data shown (dead ReLUs) are hatched.
  2. Per-node specialisation: preferred class, top z-score, specificity = top z / sqrt(K-1)
     (1.0 = the node responds to exactly one class; comparable across models with different
     class counts, since sqrt(K-1) is the largest z possible), and for ReLU layers the
     selectivity index (max class mean - mean of other classes) / (max + mean of others), in [0, 1].
     A node is a "specialist" for its preferred class when specificity >= 0.75.
  3. Linear probe per layer: standardised multinomial logistic regression fit on the TRAIN
     split's activations, scored (accuracy / macro F1) on the held-out splits. Shows where in
     the network the classes become linearly separable. The model input representation and
     the model's own output are included for reference.
  4. Node ablation: for each class, its top-k specialist nodes (chosen on the TRAIN split) are
     set to 0 and the rest of the network is replayed from the saved weights; the change in
     that class's recall is compared with ablating k random active nodes (30 draws).

Datasets: primary = the SHARED synthetic test split (synthetic_splits.py) for all three models,
so every heatmap is computed on the same held-out messages:
  intent LSTM     primary: synthetic test (rewrites of real val messages it never trained on)
                  | secondary: real test (437)
  sentiment head  primary: synthetic test | secondary: customer-support (out-of-domain)
  action DNN      primary: synthetic test (out-of-fold intent/sentiment inputs, as trained)
                  | secondary: real test (437; reference = policy on true intent + sentiment head)

Usage:
    python hidden_layer_analysis.py               # writes reports/hidden_layers/
    python hidden_layer_analysis.py --random-draws 100
"""

from __future__ import annotations

# Import first: sets the Keras backend and blocks the TensorFlow import that segfaults here.
import train_intent_lstm as intent_lstm  # isort: skip

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import action_policy as policy  # noqa: E402
import train_sentiment_head as sentiment_head  # noqa: E402
from classify import ACTION_DIR, INTENT_DIR, SEED, MessageClassifier, lstm_layer_name  # noqa: E402
from generate_synthetic import load_labelled  # noqa: E402
import synthetic_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("hidden_layer_analysis")

# Diverging blue <-> red with a neutral gray midpoint (z = 0).
DIVERGING = ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f2a6a5", "#e34948", "#8f1f1f"]
INK, INK_MUTED, SURFACE, GRID = "#1f1f1e", "#6b6a66", "#ffffff", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1-2: primary, secondary dataset
Z_CLIP = 3.0
SPECIALIST_THRESHOLD = 0.75  # specificity = top z / sqrt(K-1)
MODEL_NAMES = {"intent_lstm": "Intent LSTM", "sentiment_head": "Sentiment head", "action_dnn": "Action DNN"}


def hidden_layers(model: keras.Model) -> list[str]:
    return [layer.name for layer in model.layers if layer.name.startswith("hidden_")]


@dataclass
class Split:
    label: str             # human-readable
    inputs: list           # model inputs (list of arrays)
    y: np.ndarray          # true class ids


@dataclass
class ModelSpec:
    name: str
    model: keras.Model
    classes: list[str]
    layers: list[str]      # hidden layers to analyse, in order
    output: str            # softmax layer name
    train: Split
    primary: Split
    secondary: Split
    input_repr: callable   # (split, activations dict) -> 2D input representation for the probe
    extra_outputs: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Data per model (same splits as training / evaluation_report.py)
# --------------------------------------------------------------------------- #
def intent_spec(clf: MessageClassifier) -> ModelSpec:
    train_on = json.loads((INTENT_DIR / "metrics.json").read_text()).get("train_on", "synthetic")
    ds = intent_lstm.prepare_dataset(train_on)
    if list(ds.encoder.classes_) != clf.intent_classes:
        raise RuntimeError("Intent classes don't match the saved intent model; retrain step 3a")
    splits = ds.splits()
    texts = ds.df[intent_lstm.TEXT_COL].tolist()

    def split(key: str, label: str) -> Split:
        ids = splits[key]
        return Split(label, [clf.tokens([texts[i] for i in ids])], ds.y[ids])

    emb = clf.intent_model.get_layer("embedding").get_weights()[0]

    def mean_embedding(s: Split, _acts) -> np.ndarray:  # masked mean of the token embeddings
        X = s.inputs[0]
        mask = (X > 0)[..., None]
        return (emb[X] * mask).sum(1) / np.maximum(mask.sum(1), 1)

    # Primary: the shared synthetic test split (rewrites of real val messages, never trained on),
    # the same messages the sentiment head and action DNN are analysed on.
    syn = load_labelled()
    shared = synthetic_splits.split_indices(syn)["test"]
    syn_test = Split("synthetic test", [clf.tokens(syn["text"].iloc[shared].tolist())],
                     ds.encoder.transform(syn["intent"].iloc[shared]))
    source = {"synthetic": "synthetic", "both": "synthetic + real", "real": "real"}[train_on]
    return ModelSpec("intent_lstm", clf.intent_model, clf.intent_classes,
                     [lstm_layer_name(clf.intent_model), *hidden_layers(clf.intent_model)], "intent",
                     train=split("train", f"{source} train"),
                     primary=syn_test,
                     secondary=split("test", "real test"),
                     input_repr=mean_embedding)


def sentiment_spec(clf: MessageClassifier) -> ModelSpec:
    classes = clf.sentiment_classes
    label_id = {c: i for i, c in enumerate(classes)}
    syn = load_labelled()
    y = syn["sentiment"].map(label_id).to_numpy()
    X = clf.sentiment_scaler.transform(clf.sentiment_features(syn["text"].tolist(), cache=True)).astype(np.float32)
    shared = synthetic_splits.split_indices(syn)
    idx_train, idx_test = shared["train"], shared["test"]
    sup, _ = sentiment_head.load_support()
    Xs = clf.sentiment_scaler.transform(clf.sentiment_features(sup["text"].tolist(), cache=True)).astype(np.float32)
    return ModelSpec("sentiment_head", clf.sentiment_head, classes, hidden_layers(clf.sentiment_head), "sentiment",
                     train=Split("synthetic train", [X[idx_train]], y[idx_train]),
                     primary=Split("synthetic test", [X[idx_test]], y[idx_test]),
                     secondary=Split("customer support (out-of-domain)", [Xs],
                                     sup["sentiment"].map(label_id).to_numpy()),
                     input_repr=lambda s, _a: s.inputs[0])


def action_spec(clf: MessageClassifier) -> ModelSpec:
    actions = list(clf.actions)
    a_id = {a: i for i, a in enumerate(actions)}
    syn = load_labelled()
    y = syn["action_label"].map(a_id).to_numpy()
    inp = np.load(ACTION_DIR / "action_inputs.npz")  # train rows: out-of-fold; val/test: final models
    ip, sp = inp["intent"], inp["sentiment"]
    if len(ip) != len(syn):
        raise RuntimeError("action_inputs.npz doesn't match the synthetic data; rerun `python classify.py --train`")
    texts = syn["text"].tolist()
    text = clf.intent_state_and_probs(texts)[0] if clf.text_input == "lstm_state" else clf.action_tokens(texts)
    shared = synthetic_splits.split_indices(syn)
    idx_train, idx_test = shared["train"], shared["test"]

    def syn_split(ids, label):
        return Split(label, [text[ids], ip[ids], sp[ids]], y[ids])

    # Real test messages (same rows as classify.py --evaluate).
    ref = pd.read_csv(policy.ACTIONS_CSV)
    df, _, yi, _ = intent_lstm.prepare_data()
    if not (df["id"].to_numpy() == ref["id"].to_numpy()).all():
        raise RuntimeError(f"{policy.ACTIONS_CSV.name} is stale; rerun action_policy.py")
    _, _, idx_real = intent_lstm.split_indices(yi)
    test = ref.iloc[idx_real].reset_index(drop=True)
    t = test["text"].tolist()
    state, real_ip = clf.intent_state_and_probs(t)
    real = Split("real test", clf.action_inputs(t, real_ip, clf.sentiment_probs(t, cache=True), state=state),
                 test["action"].map(a_id).to_numpy())

    return ModelSpec("action_dnn", clf.action_model, actions, hidden_layers(clf.action_model), "action",
                     train=syn_split(idx_train, "synthetic train"),
                     primary=syn_split(idx_test, "synthetic test"),
                     secondary=real,
                     input_repr=lambda _s, acts: acts["features"],
                     extra_outputs=["features"])


# --------------------------------------------------------------------------- #
# Intermediate model, replay and statistics
# --------------------------------------------------------------------------- #
def intermediate_model(spec: ModelSpec) -> tuple[keras.Model, list[str]]:
    """One Keras model that returns every analysed layer's activations (plus the output)."""
    names = spec.layers + [spec.output] + spec.extra_outputs
    return keras.Model(inputs=spec.model.inputs, outputs=[spec.model.get_layer(n).output for n in names],
                       name=f"{spec.name}_activations"), names


def activations(act_model: keras.Model, names: list[str], split: Split) -> dict[str, np.ndarray]:
    inputs = split.inputs if len(split.inputs) > 1 else split.inputs[0]
    outs = act_model.predict(inputs, batch_size=256, verbose=0)
    return {n: np.asarray(o, dtype=np.float64) for n, o in zip(names, outs)}


def downstream_dense(model: keras.Model, layer: str) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """(W, b, activation) of every Dense layer after `layer` (dropout is identity at inference)."""
    names = [lay.name for lay in model.layers]
    after = model.layers[names.index(layer) + 1:]
    return [(*[np.asarray(w, dtype=np.float64) for w in lay.get_weights()], lay.activation.__name__)
            for lay in after if isinstance(lay, keras.layers.Dense)]


def replay(dense: list, x: np.ndarray) -> np.ndarray:
    for W, b, act in dense:
        # numpy + Apple Accelerate raises spurious divide/overflow warnings on clean matmuls;
        # silence them and check the result explicitly instead.
        with np.errstate(all="ignore"):
            x = x @ W + b
        if not np.isfinite(x).all():
            raise FloatingPointError("Non-finite values while replaying the network")
        if act == "relu":
            x = np.maximum(x, 0)
        elif act == "softmax":
            x = np.exp(x - x.max(1, keepdims=True))
            x /= x.sum(1, keepdims=True)
        elif act != "linear":
            raise ValueError(f"Unsupported activation {act}")
    return x


def class_means(a: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y, minlength=k)
    sums = np.zeros((k, a.shape[1]))
    np.add.at(sums, y, a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts[:, None], counts


def zscore(means: np.ndarray, live: np.ndarray) -> np.ndarray:
    """z-score each node across classes (classes with no examples are ignored)."""
    m = means[~np.isnan(means).any(1)]
    mu, sd = m.mean(0), m.std(0)
    z = np.full_like(means, np.nan)
    ok = live & (sd > 1e-12)
    z[:, ok] = (means[:, ok] - mu[ok]) / sd[ok]
    return z


def live_nodes(a: np.ndarray) -> np.ndarray:
    return a.std(0) > 1e-9


def recall_per_class(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    support = np.bincount(y, minlength=k)
    hits = np.bincount(y[pred == y], minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(support > 0, hits / support, np.nan)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_heatmaps(spec: ModelSpec, split: Split, stats: list[dict], path: Path) -> None:
    k = len(spec.classes)
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    panel_h = 0.55 + 0.2 * k
    fig, axes = plt.subplots(len(stats), 1, figsize=(12, panel_h * len(stats) + 0.9), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    counts = np.bincount(split.y, minlength=k)
    row_labels = [f"{c} (n={n})" for c, n in zip(spec.classes, counts)]
    for ax, s in zip(axes[:, 0], stats):
        z, live = s["z"], s["live"]
        units = z.shape[1]
        pref = np.where(live, np.nanargmax(np.nan_to_num(z, nan=-np.inf), axis=0), k)
        top = np.where(live, np.nanmax(np.nan_to_num(z, nan=-np.inf), axis=0), 0)
        order = np.lexsort((-top, pref))  # by preferred class, strongest first; dead nodes last
        zs = np.clip(z[:, order], -Z_CLIP, Z_CLIP)
        ax.imshow(np.ma.masked_invalid(zs), cmap=cmap, vmin=-Z_CLIP, vmax=Z_CLIP, aspect="auto",
                  interpolation="nearest")
        for j, node in enumerate(order):
            if not live[node]:
                ax.add_patch(Rectangle((j - 0.5, -0.5), 1, k, facecolor=SURFACE, edgecolor="#b5b4af",
                                       hatch="////", linewidth=0))
        ax.set_yticks(range(k), row_labels, fontsize=7 if k > 8 else 8, color=INK)
        ax.set_xticks([])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        n_dead = int((~live).sum())
        probe = s.get("probe_macro_f1")
        probe_txt = f"  |  probe macro F1 {probe:.3f}" if probe is not None else ""
        ax.set_title(f"{s['layer']}  -  {units - n_dead}/{units} active nodes{probe_txt}",
                     loc="left", fontsize=9, color=INK)
    axes[-1, 0].set_xlabel("Nodes, sorted by preferred class (hatched = inactive on this data)",
                           fontsize=8, color=INK_MUTED)
    fig.suptitle(f"{MODEL_NAMES[spec.name]} - class-mean activation z-scores ({split.label})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 0.93, 0.98))
    cax = fig.add_axes((0.945, 0.15, 0.012, 0.7))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-Z_CLIP, Z_CLIP))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("z-score across classes", fontsize=8, color=INK_MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_probes(probe_df: pd.DataFrame, path: Path) -> None:
    models = list(dict.fromkeys(probe_df["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.8), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, m in zip(np.atleast_1d(axes), models):
        d = probe_df[probe_df["model"] == m]
        layers = list(dict.fromkeys(d["layer"]))
        x = np.arange(len(layers))
        for color, (ds, g) in zip(SERIES, d.groupby("dataset_role", sort=False)):
            g = g.set_index("layer").loc[layers]
            ax.plot(x, g["macro_f1"], color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=g["dataset"].iloc[0])
            ax.annotate(f"{g['macro_f1'].iloc[-1]:.2f}", (x[-1], g["macro_f1"].iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK)
        ax.set_xticks(x, layers, rotation=35, ha="right", fontsize=7, color=INK)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0, labelsize=7, colors=INK_MUTED)
        ax.set_title(MODEL_NAMES[m], loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    np.atleast_1d(axes)[0].set_ylabel("Linear probe macro F1", fontsize=8, color=INK_MUTED)
    fig.suptitle("Where classes become linearly separable (probe fit on train-split activations)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse(spec: ModelSpec, out: Path, n_random: int, rng: np.random.Generator):
    k = len(spec.classes)
    act_model, names = intermediate_model(spec)
    acts = {role: activations(act_model, names, getattr(spec, role)) for role in ("train", "primary", "secondary")}

    # Sanity check: replaying the saved weights from each layer reproduces the model's predictions.
    for layer in spec.layers:
        a = acts["primary"]
        agree = (replay(downstream_dense(spec.model, layer), a[layer]).argmax(1) == a[spec.output].argmax(1)).mean()
        if agree < 0.999:
            raise RuntimeError(f"Replay from {spec.name}/{layer} disagrees with the model ({agree:.4f})")

    node_rows, layer_rows, probe_rows, ablation_rows = [], [], [], []

    # ---- Linear probes --------------------------------------------------- #
    def probe(name, f_train, role_feats):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        clf.fit(f_train, spec.train.y)
        res = {}
        for role, f in role_feats.items():
            y = getattr(spec, role).y
            p = clf.predict(f)
            res[role] = (accuracy_score(y, p), f1_score(y, p, average="macro"))
        return res

    probe_layers = [("input", {r: spec.input_repr(getattr(spec, r), acts[r]) for r in acts})] + \
                   [(lay, {r: acts[r][lay] for r in acts}) for lay in spec.layers]
    probe_scores = {}
    for name, feats in probe_layers:
        res = probe(name, feats["train"], {r: feats[r] for r in ("primary", "secondary")})
        probe_scores[name] = res
        for role, (acc, f1) in res.items():
            probe_rows.append({"model": spec.name, "layer": name, "dataset_role": role,
                               "dataset": getattr(spec, role).label, "probe_accuracy": round(acc, 4),
                               "macro_f1": round(f1, 4), "dims": feats[role].shape[1]})
    for role in ("primary", "secondary"):  # the model's own output, for reference
        y, p = getattr(spec, role).y, acts[role][spec.output].argmax(1)
        probe_rows.append({"model": spec.name, "layer": "model output", "dataset_role": role,
                           "dataset": getattr(spec, role).label, "probe_accuracy": round(accuracy_score(y, p), 4),
                           "macro_f1": round(f1_score(y, p, average="macro"), 4), "dims": k})

    # ---- Per-layer statistics, heatmaps --------------------------------- #
    for role in ("primary", "secondary"):
        split = getattr(spec, role)
        stats = []
        for layer in spec.layers:
            a = acts[role][layer]
            live = live_nodes(a)
            means, _ = class_means(a, split.y, k)
            z = zscore(means, live)
            relu = layer != "lstm"
            zz = np.nan_to_num(z, nan=-np.inf)
            pref = zz.argmax(0)
            top_z = np.where(live, zz.max(0), np.nan)
            spec_ = top_z / np.sqrt(k - 1)
            m = np.nan_to_num(means)
            top_mean = m[pref, np.arange(m.shape[1])]
            others = (m.sum(0) - top_mean) / max(k - 1, 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                sel = np.where(live & relu, (top_mean - others) / (top_mean + others), np.nan)
            for j in range(a.shape[1]):
                node_rows.append({"model": spec.name, "dataset": split.label, "layer": layer, "node": j,
                                  "active": bool(live[j]), "preferred_class": spec.classes[pref[j]] if live[j] else "",
                                  "top_z": round(float(top_z[j]), 4) if live[j] else np.nan,
                                  "specificity": round(float(spec_[j]), 4) if live[j] else np.nan,
                                  "specialist": bool(live[j] and spec_[j] >= SPECIALIST_THRESHOLD),
                                  "selectivity": round(float(sel[j]), 4) if not np.isnan(sel[j]) else np.nan,
                                  "mean_activation": round(float(a[:, j].mean()), 5)})
            is_spec = live & (np.nan_to_num(spec_) >= SPECIALIST_THRESHOLD)
            covered = len(set(pref[is_spec]))
            pf = probe_scores[layer][role][1]
            layer_rows.append({"model": spec.name, "dataset": split.label, "layer": layer, "units": a.shape[1],
                               "activation": "tanh (LSTM state)" if not relu else "relu",
                               "inactive_nodes": int((~live).sum()), "inactive_pct": round(float((~live).mean()), 4),
                               "mean_selectivity": round(float(np.nanmean(sel)), 4) if relu else np.nan,
                               "median_specificity": round(float(np.nanmedian(spec_)), 4),
                               "specialist_nodes": int(is_spec.sum()),
                               "classes_with_specialist": covered, "n_classes": k,
                               "probe_accuracy": round(probe_scores[layer][role][0], 4), "probe_macro_f1": round(pf, 4)})
            stats.append({"layer": layer, "z": z, "live": live, "probe_macro_f1": pf})
        plot_heatmaps(spec, split, stats, out / "heatmaps" / f"{spec.name}__{role}.png")

    # ---- Ablation (specialists chosen on train, measured on primary) ----- #
    y = spec.primary.y
    for layer in spec.layers:
        dense = downstream_dense(spec.model, layer)
        a_tr = acts["train"][layer]
        live_tr = live_nodes(a_tr)
        z_tr = zscore(class_means(a_tr, spec.train.y, k)[0], live_tr)
        a = acts["primary"][layer]
        base_pred = replay(dense, a).argmax(1)
        base_rec = recall_per_class(y, base_pred, k)
        base_f1 = f1_score(y, base_pred, average="macro")
        live_idx = np.flatnonzero(live_tr)
        for n_nodes in (1, 3):
            # Random control: ablate n random active nodes, n_random times.
            rand = []
            for _ in range(n_random):
                nodes = rng.choice(live_idx, size=n_nodes, replace=False)
                ab = a.copy()
                ab[:, nodes] = 0
                rand.append(recall_per_class(y, replay(dense, ab).argmax(1), k) - base_rec)
            rand = np.array(rand)
            for c, cls in enumerate(spec.classes):
                zc = np.where(live_tr, np.nan_to_num(z_tr[c], nan=-np.inf), -np.inf)
                nodes = np.argsort(-zc)[:n_nodes]
                ab = a.copy()
                ab[:, nodes] = 0
                pred = replay(dense, ab).argmax(1)
                rec = recall_per_class(y, pred, k)
                delta = rec - base_rec
                others = np.delete(delta, c)
                rand_c = rand[:, c]
                ablation_rows.append({
                    "model": spec.name, "layer": layer, "class": cls, "support": int((y == c).sum()),
                    "n_ablated": n_nodes, "nodes": " ".join(map(str, nodes)),
                    "node_z_train": " ".join(f"{z_tr[c, n]:.2f}" for n in nodes),
                    "recall_before": round(float(base_rec[c]), 4), "recall_after": round(float(rec[c]), 4),
                    "recall_change": round(float(delta[c]), 4),
                    "other_classes_mean_recall_change": round(float(np.nanmean(others)), 4),
                    "macro_f1_change": round(float(f1_score(y, pred, average="macro") - base_f1), 4),
                    "random_mean_recall_change": round(float(np.nanmean(rand_c)), 4),
                    "random_p5_recall_change": round(float(np.nanpercentile(rand_c, 5)), 4),
                    "specialist_effect_vs_random": round(float(delta[c] - np.nanmean(rand_c)), 4),
                    "beyond_random_p5": bool(delta[c] < np.nanpercentile(rand_c, 5)),
                })
    return node_rows, layer_rows, probe_rows, ablation_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--random-draws", type=int, default=30, help="Random-node ablations per layer (control)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "hidden_layers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("healthcare_ingestion", "httpx", "huggingface_hub", "generate_synthetic", "gensim",
                  "train_intent_lstm", "train_sentiment_head", "action_policy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out = args.out_dir
    (out / "heatmaps").mkdir(parents=True, exist_ok=True)
    clf = MessageClassifier()
    rng = np.random.default_rng(SEED)
    results = [[], [], [], []]
    for build in (intent_spec, sentiment_spec, action_spec):
        spec = build(clf)
        log.info("Analysing %s: layers %s | train=%d primary=%d secondary=%d", spec.name, spec.layers,
                 len(spec.train.y), len(spec.primary.y), len(spec.secondary.y))
        for acc, rows in zip(results, analyse(spec, out, args.random_draws, rng)):
            acc.extend(rows)

    nodes, layers, probes, ablation = (pd.DataFrame(r) for r in results)
    nodes.to_csv(out / "node_stats.csv", index=False)
    layers.to_csv(out / "layer_summary.csv", index=False)
    probes.to_csv(out / "linear_probe.csv", index=False)
    ablation.to_csv(out / "ablation.csv", index=False)
    plot_probes(probes, out / "linear_probe.png")

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        log.info("Layer summary:\n%s", layers.drop(columns=["n_classes"]).to_string(index=False))
        order = probes[["model", "layer"]].drop_duplicates()
        wide = probes.pivot(index=["model", "layer"], columns="dataset_role", values="macro_f1")
        log.info("Linear probe (macro F1):\n%s",
                 wide.reindex(pd.MultiIndex.from_frame(order)).to_string())
        summ = (ablation.groupby(["model", "layer", "n_ablated"], sort=False)
                .agg(mean_recall_change=("recall_change", "mean"),
                     mean_random_change=("random_mean_recall_change", "mean"),
                     classes_beyond_random=("beyond_random_p5", "sum"), n_classes=("class", "size"))
                .round(4))
        log.info("Ablation summary (specialists vs random, primary split):\n%s", summ.to_string())
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
