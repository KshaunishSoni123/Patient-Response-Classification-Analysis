"""Hyperparameter tuning for the intent LSTM (fixed 4x64 ReLU head), optimizing macro F1.

Search:
  * Optuna TPE sampler over learning rate, batch size, dropout, LSTM units,
    bidirectionality, Word2Vec dimension and embedding freezing.
  * Each trial is scored by stratified 3-fold cross-validated **macro F1** on the
    train + val portion of the fixed split (the test split is never touched).
    Word2Vec and the vocab are refit inside every fold, so there is no leakage.
    CV is used instead of the single val split because several classes have only
    3-6 val examples, making a single-split F1 too noisy to rank trials.
  * A median pruner stops clearly bad trials after their first fold(s).
  * Trials are stored in SQLite, so the study can be resumed or run by several
    processes in parallel (each pointing at the same --study-dir).

Final model (after the search, unless --skip-final):
  * Retrain the best params on the train split (early stopping on val) with
    --final-seeds seeds, report test metrics as mean ± std, and save the seed
    with the best *val* macro F1 to models/intent_lstm_tuned/.

Usage:
    python tune_intent_lstm.py --n-trials 30
    # parallel: run N workers with --skip-final, then once with --n-trials 0
    python tune_intent_lstm.py --n-trials 10 --skip-final &   (x3)
    python tune_intent_lstm.py --n-trials 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# Importing train_intent_lstm first applies its Keras backend / TensorFlow workaround.
from train_intent_lstm import ROOT, SEED, evaluate, fit, prepare_data, save, split_indices

STUDY_DIR = ROOT / "models" / "intent_lstm_tuning"
FINAL_DIR = ROOT / "models" / "intent_lstm_tuned"
STUDY_NAME = "intent_lstm_4x64_macro_f1"
CV_FOLDS = 3
FIXED = dict(hidden_layers=4, hidden_units=64, epochs=100, patience=10, w2v_init=True)

log = logging.getLogger("tune_intent_lstm")


def suggest(trial: optuna.Trial) -> dict:
    return dict(
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
        lstm_units=trial.suggest_categorical("lstm_units", [32, 64, 128]),
        bidirectional=trial.suggest_categorical("bidirectional", [False, True]),
        w2v_dim=trial.suggest_categorical("w2v_dim", [50, 100, 200]),
        freeze_embeddings=trial.suggest_categorical("freeze_embeddings", [False, True]),
    )


def make_objective(tokens, y, idx_dev, n_classes):
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(idx_dev, y[idx_dev]))

    def objective(trial: optuna.Trial) -> float:
        params = {**FIXED, **suggest(trial)}
        scores, epochs = [], []
        for k, (tr, va) in enumerate(folds):
            fitted = fit(tokens, y, idx_dev[tr], idx_dev[va], n_classes, seed=SEED, verbose=0, **params)
            pred = fitted["model"].predict(fitted["X"][idx_dev[va]], verbose=0).argmax(axis=1)
            scores.append(f1_score(y[idx_dev[va]], pred, average="macro"))
            epochs.append(fitted["best_epoch"])
            trial.report(float(np.mean(scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_macro_f1", [round(s, 4) for s in scores])
        trial.set_user_attr("fold_best_epoch", epochs)
        trial.set_user_attr("cv_macro_f1_std", round(float(np.std(scores)), 4))
        return float(np.mean(scores))

    return objective


def load_study(study_dir: Path) -> optuna.Study:
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{study_dir / 'study.db'}", engine_kwargs={"connect_args": {"timeout": 60}},
    )
    return optuna.create_study(
        study_name=STUDY_NAME, storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=8),  # random seed: parallel workers must differ
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )


def train_final(study: optuna.Study, tokens, y, encoder, seeds: list[int], out: Path) -> None:
    best = {**FIXED, **study.best_params}
    log.info("Best CV macro F1 %.4f (trial %d) with params: %s",
             study.best_value, study.best_trial.number, study.best_params)

    idx_train, idx_val, idx_test = split_indices(y)
    n_classes = len(encoder.classes_)
    runs = []
    for seed in seeds:
        fitted = fit(tokens, y, idx_train, idx_val, n_classes, seed=seed, verbose=0, **best)
        metrics = evaluate(fitted, y, {"train": idx_train, "val": idx_val, "test": idx_test})
        log.info("Final seed %d: %s", seed, metrics)
        runs.append((fitted, metrics))

    summary = {}
    for key in ["val_macro_f1", "test_accuracy", "test_macro_f1"]:
        vals = [m[key] for _, m in runs]
        summary[key] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
    log.info("Final retrain over %d seeds: %s", len(seeds), summary)

    # Choose which seed to keep by val F1, never by test F1.
    fitted, metrics = max(runs, key=lambda r: r[1]["val_macro_f1"])
    extra = {"cv_macro_f1": round(study.best_value, 4), "best_trial": study.best_trial.number,
             "seeds_summary": summary, "seeds": seeds}
    report = save(fitted, encoder, y, idx_test, metrics, out, extra=extra)
    log.info("Kept seed %d. Test classification report:\n%s", fitted["seed"], report)
    log.info("Saved tuned model to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-dir", type=Path, default=STUDY_DIR)
    parser.add_argument("--out-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--skip-final", action="store_true", help="Only run trials")
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[42, 7, 123])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("gensim").setLevel(logging.WARNING)
    logging.getLogger("healthcare_ingestion").setLevel(logging.WARNING)

    df, tokens, y, encoder = prepare_data()
    idx_train, idx_val, _ = split_indices(y)
    idx_dev = np.concatenate([idx_train, idx_val])

    study = load_study(args.study_dir)
    if args.n_trials > 0:
        study.optimize(make_objective(tokens, y, idx_dev, len(encoder.classes_)), n_trials=args.n_trials)

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    trials.sort_values("value", ascending=False).to_csv(args.study_dir / "trials.csv", index=False)
    done = trials[trials.state == "COMPLETE"]
    log.info("Trials: %d complete, %d pruned | top 5:\n%s", len(done), (trials.state == "PRUNED").sum(),
             done.sort_values("value", ascending=False).head(5).to_string(index=False))

    if not args.skip_final:
        train_final(study, tokens, y, encoder, args.final_seeds, args.out_dir)
        (args.study_dir / "best_params.json").write_text(
            json.dumps({"cv_macro_f1": study.best_value, "params": study.best_params, "fixed": FIXED}, indent=2))


if __name__ == "__main__":
    main()
