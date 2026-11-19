# Patient Response Classification

Classifies incoming patient messages into an **intent** and a **sentiment**, then maps
them to a platform **action** (book an appointment, send a payment link, route to a
human, emergency escalation, ...).

```
patient message ─┬─ intent LSTM (Word2Vec embeddings, 4x64 ReLU) ──── intent probs (17) ─────┐
                 ├─ sentiment head (frozen Hospital_Reviews BERT) ─── sentiment probs (3) ───┤
                 └─ Embedding (intent model's Word2Vec) ─ avg pool ── text embedding ────────┴─► action DNN (5x64 ReLU, softmax) ─► action
```

All three models are **trained on the Claude-generated synthetic healthcare messages**
(`llm_intent` / `llm_sentiment` / policy action, overridden by human review) and **tested
on real patient messages** from the healthcare chatbot dataset. The rule-based routing
table in `action_policy.py` is kept as a baseline.

This guide runs everything end to end: environment → data ingestion → synthetic data
generation → training → classification and evaluation.

**One command:** `./run_all.sh` runs every step below in order (calling `./setup.sh` first
if `.venv` is missing), logs each step to `logs/<step>.log`, stops at the first failure, and
prints the real test-set metrics at the end. Synthetic generation is skipped when
`data/synthetic/healthcare_synthetic.csv` already exists.

```bash
./run_all.sh                                # everything (~1.5–3 h on CPU, mostly intent tuning)
./run_all.sh --trials 40 --tune-workers 4   # parallel intent search
./run_all.sh --from sentiment               # resume from a step: ingest, generate, intent,
                                            #   sentiment, policy, action, evaluate
./run_all.sh --help                         # all options (--oof-folds, --generate, ...)
```

Each Optuna run adds trials to the existing study; `--trials 0` just retrains the best
parameters found so far.

---

## 0. Environment (one time)

Requires Python 3.13. Run once from the project root:

```bash
./setup.sh                  # creates .venv and installs pinned requirements.txt (~1 min)
source .venv/bin/activate   # every session, before running the scripts below
```

`setup.sh` finds a Python 3.13 interpreter (override with `PYTHON=/path/to/python3.13`),
creates `.venv`, installs `requirements.txt`, creates the `data/` and `models/` folders,
smoke-tests the imports, and tells you whether the healthcare dataset and Claude
credentials are in place. Re-running it is a quick no-op; `./setup.sh --force` rebuilds
the environment from scratch.

Notes:

- **Keras runs on the PyTorch backend.** TensorFlow is deliberately not installed (TF 2.20
  segfaults on import in the original Anaconda environment); every script sets
  `KERAS_BACKEND=torch` and blocks the TensorFlow import.
- **`brotli>=1.2` is required** for Hugging Face downloads (pinned in `requirements.txt`).
  Older Brotli fails with `TypeError: process() takes no keyword arguments`.
- **Claude API access** (only for synthetic data generation): create a key at
  <https://console.anthropic.com> and `export ANTHROPIC_API_KEY=sk-ant-...` in the shell you
  run the script from (or add it to `~/.zshrc`). Never commit the key.
- `data/` and `models/` are git-ignored; everything in them is produced by the steps below.

---

## 1. Data ingestion

### 1a. Customer-support data (sentiment training)

Downloads the public Kaggle dataset
[`devtayyabsajjad/nlp-customer-support-classification`](https://www.kaggle.com/datasets/devtayyabsajjad/nlp-customer-support-classification)
(no credentials needed) and cleans it.

```bash
python data_ingestion.py            # add --force to re-download
```

Output: `data/raw/nlp_customer_support_classification.csv`,
`data/processed/customer_support.csv`

### 1b. Healthcare chatbot intent data (intent training + test set)

This file is **not downloaded** — place it manually at:

```
data/raw/healthcare-chatbot-intent-dataset-10k-rows.csv
```

```bash
python healthcare_ingestion.py
```

What it does: keeps patient (`user`) messages only, maps ~480 raw intent labels onto
17 canonical intents with ordered regex rules (`INTENT_RULES`), and collapses
duplicate texts (majority label) so identical messages can't leak across splits.

Output: `data/processed/healthcare_intents.csv` (2,907 unique patient messages)

Options: `--include-bot` (keep bot messages), `--keep-duplicates`.

---

## 2. Synthetic data generation (Claude Haiku 5.5)

Creates ~10,000 pre-labelled messages for training the action model and for human
review. Requires step 1b and Claude API access.

For each real seed message (train/val split only; the 437 test messages are never used),
Claude rewrites it in a negative, neutral and positive tone with the same intent, then a
**separate blind call** labels each rewrite (intent, sentiment, emergency flag). The
action comes from `action_policy.py`; disagreements and emergencies are flagged for
human review.

```bash
python generate_synthetic.py --dry-run     # plan + cost estimate, no API calls
python generate_synthetic.py --limit 20    # pilot: ~60 messages, ~$0.01
python generate_synthetic.py               # full run: 3,334 jobs -> ~10,000 messages, ~$2
python generate_synthetic.py --build-only  # rebuild the CSV from the cache (no API calls)
```

Outputs:

| File | Contents |
|---|---|
| `data/synthetic/generations.jsonl` | Raw cached API results, one line per job. Re-runs resume from it and never re-bill a finished job. **Keep this file** — it is the reproducible artifact. |
| `data/synthetic/healthcare_synthetic.csv` | One row per message: text, seed, style, target sentiment, LLM labels, `action`, `policy_exception`, `review_status` / `review_reasons`, and blank `reviewer_*` columns |

**Human review:** open `healthcare_synthetic.csv`, filter `review_status == needs_review`,
and fill in `reviewer_intent`, `reviewer_sentiment`, `reviewer_action`, `reviewer_notes`
where the LLM label is wrong. Reviewer labels take precedence over LLM labels in training;
put `drop` in `reviewer_intent` or `reviewer_sentiment` to exclude a bad row. Re-running
the script (including `--build-only`) keeps existing reviewer columns. After reviewing,
retrain 3a and 3b.

Repeatability: the job plan is fixed by `--seed` (default 42). LLM outputs are not
bit-for-bit deterministic, so share `generations.jsonl` rather than re-generating.

---

## 3. Training

Run in this order — later steps read earlier outputs. Both models train on the synthetic
data from step 2 by default (`--train-on synthetic`). The real healthcare messages are
split 70/15/15 once (fixed seed): with synthetic training, the real **test** split (437
messages, never used as generation seeds) is the honest score. Every split groups a seed
message with its rewrites, so no model is ever scored on a message it saw a rewrite of.

### 3a. Intent model (Word2Vec + LSTM + 4x64 ReLU head)

Trains on the synthetic messages with their `llm_intent` labels. Word2Vec and the
vocabulary are trained on the synthetic text. Hyperparameter search (Optuna), then
retrain the best settings with 3 seeds and evaluate once on the real test split:

```bash
python tune_intent_lstm.py --n-trials 30
```

- **How trials are scored:** the real train+val messages are split into 3 folds. Each fold
  trains on the synthetic rewrites of the *other* folds' seeds (early stopping on a 10%
  seed-grouped synthetic hold-out) and is scored by macro F1 on the real messages in
  that fold.
- **Final model:** trained on all synthetic messages, saved to `models/intent_lstm_tuned/`
  (overwrites the earlier real-data model — copy it first if you want to compare).
  Real-val scores are optimistic here (val messages were seeds); judge on test.
- `--train-on real` reproduces the original real-data setup; `--train-on both` trains on
  synthetic + the real train split. Each source has its own study in
  `models/intent_lstm_tuning_<source>/` (real: `models/intent_lstm_tuning/`).
- ~3–5 min per trial on CPU (3× more training data than before). To parallelise, run
  several workers against the same study, then the final retrain once:
  ```bash
  python tune_intent_lstm.py --n-trials 9 --skip-final &   # x4
  python tune_intent_lstm.py --n-trials 0                  # retrain best params only
  ```
- `--n-trials 0` reuses an existing study; on a fresh checkout run a search first.
- After changing `INTENT_RULES`, re-run step 1b and then `--n-trials 0` to retrain on the
  new labels (or a fresh search).

Outputs: `models/intent_lstm_tuned/` (model, vocab, label encoder, Word2Vec, metrics,
test report) and `models/intent_lstm_tuning_synthetic/` (`study.db`, `trials.csv`,
`best_params.json`).

Optional / baselines:

```bash
python train_intent_lstm.py                  # single run with default params, synthetic training data
python train_intent_lstm.py --train-on real  # same, trained on the real train split
python train_intent_lstm.py --hidden-layers 4 --hidden-units 128 --seed 7 --out-dir models/x
python train_intent_dnn.py                   # earlier averaged-Word2Vec DNN baseline
```

### 3b. Sentiment model (frozen BERT + 2x64 head)

Freezes [`brettclaus/Hospital_Reviews`](https://huggingface.co/brettclaus/Hospital_Reviews)
(downloaded on first run, ~440 MB), embeds the synthetic messages with it, and trains a
2x64 ReLU head on their `llm_sentiment` labels. Splits are grouped by seed message, so
the negative / neutral / positive rewrites of one seed are never split across train and
test. The customer-support data (never trained on) is reported as an out-of-domain
check. Then scores every real healthcare message.

```bash
python train_sentiment_head.py
python train_sentiment_head.py --train-on support   # original customer-support setup
```

- The first run encodes ~13,500 texts with BERT on CPU (~10 min); features are
  cached in `data/processed/bert_features/`, so later runs take ~1–2 min.
- The real healthcare messages have no human sentiment labels, so sentiment accuracy is
  only measured on held-out synthetic seeds and the customer-support data.

Outputs: `models/sentiment_head/` and `data/processed/healthcare_sentiment.csv`
(per-message sentiment probabilities, joinable on `id`). Requires steps 1a, 1b and 2.

### 3c. Action labels for the real data

Applies the action policy (intent × sentiment → action) to every healthcare message.
This is the reference the classifier is evaluated against. Re-run it after 3b — the
reference sentiment comes from the sentiment head.

```bash
python action_policy.py
```

Output: `data/processed/healthcare_actions.csv` (adds `group` and `action`).
Requires 3b. Fails with a "stale" error if `healthcare_sentiment.csv` doesn't match the
current intent labels — re-run 3b.

### 3d. Action classifier (DNN, 5x64)

Input: the message's tokens through an **Embedding layer** (same vocabulary as the intent
model, initialised from its fine-tuned Word2Vec weights, trainable, masked average pooling)
concatenated with the **intent probabilities** (17) and **sentiment probabilities** (3).
Then **5 hidden ReLU layers × 64 units** and a **softmax** over the 8 actions.

```bash
python classify.py --train
```

- **Labels:** `action_label` from `healthcare_synthetic.csv` — `reviewer_action` if filled,
  otherwise the action policy on the final intent/sentiment labels, escalated to
  `route_to_human_emergency` when the LLM flagged an emergency.
- **No stacking leakage:** for the synthetic training rows, the intent and sentiment inputs
  are out-of-fold predictions — 5 seed-grouped folds, each training a fresh intent LSTM
  and sentiment head (same settings as the saved models) on the other folds. Otherwise the
  DNN would learn from over-confident in-sample probabilities it never sees on new
  messages. This takes the longest (~5 LSTM fits, ~15–30 min); results are cached in
  `models/action_dnn/oof_features.npz` and reused until the data or models change.
  `--oof-folds 0` skips it (quick experiments only).
- **Splits:** synthetic data 70/15/15 grouped by seed message (train / early stopping and
  seed choice / synthetic test), 3 seeds, best kept by val.
- Options: `--hidden-layers`, `--hidden-units`, `--dropout`, `--lr`, `--epochs`,
  `--freeze-embedding`, `--seeds`.

Output: `models/action_dnn/` (`model.keras`, `preprocessing.joblib`, `metrics.json`,
`synthetic_test_report.txt`, `oof_features.npz`). Requires 2, 3a and 3b. Retrain after
retraining either upstream model (`classify.py` refuses a mismatched action model).

---

## 4. Classification and evaluation

Classify messages with the action DNN:

```bash
python classify.py "I need to book a dental check-up next week" "What are the accepted payment methods?"
```

Returns intent, intent confidence, emergency probability, sentiment, the DNN's `action`
and `action_confidence`, plus the routing table's `policy_action` and `policy_reason`
(`table`, `emergency_override`, `low_confidence`) for comparison.

Evaluate predicted actions on the 437 real test messages (macro F1, per-action report,
confusion matrix, error list), with the routing table as a baseline:

```bash
python classify.py --evaluate
```

Output: `models/action_eval/` (`metrics.json`, `classification_report.txt`,
`confusion_matrix.csv`, `errors.csv`, and `*_routing_table.*` baseline files).
Requires 3a–3d.

The reference action for the real messages is the policy applied to the true intent and
the sentiment head's sentiment (no human sentiment/action labels exist), which favours the
routing table: it uses that same sentiment and the same rules. Read the comparison with
that in mind.

Routing-table safety rules (baseline only; constants in `action_policy.py`):

- `EMERGENCY_THRESHOLD = 0.2` — if P(emergency) ≥ 0.2, route to `route_to_human_emergency`.
- `LOW_CONFIDENCE_THRESHOLD = 0.5` — below this intent confidence, automated actions fall
  back to `route_to_human`.

---

## Quick reference

| Step | Command | Needs | Produces | Time |
|---|---|---|---|---|
| 1a | `python data_ingestion.py` | internet | `data/processed/customer_support.csv` | seconds |
| 1b | `python healthcare_ingestion.py` | raw healthcare CSV | `data/processed/healthcare_intents.csv` | seconds |
| 2 | `python generate_synthetic.py` | 1b, Claude API | `data/synthetic/*` | ~10–15 min, ~$2 |
| 3a | `python tune_intent_lstm.py --n-trials 30` | 1b, 2 | `models/intent_lstm_tuned/` | ~1.5–2.5 h (parallelise) |
| 3b | `python train_sentiment_head.py` | 1a, 1b, 2 | `models/sentiment_head/`, `healthcare_sentiment.csv` | ~10 min first run, then ~2 min |
| 3c | `python action_policy.py` | 3b | `data/processed/healthcare_actions.csv` | seconds |
| 3d | `python classify.py --train` | 2, 3a, 3b | `models/action_dnn/` | ~15–30 min (OOF), then ~2 min |
| 4 | `python classify.py --evaluate` | 3a–3d | `models/action_eval/` | ~1 min |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Exit code 139 / segfault on import | TensorFlow is being imported; run the scripts as-is (they block it) and don't `import tensorflow` yourself |
| `TypeError: process() takes no keyword arguments` during a Hugging Face download | `pip install -U "brotli>=1.2"` |
| `... is stale; rerun ...` | An upstream step changed; re-run the step named in the message |
| `Intents with no policy: [...]` | A new intent was added to `INTENT_RULES` without a row in `action_policy.POLICY` |
| Authentication error in `generate_synthetic.py` | `export ANTHROPIC_API_KEY=sk-ant-...` |
| `ModuleNotFoundError: No module named 'keras'` | The venv isn't active (prompt shows `(base)`): `source .venv/bin/activate` |
| `No action model in models/action_dnn` | Run `python classify.py --train` |
| `The action DNN was trained against different intent/sentiment models` | Retrained 3a or 3b; rerun `python classify.py --train` |
| `Raw dataset not found` | Place the healthcare CSV in `data/raw/` (step 1b) |
| `healthcare_synthetic.csv not found` | Run step 2 (or `--train-on real` / `--train-on support` for the old setups) |
| `Synthetic intents not in the real label set` | `INTENT_RULES` changed after generation; regenerate, or map the old labels |
| `Unknown labels in healthcare_synthetic.csv` | Typo in a `reviewer_intent` / `reviewer_sentiment` cell |
