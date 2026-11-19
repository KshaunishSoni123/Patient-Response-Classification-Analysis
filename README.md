# Patient Response Classification

Classifies incoming patient messages into an **intent** and a **sentiment**, then maps
them to a platform **action** (book an appointment, send a payment link, route to a
human, emergency escalation, ...).

```
patient message ─┬─ intent LSTM (Word2Vec embeddings, 4x64 ReLU)  ── 17 intents ──┐
                 └─ sentiment head (frozen Hospital_Reviews BERT) ── neg/neu/pos ─┴─► action_policy ─► action
```

This guide runs everything end to end: environment → data ingestion → synthetic data
generation → training → classification and evaluation.

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
- **Claude API access** (only for synthetic data generation): run `ant auth login`, or
  export `ANTHROPIC_API_KEY` in the shell you run the script from.
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
where the LLM label is wrong. Reviewer labels take precedence over LLM labels in training.

Repeatability: the job plan is fixed by `--seed` (default 42). LLM outputs are not
bit-for-bit deterministic, so share `generations.jsonl` rather than re-generating.

---

## 3. Training

Run in this order — later steps read earlier outputs.

### 3a. Intent model (Word2Vec + LSTM + 4x64 ReLU head)

Hyperparameter search (Optuna, 3-fold CV macro F1 on train+val), then retrain the best
settings with 3 seeds and evaluate once on the test split:

```bash
python tune_intent_lstm.py --n-trials 30
```

- ~1–2 min per trial on CPU. To parallelise, run several workers against the same study,
  then the final retrain once:
  ```bash
  python tune_intent_lstm.py --n-trials 9 --skip-final &   # x4
  python tune_intent_lstm.py --n-trials 0                  # retrain best params only
  ```
- `--n-trials 0` reuses an existing study; on a fresh checkout run a search first.
- After changing `INTENT_RULES`, re-run step 1b and then `--n-trials 0` to retrain on the
  new labels (or a fresh search).

Outputs: `models/intent_lstm_tuned/` (model, vocab, label encoder, Word2Vec, metrics,
test report) and `models/intent_lstm_tuning/` (`study.db`, `trials.csv`, `best_params.json`).

Optional / baselines:

```bash
python train_intent_lstm.py                  # single run with default params (4x64 head)
python train_intent_lstm.py --hidden-layers 4 --hidden-units 128 --seed 7 --out-dir models/x
python train_intent_dnn.py                   # earlier averaged-Word2Vec DNN baseline
```

### 3b. Sentiment model (frozen BERT + 2x64 head)

Freezes [`brettclaus/Hospital_Reviews`](https://huggingface.co/brettclaus/Hospital_Reviews)
(downloaded on first run, ~440 MB) and trains a 2x64 ReLU head on the customer-support
sentiment labels, evaluated on template-disjoint splits. Then scores every healthcare
message.

```bash
python train_sentiment_head.py
```

Outputs: `models/sentiment_head/` and `data/processed/healthcare_sentiment.csv`
(per-message sentiment probabilities, joinable on `id`). Requires steps 1a and 1b.

### 3c. Action labels for the real data

Applies the action policy (intent × sentiment → action) to every healthcare message.
This is the reference the classifier is evaluated against.

```bash
python action_policy.py
```

Output: `data/processed/healthcare_actions.csv` (adds `group` and `action`).
Requires 3b. Fails with a "stale" error if `healthcare_sentiment.csv` doesn't match the
current intent labels — re-run 3b.

### 3d. Action classifier on synthetic data — *not yet implemented*

Planned: a Dense(64)–Dense(64)–softmax model over intent probabilities, sentiment
probabilities and BERT text features, trained on `healthcare_synthetic.csv`
(reviewer labels where present) and tested on the real held-out messages.

---

## 4. Classification and evaluation

Classify messages with the trained models + action policy:

```bash
python classify.py "I need to book a dental check-up next week" "What are the accepted payment methods?"
```

Returns intent, intent confidence, emergency probability, sentiment, group, action and
the decision reason (`table`, `emergency_override`, `low_confidence`).

Evaluate predicted actions on the 437 real test messages (macro F1, per-action report,
confusion matrix, error list):

```bash
python classify.py --evaluate
```

Output: `models/action_eval/` (`metrics.json`, `classification_report.txt`,
`confusion_matrix.csv`, `errors.csv`). Requires 3a, 3b and 3c.

Safety rules applied at inference (constants in `action_policy.py`):

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
| 3a | `python tune_intent_lstm.py --n-trials 30` | 1b | `models/intent_lstm_tuned/` | ~30–60 min |
| 3b | `python train_sentiment_head.py` | 1a, 1b | `models/sentiment_head/`, `healthcare_sentiment.csv` | ~1–2 min |
| 3c | `python action_policy.py` | 3b | `data/processed/healthcare_actions.csv` | seconds |
| 4 | `python classify.py --evaluate` | 3a–3c | `models/action_eval/` | ~1 min |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Exit code 139 / segfault on import | TensorFlow is being imported; run the scripts as-is (they block it) and don't `import tensorflow` yourself |
| `TypeError: process() takes no keyword arguments` during a Hugging Face download | `pip install -U "brotli>=1.2"` |
| `... is stale; rerun ...` | An upstream step changed; re-run the step named in the message |
| `Intents with no policy: [...]` | A new intent was added to `INTENT_RULES` without a row in `action_policy.POLICY` |
| Authentication error in `generate_synthetic.py` | `ant auth login`, or export `ANTHROPIC_API_KEY` |
| `Raw dataset not found` | Place the healthcare CSV in `data/raw/` (step 1b) |
