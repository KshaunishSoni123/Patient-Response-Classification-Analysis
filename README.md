# Patient Response Classification

Classifies incoming patient messages into an **intent** and a **sentiment**, then maps
them to a platform **action** (book an appointment, send a payment link, route to a
human, emergency escalation, ...).

```
patient message ─┬─ intent LSTM (Word2Vec embeddings, 4x64 ReLU) ─┬── intent probs (17) ──────┐
                 │                                                 └── LSTM hidden state (64) ──┤
                 └─ sentiment head (frozen Hospital_Reviews BERT) ──── sentiment probs (3) ─────┴─► action DNN (1x64 ReLU, softmax) ─► action
```

**Data per model.** The intent LSTM trains on the **real healthcare intent data** (real train
split, early-stopped on real val). The sentiment head and the action DNN train on the
**Claude-generated synthetic healthcare messages** (`llm_sentiment` / policy action, overridden
by human review), using one **shared synthetic train / val / test split** (`synthetic_splits.py`):

| Synthetic split | Messages | Rewrites of | Used for |
|---|---|---|---|
| train | 8,206 | real **train** messages | intent + sentiment + action training |
| val | 844 | half of the real **val** messages (by seed) | early stopping, seed choice (all three models) |
| test | 896 | the other half of the real **val** messages | per-model and **end-to-end** test |

Because synthetic val/test only rewrite messages the intent model never trained on, all three
models (and the full pipeline) are scored on the same held-out synthetic messages without
leakage. The **437 real test messages** (never used as generation seeds) remain the final
external test, with hand-labelled sentiment in `labels/`. The rule-based routing table in
`action_policy.py` is the baseline.

**Model history.** v1: intent on synthetic data only, action DNN with an averaged trainable
embedding and 5x64 layers. v2: intent on synthetic + real train, action DNN on the intent LSTM
state with 1x64 (`experiments.py`, 120 runs: real-test action macro F1 0.736 -> 0.806).
v3: intent on the real data, the shared synthetic split for sentiment and action. **Final (v4):**
all three models trained on synthetic train, early-stopped on synthetic val. Earlier models are kept
in `models/intent_lstm_tuned_v1_synthetic/`, `models/action_dnn_v1_avgpool/` and
`models/archive/<intent train_on>/`; `reports/` holds the final (v4) results only.

This guide runs everything end to end: environment → data ingestion → synthetic data
generation → training → classification and evaluation.

**One command:** `./run_all.sh` runs every step below in order (calling `./setup.sh` first
if `.venv` is missing), logs each step to `logs/<step>.log`, stops at the first failure,
prints the real test-set metrics, and writes the report tables and figures (`reports/`).
Synthetic generation is skipped when `data/synthetic/healthcare_synthetic.csv` already exists,
and the sentiment head is reused when it is already trained on the current shared split
(`--retrain-sentiment` to force). To retrain everything on existing data without re-ingesting:
`./run_all.sh --from split`.

```bash
./run_all.sh                                # everything (~1.5–3 h on CPU, mostly intent tuning)
./run_all.sh --trials 40 --tune-workers 4   # parallel intent search
./run_all.sh --from sentiment               # resume from a step: ingest, generate, split, intent,
                                            #   sentiment, policy, action, evaluate, report
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

**This snapshot is what every later step reads** (`healthcare_ingestion.load_snapshot()`), not
the raw CSV. Splits, synthetic seeds and hand labels are keyed to its rows, so replacing the raw
file has no effect until you re-run this step - and after re-running it you must regenerate the
synthetic data and the labels too.

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

## 2b. Shared synthetic split

```bash
python synthetic_splits.py      # also built automatically on first use
```

Writes `data/processed/synthetic_splits.csv` (+ `.meta`): every synthetic row's split. Train =
rewrites of real train messages; val / test = rewrites of real val messages, divided by seed
(a seed's rewrites stay together). Whole seeds can't be stratified finely, so 200 candidate
divisions are tried and the one with the most even intent + action counts is kept (labels only,
never model outputs). Classes with very few val seeds stay lopsided (e.g. `other` has one seed,
all in val). Rebuilt automatically if the synthetic data changes.

---

## 3. Training

Run in this order — later steps read earlier outputs. The real healthcare messages are split
70/15/15 once (fixed seed); the real **test** split (437 messages, never used as generation
seeds) is the final external test. Every split groups a seed message with its rewrites.

### 3a. Intent model (Word2Vec + LSTM + 4x64 ReLU head)

**Pipeline default:** synthetic train, early stopping and seed choice on synthetic val (the shared
split), with the best params of the real-data Optuna study (`models/intent_lstm_tuning/`) so the
architecture is the same as v3 and only the training data changes; 3 seeds:

```bash
python tune_intent_lstm.py --train-on synthetic --params-from real
```

Other training sources: `--train-on real` (real train, early-stop on real val; v3) or
`--train-on both` (synthetic train + real train, early-stop on synthetic val). A fresh search: `python tune_intent_lstm.py --train-on real --n-trials 30`.

- **How trials are scored:** 3-fold CV macro F1 on the real train+val messages (for synthetic
  sources, each fold trains on the rewrites of the *other* folds' seeds).
- Each source has its own study in `models/intent_lstm_tuning_<source>/` (real:
  `models/intent_lstm_tuning/`). `run_all.sh` archives the current models to
  `models/archive/<train_on>/` before switching sources.
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
2x64 ReLU head on their `llm_sentiment` labels: **synthetic train**, early stopping on
**synthetic val**, scored on **synthetic test** (the shared split). The 5-fold CV runs on
train + val only. The customer-support data (never trained on) is reported as an out-of-domain
check. Then scores every real healthcare message.

```bash
python train_sentiment_head.py
python train_sentiment_head.py --train-on support   # original customer-support setup
```

- The first run encodes ~13,500 texts with BERT on CPU (~10 min); features are
  cached in `data/processed/bert_features/`, so later runs take ~1–2 min.
- Real-message sentiment accuracy comes from the hand labels (`labels/`, step 5).

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

### 3d. Action classifier (DNN, 1x64)

Input: the **intent LSTM's final hidden state** (64; the intent model's reading of the
message, frozen) concatenated with the **intent probabilities** (17) and **sentiment
probabilities** (3). Then **1 hidden ReLU layer × 64 units** and a **softmax** over the 8 actions.
`--text-input avg_pool --hidden-layers 5` restores v1 (trainable Embedding layer initialised
from the intent model's Word2Vec, masked average pooling, 5 x 64).

```bash
python classify.py --train
```

- **Labels:** `action_label` from `healthcare_synthetic.csv` — `reviewer_action` if filled,
  otherwise the action policy on the final intent/sentiment labels, escalated to
  `route_to_human_emergency` when the LLM flagged an emergency.
- **Inputs per split:** synthetic **val / test** rows get the **final** intent and sentiment
  models' predictions — neither model has seen them or their seed — so val/test scores are
  end-to-end pipeline scores. Synthetic **train** rows get **out-of-fold** predictions (5
  seed-grouped folds over the train rows; a fresh intent LSTM and sentiment head are fit without
  each fold, and the intent model also without the real messages that fold rewrites), so the DNN
  learns from probabilities as noisy as the ones it gets on new messages. Cached in
  `models/action_dnn/oof_features.npz`; `--oof-folds 0` skips it (quick experiments only).
  All inputs are saved to `models/action_dnn/action_inputs.npz` for the reports.
- **Splits:** the shared synthetic split (train / early stopping and seed choice / test),
  3 seeds, best kept by val.
- **LSTM state:** taken from the final intent model for every row (hidden states of different
  fold models aren't in a common basis, so they can't be stacked out-of-fold). The intent model
  trained on the real messages that the synthetic *train* rows rewrite, so those states are
  closer to in-sample; synthetic val/test and real test are unseen.
- Options: `--text-input`, `--hidden-layers`, `--hidden-units`, `--dropout`, `--lr`, `--epochs`,
  `--freeze-embedding` (avg_pool), `--seeds`.

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
the sentiment head's sentiment, which favours the routing table: it uses that same sentiment
and the same rules. Read the comparison with that in mind — or hand-label the sentiment (step 5).

Routing-table safety rules (baseline only; constants in `action_policy.py`):

- `EMERGENCY_THRESHOLD = 0.2` — if P(emergency) ≥ 0.2, route to `route_to_human_emergency`.
- `LOW_CONFIDENCE_THRESHOLD = 0.5` — below this intent confidence, automated actions fall
  back to `route_to_human`.

---

## 5. Reports, hand labels and experiments

```bash
python evaluation_report.py      # reports/: summary + per-class metrics with 95% CIs, confusion matrices
python hidden_layer_analysis.py  # reports/hidden_layers/: activation heatmaps, linear probes, node ablation
python experiments.py            # reports/experiments/action_architecture/: v1 vs v2 architecture grid (~2 h)
```

**Hand-labelled sentiment for the real test messages** (removes the circular reference):

```bash
python label_sentiment.py --annotator <your-name>   # blind, resumable, one keypress per message
```

Read `labels/GUIDELINES.md` first. The existing labels use a **situation-based** scheme
(positive = a request the practice can fulfil, negative = a health problem or unwanted
situation, neutral = informational), not the tone-based sentiment the policy table was written
for - so positive insurance questions map to `thank_you_message` in the human-sentiment reference
(see the guidelines for the counts). Labels are saved to `labels/real_test_sentiment_<name>.csv`
after every keypress (commit them). With labels present, `evaluation_report.py` adds:

- `sentiment_head` on the real test messages against the human labels;
- `action_dnn` and `routing_table` against a reference built from the true intent + the
  **human** sentiment (`data = real (human-sentiment reference)`);
- `reports/human_labels_info.json`: label counts, how many reference actions changed, and
  Cohen's kappa when two or more annotators labelled (majority vote; ties are excluded).

---

## Quick reference

| Step | Command | Needs | Produces | Time |
|---|---|---|---|---|
| 1a | `python data_ingestion.py` | internet | `data/processed/customer_support.csv` | seconds |
| 1b | `python healthcare_ingestion.py` | raw healthcare CSV | `data/processed/healthcare_intents.csv` | seconds |
| 2 | `python generate_synthetic.py` | 1b, Claude API | `data/synthetic/*` | ~10–15 min, ~$2 |
| 2b | `python synthetic_splits.py` | 1b, 2 | `data/processed/synthetic_splits.csv` | seconds |
| 3a | `python tune_intent_lstm.py --train-on synthetic --params-from real` | 1b, real study | `models/intent_lstm_tuned/` | ~5 min (a fresh search: ~1 h) |
| 3b | `python train_sentiment_head.py` | 1a, 1b, 2, 2b | `models/sentiment_head/`, `healthcare_sentiment.csv` | ~10 min first run, then ~2 min |
| 3c | `python action_policy.py` | 3b | `data/processed/healthcare_actions.csv` | seconds |
| 3d | `python classify.py --train` | 2, 3a, 3b | `models/action_dnn/` | ~15–30 min (OOF), then ~2 min |
| 4 | `python classify.py --evaluate` | 3a–3d | `models/action_eval/` | ~1 min |
| 5 | `python evaluation_report.py`, `python hidden_layer_analysis.py` | 3a–3d (+ `labels/`) | `reports/` | ~2 min |
| – | `python label_sentiment.py --annotator NAME` | 1b | `labels/` | ~20–30 min of labelling |

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
