#!/usr/bin/env bash
# End-to-end pipeline: ingestion -> synthetic data -> intent / sentiment / action models -> evaluation.
#
#   ./run_all.sh                       full pipeline (runs ./setup.sh first if .venv is missing)
#   ./run_all.sh --from sentiment      resume from a step
#   ./run_all.sh --trials 40 --tune-workers 4
#
# Steps (in order):
#   ingest     1a + 1b  data_ingestion.py, healthcare_ingestion.py
#   generate   2        generate_synthetic.py - SKIPPED if data/synthetic/healthcare_synthetic.csv
#                       exists (use --generate to resume/extend it; needs ANTHROPIC_API_KEY)
#   split      2b       synthetic_splits.py - the shared synthetic train / val / test split used by
#                       all three models (train = rewrites of real train messages, val / test =
#                       rewrites of real val messages, split by seed)
#   intent     3a       tune_intent_lstm.py - default: synthetic train, early-stopped on synthetic val
#                       (--train-on synthetic --params-from real: best params of the real-data study)
#   sentiment  3b       train_sentiment_head.py on synthetic train - SKIPPED if the existing head was
#                       already trained on the current shared split (--retrain-sentiment to force)
#   policy     3c       action_policy.py
#   action     3d       classify.py --train on synthetic train (intent LSTM state + 1x64 head)
#   evaluate   4        classify.py --evaluate
#   report     5        evaluation_report.py + hidden_layer_analysis.py (reports/); includes the
#                       hand-labelled sentiment rows when labels/real_test_sentiment_*.csv exist
#                       (label with: python label_sentiment.py --annotator NAME)
#
# Before the intent step overwrites models trained another way (different --intent-train-on), the
# current intent / sentiment / action models are copied once to models/archive/<train_on>/.
#
# Options:
#   --from STEP          start at STEP (earlier outputs must already exist)
#   --intent-train-on S  intent training data: synthetic (default), real or both
#   --intent-params-from S
#                        reuse the tuned params of S's study, no search (default real);
#                        "none" runs an Optuna search on the --intent-train-on study instead
#   --trials N           Optuna trials when searching (default 30; 0 = retrain the
#                        best params of the existing study only). Trials ADD to the study.
#   --tune-workers W     run the intent search in W parallel processes (default 1)
#   --oof-folds K        out-of-fold folds for the action DNN (default 5; 0 = fast/in-sample)
#   --action-text-input T  lstm_state (default) or avg_pool (v1)
#   --action-layers N    action DNN hidden layers (default 1; v1 used 5)
#   --action-units N     action DNN hidden units (default 64)
#   --retrain-sentiment  retrain the sentiment head even if it matches the current split
#   --experiments        also run experiments.py (architecture grid, ~2 h, resumable) at the end
#   --generate           run synthetic generation even if the CSV already exists
#
# Each step's output is also written to logs/<step>.log.
set -euo pipefail
cd "$(dirname "$0")"

STEPS=(ingest generate split intent sentiment policy action evaluate report)
FROM=ingest
TRIALS=30
INTENT_TRAIN_ON=synthetic
INTENT_PARAMS_FROM=real
TUNE_WORKERS=1
OOF_FOLDS=5
ACTION_TEXT_INPUT=lstm_state
ACTION_LAYERS=1
ACTION_UNITS=64
RETRAIN_SENTIMENT=0
RUN_EXPERIMENTS=0
FORCE_GENERATE=0

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --intent-train-on) INTENT_TRAIN_ON="$2"; shift 2 ;;
        --intent-params-from) INTENT_PARAMS_FROM="$2"; shift 2 ;;
        --tune-workers) TUNE_WORKERS="$2"; shift 2 ;;
        --oof-folds) OOF_FOLDS="$2"; shift 2 ;;
        --action-text-input) ACTION_TEXT_INPUT="$2"; shift 2 ;;
        --action-layers) ACTION_LAYERS="$2"; shift 2 ;;
        --action-units) ACTION_UNITS="$2"; shift 2 ;;
        --retrain-sentiment) RETRAIN_SENTIMENT=1; shift ;;
        --experiments) RUN_EXPERIMENTS=1; shift ;;
        --generate) FORCE_GENERATE=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done
if [[ ! " ${STEPS[*]} " == *" $FROM "* ]]; then
    echo "--from must be one of: ${STEPS[*]}" >&2; exit 1
fi

# --- Environment -------------------------------------------------------------
if [[ ! -x .venv/bin/python ]]; then
    echo "==> .venv not found; running ./setup.sh"
    ./setup.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs

# --- Helpers -----------------------------------------------------------------
started=0
should_run() {  # true once we've reached the --from step
    [[ "$1" == "$FROM" ]] && started=1
    [[ $started -eq 1 ]]
}
run() {  # run <step> <command...>: echo, time, and tee to logs/<step>.log
    local step="$1"; shift
    echo
    echo "==> [$step] $*"
    local t0=$SECONDS
    "$@" 2>&1 | tee -a "logs/$step.log"
    echo "==> [$step] done in $(( (SECONDS - t0) / 60 ))m $(( (SECONDS - t0) % 60 ))s"
}
for s in "${STEPS[@]}"; do  # fresh logs for the steps that will run
    should_run "$s" && : > "logs/$s.log"
done
started=0
T_START=$SECONDS

# --- 1. Ingestion ------------------------------------------------------------
if should_run ingest; then
    if [[ ! -f data/raw/healthcare-chatbot-intent-dataset-10k-rows.csv ]]; then
        echo "ERROR: place the healthcare dataset at data/raw/healthcare-chatbot-intent-dataset-10k-rows.csv" >&2
        exit 1
    fi
    run ingest python data_ingestion.py
    run ingest python healthcare_ingestion.py
fi

# --- 2. Synthetic data -------------------------------------------------------
if should_run generate; then
    if [[ -f data/synthetic/healthcare_synthetic.csv && $FORCE_GENERATE -eq 0 ]]; then
        echo
        echo "==> [generate] skipped: data/synthetic/healthcare_synthetic.csv exists (--generate to rerun)"
    else
        if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
            echo "ERROR: synthetic data generation needs ANTHROPIC_API_KEY (export ANTHROPIC_API_KEY=sk-ant-...)" >&2
            exit 1
        fi
        run generate python generate_synthetic.py
    fi
fi

# --- 3a. Intent model --------------------------------------------------------
archive_models() {  # keep models trained another way before they are overwritten
    local meta=models/intent_lstm_tuned/metrics.json
    [[ -f "$meta" ]] || return 0
    local current
    current=$(python -c "import json; print(json.load(open('$meta')).get('train_on', 'synthetic'))")
    [[ "$current" == "$INTENT_TRAIN_ON" ]] && return 0
    local dest="models/archive/$current"
    if [[ -e "$dest" ]]; then
        echo "==> $dest already exists; not archiving again"
        return 0
    fi
    echo "==> archiving the current (intent train_on=$current) models -> $dest/"
    mkdir -p "$dest"
    for d in intent_lstm_tuned sentiment_head action_dnn; do
        [[ -d "models/$d" ]] && cp -R "models/$d" "$dest/$d"
    done
}
# --- 2b. Shared synthetic split ----------------------------------------------
should_run split && run split python synthetic_splits.py

if should_run intent; then
    archive_models
    if [[ "$INTENT_PARAMS_FROM" != "none" ]]; then
        run intent python tune_intent_lstm.py --train-on "$INTENT_TRAIN_ON" --params-from "$INTENT_PARAMS_FROM"
    elif [[ $TRIALS -gt 0 && $TUNE_WORKERS -gt 1 ]]; then
        per_worker=$(( (TRIALS + TUNE_WORKERS - 1) / TUNE_WORKERS ))
        echo
        echo "==> [intent] $TUNE_WORKERS parallel workers x $per_worker trials"
        pids=()
        for w in $(seq 1 "$TUNE_WORKERS"); do
            python tune_intent_lstm.py --train-on "$INTENT_TRAIN_ON" --n-trials "$per_worker" --skip-final \
                > "logs/intent_worker$w.log" 2>&1 &
            pids+=($!)
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        if [[ $failed -ne 0 ]]; then
            echo "ERROR: a tuning worker failed; see logs/intent_worker*.log" >&2
            exit 1
        fi
        run intent python tune_intent_lstm.py --train-on "$INTENT_TRAIN_ON" --n-trials 0
    else
        run intent python tune_intent_lstm.py --train-on "$INTENT_TRAIN_ON" --n-trials "$TRIALS"
    fi
fi

# --- 3b-3d. Sentiment, policy reference, action DNN --------------------------
if should_run sentiment; then
    on_split=$(python -c "
import json, pathlib, synthetic_splits
m = pathlib.Path('models/sentiment_head/metrics.json')
print(int(m.exists() and json.loads(m.read_text()).get('split') == synthetic_splits.SCHEME))" 2>/dev/null || echo 0)
    if [[ $on_split -eq 1 && -f data/processed/healthcare_sentiment.csv && $RETRAIN_SENTIMENT -eq 0 ]]; then
        echo
        echo "==> [sentiment] skipped: the head is already trained on the shared split (--retrain-sentiment to retrain)"
    else
        run sentiment python train_sentiment_head.py
    fi
fi
should_run policy && run policy python action_policy.py
if should_run action; then
    run action python classify.py --train --oof-folds "$OOF_FOLDS" --text-input "$ACTION_TEXT_INPUT" \
        --hidden-layers "$ACTION_LAYERS" --hidden-units "$ACTION_UNITS"
fi

# --- 4. Evaluation -----------------------------------------------------------
if should_run evaluate; then
    run evaluate python classify.py --evaluate
    echo
    echo "==> Real test-set results (models/action_eval/metrics.json):"
    cat models/action_eval/metrics.json
fi

# --- 5. Reports (write-up tables and figures) --------------------------------
if should_run report; then
    if ! ls labels/real_test_sentiment_*.csv > /dev/null 2>&1; then
        echo
        echo "==> [report] no hand labels in labels/ - human-sentiment rows will be missing"
        echo "    (label with: python label_sentiment.py --annotator NAME)"
    fi
    run report python evaluation_report.py
    run report python hidden_layer_analysis.py
    echo
    echo "==> Real test-set headline (reports/summary_metrics.csv):"
    python - <<'PY'
import pandas as pd
s = pd.read_csv("reports/summary_metrics.csv")
s = s[s["data"].str.startswith("real") & (s["split"] == "test")]
cols = ["model", "data", "n", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]
print(s[cols].to_string(index=False))
PY
fi

# --- Optional: architecture experiments --------------------------------------
if [[ $RUN_EXPERIMENTS -eq 1 ]]; then
    run experiments python experiments.py
fi

echo
echo "==> Pipeline finished in $(( (SECONDS - T_START) / 60 ))m $(( (SECONDS - T_START) % 60 ))s. Logs in logs/."
