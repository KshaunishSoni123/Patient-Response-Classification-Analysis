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
#   intent     3a       tune_intent_lstm.py (Optuna search + final retrain)
#   sentiment  3b       train_sentiment_head.py
#   policy     3c       action_policy.py
#   action     3d       classify.py --train
#   evaluate   4        classify.py --evaluate
#
# Options:
#   --from STEP          start at STEP (earlier outputs must already exist)
#   --trials N           Optuna trials for the intent model (default 30; 0 = retrain the
#                        best params of the existing study only). Trials ADD to the study.
#   --tune-workers W     run the intent search in W parallel processes (default 1)
#   --oof-folds K        out-of-fold folds for the action DNN (default 5; 0 = fast/in-sample)
#   --generate           run synthetic generation even if the CSV already exists
#
# Each step's output is also written to logs/<step>.log.
set -euo pipefail
cd "$(dirname "$0")"

STEPS=(ingest generate intent sentiment policy action evaluate)
FROM=ingest
TRIALS=30
TUNE_WORKERS=1
OOF_FOLDS=5
FORCE_GENERATE=0

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --tune-workers) TUNE_WORKERS="$2"; shift 2 ;;
        --oof-folds) OOF_FOLDS="$2"; shift 2 ;;
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
if should_run intent; then
    if [[ $TRIALS -gt 0 && $TUNE_WORKERS -gt 1 ]]; then
        per_worker=$(( (TRIALS + TUNE_WORKERS - 1) / TUNE_WORKERS ))
        echo
        echo "==> [intent] $TUNE_WORKERS parallel workers x $per_worker trials"
        pids=()
        for w in $(seq 1 "$TUNE_WORKERS"); do
            python tune_intent_lstm.py --n-trials "$per_worker" --skip-final > "logs/intent_worker$w.log" 2>&1 &
            pids+=($!)
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        if [[ $failed -ne 0 ]]; then
            echo "ERROR: a tuning worker failed; see logs/intent_worker*.log" >&2
            exit 1
        fi
        run intent python tune_intent_lstm.py --n-trials 0
    else
        run intent python tune_intent_lstm.py --n-trials "$TRIALS"
    fi
fi

# --- 3b-3d. Sentiment, policy reference, action DNN --------------------------
should_run sentiment && run sentiment python train_sentiment_head.py
should_run policy && run policy python action_policy.py
should_run action && run action python classify.py --train --oof-folds "$OOF_FOLDS"

# --- 4. Evaluation -----------------------------------------------------------
if should_run evaluate; then
    run evaluate python classify.py --evaluate
    echo
    echo "==> Real test-set results (models/action_eval/metrics.json):"
    cat models/action_eval/metrics.json
fi

echo
echo "==> Pipeline finished in $(( (SECONDS - T_START) / 60 ))m $(( (SECONDS - T_START) % 60 ))s. Logs in logs/."
