#!/bin/bash


# cd /work/training/dfm-evals
# export OPENAI_API_KEY=dummy
# export OPENAI_BASE_URL=http://localhost:8000/v1

# EVALS=(
#  "dfm_evals/tasks/dala.py@dala"
#   "dfm_evals/tasks/danish_citizen_tests2.py@danish-citizen-tests"
#   "dfm_evals/tasks/gec_dala.py@gec_dala"
#   "dfm_evals/tasks/ifeval_da.py@ifeval-da"
#   "dfm_evals/tasks/multi_wiki_qa.py@multi_wiki_qa"
#    "dfm_evals/tasks/piqa.py@piqa"
# )

# python -m inspect_ai eval dfm_evals/tasks/dala.py@dala \
#   --model openai/my-gemma \
#   --log-dir /work/training/minimoe/logs


# #  python -m inspect_ai log dump /work/training/minimoe/logs/baseline_data.eval | grep -A 20 '"metrics"'



MINIMOE_DIR="/work/training/minimoe"
LOGS_DIR="$MINIMOE_DIR/logs"
RESULTS_DIR="$MINIMOE_DIR/results"
EVALS_DIR="/work/training/dfm-evals"

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8000/v1
export INSPECT_LOG_DIR=$LOGS_DIR
export HF_TOKEN="your_token_here"

mkdir -p $LOGS_DIR
mkdir -p $RESULTS_DIR

EVALS=(
  # "dfm_evals/tasks/dala.py@dala"
  # "dfm_evals/tasks/danish_citizen_tests2.py@danish-citizen-tests"
  # "dfm_evals/tasks/gec_dala.py@gec_dala"
  # "dfm_evals/tasks/ifeval_da.py@ifeval-da"
  # "dfm_evals/tasks/multi_wiki_qa.py@multi_wiki_qa"
  # "dfm_evals/tasks/piqa.py@piqa"
  "/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/mmlu_pro/mmlu_pro.py@mmlu_pro"
  "/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/mmlu/mmlu.py@mmlu_0_shot"
)

# # Kill any existing server
# pkill -f "serve_model.py" 2>/dev/null || true
# sleep 2

# # Start Gemma server
# python $MINIMOE_DIR/serve_model.py \
#   > $LOGS_DIR/server_baseline.log 2>&1 &
# SERVER_PID=$!

# Poll until server is up
# echo "Waiting for server to start..."
# SERVER_READY=false
# for i in $(seq 1 120); do
#   RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model": "my-gemma", "messages": [{"role": "user", "content": "ja?"}], "max_tokens": 8}')

#   if [ "$RESPONSE" == "200" ]; then
#     echo "Server is up after ${i} seconds!"
#     SERVER_READY=true
#     break
#   fi

#   if ! kill -0 $SERVER_PID 2>/dev/null; then
#     echo "ERROR: Server process died. Check server_baseline.log"
#     cat $LOGS_DIR/server_baseline.log
#     exit 1
#   fi

#   echo "Waiting... (${i}/120s)"
#   sleep 1
# done

# if [ "$SERVER_READY" != "true" ]; then
#   echo "ERROR: Server did not start, aborting."
#   exit 1
# fi

cd $EVALS_DIR



for EVAL in "${EVALS[@]}"; do
  DATASET=$(echo $EVAL | sed 's/.*@//')
  EVAL_NAME="baseline_${DATASET}"

  echo "--- Running eval: $DATASET ---"

  # Kill any existing server
  pkill -f "serve_model.py" 2>/dev/null || true
  sleep 2

  # Start server with log per dataset
  python $MINIMOE_DIR/serve_model.py \
    > $LOGS_DIR/server/baseline_${DATASET}.log 2>&1 &
  SERVER_PID=$!

  # Poll until server is up
  SERVER_READY=false
  for i in $(seq 1 120); do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "my-gemma", "messages": [{"role": "user", "content": "ja?"}], "max_tokens": 8}')

    if [ "$RESPONSE" == "200" ]; then
      echo "Server is up after ${i} seconds!"
      SERVER_READY=true
      break
    fi

    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "ERROR: Server process died. Check server_baseline_${DATASET}.log"
      cat $LOGS_DIR/server/baseline_${DATASET}.log
      break
    fi

    echo "Waiting... (${i}/120s)"
    sleep 1
  done

  if [ "$SERVER_READY" != "true" ]; then
    echo "ERROR: Server did not start for $DATASET, skipping."
    kill $SERVER_PID 2>/dev/null || true
    continue
  fi

  if [[ "$EVAL" == /* ]]; then
    python -m inspect_ai eval $EVAL \
      --model openai/my-gemma \
      --log-dir $LOGS_DIR
  else
    python -m inspect_ai eval $EVALS_DIR/$EVAL \
      --model openai/my-gemma \
      --log-dir $LOGS_DIR
  fi

  LATEST=$(ls -t $LOGS_DIR/*.eval | head -1)
  mv "$LATEST" $LOGS_DIR/$EVAL_NAME.eval

  python $MINIMOE_DIR/extract_metrics.py \
    $LOGS_DIR/$EVAL_NAME.eval \
    $RESULTS_DIR/evals.json

  kill $SERVER_PID 2>/dev/null || true
  sleep 2

  echo "--- Done: $DATASET ---"
done

kill $SERVER_PID 2>/dev/null || true

echo "All done! Results in $RESULTS_DIR/evals.json"