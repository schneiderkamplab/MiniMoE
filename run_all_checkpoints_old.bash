#!/bin/bash
#set -e
MINIMOE_DIR="/work/training/minimoe"
LOGS_DIR="$MINIMOE_DIR/logs"
RESULTS_DIR="$MINIMOE_DIR/results"
EVALS_DIR="/work/training/dfm-evals"

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8000/v1
export INSPECT_LOG_DIR=$LOGS_DIR

mkdir -p $LOGS_DIR
mkdir -p $RESULTS_DIR

# Define evals as "filename@taskname" pairs
EVALS=(
#  "dfm_evals/tasks/dala.py@dala"
  #"dfm_evals/tasks/danish_citizen_tests2.py@danish-citizen-tests"
  # "dfm_evals/tasks/gec_dala.py@gec_dala"
  # "dfm_evals/tasks/ifeval_da.py@ifeval-da"
  # "dfm_evals/tasks/multi_wiki_qa.py@multi_wiki_qa"
   "dfm_evals/tasks/piqa.py@piqa"
)

if [ -z "$1" ]; then
  CHECKPOINTS="0" # ADD 1 2 3
else
  CHECKPOINTS="$1"
fi

for CHECKPOINT_NUM in $CHECKPOINTS; do
  CHECKPOINT_PATH="$MINIMOE_DIR/checkpoints/odin-moe-${CHECKPOINT_NUM}"

  echo "=== Evaluating checkpoint $CHECKPOINT_NUM ==="

  # pkill -f "serve_model_checkpoint.py" 2>/dev/null
  # sleep 2

  # python $MINIMOE_DIR/serve_model_checkpoint.py --checkpoint $CHECKPOINT_PATH \
  #   >> $MINIMOE_DIR/logs/server_${CHECKPOINT_PATH##*/}.log 2>&1 &
  # SERVER_PID=$!

  # echo "Waiting for server to start..."
  # # Replace the fixed sleep with a polling loop
  #   echo "Waiting for server to start..."
  #   for i in $(seq 1 60); do
  #   RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/chat/completions \
  #       -H "Content-Type: application/json" \
  #       -d '{"model": "my-custom-model", "messages": [{"role": "user", "content": "ja?"}], "max_tokens": 8}')
    
  #   if [ "$RESPONSE" == "200" ]; then
  #       echo "Server is up after ${i} seconds!"
  #       break
  #   fi

  #   if ! kill -0 $SERVER_PID 2>/dev/null; then
  #       echo "ERROR: Server process died. Check logs/server.log"
  #       cat $MINIMOE_DIR/logs/server.log
  #       exit 1
  #   fi

  #   echo "Waiting... (${i}/60s)"
  #   sleep 1
  #   done

  #   if [ "$RESPONSE" != "200" ]; then
  #   echo "ERROR: Server did not start within 60 seconds"
  #   cat $MINIMOE_DIR/logs/server.log
  #   exit 1
  #   fi

  # if ! kill -0 $SERVER_PID 2>/dev/null; then
  #   echo "ERROR: Server failed to start for checkpoint $CHECKPOINT_NUM"
  #   cat $MINIMOE_DIR/logs/server.log
  #   continue
  # fi

  cd /work/training/dfm-evals

  for EVAL in "${EVALS[@]}"; do
    # Extract dataset name from eval string (e.g. "dala" from "tasks/dala.py@dala")
    DATASET=$(echo $EVAL | sed 's/.*@//')
    EVAL_NAME="minimoe${CHECKPOINT_NUM}_${DATASET}"

    echo "--- Running eval: $DATASET ---"

     # Kill any existing server
    pkill -f "serve_model_checkpoint.py" 2>/dev/null || true
    sleep 2

    # Start server with log per eval
    python $MINIMOE_DIR/serve_model_checkpoint.py --checkpoint $CHECKPOINT_PATH \
      > $LOGS_DIR/server_${EVAL_NAME}.log 2>&1 &
    SERVER_PID=$!

    # Poll until server is up
    SERVER_READY=false
    for i in $(seq 1 120); do
      RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "my-custom-model", "messages": [{"role": "user", "content": "ja?"}], "max_tokens": 8}')

      if [ "$RESPONSE" == "200" ]; then
        echo "Server is up after ${i} seconds!"
        SERVER_READY=true
        break
      fi

      if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died. Check server_${EVAL_NAME}.log"
        cat $LOGS_DIR/server_${EVAL_NAME}.log
        break
      fi

      sleep 1
    done

    if [ "$SERVER_READY" != "true" ]; then
      echo "ERROR: Server did not start for $EVAL_NAME, skipping."
      kill $SERVER_PID 2>/dev/null || true
      continue
    fi

    python -m inspect_ai eval $EVALS_DIR/$EVAL \
      --model openai/my-custom-model \
      --log-dir $LOGS_DIR
    

    LATEST=$(ls -t $LOGS_DIR/*.eval | head -1)
    mv "$LATEST" $LOGS_DIR/$EVAL_NAME.eval

    python $MINIMOE_DIR/extract_metrics.py \
      $LOGS_DIR/$EVAL_NAME.eval \
      $RESULTS_DIR/evals.json
  done

  kill $SERVER_PID 2>/dev/null
  sleep 2

  echo "=== Done with checkpoint $CHECKPOINT_NUM ==="
done

echo "All done! Results in $RESULTS_DIR/evals.json"