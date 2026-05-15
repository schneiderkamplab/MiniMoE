#!/bin/bash


pkill -f "serve_model_checkpoint.py"
sleep 2

CHECKPOINT_NUM=$1
EVAL_NAME="minimoe${CHECKPOINT_NUM}_dala"
CHECKPOINT_PATH="/work/training/minimoe/checkpoints/odin-moe-${CHECKPOINT_NUM}"

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8000/v1

LOGS_DIR="/work/training/minimoe/logs"
RESULTS_DIR="/work/training/minimoe/results"
EVALS_DIR="/work/training/dfm-evals"
MINIMOE_DIR="/work/training/minimoe"

python $MINIMOE_DIR/serve_model_checkpoint.py --checkpoint $CHECKPOINT_PATH \
  > $MINIMOE_DIR/logs/server.log 2>&1 &

# Wait for server to be ready
echo "Waiting for server to start..."
sleep 30

python -m inspect_ai eval $EVALS_DIR/dfm_evals/tasks/dala.py@dala \
  --model openai/my-custom-model \
  --log-dir $LOGS_DIR

LATEST=$(ls -t $LOGS_DIR/*.eval | head -1)
mv "$LATEST" $LOGS_DIR/$EVAL_NAME.eval

echo "Files in logs dir:"
ls -t $LOGS_DIR/*.eval | head -5
echo "Latest: $LATEST"

python $MINIMOE_DIR/extract_metrics.py \
  $LOGS_DIR/$EVAL_NAME.eval \
  $RESULTS_DIR/evals.json

