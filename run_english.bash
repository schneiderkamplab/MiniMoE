
MINIMOE_DIR="/work/training/MiniMoE"
LOGS_DIR="$MINIMOE_DIR/logs_english"
RESULTS_DIR="$MINIMOE_DIR/results"
EVALS_DIR="/work/training/dfm-evals"

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8000/v1
export INSPECT_LOG_DIR=$LOGS_DIR
#export HF_TOKEN="your_token_here"

mkdir -p $LOGS_DIR
mkdir -p $RESULTS_DIR

EVALS=(
   #"dfm_evals/tasks/dala.py@dala"
   #"dfm_evals/tasks/danish_citizen_tests2.py@danish-citizen-tests"
   #"dfm_evals/tasks/gec_dala.py@gec_dala"
   #"dfm_evals/tasks/ifeval_da.py@ifeval-da"
   #"dfm_evals/tasks/multi_wiki_qa.py@multi_wiki_qa"
   #"dfm_evals/tasks/piqa.py@piqa"
  # "/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/mmlu_pro/mmlu_pro.py@mmlu_pro"
  # "/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/mmlu/mmlu.py@mmlu_0_shot"
  #"ruler_task.py@ruler"
  #"talemaader_task.py@generative-talemaader"
  #"dfm_evals/tasks/wmt24pp.py@wmt24pp-en-da"
  #"dfm_evals/tasks/daisy.py@daisy"
  #"/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/aime2026/aime2026.py@aime2026"
  "/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/gpqa/gpqa.py@gpqa_diamond"
  "/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/tau2/tau2.py@tau2_retail"
  "/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/tau2/tau2.py@tau2_airline"
  "/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/tau2/tau2.py@tau2_telecom"
  #"/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/hle/hle.py@hle" # i get this error: UnprocessableEntityError("Error code: 422 - {'detail': [{'type': 'string_type', 'loc': ['body', 'messages', 1, 'content'], 'msg': 'Input should be a valid string', 'input': [{'type': 'text', 'text': 'Calculate a left coprime factorization of the following transfer function:\\n\\\\[\\nH(s) = \\\\begin{bmatrix} \\\\frac{s-1}{s+1} & 1 \\\\\\\\ \\\\frac{2}{s^2-1} & 0 \\\\end{bmatrix}\\n\\\\]\\nUse the following notation:\\n\\\\[\\nH(s) = D^{-1}(s) N(s)\\n\\\\]'}]}]}")     
  "/home/ucloud/miniconda3/envs/minimoe/lib/python3.14/site-packages/inspect_evals/bbeh/bbeh.py@bbeh"

)


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

#   python -m inspect_ai eval $EVAL \
#       --model openai/my-gemma \
#       --log-dir $LOGS_DIR

  if [[ "$EVAL" == /* ]]; then
    python -m inspect_ai eval $EVAL \
      --model openai/my-gemma \
      --log-dir $LOGS_DIR
  else
    python -m inspect_ai eval $EVALS_DIR/$EVAL \
      --model openai/my-custom-model \
      --max-connections 8 \
      --max-samples 8 \
      --log-dir $LOGS_DIR
  fi

  LATEST=$(ls -t $LOGS_DIR/*.eval | head -1)
  mv "$LATEST" $LOGS_DIR/$EVAL_NAME.eval

  python $MINIMOE_DIR/extract_metrics.py \
    $LOGS_DIR/$EVAL_NAME.eval \
    $RESULTS_DIR/evals_english.json

  kill $SERVER_PID 2>/dev/null || true
  sleep 2

  echo "--- Done: $DATASET ---"
done

kill $SERVER_PID 2>/dev/null || true

echo "All done! Results in $RESULTS_DIR/evals_english.json"