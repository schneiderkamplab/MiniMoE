# tokcleanse
tokenizer sanitizing

The installed entry point is `tokcleanse`.

Examples:

```bash
tokcleanse download google/gemma-4-E2B-it
tokcleanse sanitize google/gemma-4-E2B-it models/odin-id --overwrite
tokcleanse sanitize google/gemma-4-E2B-it models/odin-alpha --overwrite --reassign
tokcleanse compare models/odin-id models/odin-alpha --prompt "Who are you?"
tokcleanse compare models/google/gemma-4-E2B-it models/odin-id
tokcleanse compare models/google/gemma-4-E2B-it models/odin-id --prompt-file examples/prompts.jsonl
tokcleanse compare models/google/gemma-4-E2B-it models/odin-id --prompt-file examples/prompts.jsonl --device mps
tokcleanse compare models/google/gemma-4-E2B-it models/odin-id --prompt-file examples/prompts.jsonl --batch-size 8
tokcleanse compare models/google/gemma-4-E2B-it models/odin-id --prompt-file examples/prompts.jsonl --output-file compare.jsonl
tokcleanse compare models/google/gemma-4-E2B-it models/odin-id --llm-judge google/gemma-4-E2B-it
tokcleanse compare models/google/gemma-4-E2B-it models/odin-id --llm-judge google/gemma-4-E2B-it --judge-batch-size 4
tokcleanse compare models/google/gemma-4-E2B-it models/odin-id --llm-judge google/gemma-4-E2B-it --judge-all
```

`tokcleanse download HF_ID` downloads the snapshot into `models/HF_ID`.

For `compare`, if `--prompt-file` is provided then repeated `--prompt` values are ignored. Prompt comparisons show `tqdm` progress bars during generation and scoring, report exact and semantic-close match statistics, print the resolved runtime device before model loading starts, and support `--device auto|cpu|cuda|mps` with `auto` as the default. Main prompt generation for models `a` and `b` is batched with `--batch-size` (default `8`). `--llm-judge MODEL` judges only unresolved mismatches by default; add `--judge-all` to force a judge decision for every prompt. Judge generations are batched with `--judge-batch-size`, which defaults to `--batch-size`. With `--output-file`, all per-prompt JSON rows are written to that JSONL file, stdout keeps only mismatch rows plus the summary.
