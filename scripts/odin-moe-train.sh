#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}"

exp=$1
shift

default_args=()
has_batch_size=0
has_gradient_accumulation=0
has_max_length=0
has_gradient_checkpointing_flag=0

for arg in "$@"; do
  case "${arg}" in
    --batch-size)
      has_batch_size=1
      ;;
    --gradient-accumulation)
      has_gradient_accumulation=1
      ;;
    --max-length)
      has_max_length=1
      ;;
    --gradient-checkpointing|--no-gradient-checkpointing)
      has_gradient_checkpointing_flag=1
      ;;
  esac
done

if [[ ${has_batch_size} -eq 0 ]]; then
  default_args+=(--batch-size 4)
fi
if [[ ${has_gradient_accumulation} -eq 0 ]]; then
  default_args+=(--gradient-accumulation 2)
fi
if [[ ${has_max_length} -eq 0 ]]; then
  default_args+=(--max-length 256)
fi
if [[ ${has_gradient_checkpointing_flag} -eq 0 ]]; then
  default_args+=(--no-gradient-checkpointing)
fi

if [[ ${#default_args[@]} -gt 0 ]]; then
  set -- "${default_args[@]}" "$@"
fi

tokcleanse train models/odin-moe models/odin-danish models/odin-moe-trained-$1 \
  --ind-file data/just-cp-cp-0-of-7-train.jinx \
  --ood-file data/just-dyna-dyna-0-of-1-train.jinx \
  --eval-file data/just-dyna-dyna-0-of-1-test.jinx \
  "$@"
