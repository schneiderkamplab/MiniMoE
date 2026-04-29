#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}"

tokcleanse upcycle models/odin-danish models/odin-moe --overwrite --num-experts 2 --top-k-experts 2 --expert-init copy "$@"
