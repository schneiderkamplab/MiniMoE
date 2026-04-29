#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}"

tokcleanse sanitize google/gemma-4-E2B-it models/odin-danish --overwrite --reassign --token-map examples/danish_tokens.json --special-token-map examples/html_tokens.json "$@"
