#!/usr/bin/env bash
set -euo pipefail

tokcleanse sanitize models/odin-id models/odin-danish --overwrite --reassign --token-map examples/danish_tokens.json --special-token-map examples/html_tokens.json "$@"
