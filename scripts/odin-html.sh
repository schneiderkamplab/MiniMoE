#!/usr/bin/env bash
set -euo pipefail

tokcleanse sanitize models/odin-id models/odin-html --overwrite --reassign --special-token-map-file examples/html_tokens.json
