#!/usr/bin/env bash
set -euo pipefail

tokcleanse sanitize google/gemma-4-E2B-it models/odin-alpha --overwrite --reassign "$@"
