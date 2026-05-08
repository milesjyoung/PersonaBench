#!/usr/bin/env bash
# Run a single pipeline step for one persona.
#
# Usage:
#   bash run_step.sh <step> <persona> [provider] [model]
#
# Examples:
#   bash run_step.sh 6 julio_simmons openai gpt-5.5
#   bash run_step.sh 4 maria_buendia anthropic claude-opus-4-7
#   bash run_step.sh 6 julio_simmons              # defaults: anthropic, claude-opus-4-7

set -euo pipefail

STEP="${1:-}"
PERSONA="${2:-}"
PROVIDER="${3:-anthropic}"
MODEL="${4:-claude-opus-4-7}"

if [[ -z "$STEP" || -z "$PERSONA" ]]; then
  echo "usage: bash run_step.sh <step> <persona> [provider] [model]" >&2
  echo "example: bash run_step.sh 6 julio_simmons openai gpt-5.5" >&2
  exit 2
fi

cd "$(dirname "$0")"

python openclaw_pipeline.py \
  --persona "${PERSONA}" \
  --start "${STEP}" \
  --stop "${STEP}" \
  --provider "${PROVIDER}" \
  --model "${MODEL}" \
  --verifier-model "${MODEL}"
