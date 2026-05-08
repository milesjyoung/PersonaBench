#!/usr/bin/env bash
# Run all pipeline steps (2 through 6) for one persona.
#
# Usage:
#   bash run_pipeline.sh <persona> [provider] [model]
#
# Examples:
#   bash run_pipeline.sh julio_simmons openai gpt-5.5
#   bash run_pipeline.sh julio_simmons anthropic claude-opus-4-7
#   bash run_pipeline.sh julio_simmons            # defaults: anthropic, claude-opus-4-7

set -euo pipefail

PERSONA="${1:-}"
PROVIDER="${2:-anthropic}"
MODEL="${3:-claude-opus-4-7}"

if [[ -z "$PERSONA" ]]; then
  echo "usage: bash run_pipeline.sh <persona> [provider] [model]" >&2
  echo "example: bash run_pipeline.sh julio_simmons openai gpt-5.5" >&2
  exit 2
fi

cd "$(dirname "$0")"

for STEP in 2 3 4 5 6; do
  echo "--- Step ${STEP} ---"
  bash run_step.sh "${STEP}" "${PERSONA}" "${PROVIDER}" "${MODEL}"
done

echo "=== Pipeline complete for ${PERSONA} ==="
