#!/usr/bin/env bash
# PersonaBench pipeline orchestrator.
#
# Walks Steps 2 through 6 for one persona. Each step is a fresh OS subprocess
# invocation of the selected model CLI. The orchestrator (Python,
# openclaw_pipeline.py) handles all I/O.
#
# `set -euo pipefail` ensures step N+1 only fires if step N exited 0. That is
# how "step n waits for step n-1" is enforced.
#
# Usage:
#   bash run_pipeline.sh <persona>
#   bash run_pipeline.sh julio_simmons
#   bash run_pipeline.sh julio_simmons codex gpt-5.5
#   bash run_pipeline.sh julio_simmons anthropic-api claude-opus-4-7
#
# To run a different model selection, edit the CONFIG block below or pass
# the corresponding flags through to openclaw_pipeline.py directly.
#
# Outputs: each step writes to its own data_samples/output/ directory.

set -euo pipefail

# ---- CONFIG ----------------------------------------------------------------
# Model strings are pinned to exact versions, never aliases. Aliases drift
# across CLI releases and break artifact reproducibility.
BACKEND="${2:-${PERSONABENCH_BACKEND:-claude}}"
MODEL_ARG="${3:-}"

case "$BACKEND" in
  claude)
    GENERATOR_MODEL="${GENERATOR_MODEL:-claude-opus-4-7}"
    VERIFIER_MODEL="${VERIFIER_MODEL:-claude-sonnet-4-6}"
    JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-6}"
    ;;
  codex)
    GENERATOR_MODEL="${GENERATOR_MODEL:-${MODEL_ARG:-gpt-5.5}}"
    VERIFIER_MODEL="${VERIFIER_MODEL:-gpt-5}"
    JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4}"
    ;;
  anthropic-api)
    GENERATOR_MODEL="${GENERATOR_MODEL:-${MODEL_ARG:-claude-opus-4-7}}"
    VERIFIER_MODEL="${VERIFIER_MODEL:-claude-sonnet-4-6}"
    JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-6}"
    ;;
  openai-api)
    GENERATOR_MODEL="${GENERATOR_MODEL:-${MODEL_ARG:-gpt-5.5}}"
    VERIFIER_MODEL="${VERIFIER_MODEL:-gpt-5}"
    JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4}"
    ;;
  *) echo "error: backend must be claude, codex, anthropic-api, or openai-api (got '${BACKEND}')" >&2; exit 2 ;;
esac

# Log window that Step 4 generates against. Pinned for reproducibility.
LOG_START="${LOG_START:-2026-03-01}"
LOG_END="${LOG_END:-2026-03-31}"

# Per-fact retry budget for Step 4's reverse-inferability gate.
MAX_FACT_ATTEMPTS=3
# ----------------------------------------------------------------------------

PERSONA="${1:-}"
if [[ -z "$PERSONA" ]]; then
  echo "usage: bash run_pipeline.sh <persona>" >&2
  echo "example: bash run_pipeline.sh julio_simmons" >&2
  echo "" >&2
  echo "available personas:" >&2
  for f in step1_seed/data_samples/output/*_seed.json; do
    echo "  $(basename "$f" _seed.json)" >&2
  done
  exit 2
fi

cd "$(dirname "$0")"

if [[ ! -f "step1_seed/data_samples/output/${PERSONA}_seed.json" ]]; then
  echo "error: persona '${PERSONA}' has no seed at step1_seed/data_samples/output/${PERSONA}_seed.json" >&2
  exit 2
fi

echo "=== PersonaBench pipeline ==="
echo "persona:         ${PERSONA}"
echo "backend:         ${BACKEND}"
echo "generator_model: ${GENERATOR_MODEL}"
echo "verifier_model:  ${VERIFIER_MODEL}"
echo "judge_model:     ${JUDGE_MODEL}"
echo "log_window:      ${LOG_START} to ${LOG_END}"
echo "started:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

for STEP in 2 3 4 5 6; do
  echo "--- Step ${STEP} ---"
  PERSONABENCH_BACKEND="${BACKEND}" \
  GENERATOR_MODEL="${GENERATOR_MODEL}" \
  VERIFIER_MODEL="${VERIFIER_MODEL}" \
  JUDGE_MODEL="${JUDGE_MODEL}" \
  bash run_step.sh "${STEP}" "${PERSONA}"
done

echo ""
echo "=== Pipeline complete for ${PERSONA} ==="
echo "ended: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
