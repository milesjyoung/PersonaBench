#!/usr/bin/env bash
# Run a single PersonaBench pipeline step for one persona.
#
# Each invocation is a separate OS subprocess (this script forks Python which
# forks `claude -p` per LLM call). Each `claude -p` call has --tools "" so the
# LLM has no filesystem, shell, or network access during inference. This is
# the benchmark's structural isolation guarantee.
#
# Usage:
#   bash run_step.sh <step> <persona>
#   bash run_step.sh 4 julio_simmons
#   bash run_step.sh 6 alicia_gonzalez
#
# Steps:
#   2   Interview and verification
#   3   Social circle and verification
#   4   App log synthesis (per-fact reverse-inferability gate)
#   5   Test case synthesis and verification
#   6   Benchmark runner (two-pass, ground-truth isolated)
#
# This script is a thin bash wrapper around openclaw_pipeline.py. The Python
# orchestrator handles file I/O, prompt template filling, the per-fact loop in
# Step 4, and the iterative refinement loop in Step 5. Every LLM call inside
# the orchestrator is a fresh `claude -p --tools ""` subprocess.

set -euo pipefail

STEP="${1:-}"
PERSONA="${2:-}"

if [[ -z "$STEP" || -z "$PERSONA" ]]; then
  echo "usage: bash run_step.sh <step> <persona>" >&2
  echo "example: bash run_step.sh 4 julio_simmons" >&2
  exit 2
fi

case "$STEP" in
  2|3|4|5|6) ;;
  *) echo "error: step must be 2, 3, 4, 5, or 6 (got '${STEP}')" >&2; exit 2 ;;
esac

cd "$(dirname "$0")"

# Pinned model strings. Edit run_pipeline.sh's CONFIG to change in one place.
GENERATOR_MODEL="${GENERATOR_MODEL:-claude-opus-4-7}"
VERIFIER_MODEL="${VERIFIER_MODEL:-claude-sonnet-4-6}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-6}"
LOG_START="${LOG_START:-2026-02-20}"
LOG_END="${LOG_END:-2026-04-20}"
MAX_FACT_ATTEMPTS="${MAX_FACT_ATTEMPTS:-3}"

python openclaw_pipeline.py \
  --persona "${PERSONA}" \
  --start "${STEP}" \
  --stop "${STEP}" \
  --model "${GENERATOR_MODEL}" \
  --verifier-model "${VERIFIER_MODEL}" \
  --judge-model "${JUDGE_MODEL}" \
  --log-start "${LOG_START}" \
  --log-end "${LOG_END}" \
  --per-fact-max-attempts "${MAX_FACT_ATTEMPTS}"
