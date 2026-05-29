#!/usr/bin/env bash
# Run a single PersonaBench pipeline step for one persona.
#
# Each invocation is a separate OS subprocess (this script forks Python which
# forks the selected model CLI per LLM call). The default backend is `claude`;
# pass `codex` as the third argument to use `codex exec`, or an API backend
# if you want metered SDK calls.
#
# Usage:
#   bash run_step.sh <step> <persona> [backend] [model]
#   bash run_step.sh 4 julio_simmons
#   bash run_step.sh 6 alicia_gonzalez
#   bash run_step.sh 6 julio_simmons codex gpt-5.4-mini
#   bash run_step.sh 6 julio_simmons anthropic-api claude-opus-4-7
#
# Steps:
#   2   Interview and verification
#   3   Social circle and verification
#   4   App log synthesis (cluster reverse-inferability gate)
#   5   Test case synthesis and verification
#   6   Benchmark runner (two-pass, ground-truth isolated)
#
# This script is a thin bash wrapper around openclaw_pipeline.py. The Python
# orchestrator handles file I/O, prompt template filling, the Step 4 cluster
# gate, and the iterative refinement loop in Step 5.

set -euo pipefail

STEP="${1:-}"
PERSONA="${2:-}"
BACKEND="${3:-${PERSONABENCH_BACKEND:-claude}}"
MODEL_ARG="${4:-}"

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
case "$BACKEND" in
  claude)
    DEFAULT_GENERATOR_MODEL="claude-opus-4-7"
    DEFAULT_VERIFIER_MODEL="claude-sonnet-4-6"
    DEFAULT_JUDGE_MODEL="claude-sonnet-4-6"
    ;;
  codex)
    DEFAULT_GENERATOR_MODEL="${MODEL_ARG:-gpt-5.5}"
    DEFAULT_VERIFIER_MODEL="gpt-5"
    DEFAULT_JUDGE_MODEL="gpt-5.4"
    ;;
  anthropic-api)
    DEFAULT_GENERATOR_MODEL="${MODEL_ARG:-claude-opus-4-7}"
    DEFAULT_VERIFIER_MODEL="claude-sonnet-4-6"
    DEFAULT_JUDGE_MODEL="claude-sonnet-4-6"
    ;;
  openai-api)
    DEFAULT_GENERATOR_MODEL="${MODEL_ARG:-gpt-5.5}"
    DEFAULT_VERIFIER_MODEL="gpt-5"
    DEFAULT_JUDGE_MODEL="gpt-5.4"
    ;;
  *) echo "error: backend must be claude, codex, anthropic-api, or openai-api (got '${BACKEND}')" >&2; exit 2 ;;
esac

# Step 6 uses a smaller evaluator model by default.
if [[ "$STEP" == "6" ]]; then
  case "$BACKEND" in
    codex|openai-api) DEFAULT_GENERATOR_MODEL="${MODEL_ARG:-gpt-5.4-mini}" ;;
  esac
fi

GENERATOR_MODEL="${GENERATOR_MODEL:-${MODEL_ARG:-$DEFAULT_GENERATOR_MODEL}}"
VERIFIER_MODEL="${VERIFIER_MODEL:-$DEFAULT_VERIFIER_MODEL}"
JUDGE_MODEL="${JUDGE_MODEL:-$DEFAULT_JUDGE_MODEL}"
LOG_START="${LOG_START:-2026-03-01}"
LOG_END="${LOG_END:-2026-03-31}"
MAX_CLUSTER_ATTEMPTS="${MAX_CLUSTER_ATTEMPTS:-${MAX_FACT_ATTEMPTS:-3}}"
MAX_FACTS="${MAX_FACTS:-100}"
FILLER_RATIO="${FILLER_RATIO:-2.5}"
DECOY_COUNT="${DECOY_COUNT:-0}"
DECOY_POOL_SIZE="${DECOY_POOL_SIZE:-40}"
VERIFIED_DECOYS="${VERIFIED_DECOYS:-}"

DECOY_ARGS=()
if [[ -n "$VERIFIED_DECOYS" ]]; then
  DECOY_ARGS+=(--verified-decoys "$VERIFIED_DECOYS")
fi
if [[ "$DECOY_COUNT" != "0" ]]; then
  DECOY_ARGS+=(--decoy-count "$DECOY_COUNT")
fi

python openclaw_pipeline.py \
  --persona "${PERSONA}" \
  --start "${STEP}" \
  --stop "${STEP}" \
  --backend "${BACKEND}" \
  --model "${GENERATOR_MODEL}" \
  --verifier-model "${VERIFIER_MODEL}" \
  --judge-model "${JUDGE_MODEL}" \
  --log-start "${LOG_START}" \
  --log-end "${LOG_END}" \
  --per-cluster-max-attempts "${MAX_CLUSTER_ATTEMPTS}" \
  --max-facts "${MAX_FACTS}" \
  --filler-ratio "${FILLER_RATIO}" \
  --decoy-pool-size "${DECOY_POOL_SIZE}" \
  "${DECOY_ARGS[@]}"
