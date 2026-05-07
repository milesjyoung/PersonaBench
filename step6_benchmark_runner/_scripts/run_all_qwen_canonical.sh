#!/usr/bin/env bash
# Canonical Qwen3.5:9b run for week-1 deliverable:
#   - full 262K context (no summarization of meaningful or filler)
#   - CoT scaffold (prompt_cot_v2.txt) for Type 5
#   - 3-judge majority Pass 2 (Haiku + Sonnet + Opus)
# Waits for qwen3.5:9b to be available before firing.
set -e

cd "$(dirname "$0")/../.."

# Block until the model is pulled and listed.
until ollama list | grep -q "qwen3.5:9b"; do
  echo "[wait] qwen3.5:9b not pulled yet, sleeping 30s..."
  sleep 30
done
echo "[ready] qwen3.5:9b is pulled."

PERSONAS=(julio_simmons mary_alberti alicia_gonzalez deeva_cintron maria_buendia)

for P in "${PERSONAS[@]}"; do
  echo ""
  echo "=================================================="
  echo "=== $P ==="
  echo "=================================================="
  python -m step6_benchmark_runner._scripts.run_small_model \
    --persona "$P" \
    --provider ollama \
    --model qwen3.5:9b \
    --truncation meaningful_preserved \
    --log-budget 110000 \
    --cot \
    --judges haiku,sonnet,opus \
    --batch-size 3 \
    --num-ctx 131072
done

echo ""
echo "[done] all 5 personas, qwen3.5:9b + CoT + 3-judge majority"
