#!/usr/bin/env bash
# Fire 5 personas in parallel, each running 3-judge Pass 2 sequentially.
# Total wall time = max(per-persona time) instead of 5 * per-persona.
set -e
cd "$(dirname "$0")/../.."

PERSONAS=(julio_simmons mary_alberti alicia_gonzalez deeva_cintron maria_buendia)
PIDS=()
LOG_DIR="step6_benchmark_runner/data_samples/output_three_judge/_logs"
mkdir -p "$LOG_DIR"

for P in "${PERSONAS[@]}"; do
  LOG="$LOG_DIR/${P}.log"
  echo "[fire] $P -> $LOG"
  python -m step6_benchmark_runner._scripts.run_three_judge_majority --persona "$P" \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

echo "[fire] launched ${#PIDS[@]} parallel persona runs: ${PIDS[*]}"
echo "[fire] waiting for all to complete..."
FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    echo "[fail] PID $PID exited non-zero"
    FAIL=1
  fi
done

if [ "$FAIL" -eq 0 ]; then
  echo "[done] all 5 personas done"
else
  echo "[done] some personas failed; check logs in $LOG_DIR"
fi
