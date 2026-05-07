"""Run batched Opus-judge Pass 2 across all 5 personas, sequentially. Skips
any persona where the result file already exists.

Usage:
  python -m step6_benchmark_runner._scripts.run_dual_judge_all \\
    --judge-model opus --batch-size 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_pass2_batched import run as run_one

PERSONAS = ["julio_simmons", "mary_alberti", "alicia_gonzalez", "deeva_cintron", "maria_buendia"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge-model", default="opus")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    OUT = Path(__file__).resolve().parents[2] / "step6_benchmark_runner" / "data_samples" / "output_dual_judge"
    suffix = args.judge_model.split("-")[1] if "-" in args.judge_model else args.judge_model

    for p in PERSONAS:
        target = OUT / f"{p}_benchmark_results_{suffix}_judge.json"
        if args.skip_existing and target.exists():
            print(f"[skip] {p} already done -> {target}", flush=True)
            continue
        print(f"\n=== {p} judge={args.judge_model} ===", flush=True)
        try:
            run_one(p, args.judge_model, args.batch_size)
        except Exception as e:
            print(f"[error/{p}] {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
