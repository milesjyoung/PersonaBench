"""Run CoT-scaffolded Pass 1 + batched Pass 2 across all 5 personas.

Usage:
  python -m step6_benchmark_runner._scripts.run_cot_all \\
    --pass1-model opus --pass2-model sonnet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_cot_pass1 import run as run_one

PERSONAS = ["julio_simmons", "mary_alberti", "alicia_gonzalez", "deeva_cintron", "maria_buendia"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pass1-model", default="opus")
    ap.add_argument("--pass2-model", default="sonnet")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--type5-only", action="store_true", default=False)
    args = ap.parse_args()

    OUT = Path(__file__).resolve().parents[2] / "step6_benchmark_runner" / "data_samples" / "output_cot"
    suffix = "_type5" if args.type5_only else ""

    for p in PERSONAS:
        target = OUT / f"{p}_benchmark_results_cot{suffix}.json"
        if args.skip_existing and target.exists():
            print(f"[skip] {p} already done -> {target}", flush=True)
            continue
        print(f"\n=== {p} CoT pass1={args.pass1_model} pass2={args.pass2_model} type5={args.type5_only} ===", flush=True)
        try:
            run_one(p, args.pass1_model, args.pass2_model, args.type5_only)
        except Exception as e:
            print(f"[error/{p}] {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
