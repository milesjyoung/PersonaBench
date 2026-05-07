"""Three-judge majority Pass 2 against existing Pass 1 answers.

Runs Haiku + Sonnet + Opus as Pass 2 judges against the canonical Pass 1
answers for every persona, then aggregates with majority vote (mode for
Types 1-4 with harshness tie-break, median-per-dimension for Type 5).

Outputs:
  step6_benchmark_runner/data_samples/output_three_judge/
    {persona}_benchmark_results_haiku.json
    {persona}_benchmark_results_sonnet.json
    {persona}_benchmark_results_opus.json
    {persona}_benchmark_results_majority.json

Usage:
  python -m step6_benchmark_runner._scripts.run_three_judge_majority --all
  python -m step6_benchmark_runner._scripts.run_three_judge_majority --persona julio_simmons
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_small_model import aggregate_majority, run_pass2_via_claude

REPO = Path(__file__).resolve().parents[2]
STEP6 = REPO / "step6_benchmark_runner"
CANONICAL_OUT = STEP6 / "data_samples" / "output"
THREE_JUDGE_OUT = STEP6 / "data_samples" / "output_three_judge"

JUDGES = ["haiku", "sonnet", "opus"]
PERSONAS = [
    "julio_simmons",
    "mary_alberti",
    "alicia_gonzalez",
    "deeva_cintron",
    "maria_buendia",
]


def run_one(persona: str) -> None:
    pass1_path = CANONICAL_OUT / f"{persona}_pass1_answers.json"
    if not pass1_path.exists():
        print(f"[skip] {persona}: no canonical Pass 1 answers at {pass1_path}", flush=True)
        return

    pass1 = json.loads(pass1_path.read_text(encoding="utf-8"))
    THREE_JUDGE_OUT.mkdir(parents=True, exist_ok=True)

    per_judge: dict[str, dict] = {}
    for j in JUDGES:
        print(f"[{persona}] judge={j}", flush=True)
        result = run_pass2_via_claude(persona, pass1, j)
        per_judge[j] = result
        out = THREE_JUDGE_OUT / f"{persona}_benchmark_results_{j}.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    majority = aggregate_majority(per_judge, JUDGES)
    out = THREE_JUDGE_OUT / f"{persona}_benchmark_results_majority.json"
    out.write_text(json.dumps(majority, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[{persona}] DONE majority={majority.get('overall_accuracy')} "
        f"(haiku={per_judge['haiku'].get('overall_accuracy')} "
        f"sonnet={per_judge['sonnet'].get('overall_accuracy')} "
        f"opus={per_judge['opus'].get('overall_accuracy')})",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persona", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for p in PERSONAS:
            run_one(p)
    elif args.persona:
        run_one(args.persona)
    else:
        ap.error("use --persona <name> or --all")


if __name__ == "__main__":
    main()
