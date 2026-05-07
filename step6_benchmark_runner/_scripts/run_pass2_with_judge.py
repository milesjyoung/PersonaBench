"""Rerun Pass 2 only against an existing pass1_answers.json with an explicit
judge model. Used for the dual-judge experiment: same Pass 1 answers, scored
by Sonnet (canonical) and by Opus, reported as a band.

Usage:
  python -m step6_benchmark_runner._scripts.run_pass2_with_judge \\
    --persona julio_simmons --judge-model opus

Outputs to step6_benchmark_runner/data_samples/output_dual_judge/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _cli_helpers import (
    call_claude_cli,
    extract_json,
    fill,
    load_prompt_sections,
)

REPO = Path(__file__).resolve().parents[2]
STEP6 = REPO / "step6_benchmark_runner"
PROMPT = STEP6 / "prompt.txt"
OUT_DIR = STEP6 / "data_samples" / "output_dual_judge"
TEST_CASES_DIR = REPO / "step5_testcases_synthesis" / "data_samples" / "output"
PASS1_DIR = STEP6 / "data_samples" / "output"


def run(persona: str, judge_model: str) -> None:
    test_cases_path = TEST_CASES_DIR / f"{persona}_test_cases.json"
    pass1_path = PASS1_DIR / f"{persona}_pass1_answers.json"
    if not test_cases_path.exists():
        raise FileNotFoundError(test_cases_path)
    if not pass1_path.exists():
        raise FileNotFoundError(pass1_path)

    test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))
    pass1 = json.loads(pass1_path.read_text(encoding="utf-8"))

    sections = load_prompt_sections(PROMPT)
    prompt = fill(sections["pass2"], "{{INSERT_FULL_TEST_CASES_JSON_HERE}}", test_cases)
    prompt = fill(prompt, "{{INSERT_PASS_1_ANSWERS_JSON_HERE}}", pass1)

    print(f"[pass2/{persona}] judge={judge_model}; calling claude CLI...")
    raw = call_claude_cli(prompt, model=judge_model)
    result = extract_json(raw)

    result.setdefault("metadata", {})
    result["metadata"]["pass_2_date"] = date.today().isoformat()
    result["metadata"]["model_used_pass2"] = judge_model
    result["metadata"]["pass_1_source"] = str(pass1_path.relative_to(REPO))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = judge_model.split("-")[1] if "-" in judge_model else judge_model
    out_path = OUT_DIR / f"{persona}_benchmark_results_{suffix}_judge.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] overall={result.get('overall_accuracy')} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persona", required=True, help="e.g. julio_simmons")
    ap.add_argument("--judge-model", required=True, help="opus | sonnet | full model id")
    args = ap.parse_args()
    run(args.persona, args.judge_model)


if __name__ == "__main__":
    main()
