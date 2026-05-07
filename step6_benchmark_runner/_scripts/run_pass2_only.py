"""Run batched Pass 2 against an arbitrary Pass 1 answers file.

Used when Pass 1 already exists on disk (e.g., from a prior small-model run
or a CoT run) and we just need scoring. Avoids redoing expensive Pass 1.

Usage:
  python -m step6_benchmark_runner._scripts.run_pass2_only \\
    --persona julio_simmons \\
    --pass1 step6_benchmark_runner/data_samples/output_small_models/julio_simmons_pass1_gpt4omini.json \\
    --judge-model sonnet \\
    --output step6_benchmark_runner/data_samples/output_small_models/julio_simmons_benchmark_results_gpt4omini.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _cli_helpers import (
    call_and_parse_json,
    call_claude_cli,
    extract_json,
    fill,
    load_prompt_sections,
)
from run_pass2_batched import aggregate as aggregate_pass2, chunk as chunk_cases

REPO = Path(__file__).resolve().parents[2]
STEP6 = REPO / "step6_benchmark_runner"
PROMPT = STEP6 / "prompt.txt"
TC_DIR = REPO / "step5_testcases_synthesis" / "data_samples" / "output"


def run(persona: str, pass1_path: Path, judge_model: str, batch_size: int, out_path: Path) -> None:
    test_cases = json.loads((TC_DIR / f"{persona}_test_cases.json").read_text(encoding="utf-8"))
    pass1 = json.loads(pass1_path.read_text(encoding="utf-8"))
    sections = load_prompt_sections(PROMPT)

    cases = test_cases["test_cases"]
    answers_by_id = {a["test_case_id"]: a for a in pass1.get("answers", [])}
    persona_name = test_cases.get("metadata", {}).get("persona", persona)
    participant_id = test_cases.get("metadata", {}).get("participant_id", "")
    pass1_model = pass1.get("metadata", {}).get("model_used", "unknown")

    all_evals = []
    batches = chunk_cases(cases, batch_size)
    for bi, batch in enumerate(batches, start=1):
        batch_payload = {"metadata": test_cases.get("metadata", {}), "test_cases": batch}
        batch_pass1 = {
            "metadata": pass1.get("metadata", {}),
            "answers": [answers_by_id[c["id"]] for c in batch if c["id"] in answers_by_id],
        }
        prompt = fill(sections["pass2"], "{{INSERT_FULL_TEST_CASES_JSON_HERE}}", batch_payload)
        prompt = fill(prompt, "{{INSERT_PASS_1_ANSWERS_JSON_HERE}}", batch_pass1)
        prompt += (
            f"\n\nIMPORTANT: This batch contains {len(batch)} cases "
            f"({batch[0]['id']}..{batch[-1]['id']}). Score every case in this "
            f"batch. Return JSON with an `evaluations` array containing exactly "
            f"{len(batch)} entries."
        )
        print(f"[pass2/{persona}] batch {bi}/{len(batches)} judge={judge_model}", flush=True)
        parsed = call_and_parse_json(prompt, model=judge_model, timeout_sec=1500, max_retries=3)
        evals = parsed.get("evaluations", [])
        case_meta = {c["id"]: c for c in batch}
        for ev in evals:
            tc = case_meta.get(ev.get("test_case_id"), {})
            if "subtype" not in ev and tc.get("subtype"):
                ev["subtype"] = tc["subtype"]
            if "is_capstone" not in ev and tc.get("is_capstone") is not None:
                ev["is_capstone"] = tc["is_capstone"]
        all_evals.extend(evals)

    result = aggregate_pass2(all_evals, persona_name, participant_id, pass1_model, judge_model)
    result["metadata"]["pass_1_source"] = str(pass1_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] overall={result['overall_accuracy']} -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--pass1", type=Path, required=True)
    ap.add_argument("--judge-model", default="sonnet")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run(args.persona, args.pass1, args.judge_model, args.batch_size, args.output)


if __name__ == "__main__":
    main()
