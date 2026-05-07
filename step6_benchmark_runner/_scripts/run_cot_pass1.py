"""Run Pass 1 with the CoT-scaffolded prompt (prompt_cot.txt) on a persona,
then chain Pass 2 through the canonical scoring prompt for an apples-to-apples
comparison vs the single-shot baseline.

Outputs:
  step6_benchmark_runner/data_samples/output_cot/{persona}_pass1_answers_cot.json
  step6_benchmark_runner/data_samples/output_cot/{persona}_benchmark_results_cot.json

Usage:
  python -m step6_benchmark_runner._scripts.run_cot_pass1 \\
    --persona julio_simmons --pass1-model opus --pass2-model sonnet
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _cli_helpers import (
    approx_token_count,
    call_and_parse_json,
    call_claude_cli,
    extract_json,
    fill,
    load_prompt_sections,
    strip_ground_truth,
    strip_to_raw_logs,
)
from run_pass2_batched import aggregate as aggregate_pass2, chunk as chunk_cases

REPO = Path(__file__).resolve().parents[2]
STEP4_OUT = REPO / "step4_app_log_synthesizer" / "data_samples" / "output"
STEP5_OUT = REPO / "step5_testcases_synthesis" / "data_samples" / "output"
STEP6 = REPO / "step6_benchmark_runner"
COT_PROMPT = STEP6 / "prompt_cot.txt"
CANONICAL_PROMPT = STEP6 / "prompt.txt"
OUT_DIR = STEP6 / "data_samples" / "output_cot"


def run_pass_1_cot(persona: str, model: str, type5_only: bool = False) -> dict:
    app_logs = json.loads(
        (STEP4_OUT / f"{persona}_app_logs.json").read_text(encoding="utf-8")
    )
    test_cases = json.loads(
        (STEP5_OUT / f"{persona}_test_cases.json").read_text(encoding="utf-8")
    )
    raw_logs = strip_to_raw_logs(app_logs)
    stripped = strip_ground_truth(test_cases)

    if type5_only:
        stripped["test_cases"] = [
            tc for tc in stripped["test_cases"]
            if tc.get("type") == "type_5_agent_behavior"
        ]
        print(f"[cot/{persona}] type5_only: {len(stripped['test_cases'])} cases", flush=True)

    sections = load_prompt_sections(COT_PROMPT)
    prompt = fill(sections["pass1"], "{{INSERT_RAW_LOGS_TEXT_HERE}}", raw_logs)
    prompt = fill(prompt, "{{INSERT_TEST_CASE_QUESTIONS_JSON_HERE}}", stripped)

    print(
        f"[cot/{persona}] pass1 model={model} "
        f"raw_log_tokens~{approx_token_count(raw_logs)} "
        f"cases={len(stripped['test_cases'])}",
        flush=True,
    )
    last_err = None
    answers = None
    for attempt in range(1, 4):
        try:
            raw = call_claude_cli(prompt, model=model, timeout_sec=1800, max_retries=1)
            answers = extract_json(raw)
            break
        except (ValueError, RuntimeError) as e:
            last_err = e
            print(f"  [cot/{persona}] pass1 attempt {attempt}/3 failed: {e}", flush=True)
    if answers is None:
        raise RuntimeError(f"Pass 1 failed after 3 attempts: {last_err}")
    answers.setdefault("metadata", {})
    answers["metadata"]["pass_1_date"] = date.today().isoformat()
    answers["metadata"]["model_used"] = model
    answers["metadata"]["raw_log_token_count"] = approx_token_count(raw_logs)
    answers["metadata"]["cot_variant"] = "type5_only_v1"
    answers["metadata"]["cases_answered"] = len(answers.get("answers", []))
    return answers


def run_pass_2(persona: str, pass1_answers: dict, judge_model: str, type5_only: bool = False) -> dict:
    test_cases = json.loads(
        (STEP5_OUT / f"{persona}_test_cases.json").read_text(encoding="utf-8")
    )
    sections = load_prompt_sections(CANONICAL_PROMPT)

    cases = test_cases["test_cases"]
    if type5_only:
        cases = [c for c in cases if c.get("type") == "type_5_agent_behavior"]
    answers_by_id = {a["test_case_id"]: a for a in pass1_answers.get("answers", [])}
    persona_name = test_cases.get("metadata", {}).get("persona", persona)
    participant_id = test_cases.get("metadata", {}).get("participant_id", "")
    pass1_model = pass1_answers.get("metadata", {}).get("model_used", "unknown")

    all_evals = []
    batch_size = 20
    batches = chunk_cases(cases, batch_size)
    for bi, batch in enumerate(batches, start=1):
        batch_payload = {"metadata": test_cases.get("metadata", {}), "test_cases": batch}
        batch_pass1 = {
            "metadata": pass1_answers.get("metadata", {}),
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
        print(f"[cot/{persona}] pass2 batch {bi}/{len(batches)} judge={judge_model}", flush=True)
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
    result["metadata"]["cot_variant"] = "type5_only_v1"
    return result


def run(persona: str, pass1_model: str, pass2_model: str, type5_only: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    answers = run_pass_1_cot(persona, pass1_model, type5_only=type5_only)
    suffix = "_type5" if type5_only else ""
    pass1_path = OUT_DIR / f"{persona}_pass1_answers_cot{suffix}.json"
    pass1_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")

    result = run_pass_2(persona, answers, pass2_model, type5_only=type5_only)
    pass2_path = OUT_DIR / f"{persona}_benchmark_results_cot{suffix}.json"
    pass2_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done/{persona}] cot overall={result.get('overall_accuracy')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--pass1-model", default="opus")
    ap.add_argument("--pass2-model", default="sonnet")
    ap.add_argument("--type5-only", action="store_true",
                    help="Only run Type 5 cases (faster A/B since CoT only "
                    "affects Type 5)")
    args = ap.parse_args()
    run(args.persona, args.pass1_model, args.pass2_model, args.type5_only)


if __name__ == "__main__":
    main()
