"""Context-window ablation: rerun Pass 1 on Julio with raw logs truncated to
{32K, 64K, 128K, full} tokens, then Pass 2 with the canonical scoring prompt.

Answers the question: is Opus 4.7's high accuracy driven by context length
(retrieval over a long haystack) or by reasoning (recovering signals from a
smaller window)?

Hypothesis: Type 2 / Type 3 (cross-source / temporal) drop with smaller
context — they need retrieval across the full log window. Type 5 (reasoning +
safety) holds up — it is reasoning-bound, not retrieval-bound.

Outputs:
  step6_benchmark_runner/data_samples/output_ablation/
    {persona}_pass1_{budget}.json
    {persona}_benchmark_results_{budget}.json
  4.28.26/context_window_ablation.md   (final report; written by analyze step)

Usage:
  python -m step6_benchmark_runner._scripts.context_window_ablation \\
    --persona julio_simmons --budgets 32000,64000,128000,full
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
    truncate_to_tokens,
)
from run_pass2_batched import aggregate as aggregate_pass2, chunk as chunk_cases

REPO = Path(__file__).resolve().parents[2]
STEP4_OUT = REPO / "step4_app_log_synthesizer" / "data_samples" / "output"
STEP5_OUT = REPO / "step5_testcases_synthesis" / "data_samples" / "output"
STEP6 = REPO / "step6_benchmark_runner"
PROMPT = STEP6 / "prompt.txt"
OUT_DIR = STEP6 / "data_samples" / "output_ablation"


def run_one_budget(
    persona: str,
    raw_logs_full: str,
    test_cases: dict,
    budget: int | None,
    pass1_model: str,
    pass2_model: str,
) -> dict:
    label = "full" if budget is None else f"{budget}"
    truncated = raw_logs_full if budget is None else truncate_to_tokens(raw_logs_full, budget)
    actual_tokens = approx_token_count(truncated)
    stripped = strip_ground_truth(test_cases)

    sections = load_prompt_sections(PROMPT)
    p1 = fill(sections["pass1"], "{{INSERT_RAW_LOGS_TEXT_HERE}}", truncated)
    p1 = fill(p1, "{{INSERT_TEST_CASE_QUESTIONS_JSON_HERE}}", stripped)

    print(f"[ablation/{persona}/{label}] pass1 tokens~{actual_tokens}", flush=True)
    answers = call_and_parse_json(p1, model=pass1_model, max_retries=3, timeout_sec=3600)
    answers.setdefault("metadata", {})
    answers["metadata"]["pass_1_date"] = date.today().isoformat()
    answers["metadata"]["model_used"] = pass1_model
    answers["metadata"]["raw_log_token_count"] = actual_tokens
    answers["metadata"]["context_budget"] = label
    answers["metadata"]["cases_answered"] = len(answers.get("answers", []))

    p1_path = OUT_DIR / f"{persona}_pass1_{label}.json"
    p1_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")

    cases = test_cases["test_cases"]
    answers_by_id = {a["test_case_id"]: a for a in answers.get("answers", [])}
    persona_name = test_cases.get("metadata", {}).get("persona", persona)
    participant_id = test_cases.get("metadata", {}).get("participant_id", "")

    all_evals = []
    batch_size = 20
    batches = chunk_cases(cases, batch_size)
    for bi, batch in enumerate(batches, start=1):
        batch_payload = {"metadata": test_cases.get("metadata", {}), "test_cases": batch}
        batch_pass1 = {
            "metadata": answers.get("metadata", {}),
            "answers": [answers_by_id[c["id"]] for c in batch if c["id"] in answers_by_id],
        }
        bp = fill(sections["pass2"], "{{INSERT_FULL_TEST_CASES_JSON_HERE}}", batch_payload)
        bp = fill(bp, "{{INSERT_PASS_1_ANSWERS_JSON_HERE}}", batch_pass1)
        bp += (
            f"\n\nIMPORTANT: This batch contains {len(batch)} cases "
            f"({batch[0]['id']}..{batch[-1]['id']}). Score every case in this "
            f"batch. Return JSON with an `evaluations` array containing exactly "
            f"{len(batch)} entries."
        )
        print(f"[ablation/{persona}/{label}] pass2 batch {bi}/{len(batches)}", flush=True)
        parsed = call_and_parse_json(bp, model=pass2_model, timeout_sec=1500, max_retries=3)
        evals = parsed.get("evaluations", [])
        case_meta = {c["id"]: c for c in batch}
        for ev in evals:
            tc = case_meta.get(ev.get("test_case_id"), {})
            if "subtype" not in ev and tc.get("subtype"):
                ev["subtype"] = tc["subtype"]
            if "is_capstone" not in ev and tc.get("is_capstone") is not None:
                ev["is_capstone"] = tc["is_capstone"]
        all_evals.extend(evals)

    result = aggregate_pass2(
        all_evals, persona_name, participant_id, pass1_model, pass2_model
    )
    result["metadata"]["context_budget"] = label
    result["metadata"]["raw_log_token_count"] = actual_tokens

    p2_path = OUT_DIR / f"{persona}_benchmark_results_{label}.json"
    p2_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done/{persona}/{label}] overall={result.get('overall_accuracy')}", flush=True)
    return result


def run(persona: str, budgets: list[int | None], pass1_model: str, pass2_model: str, skip_existing: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app_logs = json.loads((STEP4_OUT / f"{persona}_app_logs.json").read_text(encoding="utf-8"))
    test_cases = json.loads((STEP5_OUT / f"{persona}_test_cases.json").read_text(encoding="utf-8"))
    raw_full = strip_to_raw_logs(app_logs)
    print(f"[ablation/{persona}] full raw_log tokens~{approx_token_count(raw_full)}", flush=True)

    summary = []
    for b in budgets:
        label = "full" if b is None else f"{b}"
        target = OUT_DIR / f"{persona}_benchmark_results_{label}.json"
        if skip_existing and target.exists():
            print(f"[skip] {persona}/{label} already done -> {target}", flush=True)
            continue
        result = run_one_budget(persona, raw_full, test_cases, b, pass1_model, pass2_model)
        summary.append(
            {
                "budget": "full" if b is None else b,
                "overall_accuracy": result.get("overall_accuracy"),
                "accuracy_by_type": result.get("accuracy_by_type"),
                "type_5_dimensions": result.get("type_5_breakdown", {}).get("by_dimension"),
                "capstone": result.get("type_5_breakdown", {}).get("capstone_result"),
            }
        )

    summary_path = OUT_DIR / f"{persona}_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[summary]\n{json.dumps(summary, indent=2)}")


def parse_budgets(arg: str) -> list[int | None]:
    out: list[int | None] = []
    for part in arg.split(","):
        part = part.strip().lower()
        if part in {"full", "all", ""}:
            out.append(None)
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--budgets", default="32000,64000,128000,full")
    ap.add_argument("--pass1-model", default="opus")
    ap.add_argument("--pass2-model", default="sonnet")
    args = ap.parse_args()
    run(args.persona, parse_budgets(args.budgets), args.pass1_model, args.pass2_model)


if __name__ == "__main__":
    main()
