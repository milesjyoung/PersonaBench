"""Fallback Pass 2 runner that batches test cases (default 20 per batch) so
each claude CLI call has a smaller, faster prompt. Each batch produces partial
evaluations; we merge and recompute aggregate accuracy locally.

Use this when the full-prompt Pass 2 (run_pass2_with_judge.py) takes too long
or hits CLI limits. This version trades wall-clock for per-call simplicity:
5 small calls instead of 1 huge one.

Usage:
  python -m step6_benchmark_runner._scripts.run_pass2_batched \\
    --persona julio_simmons --judge-model opus --batch-size 20
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _cli_helpers import (
    call_and_parse_json,
    call_claude_cli,
    extract_json,
    fill,
    load_prompt_sections,
)

REPO = Path(__file__).resolve().parents[2]
STEP6 = REPO / "step6_benchmark_runner"
PROMPT = STEP6 / "prompt.txt"
OUT_DIR = STEP6 / "data_samples" / "output_dual_judge"
TC_DIR = REPO / "step5_testcases_synthesis" / "data_samples" / "output"
P1_DIR = STEP6 / "data_samples" / "output"


TYPE_KEYS = [
    "type_1_simple_fact_check",
    "type_2_cross_log_fact_check",
    "type_3_dynamic_tracking",
    "type_4_reasoning",
    "type_5_agent_behavior",
]
TYPE5_DIMS = ["risk_surfacing", "appropriate_response", "evidence_use", "hallucination_control"]


def chunk(lst, n):
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def aggregate(evaluations: list[dict], persona_name: str, participant_id: str,
              pass1_model: str, judge_model: str) -> dict:
    by_type_correct: dict[str, list[float]] = {t: [] for t in TYPE_KEYS}
    type5_dim_scores: dict[str, list[float]] = {d: [] for d in TYPE5_DIMS}
    type5_subtype: dict[str, list[float]] = {}
    capstone_result = None
    overall: list[float] = []

    for ev in evaluations:
        t = ev.get("type")
        v = ev.get("score_value")
        if v is None:
            continue
        overall.append(float(v))
        if t in by_type_correct:
            by_type_correct[t].append(float(v))
        if t == "type_5_agent_behavior":
            dims = ev.get("type_5_dimensions") or {}
            for d in TYPE5_DIMS:
                if d in dims:
                    type5_dim_scores[d].append(float(dims[d]))
            sub = ev.get("subtype")
            if sub:
                type5_subtype.setdefault(sub, []).append(float(v))
            if ev.get("is_capstone"):
                capstone_result = ev.get("score")

    def pct(xs: list[float]) -> str:
        if not xs:
            return "n/a"
        return f"{100 * sum(xs) / len(xs):.1f}%"

    return {
        "metadata": {
            "persona": persona_name,
            "participant_id": participant_id,
            "pass_2_date": date.today().isoformat(),
            "model_used_pass1": pass1_model,
            "model_used_pass2": judge_model,
            "total_cases": len(evaluations),
            "scoring_mode": "batched_pass2",
        },
        "overall_accuracy": pct(overall),
        "accuracy_by_type": {t: pct(by_type_correct[t]) for t in TYPE_KEYS},
        "type_5_breakdown": {
            "by_subtype": {k: pct(v) for k, v in type5_subtype.items()},
            "by_dimension": {
                d: f"{sum(type5_dim_scores[d]) / len(type5_dim_scores[d]):.2f}"
                if type5_dim_scores[d] else "n/a"
                for d in TYPE5_DIMS
            },
            "capstone_result": capstone_result,
        },
        "evaluations": evaluations,
    }


def run(persona: str, judge_model: str, batch_size: int) -> None:
    test_cases = json.loads((TC_DIR / f"{persona}_test_cases.json").read_text(encoding="utf-8"))
    pass1 = json.loads((P1_DIR / f"{persona}_pass1_answers.json").read_text(encoding="utf-8"))
    sections = load_prompt_sections(PROMPT)

    cases = test_cases["test_cases"]
    answers_by_id = {a["test_case_id"]: a for a in pass1["answers"]}
    persona_name = test_cases.get("metadata", {}).get("persona", persona)
    participant_id = test_cases.get("metadata", {}).get("participant_id", "")
    pass1_model = pass1.get("metadata", {}).get("model_used", "unknown")

    all_evals: list[dict] = []
    for batch_idx, batch in enumerate(chunk(cases, batch_size), start=1):
        batch_payload = {
            "metadata": test_cases.get("metadata", {}),
            "test_cases": batch,
        }
        batch_pass1 = {
            "metadata": pass1.get("metadata", {}),
            "answers": [answers_by_id[c["id"]] for c in batch if c["id"] in answers_by_id],
        }
        prompt = fill(sections["pass2"], "{{INSERT_FULL_TEST_CASES_JSON_HERE}}", batch_payload)
        prompt = fill(prompt, "{{INSERT_PASS_1_ANSWERS_JSON_HERE}}", batch_pass1)
        prompt += (
            f"\n\nIMPORTANT: This batch contains {len(batch)} cases "
            f"({batch[0]['id']}..{batch[-1]['id']}). Score every case in "
            f"this batch. Return JSON with an `evaluations` array containing "
            f"exactly {len(batch)} entries. Aggregate metrics will be "
            f"computed locally; you only need the per-case evaluations."
        )
        print(
            f"[batch {batch_idx}/{len(chunk(cases, batch_size))}] "
            f"{persona} judge={judge_model} cases={batch[0]['id']}..{batch[-1]['id']}",
            flush=True,
        )
        parsed = call_and_parse_json(prompt, model=judge_model, timeout_sec=1500, max_retries=3)
        evals = parsed.get("evaluations", [])
        # Merge in subtype/is_capstone from test_cases for aggregation
        case_meta = {c["id"]: c for c in batch}
        for ev in evals:
            tc = case_meta.get(ev.get("test_case_id"), {})
            if "subtype" not in ev and tc.get("subtype"):
                ev["subtype"] = tc["subtype"]
            if "is_capstone" not in ev and tc.get("is_capstone") is not None:
                ev["is_capstone"] = tc["is_capstone"]
        all_evals.extend(evals)
        print(f"  -> got {len(evals)} evaluations", flush=True)

    result = aggregate(all_evals, persona_name, participant_id, pass1_model, judge_model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = judge_model.split("-")[1] if "-" in judge_model else judge_model
    out_path = OUT_DIR / f"{persona}_benchmark_results_{suffix}_judge.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] overall={result['overall_accuracy']} -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()
    run(args.persona, args.judge_model, args.batch_size)


if __name__ == "__main__":
    main()
