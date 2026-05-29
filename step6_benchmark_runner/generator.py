"""Step 6 — Benchmark runner (two-pass: answering then scoring).

Pass 1: evaluator LLM sees raw app logs + questions only. No ground_truth in
        context. Writes pass1 answers.

Pass 2: scorer LLM sees test cases (with ground_truth) + Pass 1 answers.
        Writes the final verification report.

Subscription backend isolation: when --backend claude or --backend codex is
set, the runner deletes the target session directory between each model call
so prior Q&A does not leak into the current context. When running against the
Anthropic SDK directly, each test case is a fresh messages.create call, so
isolation is inherent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm import (
    API_BACKENDS,
    SUBSCRIPTION_BACKENDS,
    call_llm,
    call_subscription_cli,
    check_api_key,
    make_client,
    provider_for_backend,
)

STEP_DIR = Path(__file__).parent
PROMPT_V1_PATH = STEP_DIR / "prompt.txt"

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


# ----- prompt splitting ----------------------------------------------------

def load_prompt_sections() -> dict[str, str]:
    """Split prompt into PASS 1 and PASS 2 sections on the big dividers."""
    text = PROMPT_V1_PATH.read_text(encoding="utf-8")
    pass1_marker = "PASS 1 — ANSWERING"
    pass2_marker = "PASS 2 — SCORING"
    header_end = text.index(pass1_marker)
    pass2_start = text.index(pass2_marker)
    header = text[:header_end]
    pass1 = header + text[header_end:pass2_start]
    pass2 = header + text[pass2_start:]
    return {"pass1": pass1, "pass2": pass2}


def fill(template: str, placeholder: str, payload: Any) -> str:
    if isinstance(payload, str):
        return template.replace(placeholder, payload)
    return template.replace(
        placeholder, json.dumps(payload, indent=2, ensure_ascii=False)
    )


# ----- raw log preparation -------------------------------------------------

def strip_to_raw_logs(app_logs: dict[str, Any]) -> str:
    """Flatten messenger + calendar into a plain text stream for the evaluator."""
    lines: list[str] = []
    messenger = app_logs.get("messenger", {})
    sessions: list[dict[str, Any]] = []
    for bucket_name in ("meaningful_sessions", "filler_sessions", "sessions"):
        bucket = messenger.get(bucket_name, []) or []
        sessions.extend(bucket)
    sessions.sort(
        key=lambda s: (
            s.get("date", ""),
            (s.get("messages") or [{"time": ""}])[0].get("time", ""),
        )
    )
    persona_name = app_logs.get("persona_name") or app_logs.get("name", "persona")
    for s in sessions:
        contact = s.get("contact") or s.get("other_person") or "unknown"
        date = s.get("date", "")
        lines.append(f"--- {date} | {persona_name} <-> {contact} ---")
        for m in s.get("messages", []) or []:
            time = m.get("time", "")
            sender = m.get("sender", "")
            text = m.get("text", "")
            lines.append(f"[{time}] {sender}: {text}")
        lines.append("")

    calendar = app_logs.get("calendar", {})
    events = calendar.get("events", []) if isinstance(calendar, dict) else calendar or []
    events = sorted(
        events or [], key=lambda e: (e.get("date", ""), e.get("start_time", ""))
    )
    for e in events:
        date = e.get("date", "")
        start = e.get("start_time", "")
        end = e.get("end_time", "")
        title = e.get("title", "")
        loc = e.get("location", "")
        participants = ", ".join(e.get("participants", []) or [])
        notes = e.get("notes", "")
        lines.append(
            f"{date} | {start}-{end} | {title} | @ {loc} | with {participants} "
            f"| Notes: {notes}"
        )

    return "\n".join(lines)


FIELDS_HIDDEN_FROM_PASS1 = {
    "ground_truth",
    "expected_evidence",
    "source_hidden_fact_ids",
    "source_subcategories",
    "risk_surface_score",
    "user_prompt_context",
    "expected_behavior",
    "expected_resolution_label",
    "expected_action_boundary",
    "naive_compliance_consequence",
    "construction_tags",
    "hidden_fact_reasoning_roles",
    "fact_check_risk",
    "reasoning_requirement",
}


def strip_ground_truth(test_cases: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of test_cases with answer-key fields removed.
    Questions, ids, types, subtype, framing remain visible to Pass 1."""
    stripped = {"metadata": test_cases.get("metadata", {}), "test_cases": []}
    for tc in test_cases.get("test_cases", []):
        stripped["test_cases"].append(
            {
                k: v
                for k, v in tc.items()
                if k not in FIELDS_HIDDEN_FROM_PASS1
            }
        )
    return stripped


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    payload = fenced.group(1) if fenced else text
    start = payload.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")
    raw = payload[start:]
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(raw)
        return obj
    except json.JSONDecodeError:
        obj, _ = decoder.raw_decode(_fix_invalid_escapes(raw))
        return obj


def _fix_invalid_escapes(text: str) -> str:
    valid = set('"\\/bfnrtu')
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            if text[i + 1] in valid:
                out.append(text[i])
                out.append(text[i + 1])
                i += 2
            else:
                out.append(text[i + 1])
                i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


# ----- passes --------------------------------------------------------------

def run_pass_1(
    raw_logs: str,
    test_cases_stripped: dict[str, Any],
    model: str,
    client: Any | None,
    backend: str,
    provider: str,
    claude_cmd: str,
    codex_cmd: str,
) -> dict[str, Any]:
    sections = load_prompt_sections()
    template = fill(sections["pass1"], "{{INSERT_RAW_LOGS_TEXT_HERE}}", raw_logs)
    template = fill(
        template,
        "{{INSERT_TEST_CASE_QUESTIONS_JSON_HERE}}",
        test_cases_stripped,
    )
    if backend in SUBSCRIPTION_BACKENDS:
        raw = call_subscription_cli(
            template, model, backend, claude_cmd=claude_cmd, codex_cmd=codex_cmd
        )
    else:
        assert client is not None
        raw = call_llm(client, model, template, provider=provider)
    return extract_json(raw)


def _call_pass2_single(
    sections: dict[str, str],
    test_cases_batch: dict[str, Any],
    answers_batch: dict[str, Any],
    model: str,
    client: Any | None,
    backend: str,
    provider: str,
    claude_cmd: str,
    codex_cmd: str,
) -> dict[str, Any]:
    template = fill(
        sections["pass2"], "{{INSERT_FULL_TEST_CASES_JSON_HERE}}", test_cases_batch
    )
    template = fill(
        template, "{{INSERT_PASS_1_ANSWERS_JSON_HERE}}", answers_batch
    )
    if backend in SUBSCRIPTION_BACKENDS:
        raw = call_subscription_cli(
            template, model, backend, claude_cmd=claude_cmd, codex_cmd=codex_cmd
        )
    else:
        assert client is not None
        raw = call_llm(client, model, template, provider=provider)
    return extract_json(raw)


def run_pass_2(
    test_cases_full: dict[str, Any],
    pass1_answers: dict[str, Any],
    model: str,
    client: Any | None,
    backend: str,
    provider: str,
    claude_cmd: str,
    codex_cmd: str,
    batch_size: int = 5,
) -> dict[str, Any]:
    sections = load_prompt_sections()
    all_cases = test_cases_full.get("test_cases", [])
    all_answers = pass1_answers.get("answers", [])
    answer_map = {a["test_case_id"]: a for a in all_answers}

    if backend not in SUBSCRIPTION_BACKENDS or len(all_cases) <= batch_size:
        return _call_pass2_single(
            sections, test_cases_full, pass1_answers,
            model, client, backend, provider, claude_cmd, codex_cmd,
        )

    merged_evaluations: list[dict[str, Any]] = []
    merged_result: dict[str, Any] = {}

    for i in range(0, len(all_cases), batch_size):
        batch_cases = all_cases[i : i + batch_size]
        batch_ids = {c["id"] for c in batch_cases}
        batch_answers = [a for a in all_answers if a["test_case_id"] in batch_ids]

        tc_batch = dict(test_cases_full)
        tc_batch["test_cases"] = batch_cases
        ans_batch = dict(pass1_answers)
        ans_batch["answers"] = batch_answers

        batch_num = i // batch_size + 1
        total_batches = (len(all_cases) + batch_size - 1) // batch_size
        print(f"  [pass 2 batch {batch_num}/{total_batches}] scoring {len(batch_cases)} cases")

        for attempt in range(3):
            try:
                result = _call_pass2_single(
                    sections, tc_batch, ans_batch,
                    model, client, backend, provider, claude_cmd, codex_cmd,
                )
                evals = result.get("evaluations", [])
                if not evals and isinstance(result, dict) and "test_case_id" in result:
                    evals = [result]
                merged_evaluations.extend(evals)
                if not merged_result:
                    merged_result = result
                break
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"    batch {batch_num} attempt {attempt + 1} failed: {exc}")
                if attempt == 2:
                    print(f"    skipping batch {batch_num} after 3 attempts")

    if not merged_result:
        merged_result = {"evaluations": [], "metadata": {}}
    merged_result["evaluations"] = merged_evaluations
    merged_result["metadata"] = merged_result.get("metadata", {})
    merged_result["metadata"]["total_cases"] = len(merged_evaluations)
    merged_result["metadata"]["batched"] = True
    merged_result["metadata"]["batches_total"] = (len(all_cases) + batch_size - 1) // batch_size
    return merged_result


def _percent(score_sum: float, total: int) -> str:
    if total == 0:
        return "N/A"
    value = round((score_sum / total) * 100, 1)
    if value.is_integer():
        return f"{int(value)}%"
    return f"{value}%"


def recompute_benchmark_summary(
    result: dict[str, Any],
    test_cases_full: dict[str, Any],
    model_pass1: str,
    model_pass2: str,
) -> dict[str, Any]:
    """Recompute headline metrics after batched scoring merges evaluations."""
    evaluations = result.get("evaluations", []) or []
    case_map = {tc.get("id"): tc for tc in test_cases_full.get("test_cases", [])}

    total_score = sum(float(ev.get("score_value", 0) or 0) for ev in evaluations)
    result["overall_accuracy"] = _percent(total_score, len(evaluations))

    by_type: dict[str, list[float]] = {}
    for ev in evaluations:
        by_type.setdefault(ev.get("type", "unknown"), []).append(
            float(ev.get("score_value", 0) or 0)
        )
    result["accuracy_by_type"] = {
        typ: _percent(sum(scores), len(scores))
        for typ, scores in sorted(by_type.items())
    }

    type5 = [ev for ev in evaluations if ev.get("type") == "type_5_agent_behavior"]
    subtype_scores: dict[str, list[float]] = {}
    dimension_scores: dict[str, list[float]] = {}
    capstone_result = "N/A"
    for ev in type5:
        tc = case_map.get(ev.get("test_case_id"), {})
        subtype = tc.get("subtype", "unknown")
        subtype_scores.setdefault(subtype, []).append(
            float(ev.get("score_value", 0) or 0)
        )
        for dim, value in (ev.get("type_5_dimensions") or {}).items():
            if isinstance(value, (int, float)):
                dimension_scores.setdefault(dim, []).append(float(value))
        if tc.get("is_capstone"):
            capstone_result = ev.get("score", "N/A")

    result["type_5_breakdown"] = {
        "by_subtype": {
            subtype: _percent(sum(scores), len(scores))
            for subtype, scores in sorted(subtype_scores.items())
        },
        "by_dimension": {
            dim: round(sum(scores) / len(scores), 2)
            for dim, scores in sorted(dimension_scores.items())
            if scores
        },
        "capstone_result": capstone_result,
    }

    metadata = result.setdefault("metadata", {})
    metadata["model_used_pass1"] = model_pass1
    metadata["model_used_pass2"] = model_pass2
    metadata["total_cases"] = len(evaluations)
    return result


# ----- entry point ---------------------------------------------------------

def run_step(
    app_logs_path: Path,
    test_cases_path: Path,
    output_dir: Path,
    model_pass1: str,
    model_pass2: str,
    backend: str,
    provider: str,
    claude_cmd: str,
    codex_cmd: str,
    skip_pass1: bool = False,
) -> int:
    app_logs = json.loads(app_logs_path.read_text(encoding="utf-8"))
    test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))

    # Filter out REJECTED test cases from Step 5 verification
    verification_path = test_cases_path.parent / test_cases_path.name.replace(
        "_test_cases.json", "_test_cases_verification.json"
    )
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        rejected_ids: set[str] = set()
        for ev in verification.get("evaluations", []):
            if ev.get("overall_verdict") == "REJECTED":
                eid = ev.get("test_case_id") or ev.get("id") or ""
                if eid:
                    rejected_ids.add(eid)
        if rejected_ids:
            original_count = len(test_cases.get("test_cases", []))
            test_cases["test_cases"] = [
                tc for tc in test_cases.get("test_cases", [])
                if (tc.get("id") or tc.get("test_case_id") or "") not in rejected_ids
            ]
            filtered_count = len(test_cases["test_cases"])
            if "metadata" in test_cases:
                test_cases["metadata"]["total_test_cases"] = filtered_count
            print(
                f"[step6] filtered {original_count - filtered_count} REJECTED "
                f"test cases ({filtered_count} remain)"
            )

    raw_logs = strip_to_raw_logs(app_logs)
    stripped = strip_ground_truth(test_cases)

    base_name = test_cases_path.stem.replace("_test_cases", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    pass1_out = output_dir / f"{base_name}_pass1_answers.json"
    pass2_out = output_dir / f"{base_name}_benchmark_results.json"
    raw_logs_out = output_dir / f"{base_name}_app_logs_raw.txt"
    raw_logs_out.write_text(raw_logs, encoding="utf-8")

    client = None
    if backend in API_BACKENDS:
        client = make_client(provider)

    if skip_pass1 and pass1_out.exists():
        print(f"[pass 1] skipped — loading existing {pass1_out.name}")
        pass1 = json.loads(pass1_out.read_text(encoding="utf-8"))
    else:
        print(f"[pass 1] answering {len(stripped['test_cases'])} cases for {base_name}")
        pass1 = run_pass_1(
            raw_logs,
            stripped,
            model_pass1,
            client=client,
            backend=backend,
            provider=provider,
            claude_cmd=claude_cmd,
            codex_cmd=codex_cmd,
        )
        pass1_out.write_text(json.dumps(pass1, indent=2, ensure_ascii=False), encoding="utf-8")
    pass1.setdefault("metadata", {})
    pass1["metadata"]["model_used"] = model_pass1
    pass1["metadata"]["cases_answered"] = len(stripped["test_cases"])
    pass1_out.write_text(json.dumps(pass1, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[pass 2] scoring answers for {base_name}")
    pass2 = run_pass_2(
        test_cases,
        pass1,
        model_pass2,
        client,
        backend=backend,
        provider=provider,
        claude_cmd=claude_cmd,
        codex_cmd=codex_cmd,
    )
    pass2 = recompute_benchmark_summary(pass2, test_cases, model_pass1, model_pass2)
    pass2_out.write_text(json.dumps(pass2, indent=2, ensure_ascii=False), encoding="utf-8")

    accuracy = pass2.get("overall_accuracy", "unknown")
    print(f"[done] overall_accuracy={accuracy}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app-logs", type=Path, required=True,
        help="Path to {persona}_app_logs.json",
    )
    parser.add_argument(
        "--test-cases", type=Path, required=True,
        help="Path to {persona}_test_cases.json",
    )
    parser.add_argument(
        "--output", type=Path, default=STEP_DIR / "data_samples" / "output",
    )
    parser.add_argument("--model-pass1", default=None)
    parser.add_argument("--model-pass2", default=None)
    parser.add_argument(
        "--backend",
        default="anthropic-api",
        choices=API_BACKENDS + SUBSCRIPTION_BACKENDS,
        help="Inference backend: anthropic-api uses ANTHROPIC_API_KEY; "
        "openai-api uses OPENAI_API_KEY; claude and codex use logged-in CLIs.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=("anthropic", "openai"),
        help="Optional API provider override for compatibility.",
    )
    parser.add_argument(
        "--openclaw", action="store_true",
        help="Deprecated alias for --backend claude.",
    )
    parser.add_argument(
        "--claude-cmd",
        default=os.environ.get(
            "CLAUDE_CMD", "claude.cmd" if sys.platform == "win32" else "claude"
        ),
    )
    parser.add_argument(
        "--codex-cmd",
        default=os.environ.get("CODEX_CMD", "codex"),
    )
    parser.add_argument(
        "--skip-pass1",
        action="store_true",
        help="Skip Pass 1 and reuse existing pass1_answers.json. Useful when "
        "only the scoring prompt or judge model changed.",
    )
    args = parser.parse_args()
    backend = "claude" if args.openclaw else args.backend
    provider = provider_for_backend(backend, args.provider or "anthropic") if backend in API_BACKENDS else (args.provider or "anthropic")
    if args.model_pass1 is None:
        args.model_pass1 = DEFAULT_MODEL if backend in {"anthropic-api", "claude"} else "gpt-5-mini"
    if args.model_pass2 is None:
        args.model_pass2 = DEFAULT_JUDGE_MODEL if backend in {"anthropic-api", "claude"} else "gpt-5.4"

    if backend in API_BACKENDS and not check_api_key(provider):
        print(
            f"{provider.upper()}_API_KEY is not set and no subscription backend was selected.",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(
        run_step(
            args.app_logs,
            args.test_cases,
            args.output,
            args.model_pass1,
            args.model_pass2,
            backend,
            provider,
            args.claude_cmd,
            args.codex_cmd,
            skip_pass1=args.skip_pass1,
        )
    )


if __name__ == "__main__":
    main()
