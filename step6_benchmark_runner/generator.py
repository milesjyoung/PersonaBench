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
PROMPT_PATH = STEP_DIR / "prompt.txt"

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


# ----- prompt splitting ----------------------------------------------------

def load_prompt_sections() -> dict[str, str]:
    """Split prompt.txt into PASS 1 and PASS 2 sections on the big dividers."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
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


def strip_ground_truth(test_cases: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of test_cases with ground_truth/expected_evidence
    removed from each case. Questions, ids, types, subtype, framing remain."""
    stripped = {"metadata": test_cases.get("metadata", {}), "test_cases": []}
    for tc in test_cases.get("test_cases", []):
        stripped["test_cases"].append(
            {
                k: v
                for k, v in tc.items()
                if k
                not in {
                    "ground_truth",
                    "expected_evidence",
                    "source_hidden_fact_ids",
                    "source_subcategories",
                    "risk_surface_score",
                    "user_prompt_context",
                    "expected_behavior",
                }
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
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_fix_invalid_escapes(raw))


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


def run_pass_2(
    test_cases_full: dict[str, Any],
    pass1_answers: dict[str, Any],
    model: str,
    client: Any | None,
    backend: str,
    provider: str,
    claude_cmd: str,
    codex_cmd: str,
) -> dict[str, Any]:
    sections = load_prompt_sections()
    template = fill(
        sections["pass2"], "{{INSERT_FULL_TEST_CASES_JSON_HERE}}", test_cases_full
    )
    template = fill(
        template, "{{INSERT_PASS_1_ANSWERS_JSON_HERE}}", pass1_answers
    )
    if backend in SUBSCRIPTION_BACKENDS:
        raw = call_subscription_cli(
            template, model, backend, claude_cmd=claude_cmd, codex_cmd=codex_cmd
        )
    else:
        assert client is not None
        raw = call_llm(client, model, template, provider=provider)
    return extract_json(raw)


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
) -> int:
    app_logs = json.loads(app_logs_path.read_text(encoding="utf-8"))
    test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))

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
        default=os.environ.get("CODEX_CMD", "codex.exe" if sys.platform == "win32" else "codex"),
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
        )
    )


if __name__ == "__main__":
    main()
