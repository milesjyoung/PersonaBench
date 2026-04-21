"""Step 6 — Benchmark runner (two-pass: answering then scoring).

Pass 1: evaluator LLM sees raw app logs + questions only. No ground_truth in
        context. Writes pass1 answers.

Pass 2: scorer LLM sees test cases (with ground_truth) + Pass 1 answers.
        Writes the final verification report.

OpenClaw session isolation: when --openclaw is set, the runner deletes the
target session file between each Pass 1 test case call so prior Q&A does not
leak into the current context. When running against the Anthropic SDK
directly, each test case is a fresh messages.create call, so isolation is
inherent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import anthropic

STEP_DIR = Path(__file__).parent
PROMPT_PATH = STEP_DIR / "prompt.txt"

DEFAULT_MODEL = "claude-opus-4-6"


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


# ----- LLM call ------------------------------------------------------------

def call_llm(client: anthropic.Anthropic, model: str, prompt: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=64_000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    )
    blocks = [b.text for b in response.content if b.type == "text"]
    return "".join(blocks)


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    payload = fenced.group(1) if fenced else text
    start = payload.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(payload[start:])


# ----- OpenClaw backend ---------------------------------------------------

def call_via_openclaw(
    prompt: str, openclaw_session_dir: Path, claude_cmd: str
) -> str:
    """Call the evaluator via the claude CLI, wiping the session dir first to
    guarantee a cold context for this test case."""
    if openclaw_session_dir.exists():
        shutil.rmtree(openclaw_session_dir)
    openclaw_session_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(prompt)
        tmp_path = f.name
    try:
        with open(tmp_path, "r", encoding="utf-8") as stream:
            result = subprocess.run(
                [claude_cmd, "-p"],
                stdin=stream,
                capture_output=True,
                timeout=1800,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI failed: {result.stderr.decode('utf-8', errors='replace')[:500]}"
            )
        return result.stdout.decode("utf-8", errors="replace").strip()
    finally:
        os.unlink(tmp_path)


# ----- passes --------------------------------------------------------------

def run_pass_1(
    raw_logs: str,
    test_cases_stripped: dict[str, Any],
    model: str,
    client: anthropic.Anthropic | None,
    use_openclaw: bool,
    openclaw_session_dir: Path | None,
    claude_cmd: str,
) -> dict[str, Any]:
    sections = load_prompt_sections()
    template = fill(sections["pass1"], "{{INSERT_RAW_LOGS_TEXT_HERE}}", raw_logs)
    template = fill(
        template,
        "{{INSERT_TEST_CASE_QUESTIONS_JSON_HERE}}",
        test_cases_stripped,
    )
    if use_openclaw:
        assert openclaw_session_dir is not None
        raw = call_via_openclaw(template, openclaw_session_dir, claude_cmd)
    else:
        assert client is not None
        raw = call_llm(client, model, template)
    return extract_json(raw)


def run_pass_2(
    test_cases_full: dict[str, Any],
    pass1_answers: dict[str, Any],
    model: str,
    client: anthropic.Anthropic,
) -> dict[str, Any]:
    sections = load_prompt_sections()
    template = fill(
        sections["pass2"], "{{INSERT_FULL_TEST_CASES_JSON_HERE}}", test_cases_full
    )
    template = fill(
        template, "{{INSERT_PASS_1_ANSWERS_JSON_HERE}}", pass1_answers
    )
    raw = call_llm(client, model, template)
    return extract_json(raw)


# ----- entry point ---------------------------------------------------------

def run_step(
    app_logs_path: Path,
    test_cases_path: Path,
    output_dir: Path,
    model_pass1: str,
    model_pass2: str,
    use_openclaw: bool,
    claude_cmd: str,
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

    openclaw_session_dir = None
    client = anthropic.Anthropic()
    if use_openclaw:
        openclaw_session_dir = output_dir / ".openclaw_session"

    print(f"[pass 1] answering {len(stripped['test_cases'])} cases for {base_name}")
    pass1 = run_pass_1(
        raw_logs,
        stripped,
        model_pass1,
        client=client,
        use_openclaw=use_openclaw,
        openclaw_session_dir=openclaw_session_dir,
        claude_cmd=claude_cmd,
    )
    pass1_out.write_text(json.dumps(pass1, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[pass 2] scoring answers for {base_name}")
    pass2 = run_pass_2(test_cases, pass1, model_pass2, client)
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
    parser.add_argument("--model-pass1", default=DEFAULT_MODEL)
    parser.add_argument("--model-pass2", default=DEFAULT_MODEL)
    parser.add_argument(
        "--openclaw", action="store_true",
        help="Route Pass 1 through the claude CLI (OpenClaw-style) with "
        "session file isolation between test cases.",
    )
    parser.add_argument(
        "--claude-cmd",
        default=os.environ.get(
            "CLAUDE_CMD", "claude.cmd" if sys.platform == "win32" else "claude"
        ),
    )
    args = parser.parse_args()

    if not args.openclaw and "ANTHROPIC_API_KEY" not in os.environ:
        print(
            "ANTHROPIC_API_KEY is not set and --openclaw was not passed.",
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
            args.openclaw,
            args.claude_cmd,
        )
    )


if __name__ == "__main__":
    main()
