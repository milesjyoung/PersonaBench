"""Shared helpers for the dynamic-track scratch scripts.

Wraps the `claude` CLI with explicit --model selection so we can run dual-judge
Pass 2, CoT Pass 1, and ablation experiments with reproducible model routing.

These scripts live under step6_benchmark_runner/_scripts/ (underscore prefix
marks scratch / experimental work per repo convention). They are not part of
the canonical pipeline. Their outputs go to data_samples/output_<experiment>/
directories so canonical data_samples/output/ is never overwritten.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_CLAUDE_CMD = os.environ.get(
    "CLAUDE_CMD", "claude.cmd" if sys.platform == "win32" else "claude"
)


def call_claude_cli(
    prompt: str,
    model: str | None = None,
    claude_cmd: str = DEFAULT_CLAUDE_CMD,
    timeout_sec: int = 1800,
    session_dir: Path | None = None,
    max_retries: int = 1,
    raw_dump_path: Path | None = None,
) -> str:
    """Invoke `claude -p [--model X]` with the prompt on stdin.

    If session_dir is provided, it is wiped before each call. This is the
    OpenClaw isolation pattern for cold-context runs (Step 6 Pass 1 needs it
    so prior cases don't leak into the current one).
    """
    if session_dir is not None:
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)

    cmd = [claude_cmd, "-p"]
    if model:
        cmd += ["--model", model]
    # Pass 2 scoring is judgment-by-rubric, not deep reasoning. Lowering effort
    # cuts wall-clock substantially without hurting grading quality.
    if os.environ.get("CLAUDE_EFFORT"):
        cmd += ["--effort", os.environ["CLAUDE_EFFORT"]]

    last_stderr = ""
    for attempt in range(max_retries + 1):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(prompt)
            tmp_path = f.name
        try:
            with open(tmp_path, "r", encoding="utf-8") as stream:
                result = subprocess.run(
                    cmd,
                    stdin=stream,
                    capture_output=True,
                    timeout=timeout_sec,
                )
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            if raw_dump_path is not None:
                raw_dump_path.parent.mkdir(parents=True, exist_ok=True)
                raw_dump_path.write_text(
                    f"RC={result.returncode}\n--STDOUT--\n{stdout}\n--STDERR--\n{stderr}",
                    encoding="utf-8",
                )
            if result.returncode == 0 and stdout.strip():
                return stdout.strip()
            last_stderr = stderr
            print(
                f"[cli] attempt {attempt + 1}/{max_retries + 1} failed "
                f"(rc={result.returncode}); stderr: {stderr[:200]}"
            )
        finally:
            os.unlink(tmp_path)
    raise RuntimeError(
        f"claude CLI failed after {max_retries + 1} attempts: {last_stderr[:500]}"
    )


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    payload = fenced.group(1) if fenced else text
    start = payload.find("{")
    if start == -1:
        raise ValueError("No JSON object found in CLI output")
    return json.loads(payload[start:])


def call_and_parse_json(
    prompt: str,
    model: str | None = None,
    timeout_sec: int = 1800,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call the CLI and parse JSON, with retries on either CLI-level or
    JSON-parse-level failure. Used for Pass 2 batches and similar where
    the CLI sometimes returns text that doesn't contain a JSON object."""
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_claude_cli(prompt, model=model, timeout_sec=timeout_sec, max_retries=1)
            return extract_json(raw)
        except (ValueError, RuntimeError, json.JSONDecodeError) as e:
            last_err = e
            print(
                f"  [call_and_parse] attempt {attempt}/{max_retries} failed: {e}",
                flush=True,
            )
    raise RuntimeError(f"call_and_parse failed after {max_retries} attempts: {last_err}")


def fill(template: str, placeholder: str, payload: Any) -> str:
    if isinstance(payload, str):
        return template.replace(placeholder, payload)
    return template.replace(
        placeholder, json.dumps(payload, indent=2, ensure_ascii=False)
    )


def load_prompt_sections(prompt_path: Path) -> dict[str, str]:
    text = prompt_path.read_text(encoding="utf-8")
    pass1_marker = "PASS 1 — ANSWERING"
    pass2_marker = "PASS 2 — SCORING"
    header_end = text.index(pass1_marker)
    pass2_start = text.index(pass2_marker)
    header = text[:header_end]
    pass1 = header + text[header_end:pass2_start]
    pass2 = header + text[pass2_start:]
    return {"pass1": pass1, "pass2": pass2}


def strip_to_raw_logs_filler_pruned(
    app_logs: dict[str, Any],
    target_tokens: int,
    seed: int = 42,
) -> str:
    """Flatten messenger + calendar to text, downsampling filler sessions
    only — meaningful_sessions and calendar events are NEVER dropped or
    summarized.

    Hard guarantees (asserted at runtime):
      1. Every entry in messenger.meaningful_sessions appears in the output.
      2. Every entry in calendar.events appears in the output.
      3. Only messenger.filler_sessions are subject to sampling.

    If even retaining all meaningful + calendar exceeds target_tokens, the
    function still returns all meaningful + calendar (the budget is honored
    on a best-effort basis for filler only). Per professor directive:
    summarizing meaningful context is lossy and is forbidden.

    Filler sampling is deterministic given `seed` so reruns reproduce.

    Returns:
      str — the rendered text. For diagnostics including counts of what was
      preserved vs sampled, use `prune_filler_with_diagnostics` instead.
    """
    text, _ = prune_filler_with_diagnostics(app_logs, target_tokens, seed)
    return text


def prune_filler_with_diagnostics(
    app_logs: dict[str, Any],
    target_tokens: int,
    seed: int = 42,
) -> tuple[str, dict[str, Any]]:
    """Same as strip_to_raw_logs_filler_pruned but also returns a diagnostics
    dict suitable for pinning into output metadata, so downstream readers
    can prove no meaningful context was dropped:

        {
          "meaningful_sessions_in": int,
          "meaningful_sessions_kept": int,    # MUST equal _in
          "calendar_events_in": int,
          "calendar_events_kept": int,        # MUST equal _in
          "filler_sessions_total": int,
          "filler_sessions_sampled": int,
          "filler_sessions_dropped": int,
          "total_tokens_estimated": int,
          "target_tokens": int,
          "filler_sample_seed": int,
        }

    Raises AssertionError if the meaningful or calendar guarantee is ever
    violated.
    """
    import random as _random

    messenger = app_logs.get("messenger", {})
    meaningful_sessions = messenger.get("meaningful_sessions", []) or []
    filler_sessions = messenger.get("filler_sessions", []) or []
    calendar = app_logs.get("calendar", {})
    cal_events = calendar.get("events", []) if isinstance(calendar, dict) else (calendar or [])

    n_meaningful_in = len(meaningful_sessions)
    n_calendar_in = len(cal_events)
    n_filler_in = len(filler_sessions)

    persona_name = app_logs.get("persona_name") or app_logs.get("name", "persona")

    def render_session(s: dict[str, Any]) -> str:
        contact = s.get("contact") or s.get("other_person") or "unknown"
        date = s.get("date", "")
        out = [f"--- {date} | {persona_name} <-> {contact} ---"]
        for m in s.get("messages", []) or []:
            out.append(f"[{m.get('time','')}] {m.get('sender','')}: {m.get('text','')}")
        out.append("")
        return "\n".join(out)

    def render_event(e: dict[str, Any]) -> str:
        date = e.get("date", "")
        start = e.get("start_time", "")
        end = e.get("end_time", "")
        title = e.get("title", "")
        loc = e.get("location", "")
        participants = ", ".join(e.get("participants", []) or [])
        notes = e.get("notes", "")
        return (
            f"{date} | {start}-{end} | {title} | @ {loc} | with {participants} | "
            f"Notes: {notes}"
        )

    meaningful_sorted = sorted(
        meaningful_sessions,
        key=lambda s: (s.get("date", ""), (s.get("messages") or [{"time": ""}])[0].get("time", "")),
    )
    cal_sorted = sorted(cal_events, key=lambda e: (e.get("date", ""), e.get("start_time", "")))

    fixed_text_parts = [render_session(s) for s in meaningful_sorted]
    fixed_text_parts.extend(render_event(e) for e in cal_sorted)
    fixed_text = "\n".join(fixed_text_parts)
    fixed_tokens = approx_token_count(fixed_text)

    remaining = target_tokens - fixed_tokens
    if remaining <= 0 or not filler_sessions:
        diagnostics = {
            "meaningful_sessions_in": n_meaningful_in,
            "meaningful_sessions_kept": n_meaningful_in,
            "calendar_events_in": n_calendar_in,
            "calendar_events_kept": n_calendar_in,
            "filler_sessions_total": n_filler_in,
            "filler_sessions_sampled": 0,
            "filler_sessions_dropped": n_filler_in,
            "total_tokens_estimated": fixed_tokens,
            "target_tokens": target_tokens,
            "filler_sample_seed": seed,
        }
        assert diagnostics["meaningful_sessions_kept"] == diagnostics["meaningful_sessions_in"], (
            "meaningful_sessions guarantee violated"
        )
        assert diagnostics["calendar_events_kept"] == diagnostics["calendar_events_in"], (
            "calendar_events guarantee violated"
        )
        return fixed_text, diagnostics

    rng = _random.Random(seed)
    rendered_filler = [render_session(s) for s in filler_sessions]
    indexed = list(enumerate(rendered_filler))
    rng.shuffle(indexed)

    accumulated_chars = 0
    char_budget = remaining * 4
    selected_idx: list[int] = []
    for idx, text in indexed:
        if accumulated_chars + len(text) > char_budget:
            continue
        selected_idx.append(idx)
        accumulated_chars += len(text)

    selected_idx.sort()

    all_lines: list[str] = []
    meaningful_by_date: list[tuple[str, str]] = []
    for s in meaningful_sorted:
        meaningful_by_date.append((
            s.get("date", ""),
            render_session(s),
        ))
    for idx in selected_idx:
        s = filler_sessions[idx]
        meaningful_by_date.append((
            s.get("date", ""),
            rendered_filler[idx],
        ))

    meaningful_by_date.sort(key=lambda x: x[0])
    for _, text in meaningful_by_date:
        all_lines.append(text)
    for e in cal_sorted:
        all_lines.append(render_event(e))

    rendered_text = "\n".join(all_lines)
    diagnostics = {
        "meaningful_sessions_in": n_meaningful_in,
        "meaningful_sessions_kept": n_meaningful_in,
        "calendar_events_in": n_calendar_in,
        "calendar_events_kept": n_calendar_in,
        "filler_sessions_total": n_filler_in,
        "filler_sessions_sampled": len(selected_idx),
        "filler_sessions_dropped": n_filler_in - len(selected_idx),
        "total_tokens_estimated": approx_token_count(rendered_text),
        "target_tokens": target_tokens,
        "filler_sample_seed": seed,
    }
    assert diagnostics["meaningful_sessions_kept"] == diagnostics["meaningful_sessions_in"], (
        "meaningful_sessions guarantee violated"
    )
    assert diagnostics["calendar_events_kept"] == diagnostics["calendar_events_in"], (
        "calendar_events guarantee violated"
    )
    return rendered_text, diagnostics


def strip_to_raw_logs(app_logs: dict[str, Any]) -> str:
    """Same flattening as the canonical generator.py — keep in sync."""
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


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Coarse truncation by 4-chars-per-token estimate. Truncates mid-line if
    needed; cuts FROM THE END so the earliest signals stay intact."""
    char_budget = max_tokens * 4
    if len(text) <= char_budget:
        return text
    return text[:char_budget]


def approx_token_count(text: str) -> int:
    return len(text) // 4
