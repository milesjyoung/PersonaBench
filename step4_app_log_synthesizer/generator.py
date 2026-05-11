"""Step 4 — App log synthesis with per-fact reverse-inferability gate.

Each hidden fact is converted into 2-3 implicit fragments; those fragments
are fed to an independent cold LLM that tries to infer the fact without
seeing the ground truth. If the inference succeeds, the fragments are
accepted. If not, the fragments are regenerated. Only once all facts pass
the gate are the fragments merged with filler into the final app log.

Three phases:
  1. Per-fact fragment generation     (prompt.txt)
  2. Per-fact reverse-inferability    (verification_prompt.txt)
  3. Global merge + filler + surprises (merge_prompt.txt)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm import (
    API_BACKENDS, SUBSCRIPTION_BACKENDS, SUPPORTED_BACKENDS,
    make_client, call_llm as _call_llm, call_subscription_cli,
    check_api_key, provider_for_backend, default_model_for_backend,
    SUPPORTED_PROVIDERS,
)

STEP_DIR = Path(__file__).parent
PROMPT_PATH = STEP_DIR / "prompt.txt"
VERIFICATION_PROMPT_PATH = STEP_DIR / "verification_prompt.txt"
MERGE_PROMPT_PATH = STEP_DIR / "merge_prompt.txt"

DEFAULT_MODEL = None
DEFAULT_VERIFIER_MODEL = None
DEFAULT_PER_FACT_MAX_ATTEMPTS = 3
DEFAULT_LOG_WINDOW_DAYS = 60


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill(template: str, placeholder: str, payload: Any) -> str:
    if isinstance(payload, str):
        return template.replace(placeholder, payload)
    return template.replace(
        placeholder, json.dumps(payload, indent=2, ensure_ascii=False)
    )


CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude.cmd" if sys.platform == "win32" else "claude")
CODEX_CMD = os.environ.get("CODEX_CMD", "codex.exe" if sys.platform == "win32" else "codex")


def call_llm(client, model: str, prompt: str, provider: str = "anthropic",
             backend: str = "anthropic-api") -> str:
    if backend in SUBSCRIPTION_BACKENDS:
        return call_subscription_cli(prompt, model, backend,
                                     claude_cmd=CLAUDE_CMD, codex_cmd=CODEX_CMD)
    return _call_llm(client, model, prompt, provider=provider)


def _fix_invalid_escapes(text: str) -> str:
    VALID = set('"\\/bfnrtu')
    out = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            if text[i + 1] in VALID:
                out.append(text[i])
                out.append(text[i + 1])
                i += 2
            else:
                out.append(text[i + 1])
                i += 2
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)


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


# ---- per-fact generation + verification ----------------------------------


def generate_fragments(
    client,
    model: str,
    hidden_fact: dict[str, Any],
    social_circle: dict[str, Any],
    log_start: str,
    log_end: str,
    contact_usage: dict[str, int],
    provider: str = "anthropic",
    backend: str = "anthropic-api",
) -> dict[str, Any]:
    template = load_prompt(PROMPT_PATH)
    template = fill(template, "{{INSERT_HIDDEN_FACT_JSON_HERE}}", hidden_fact)
    template = fill(
        template, "{{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}}", social_circle
    )
    template = fill(template, "{{LOG_START_DATE}}", log_start)
    template = fill(template, "{{LOG_END_DATE}}", log_end)
    template = fill(template, "{{INSERT_CONTACT_USAGE_COUNTS_JSON_HERE}}", contact_usage)
    raw = call_llm(client, model, template, provider=provider, backend=backend)
    return extract_json(raw)


def verify_fragments(
    client,
    verifier_model: str,
    fragments: list[dict[str, Any]],
    banned_token_set: list[str],
    persona: dict[str, Any],
    provider: str = "anthropic",
    backend: str = "anthropic-api",
) -> dict[str, Any]:
    verifier_input = {
        "fragments": fragments,
        "banned_token_set": banned_token_set,
    }
    template = load_prompt(VERIFICATION_PROMPT_PATH)
    template = fill(template, "{{INSERT_FRAGMENTS_JSON_HERE}}", verifier_input)
    template = fill(template, "{{PERSONA_NAME}}", persona["name"])
    template = fill(template, "{{PERSONA_AGE}}", str(persona["age"]))
    template = fill(template, "{{PERSONA_OCCUPATION}}", persona["occupation"])
    template = fill(template, "{{PERSONA_LOCATION}}", persona["location"])
    raw = call_llm(client, verifier_model, template, provider=provider, backend=backend)
    return extract_json(raw)


def fact_passed(
    verification: dict[str, Any],
    hidden_fact: dict[str, Any],
) -> bool:
    if verification.get("verdict") != "RECOVERED":
        return False
    recovered = (verification.get("candidate_label") or "").lower()
    target = (hidden_fact.get("ground_truth_label") or "").lower()
    if not recovered or not target:
        return False
    # Cheap semantic proxy: any meaningful token overlap. The new Step 4
    # prompts ban ground_truth_label tokens from fragments, so the verifier
    # must paraphrase even when correct. A correct paraphrase ("daily SSRI
    # at 100mg") carries 1-2 ground-truth tokens, so the threshold is 0.15.
    # This is a hallucination safety net only; the verifier's RECOVERED
    # verdict + structure_check + triviality_check do the real gate work.
    rec_tokens = set(re.findall(r"[a-z0-9]{3,}", recovered))
    tgt_tokens = set(re.findall(r"[a-z0-9]{3,}", target))
    if not tgt_tokens:
        return False
    overlap = len(rec_tokens & tgt_tokens) / len(tgt_tokens)
    return overlap >= 0.15


def update_contact_usage(
    contact_usage: dict[str, int], fragment_bundle: dict[str, Any]
) -> None:
    for name in fragment_bundle.get("contacts_used", []) or []:
        contact_usage[name] = contact_usage.get(name, 0) + 1


# ---- merge (deterministic assembly + per-contact filler LLM calls) -------


FILLER_PROMPT_TEMPLATE = """Generate {count} filler messenger sessions between {persona_name} and {contact_name} spread across these dates: {dates}.

Contact voice profile:
- Relationship: {relationship}
- Communication style: {comm_style}
- Typical topics: {topics}

Rules:
- Each session is 2-6 mundane messages (3-15 words each).
- Content: logistics, weather, food, pets, running late, errands, casual check-ins.
- NO hidden facts, NO deep reflection, NO essay-style messages.
- {contact_name}'s messages must match their voice profile above.
- Vary openers across sessions. Do not recycle greetings.
- Use realistic timestamps (HH:MM) within each session.

Return a JSON array of sessions:
[
  {{
    "date": "YYYY-MM-DD",
    "contact": "{contact_name}",
    "messages": [
      {{"time": "HH:MM", "sender": "...", "text": "..."}}
    ]
  }}
]

Return ONLY the JSON array, no other text."""


def _group_fragments_into_sessions(
    fragments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messenger_groups: dict[str, list[dict[str, Any]]] = {}
    calendar_events: list[dict[str, Any]] = []

    for frag in fragments:
        if frag.get("type") == "calendar" and frag.get("calendar_event"):
            evt = dict(frag["calendar_event"])
            evt["date"] = frag["date"]
            calendar_events.append(evt)
        elif frag.get("type") == "messenger" and frag.get("messages"):
            key = f"{frag['date']}|{frag.get('source', {}).get('contact_name', 'unknown')}"
            messenger_groups.setdefault(key, []).append(frag)

    sessions = []
    for key, frags in sorted(messenger_groups.items()):
        date, contact = key.split("|", 1)
        messages = []
        for f in frags:
            messages.extend(f.get("messages", []))
        messages.sort(key=lambda m: m.get("time", ""))
        sessions.append({
            "date": date,
            "contact": contact,
            "messages": messages,
        })

    sessions.sort(key=lambda s: (s["date"], (s["messages"] or [{}])[0].get("time", "")))
    calendar_events.sort(key=lambda e: (e.get("date", ""), e.get("start_time", "")))
    return sessions, calendar_events


def _compute_filler_budget(
    meaningful_sessions: list[dict[str, Any]],
    contacts: list[str],
    ratio: float = 2.5,
) -> dict[str, int]:
    total_filler_sessions = max(1, int(len(meaningful_sessions) * ratio))

    contact_meaningful = {}
    for s in meaningful_sessions:
        c = s.get("contact", "unknown")
        contact_meaningful[c] = contact_meaningful.get(c, 0) + 1
    total_meaningful = max(len(meaningful_sessions), 1)

    budget: dict[str, int] = {}
    for c in contacts:
        share = contact_meaningful.get(c, 0) / total_meaningful
        if share > 0.35:
            budget[c] = max(1, int(total_filler_sessions * 0.15))
        elif share < 0.10:
            budget[c] = max(3, int(total_filler_sessions * 0.25))
        else:
            budget[c] = max(1, int(total_filler_sessions / len(contacts)))

    assigned = sum(budget.values())
    if assigned < total_filler_sessions and contacts:
        budget[contacts[0]] += total_filler_sessions - assigned

    return budget


def _dates_for_contact(
    meaningful_sessions: list[dict[str, Any]],
    contact: str,
    log_start: str,
    log_end: str,
) -> list[str]:
    from datetime import date, timedelta
    start = date.fromisoformat(log_start)
    end = date.fromisoformat(log_end)
    meaningful_dates = {
        s["date"] for s in meaningful_sessions if s.get("contact") == contact
    }
    all_dates = []
    d = start
    while d <= end:
        ds = d.isoformat()
        if ds not in meaningful_dates:
            all_dates.append(ds)
        d += timedelta(days=1)
    return all_dates


def _generate_filler_for_contact(
    client,
    model: str,
    persona_name: str,
    contact: dict[str, Any],
    count: int,
    available_dates: list[str],
    provider: str = "anthropic",
    backend: str = "anthropic-api",
) -> list[dict[str, Any]]:
    contact_name = contact["name"]
    relationship = contact.get("relationship", "friend")
    personality = contact.get("personality_mini_profile", {})
    comm_style = personality.get("communication_style", "casual texting")
    topics = ", ".join(contact.get("recent_text_topics", [])[:3]) or "daily life"

    sample_dates = available_dates[:count] if len(available_dates) >= count else (
        available_dates * ((count // max(len(available_dates), 1)) + 1)
    )[:count]
    dates_str = ", ".join(sample_dates[:15])
    if len(sample_dates) > 15:
        dates_str += f" (and {len(sample_dates) - 15} more)"

    prompt = FILLER_PROMPT_TEMPLATE.format(
        count=count,
        persona_name=persona_name,
        contact_name=contact_name,
        dates=dates_str,
        relationship=relationship,
        comm_style=comm_style,
        topics=topics,
    )

    raw = call_llm(client, model, prompt, provider=provider, backend=backend)
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    payload = fenced.group(1) if fenced else raw
    start = payload.find("[")
    if start == -1:
        return []
    raw_json = payload[start:]
    try:
        sessions = json.loads(raw_json)
    except json.JSONDecodeError:
        try:
            sessions = json.loads(_fix_invalid_escapes(raw_json))
        except json.JSONDecodeError:
            return []

    if not isinstance(sessions, list):
        return []
    for s in sessions:
        s["contact"] = contact_name
    return sessions


def _build_hidden_facts_registry(
    hidden_facts: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trace_by_id = {t["fact_id"]: t for t in trace}
    frags_by_fact = {}
    for f in fragments:
        fid = f.get("source_fact_id", "")
        frags_by_fact.setdefault(fid, []).append(f)

    registry = []
    for hf in hidden_facts:
        fid = hf.get("fact_id", "")
        t = trace_by_id.get(fid, {})
        fact_frags = frags_by_fact.get(fid, [])

        chat_signals = {}
        calendar_signals = []
        for f in fact_frags:
            if f.get("type") == "messenger":
                contact = f.get("source", {}).get("contact_name", "unknown")
                chat_signals[contact] = f.get("implicit_signal_explanation", "")[:120]
            elif f.get("type") == "calendar":
                calendar_signals.append(f.get("implicit_signal_explanation", "")[:120])

        registry.append({
            "fact_id": fid,
            "ground_truth_label": hf.get("ground_truth_label", ""),
            "category": hf.get("category", ""),
            "source_subcategory": hf.get("source_subcategory", ""),
            "verification": {
                "reverse_inferable": t.get("passed", False),
                "recovered_label": (t.get("final_verification") or {}).get("candidate_label", ""),
                "reviewer_model": "cold independent verifier",
                "fragments_used": [f.get("fragment_id", "") for f in fact_frags],
            },
            "embedding_strategy": {
                "chat_signals": chat_signals,
                "calendar_signals": calendar_signals,
            },
        })
    return registry


def _build_cross_app_index(
    meaningful_sessions: list[dict[str, Any]],
    calendar_events: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    frags_by_date: dict[str, set] = {}
    for f in fragments:
        d = f.get("date", "")
        fid = f.get("source_fact_id", "")
        if d and fid:
            frags_by_date.setdefault(d, set()).add(fid)

    index: dict[str, dict[str, list[str]]] = {}
    all_dates = set()
    for s in meaningful_sessions:
        all_dates.add(s.get("date", ""))
    for e in calendar_events:
        all_dates.add(e.get("date", ""))

    for d in sorted(all_dates):
        if not d:
            continue
        m_ids = [s["session_id"] for s in meaningful_sessions if s.get("date") == d]
        e_ids = [e["event_id"] for e in calendar_events if e.get("date") == d]
        hf_ids = sorted(frags_by_date.get(d, set()))
        index[d] = {
            "messenger_session_ids": m_ids,
            "calendar_event_ids": e_ids,
            "hidden_fact_ids_touched": hf_ids,
        }
    return index


def merge_app_log(
    client,
    model: str,
    verified_fragments: list[dict[str, Any]],
    hidden_facts: list[dict[str, Any]],
    social_circle: dict[str, Any],
    persona: dict[str, Any],
    log_start: str,
    log_end: str,
    news_events: list[dict[str, Any]],
    provider: str = "anthropic",
    backend: str = "anthropic-api",
    trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    print("[merge] assembling fragments into sessions (deterministic)")
    meaningful_sessions, calendar_events = _group_fragments_into_sessions(
        verified_fragments
    )

    for i, s in enumerate(meaningful_sessions, 1):
        s["session_id"] = f"M-{i:03d}"
    for i, e in enumerate(calendar_events, 1):
        e["event_id"] = f"E-{i:03d}"

    contacts_list = social_circle.get("social_circle", social_circle.get("contacts", []))
    contact_names = [c["name"] for c in contacts_list]

    budget = _compute_filler_budget(meaningful_sessions, contact_names)
    print(f"[merge] filler budget: {sum(budget.values())} sessions across {len(budget)} contacts")

    filler_sessions: list[dict[str, Any]] = []
    batch_size = 50
    for contact_data in contacts_list:
        cname = contact_data["name"]
        count = budget.get(cname, 0)
        if count == 0:
            continue
        available = _dates_for_contact(meaningful_sessions, cname, log_start, log_end)
        contact_filler: list[dict[str, Any]] = []
        remaining = count
        batch_num = 0
        while remaining > 0:
            batch_num += 1
            this_batch = min(remaining, batch_size)
            print(f"[merge] {cname} batch {batch_num}: {this_batch} sessions ({remaining} remaining)")
            sessions = _generate_filler_for_contact(
                client, model, persona["name"], contact_data, this_batch, available,
                provider=provider, backend=backend,
            )
            contact_filler.extend(sessions)
            remaining -= this_batch
        filler_sessions.extend(contact_filler)
        print(f"[merge] got {len(contact_filler)} filler sessions for {cname}")

    filler_sessions.sort(
        key=lambda s: (s.get("date", ""), (s.get("messages") or [{}])[0].get("time", ""))
    )
    for i, s in enumerate(filler_sessions, 1):
        s["session_id"] = f"F-{i:03d}"

    print("[merge] building metadata")
    hf_registry = _build_hidden_facts_registry(
        hidden_facts, trace or [], verified_fragments
    )
    cross_app = _build_cross_app_index(
        meaningful_sessions, calendar_events, verified_fragments
    )

    m_tokens = sum(len(json.dumps(s)) for s in meaningful_sessions) // 4
    f_tokens = sum(len(json.dumps(s)) for s in filler_sessions) // 4
    actual_ratio = f_tokens / max(m_tokens, 1)

    app_log = {
        "persona_name": persona["name"],
        "log_window": {"start_date": log_start, "end_date": log_end},
        "messenger": {
            "meaningful_sessions": meaningful_sessions,
            "filler_sessions": filler_sessions,
        },
        "calendar": {"events": calendar_events},
        "hidden_facts": hf_registry,
        "cross_app_index": cross_app,
        "token_stats": {
            "approximate_tokens": m_tokens + f_tokens,
            "meaningful_token_count": m_tokens,
            "filler_token_count": f_tokens,
            "ratio_filler_to_meaningful": f"{actual_ratio:.1f}:1",
        },
        "surprises_woven": [],
    }

    print(
        f"[merge] done: {len(meaningful_sessions)} meaningful, "
        f"{len(filler_sessions)} filler, {len(calendar_events)} events, "
        f"ratio {actual_ratio:.1f}:1"
    )
    return sanitize_app_log(app_log)


def sanitize_app_log(app_log: dict[str, Any]) -> dict[str, Any]:
    """Defense-in-depth: strip per-session/per-event `source_fact_ids` so
    Backend C orchestrators that hand the raw JSON to a Pass 1 subagent
    cannot accidentally leak fact-anchored sessions to the evaluator.
    Traceability is preserved via `cross_app_index` only.
    """
    messenger = app_log.get("messenger", {})
    for bucket in ("meaningful_sessions", "filler_sessions", "sessions"):
        for s in messenger.get(bucket, []) or []:
            s.pop("source_fact_ids", None)
    calendar = app_log.get("calendar", {})
    if isinstance(calendar, dict):
        for e in calendar.get("events", []) or []:
            e.pop("source_fact_ids", None)
    return app_log


# ---- entry point ---------------------------------------------------------


def derive_persona_identity(corrected_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": corrected_profile.get("name", ""),
        "age": corrected_profile.get("age", 0),
        "occupation": corrected_profile.get("occupation", ""),
        "location": corrected_profile.get("location", ""),
    }


def run_step(
    profile_path: Path,
    social_circle_path: Path,
    news_events_path: Path | None,
    output_dir: Path,
    model: str,
    verifier_model: str,
    per_fact_max_attempts: int,
    log_start: str,
    log_end: str,
    provider: str = "anthropic",
    backend: str = "anthropic-api",
) -> int:
    profile_file = json.loads(profile_path.read_text(encoding="utf-8"))
    corrected_profile = profile_file["corrected_extracted_profile"]
    hidden_facts = corrected_profile["hidden_facts"]

    circle_file = json.loads(social_circle_path.read_text(encoding="utf-8"))
    corrected_social_circle = circle_file["corrected_social_circle"]

    news_events: list[dict[str, Any]] = []
    if news_events_path and news_events_path.exists():
        news_events = json.loads(news_events_path.read_text(encoding="utf-8"))

    client = make_client(provider) if backend in API_BACKENDS else None
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = profile_path.stem.replace("_verification", "")
    app_log_out = output_dir / f"{base_name}_app_logs.json"
    trace_out = output_dir / f"{base_name}_app_logs_trace.json"

    persona = derive_persona_identity(corrected_profile)

    verified_fragments: list[dict[str, Any]] = []
    contact_usage: dict[str, int] = {}
    trace: list[dict[str, Any]] = []

    print(f"[step4] persona {persona['name']}  hidden_facts={len(hidden_facts)}")

    for i, hf in enumerate(hidden_facts, start=1):
        fact_id = hf.get("fact_id", f"HF-{i:03d}")
        last_fragments: list[dict[str, Any]] = []
        last_verification: dict[str, Any] | None = None
        passed = False

        for attempt in range(1, per_fact_max_attempts + 1):
            print(
                f"[step4/{fact_id}] attempt {attempt}/{per_fact_max_attempts} "
                f"({i}/{len(hidden_facts)})"
            )
            bundle = generate_fragments(
                client, model, hf, corrected_social_circle,
                log_start, log_end, contact_usage, provider=provider, backend=backend,
            )
            fragments = bundle.get("fragments", [])
            banned_token_set = bundle.get("banned_token_set", []) or []
            verification = verify_fragments(
                client, verifier_model, fragments, banned_token_set, persona,
                provider=provider, backend=backend,
            )
            last_fragments = fragments
            last_verification = verification
            if fact_passed(verification, hf):
                passed = True
                update_contact_usage(contact_usage, bundle)
                break
            else:
                verdict = verification.get("verdict", "UNKNOWN")
                print(
                    f"[step4/{fact_id}] verdict={verdict} "
                    f"candidate={verification.get('candidate_label', '')!r} "
                    f"regenerating"
                )

        verified_fragments.extend(
            {**f, "source_fact_id": fact_id, "passed_verification": passed}
            for f in last_fragments
        )
        trace.append(
            {
                "fact_id": fact_id,
                "ground_truth_label": hf.get("ground_truth_label"),
                "attempts": attempt,
                "passed": passed,
                "final_verification": last_verification,
            }
        )
        trace_out.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    trace_out.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fragments_out = output_dir / f"{base_name}_verified_fragments.json"
    fragments_out.write_text(
        json.dumps(verified_fragments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[step4] saved {len(verified_fragments)} fragments to {fragments_out}")

    passed_count = sum(1 for t in trace if t["passed"])
    print(
        f"[step4] fragment phase done: {passed_count}/{len(hidden_facts)} "
        "hidden facts passed reverse-inferability"
    )
    if passed_count < len(hidden_facts):
        print(
            "[step4] WARNING: some hidden facts are unrecoverable from their "
            "fragments. They will still be merged but flagged in the trace."
        )

    print(f"[step4] merging fragments + filler for {persona['name']}")
    app_log = merge_app_log(
        client, model, verified_fragments, hidden_facts,
        corrected_social_circle, persona, log_start, log_end,
        news_events, provider=provider, backend=backend,
        trace=trace,
    )
    app_log_out.write_text(
        json.dumps(app_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[step4] wrote {app_log_out}")

    return 0 if passed_count == len(hidden_facts) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, required=True,
        help="Path to {persona}_verification.json from step2",
    )
    parser.add_argument(
        "--social-circle", type=Path, required=True,
        help="Path to {persona}_social_circle_verification.json from step3",
    )
    parser.add_argument(
        "--news-events", type=Path, default=None,
        help="Optional path to news events JSON for surprise weaving",
    )
    parser.add_argument(
        "--output", type=Path,
        default=STEP_DIR / "data_samples" / "output",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--verifier-model", default=DEFAULT_VERIFIER_MODEL,
        help="Independent model for the reverse-inferability gate",
    )
    parser.add_argument(
        "--per-fact-max-attempts", type=int, default=DEFAULT_PER_FACT_MAX_ATTEMPTS,
    )
    parser.add_argument("--log-start", default="2026-03-01")
    parser.add_argument("--log-end", default="2026-03-31")
    parser.add_argument("--provider", default=None, choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--backend", default=None, choices=SUPPORTED_BACKENDS)
    args = parser.parse_args()

    if args.backend is None:
        args.backend = f"{args.provider or 'anthropic'}-api" if args.provider else os.environ.get("PERSONABENCH_BACKEND", "claude")
    provider = provider_for_backend(args.backend, args.provider or "anthropic") if args.backend in API_BACKENDS else (args.provider or "anthropic")

    if args.model is None:
        args.model = default_model_for_backend(args.backend, "generator")
    if args.verifier_model is None:
        args.verifier_model = default_model_for_backend(args.backend, "verifier")

    if args.backend in API_BACKENDS and not check_api_key(provider):
        print(f"API key for {provider} is not set.", file=sys.stderr)
        sys.exit(2)

    sys.exit(
        run_step(
            args.profile,
            args.social_circle,
            args.news_events,
            args.output,
            args.model,
            args.verifier_model,
            args.per_fact_max_attempts,
            args.log_start,
            args.log_end,
            provider,
            args.backend,
        )
    )


if __name__ == "__main__":
    main()
