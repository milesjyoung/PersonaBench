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

import anthropic

STEP_DIR = Path(__file__).parent
PROMPT_PATH = STEP_DIR / "prompt.txt"
VERIFICATION_PROMPT_PATH = STEP_DIR / "verification_prompt.txt"
MERGE_PROMPT_PATH = STEP_DIR / "merge_prompt.txt"

DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_VERIFIER_MODEL = "claude-sonnet-4-6"  # cold, independent verifier
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


def call_llm(client: anthropic.Anthropic, model: str, prompt: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=32_000,
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


# ---- per-fact generation + verification ----------------------------------


def generate_fragments(
    client: anthropic.Anthropic,
    model: str,
    hidden_fact: dict[str, Any],
    social_circle: dict[str, Any],
    log_start: str,
    log_end: str,
    contact_usage: dict[str, int],
) -> dict[str, Any]:
    template = load_prompt(PROMPT_PATH)
    template = fill(template, "{{INSERT_HIDDEN_FACT_JSON_HERE}}", hidden_fact)
    template = fill(
        template, "{{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}}", social_circle
    )
    template = fill(template, "{{LOG_START_DATE}}", log_start)
    template = fill(template, "{{LOG_END_DATE}}", log_end)
    template = fill(template, "{{INSERT_CONTACT_USAGE_COUNTS_JSON_HERE}}", contact_usage)
    raw = call_llm(client, model, template)
    return extract_json(raw)


def verify_fragments(
    client: anthropic.Anthropic,
    verifier_model: str,
    fragments: list[dict[str, Any]],
    persona: dict[str, Any],
) -> dict[str, Any]:
    template = load_prompt(VERIFICATION_PROMPT_PATH)
    template = fill(template, "{{INSERT_FRAGMENTS_JSON_HERE}}", fragments)
    template = fill(template, "{{PERSONA_NAME}}", persona["name"])
    template = fill(template, "{{PERSONA_AGE}}", str(persona["age"]))
    template = fill(template, "{{PERSONA_OCCUPATION}}", persona["occupation"])
    template = fill(template, "{{PERSONA_LOCATION}}", persona["location"])
    raw = call_llm(client, verifier_model, template)
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
    # Cheap semantic proxy: significant token overlap.
    rec_tokens = set(re.findall(r"[a-z0-9]{3,}", recovered))
    tgt_tokens = set(re.findall(r"[a-z0-9]{3,}", target))
    if not tgt_tokens:
        return False
    overlap = len(rec_tokens & tgt_tokens) / len(tgt_tokens)
    return overlap >= 0.4


def update_contact_usage(
    contact_usage: dict[str, int], fragment_bundle: dict[str, Any]
) -> None:
    for name in fragment_bundle.get("contacts_used", []) or []:
        contact_usage[name] = contact_usage.get(name, 0) + 1


# ---- merge ---------------------------------------------------------------


def merge_app_log(
    client: anthropic.Anthropic,
    model: str,
    verified_fragments: list[dict[str, Any]],
    hidden_facts: list[dict[str, Any]],
    social_circle: dict[str, Any],
    persona: dict[str, Any],
    log_start: str,
    log_end: str,
    news_events: list[dict[str, Any]],
) -> dict[str, Any]:
    template = load_prompt(MERGE_PROMPT_PATH)
    template = fill(template, "{{PERSONA_NAME}}", persona["name"])
    template = fill(template, "{{PERSONA_AGE}}", str(persona["age"]))
    template = fill(template, "{{PERSONA_OCCUPATION}}", persona["occupation"])
    template = fill(template, "{{PERSONA_LOCATION}}", persona["location"])
    template = fill(
        template, "{{INSERT_VERIFIED_FRAGMENTS_JSON_HERE}}", verified_fragments
    )
    template = fill(template, "{{INSERT_HIDDEN_FACTS_JSON_HERE}}", hidden_facts)
    template = fill(
        template, "{{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}}", social_circle
    )
    template = fill(template, "{{LOG_START_DATE}}", log_start)
    template = fill(template, "{{LOG_END_DATE}}", log_end)
    template = fill(template, "{{INSERT_NEWS_EVENTS_JSON_HERE}}", news_events)
    raw = call_llm(client, model, template)
    return sanitize_app_log(extract_json(raw))


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
) -> int:
    profile_file = json.loads(profile_path.read_text(encoding="utf-8"))
    corrected_profile = profile_file["corrected_extracted_profile"]
    hidden_facts = corrected_profile["hidden_facts"]

    circle_file = json.loads(social_circle_path.read_text(encoding="utf-8"))
    corrected_social_circle = circle_file["corrected_social_circle"]

    news_events: list[dict[str, Any]] = []
    if news_events_path and news_events_path.exists():
        news_events = json.loads(news_events_path.read_text(encoding="utf-8"))

    client = anthropic.Anthropic()
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
                log_start, log_end, contact_usage,
            )
            fragments = bundle.get("fragments", [])
            verification = verify_fragments(client, verifier_model, fragments, persona)
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
        corrected_social_circle, persona, log_start, log_end, news_events,
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
    parser.add_argument("--log-start", default="2026-02-20")
    parser.add_argument("--log-end", default="2026-04-20")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
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
        )
    )


if __name__ == "__main__":
    main()
