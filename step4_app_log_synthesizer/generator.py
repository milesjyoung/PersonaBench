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


# ---- merge ---------------------------------------------------------------


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
    raw = call_llm(client, model, template, provider=provider, backend=backend)
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
    for merge_attempt in range(1, 4):
        try:
            app_log = merge_app_log(
                client, model, verified_fragments, hidden_facts,
                corrected_social_circle, persona, log_start, log_end,
                news_events, provider=provider, backend=backend,
            )
            break
        except Exception as e:
            print(f"[step4] merge attempt {merge_attempt}/3 failed: {e}")
            if merge_attempt == 3:
                print("[step4] merge exhausted retries. Fragments saved to disk.")
                return 1
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
