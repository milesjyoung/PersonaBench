"""Pipeline runner driving the six steps via subscription CLI or API.

Supports cheap subscription backends (`claude`, `codex`) and metered API
backends (`anthropic-api`, `openai-api`). Each subscription inference call is
routed to a fresh CLI invocation so context does not leak between calls.

Usage:
    # Single persona, steps 2-6
    python openclaw_pipeline.py --seed step1_seed/data_samples/output/julio_simmons_seed.json

    # All 5 personas end-to-end
    python openclaw_pipeline.py --all --start 2 --stop 6

    # Just Step 4 (app log synthesis) for one persona
    python openclaw_pipeline.py --seed ... --start 4 --stop 4

Requires the selected CLI on PATH, or the selected API key in the environment.
Set CLAUDE_CMD or CODEX_CMD to override CLI paths.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from llm import (
    API_BACKENDS,
    SUBSCRIPTION_BACKENDS,
    SUPPORTED_BACKENDS,
    SUPPORTED_PROVIDERS,
    call_llm,
    call_subscription_cli,
    check_api_key,
    make_client,
    provider_for_backend,
)

REPO_ROOT = Path(__file__).parent
CLAUDE_CMD = os.environ.get(
    "CLAUDE_CMD", "claude.cmd" if sys.platform == "win32" else "claude"
)
CODEX_CMD = os.environ.get("CODEX_CMD", "codex.exe" if sys.platform == "win32" else "codex")

STEP1_DIR = REPO_ROOT / "step1_seed"
STEP2_DIR = REPO_ROOT / "step2_interview"
STEP3_DIR = REPO_ROOT / "step3_social_circle"
STEP4_DIR = REPO_ROOT / "step4_app_log_synthesizer"
STEP5_DIR = REPO_ROOT / "step5_testcases_synthesis"
STEP6_DIR = REPO_ROOT / "step6_benchmark_runner"


class ModelCaller:
    def __init__(self, backend: str, provider: str = "anthropic") -> None:
        self.backend = backend
        self.provider = provider_for_backend(backend, provider) if backend in API_BACKENDS else provider
        self.client = make_client(self.provider) if backend in API_BACKENDS else None

    def call(self, prompt: str, model: str | None = None) -> str:
        if self.backend in SUBSCRIPTION_BACKENDS:
            return call_subscription_cli(
                prompt, model, self.backend,
                claude_cmd=CLAUDE_CMD, codex_cmd=CODEX_CMD,
            )
        if self.client is None:
            raise RuntimeError("API client is not initialized")
        if model is None:
            raise ValueError("API backend requires an explicit model")
        return call_llm(self.client, model, prompt, provider=self.provider)


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    payload = fenced.group(1) if fenced else text
    start = payload.find("{")
    if start == -1:
        raise ValueError("No JSON object found in CLI output")
    return json.loads(payload[start:])


# ----- helpers --------------------------------------------------------------

def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill(template: str, placeholder: str, payload: Any) -> str:
    if isinstance(payload, str):
        return template.replace(placeholder, payload)
    return template.replace(
        placeholder, json.dumps(payload, indent=2, ensure_ascii=False)
    )


def extract_name(seed: dict) -> str:
    persona_text = seed.get("persona", "") or seed.get("professional_persona", "")
    match = re.match(r"\s*([A-Z][a-z]+\s+[A-Z][a-z]+)", persona_text)
    return match.group(1) if match else seed.get("uuid", "persona")


def base(name: str) -> str:
    return name.lower().replace(" ", "_")


def fill_seed_placeholders(template: str, seed: dict) -> str:
    mapping = {
        "{{participant_id}}": seed.get("uuid", ""),
        "{{name}}": extract_name(seed),
        "{{age}}": str(seed.get("age", "")),
        "{{sex}}": seed.get("sex", ""),
        "{{marital_status}}": seed.get("marital_status", ""),
        "{{education_level}}": seed.get("education_level", ""),
        "{{occupation}}": seed.get("occupation", ""),
        "{{city}}": seed.get("city", ""),
        "{{state}}": seed.get("state", ""),
        "{{zipcode}}": seed.get("zipcode", ""),
        "{{country}}": seed.get("country", ""),
        "{{professional_persona}}": seed.get("professional_persona", ""),
        "{{sports_persona}}": seed.get("sports_persona", ""),
        "{{arts_persona}}": seed.get("arts_persona", ""),
        "{{travel_persona}}": seed.get("travel_persona", ""),
        "{{culinary_persona}}": seed.get("culinary_persona", ""),
        "{{persona}}": seed.get("persona", ""),
        "{{cultural_background}}": seed.get("cultural_background", ""),
        "{{skills_and_expertise}}": seed.get("skills_and_expertise", ""),
        "{{hobbies_and_interests}}": seed.get("hobbies_and_interests", ""),
        "{{career_goals_and_ambitions}}": seed.get("career_goals_and_ambitions", ""),
    }
    for key, value in mapping.items():
        template = template.replace(key, value)
    return template


# ----- step 2 ---------------------------------------------------------------

def run_step2_gen(
    seed: dict, output_dir: Path, caller: ModelCaller, model: str | None = None
) -> Path:
    template = fill_seed_placeholders(load_prompt(STEP2_DIR / "prompt.txt"), seed)
    raw = caller.call(template, model=model)
    interview = extract_json(raw)
    out = output_dir / f"{base(extract_name(seed))}_interview.json"
    out.write_text(json.dumps(interview, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_step2_verify(
    seed: dict, interview: dict, output_dir: Path, caller: ModelCaller,
    model: str | None = None,
) -> Path:
    template = fill_seed_placeholders(
        load_prompt(STEP2_DIR / "verification_prompt.txt"), seed
    )
    template = fill(template, "{{INSERT_SEED_JSON_HERE}}", seed)
    template = fill(template, "{{INSERT_TRANSCRIPT_JSON_HERE}}", interview["transcript"])
    template = fill(
        template, "{{INSERT_EXTRACTED_PROFILE_JSON_HERE}}", interview["extracted_profile"]
    )
    raw = caller.call(template, model=model)
    verification = extract_json(raw)
    out = output_dir / f"{base(extract_name(seed))}_verification.json"
    out.write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ----- step 3 ---------------------------------------------------------------

def run_step3_gen(
    profile: dict, transcript: list, output_dir: Path, name: str,
    caller: ModelCaller,
    model: str | None = None,
) -> Path:
    template = load_prompt(STEP3_DIR / "prompt.txt")
    template = fill(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", profile
    )
    template = fill(template, "{{INSERT_TRANSCRIPT_JSON_HERE}}", transcript)
    template = template.replace("{{participant_id}}", profile.get("participant_id", ""))
    template = template.replace("{{name}}", profile.get("name", name))
    raw = caller.call(template, model=model)
    circle = extract_json(raw)
    out = output_dir / f"{base(name)}_social_circle.json"
    out.write_text(json.dumps(circle, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_step3_verify(
    profile: dict, transcript: list, circle: dict, output_dir: Path, name: str,
    caller: ModelCaller,
    model: str | None = None,
) -> Path:
    template = load_prompt(STEP3_DIR / "verification_prompt.txt")
    template = fill(template, "{{INSERT_TRANSCRIPT_JSON_HERE}}", transcript)
    template = fill(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", profile
    )
    template = fill(template, "{{INSERT_SOCIAL_CIRCLE_JSON_HERE}}", circle)
    template = template.replace("{{participant_id}}", profile.get("participant_id", ""))
    template = template.replace("{{name}}", profile.get("name", name))
    raw = caller.call(template, model=model)
    verification = extract_json(raw)
    out = output_dir / f"{base(name)}_social_circle_verification.json"
    out.write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ----- step 4 ---------------------------------------------------------------

def run_step4(
    profile: dict,
    corrected_social_circle: dict,
    output_dir: Path,
    name: str,
    log_start: str,
    log_end: str,
    per_fact_max_attempts: int,
    caller: ModelCaller,
    model: str | None = None,
    verifier_model: str | None = None,
) -> Path:
    """Per-fact generate-then-verify loop via the CLI, then merge.

    The reverse-inferability gate is the structural defense: each hidden fact's
    fragments must be re-recoverable by an independent verifier model that does
    not see the ground-truth label. Pass a different `verifier_model` from
    `model` so the gate is not the generator pattern-matching its own output.
    Both invocations are fresh backend CLI subprocesses, so the verifier has
    no access to the generator's prompt or response history.
    """
    if verifier_model is None:
        verifier_model = model
    hidden_facts = profile["hidden_facts"]
    persona = {
        "name": profile.get("name", ""),
        "age": profile.get("age", ""),
        "occupation": profile.get("occupation", ""),
        "location": profile.get("location", ""),
    }

    verified_fragments: list[dict[str, Any]] = []
    contact_usage: dict[str, int] = {}
    trace: list[dict[str, Any]] = []

    gen_template = load_prompt(STEP4_DIR / "prompt.txt")
    ver_template = load_prompt(STEP4_DIR / "verification_prompt.txt")
    merge_template = load_prompt(STEP4_DIR / "merge_prompt.txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{base(name)}_app_logs_trace.json"
    fragments_path = output_dir / f"{base(name)}_verified_fragments.json"

    for i, hf in enumerate(hidden_facts, start=1):
        fact_id = hf.get("fact_id", f"HF-{i:03d}")
        passed = False
        last_fragments: list[dict[str, Any]] = []
        last_verification: dict[str, Any] | None = None

        for attempt in range(1, per_fact_max_attempts + 1):
            print(
                f"[step4/{fact_id}] attempt {attempt}/{per_fact_max_attempts} "
                f"({i}/{len(hidden_facts)})"
            )
            gen_prompt = fill(gen_template, "{{INSERT_HIDDEN_FACT_JSON_HERE}}", hf)
            gen_prompt = fill(
                gen_prompt,
                "{{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}}",
                corrected_social_circle,
            )
            gen_prompt = gen_prompt.replace("{{LOG_START_DATE}}", log_start)
            gen_prompt = gen_prompt.replace("{{LOG_END_DATE}}", log_end)
            gen_prompt = fill(
                gen_prompt, "{{INSERT_CONTACT_USAGE_COUNTS_JSON_HERE}}", contact_usage
            )
            bundle = extract_json(caller.call(gen_prompt, model=model))
            fragments = bundle.get("fragments", [])

            ver_prompt = fill(ver_template, "{{INSERT_FRAGMENTS_JSON_HERE}}", fragments)
            ver_prompt = ver_prompt.replace("{{PERSONA_NAME}}", str(persona["name"]))
            ver_prompt = ver_prompt.replace("{{PERSONA_AGE}}", str(persona["age"]))
            ver_prompt = ver_prompt.replace(
                "{{PERSONA_OCCUPATION}}", str(persona["occupation"])
            )
            ver_prompt = ver_prompt.replace("{{PERSONA_LOCATION}}", str(persona["location"]))
            verification = extract_json(caller.call(ver_prompt, model=verifier_model))
            last_fragments = fragments
            last_verification = verification

            if verification.get("verdict") == "RECOVERED":
                recovered = (verification.get("candidate_label") or "").lower()
                target = (hf.get("ground_truth_label") or "").lower()
                rec_tokens = set(re.findall(r"[a-z0-9]{3,}", recovered))
                tgt_tokens = set(re.findall(r"[a-z0-9]{3,}", target))
                if tgt_tokens and len(rec_tokens & tgt_tokens) / len(tgt_tokens) >= 0.15:
                    passed = True
                    for c in bundle.get("contacts_used", []) or []:
                        contact_usage[c] = contact_usage.get(c, 0) + 1
                    break

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
        trace_path.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        fragments_path.write_text(
            json.dumps(verified_fragments, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    trace_path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fragments_path.write_text(
        json.dumps(verified_fragments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    merge_prompt = merge_template
    merge_prompt = merge_prompt.replace("{{PERSONA_NAME}}", str(persona["name"]))
    merge_prompt = merge_prompt.replace("{{PERSONA_AGE}}", str(persona["age"]))
    merge_prompt = merge_prompt.replace(
        "{{PERSONA_OCCUPATION}}", str(persona["occupation"])
    )
    merge_prompt = merge_prompt.replace("{{PERSONA_LOCATION}}", str(persona["location"]))
    merge_prompt = fill(
        merge_prompt, "{{INSERT_VERIFIED_FRAGMENTS_JSON_HERE}}", verified_fragments
    )
    merge_prompt = fill(
        merge_prompt, "{{INSERT_HIDDEN_FACTS_JSON_HERE}}", hidden_facts
    )
    merge_prompt = fill(
        merge_prompt,
        "{{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}}",
        corrected_social_circle,
    )
    merge_prompt = merge_prompt.replace("{{LOG_START_DATE}}", log_start)
    merge_prompt = merge_prompt.replace("{{LOG_END_DATE}}", log_end)
    merge_prompt = fill(merge_prompt, "{{INSERT_NEWS_EVENTS_JSON_HERE}}", [])

    app_log = extract_json(caller.call(merge_prompt, model=model))
    # Defense-in-depth: strip per-session/per-event source_fact_ids so any
    # downstream consumer that hands the raw JSON to a Pass 1 inference call
    # cannot accidentally leak fact-anchored sessions. Traceability is
    # preserved via cross_app_index.
    messenger = app_log.get("messenger", {})
    for bucket in ("meaningful_sessions", "filler_sessions", "sessions"):
        for s in messenger.get(bucket, []) or []:
            s.pop("source_fact_ids", None)
    calendar = app_log.get("calendar", {})
    if isinstance(calendar, dict):
        for e in calendar.get("events", []) or []:
            e.pop("source_fact_ids", None)
    out = output_dir / f"{base(name)}_app_logs.json"
    out.write_text(json.dumps(app_log, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ----- step 5 ---------------------------------------------------------------

def run_step5_gen(
    profile: dict,
    app_logs: dict,
    corrected_social_circle: dict,
    output_dir: Path,
    name: str,
    caller: ModelCaller,
    model: str | None = None,
) -> Path:
    template = load_prompt(STEP5_DIR / "prompt.txt")
    template = fill(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", profile
    )
    template = fill(template, "{{INSERT_APP_LOGS_JSON_HERE}}", app_logs)
    template = fill(
        template,
        "{{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}}",
        corrected_social_circle,
    )
    cases = extract_json(caller.call(template, model=model))
    out = output_dir / f"{base(name)}_test_cases.json"
    out.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_step5_verify(
    test_cases: dict,
    profile: dict,
    app_logs: dict,
    output_dir: Path,
    name: str,
    caller: ModelCaller,
    model: str | None = None,
) -> Path:
    template = load_prompt(STEP5_DIR / "verification_prompt.txt")
    template = fill(template, "{{INSERT_TEST_CASES_JSON_HERE}}", test_cases)
    template = fill(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", profile
    )
    template = fill(template, "{{INSERT_APP_LOGS_JSON_HERE}}", app_logs)
    verification = extract_json(caller.call(template, model=model))
    out = output_dir / f"{base(name)}_test_cases_verification.json"
    out.write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ----- step 6 ---------------------------------------------------------------

def run_step6(
    app_logs_path: Path, test_cases_path: Path, output_dir: Path,
    backend: str,
    provider: str,
    model_pass1: str | None = None, model_pass2: str | None = None,
) -> int:
    """Delegate to step6_benchmark_runner/generator.py with the selected backend.

    Step 6 is CLI-aware and keeps the two-pass split encapsulated. Calling it
    as a subprocess preserves cold-context behavior for each pass.
    """
    cmd = [
        sys.executable,
        str(STEP6_DIR / "generator.py"),
        "--app-logs", str(app_logs_path),
        "--test-cases", str(test_cases_path),
        "--output", str(output_dir),
        "--backend", backend,
        "--provider", provider,
        "--claude-cmd", CLAUDE_CMD,
        "--codex-cmd", CODEX_CMD,
    ]
    if model_pass1:
        cmd += ["--model-pass1", model_pass1]
    if model_pass2:
        cmd += ["--model-pass2", model_pass2]
    return subprocess.run(cmd, check=False).returncode


# ----- pipeline orchestration -----------------------------------------------

def run_pipeline(
    seed_path: Path,
    caller: ModelCaller,
    backend: str,
    provider: str,
    start: int = 2,
    stop: int = 6,
    log_start: str = "2026-03-01",
    log_end: str = "2026-03-31",
    per_fact_max_attempts: int = 3,
    model: str | None = None,
    verifier_model: str | None = None,
    judge_model: str | None = None,
) -> None:
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    name = extract_name(seed)
    n = base(name)
    print(f"=== Pipeline for {name} (steps {start}-{stop}) ===")

    step2_out = STEP2_DIR / "data_samples" / "output"
    step3_out = STEP3_DIR / "data_samples" / "output"
    step4_out = STEP4_DIR / "data_samples" / "output"
    step5_out = STEP5_DIR / "data_samples" / "output"
    step6_out = STEP6_DIR / "data_samples" / "output"
    for d in (step2_out, step3_out, step4_out, step5_out, step6_out):
        d.mkdir(parents=True, exist_ok=True)

    # Load or generate Step 2 outputs
    if start <= 2 <= stop:
        interview_path = run_step2_gen(seed, step2_out, caller, model=model)
        interview = json.loads(interview_path.read_text(encoding="utf-8"))
        verification_path = run_step2_verify(seed, interview, step2_out, caller, model=model)
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    else:
        interview = json.loads((step2_out / f"{n}_interview.json").read_text(encoding="utf-8"))
        verification = json.loads((step2_out / f"{n}_verification.json").read_text(encoding="utf-8"))

    profile = verification.get("corrected_extracted_profile", verification)
    transcript = interview.get("transcript", interview)

    # Step 3
    if start <= 3 <= stop:
        circle_path = run_step3_gen(profile, transcript, step3_out, name, caller, model=model)
        circle = json.loads(circle_path.read_text(encoding="utf-8"))
        run_step3_verify(profile, transcript, circle, step3_out, name, caller, model=model)

    circle_verification_path = step3_out / f"{n}_social_circle_verification.json"
    circle_verification = json.loads(circle_verification_path.read_text(encoding="utf-8"))
    corrected_social_circle = circle_verification.get(
        "corrected_social_circle", circle_verification
    )

    # Step 4
    if start <= 4 <= stop:
        run_step4(
            profile, corrected_social_circle, step4_out, name,
            log_start, log_end, per_fact_max_attempts,
            caller, model=model, verifier_model=verifier_model,
        )

    app_logs_path = step4_out / f"{n}_app_logs.json"

    # Step 5
    if start <= 5 <= stop:
        if not app_logs_path.exists():
            print(f"[step5] skipping — {app_logs_path.name} not found")
        else:
            app_logs = json.loads(app_logs_path.read_text(encoding="utf-8"))
            test_cases_path = run_step5_gen(
                profile, app_logs, corrected_social_circle, step5_out, name,
                caller, model=model,
            )
            test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))
            run_step5_verify(test_cases, profile, app_logs, step5_out, name, caller, model=model)

    test_cases_path = step5_out / f"{n}_test_cases.json"

    # Step 6
    if start <= 6 <= stop:
        if not app_logs_path.exists() or not test_cases_path.exists():
            print("[step6] skipping — required inputs missing")
        else:
            run_step6(
                app_logs_path, test_cases_path, step6_out, backend, provider,
                model_pass1=model,
                model_pass2=judge_model or model,
            )

    print(f"=== Pipeline complete for {name} ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=Path, help="Path to a seed JSON")
    parser.add_argument(
        "--persona", type=str, default=None,
        help="Persona name shorthand (e.g. julio_simmons); resolves to "
        "step1_seed/data_samples/output/<persona>_seed.json. Mutually "
        "exclusive with --seed and --all.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run against every seed JSON in step1_seed/"
    )
    parser.add_argument("--start", type=int, default=2, choices=[2, 3, 4, 5, 6])
    parser.add_argument("--stop", type=int, default=6, choices=[2, 3, 4, 5, 6])
    parser.add_argument("--log-start", default="2026-03-01")
    parser.add_argument("--log-end", default="2026-03-31")
    parser.add_argument("--per-fact-max-attempts", type=int, default=3)
    parser.add_argument(
        "--backend",
        default=None,
        choices=SUBSCRIPTION_BACKENDS + ("api",) + API_BACKENDS,
        help="Model backend: claude/codex for subscription CLI, or "
        "anthropic-api/openai-api for metered API calls. `api` uses --provider.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=SUPPORTED_PROVIDERS,
        help="API provider used when --backend api is selected, or as a "
        "backward-compatible shortcut for --backend <provider>-api.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Generator model passed to every backend CLI call. MUST be an "
        "exact pinned string, never an alias. Defaults to claude-opus-4-7 for "
        "the Claude backend and gpt-5.5 for the Codex backend.",
    )
    parser.add_argument(
        "--verifier-model", default=None,
        help="Verifier model for Step 4's reverse-inferability gate. MUST be "
        "an exact pinned string. Should differ from --model when possible so "
        "the verifier is independent.",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Judge model for Step 6 Pass 2 scoring. MUST be an exact pinned "
        "string. Should differ from --model when possible to keep the scorer "
        "independent of the answerer.",
    )
    args = parser.parse_args()

    if args.backend is None:
        if args.provider:
            args.backend = f"{args.provider}-api"
        else:
            args.backend = os.environ.get("PERSONABENCH_BACKEND", "claude")
    elif args.backend == "api":
        args.backend = f"{args.provider or 'anthropic'}-api"
    provider = provider_for_backend(args.backend, args.provider or "anthropic") if args.backend in API_BACKENDS else (args.provider or "anthropic")

    if args.model is None:
        if args.backend in {"claude", "anthropic-api"}:
            args.model = "claude-opus-4-7"
        else:
            args.model = "gpt-5.5"
    if args.verifier_model is None:
        if args.backend in {"claude", "anthropic-api"}:
            args.verifier_model = "claude-sonnet-4-6"
        else:
            args.verifier_model = "gpt-5"
    if args.judge_model is None:
        if args.backend in {"claude", "anthropic-api"}:
            args.judge_model = "claude-sonnet-4-6"
        else:
            args.judge_model = "gpt-5.4"

    # Reject obvious aliases. CLIs may resolve aliases internally but the
    # artifact metadata won't record which version was actually used, which
    # breaks the reproducibility contract.
    for flag, val in [
        ("--model", args.model),
        ("--verifier-model", args.verifier_model),
        ("--judge-model", args.judge_model),
    ]:
        if val in {"opus", "sonnet", "haiku"}:
            parser.error(
                f"{flag} {val!r} is an alias. Use the exact pinned model "
                f"string (e.g. claude-opus-4-7). Aliases drift across CLI "
                f"versions and break reproducibility."
            )

    if args.start > args.stop:
        parser.error("--start must be <= --stop")

    if args.backend in API_BACKENDS and not check_api_key(provider):
        parser.error(f"{provider.upper()}_API_KEY is not set")

    caller = ModelCaller(args.backend, provider=provider)

    # Resolve --persona to --seed
    if args.persona:
        if args.seed or args.all:
            parser.error("--persona is mutually exclusive with --seed and --all")
        args.seed = STEP1_DIR / "data_samples" / "output" / f"{args.persona}_seed.json"
        if not args.seed.exists():
            parser.error(f"Persona seed not found: {args.seed}")

    if args.all:
        seed_dir = STEP1_DIR / "data_samples" / "output"
        for seed_file in sorted(seed_dir.glob("*_seed.json")):
            run_pipeline(
                seed_file, caller, args.backend, provider, args.start, args.stop,
                args.log_start, args.log_end, args.per_fact_max_attempts,
                model=args.model,
                verifier_model=args.verifier_model,
                judge_model=args.judge_model,
            )
    elif args.seed:
        run_pipeline(
            args.seed, caller, args.backend, provider, args.start, args.stop,
            args.log_start, args.log_end, args.per_fact_max_attempts,
            model=args.model,
            verifier_model=args.verifier_model,
            judge_model=args.judge_model,
        )
    else:
        parser.error("Provide --persona <name>, --seed <path>, or --all")


if __name__ == "__main__":
    main()
