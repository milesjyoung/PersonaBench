"""OpenClaw-style pipeline runner driving the six steps via the `claude` CLI.

Uses the Claude Code CLI (`claude -p`) for inference, which runs against a
subscription rather than a metered API. Each inference call is routed to a
fresh CLI invocation so context does not leak between calls; for Step 6's
benchmark runner this matters especially (see --openclaw flag on
step6_benchmark_runner/generator.py for the session-delete equivalent).

Usage:
    # Single persona, steps 2-6
    python openclaw_pipeline.py --seed step1_seed/data_samples/output/julio_simmons_seed.json

    # All 5 personas end-to-end
    python openclaw_pipeline.py --all --start 2 --stop 6

    # Just Step 4 (app log synthesis) for one persona
    python openclaw_pipeline.py --seed ... --start 4 --stop 4

Requires the `claude` CLI on PATH. Set CLAUDE_CMD env var to override.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent
CLAUDE_CMD = os.environ.get(
    "CLAUDE_CMD", "claude.cmd" if sys.platform == "win32" else "claude"
)

STEP1_DIR = REPO_ROOT / "step1_seed"
STEP2_DIR = REPO_ROOT / "step2_interview"
STEP3_DIR = REPO_ROOT / "step3_social_circle"
STEP4_DIR = REPO_ROOT / "step4_app_log_synthesizer"
STEP5_DIR = REPO_ROOT / "step5_testcases_synthesis"
STEP6_DIR = REPO_ROOT / "step6_benchmark_runner"


# ----- CLI call -------------------------------------------------------------

def call_claude(prompt: str, max_retries: int = 2) -> str:
    """Call the `claude` CLI with a prompt written to a temp file.

    Uses a temp file to dodge Windows cmdline / pipe length limits on long
    prompts (the pipeline's merge prompt can exceed 100KB).
    """
    last_stderr = ""
    for attempt in range(max_retries + 1):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write(prompt)
        tmp.close()
        try:
            with open(tmp.name, "r", encoding="utf-8") as stream:
                result = subprocess.run(
                    [CLAUDE_CMD, "-p"],
                    stdin=stream,
                    capture_output=True,
                    timeout=1800,
                )
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            if result.returncode == 0 and stdout.strip():
                return stdout.strip()
            last_stderr = stderr
            if attempt < max_retries:
                print(
                    f"  retry {attempt + 1}/{max_retries} (stderr: {stderr[:200]})"
                )
        finally:
            os.unlink(tmp.name)
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

def run_step2_gen(seed: dict, output_dir: Path) -> Path:
    template = fill_seed_placeholders(load_prompt(STEP2_DIR / "prompt.txt"), seed)
    raw = call_claude(template)
    interview = extract_json(raw)
    out = output_dir / f"{base(extract_name(seed))}_interview.json"
    out.write_text(json.dumps(interview, indent=2, ensure_ascii=False))
    return out


def run_step2_verify(seed: dict, interview: dict, output_dir: Path) -> Path:
    template = fill_seed_placeholders(
        load_prompt(STEP2_DIR / "verification_prompt.txt"), seed
    )
    template = fill(template, "{{INSERT_SEED_JSON_HERE}}", seed)
    template = fill(template, "{{INSERT_TRANSCRIPT_JSON_HERE}}", interview["transcript"])
    template = fill(
        template, "{{INSERT_EXTRACTED_PROFILE_JSON_HERE}}", interview["extracted_profile"]
    )
    raw = call_claude(template)
    verification = extract_json(raw)
    out = output_dir / f"{base(extract_name(seed))}_verification.json"
    out.write_text(json.dumps(verification, indent=2, ensure_ascii=False))
    return out


# ----- step 3 ---------------------------------------------------------------

def run_step3_gen(
    profile: dict, transcript: list, output_dir: Path, name: str
) -> Path:
    template = load_prompt(STEP3_DIR / "prompt.txt")
    template = fill(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", profile
    )
    template = fill(template, "{{INSERT_TRANSCRIPT_JSON_HERE}}", transcript)
    template = template.replace("{{participant_id}}", profile.get("participant_id", ""))
    template = template.replace("{{name}}", profile.get("name", name))
    raw = call_claude(template)
    circle = extract_json(raw)
    out = output_dir / f"{base(name)}_social_circle.json"
    out.write_text(json.dumps(circle, indent=2, ensure_ascii=False))
    return out


def run_step3_verify(
    profile: dict, transcript: list, circle: dict, output_dir: Path, name: str
) -> Path:
    template = load_prompt(STEP3_DIR / "verification_prompt.txt")
    template = fill(template, "{{INSERT_TRANSCRIPT_JSON_HERE}}", transcript)
    template = fill(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", profile
    )
    template = fill(template, "{{INSERT_SOCIAL_CIRCLE_JSON_HERE}}", circle)
    template = template.replace("{{participant_id}}", profile.get("participant_id", ""))
    template = template.replace("{{name}}", profile.get("name", name))
    raw = call_claude(template)
    verification = extract_json(raw)
    out = output_dir / f"{base(name)}_social_circle_verification.json"
    out.write_text(json.dumps(verification, indent=2, ensure_ascii=False))
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
) -> Path:
    """Per-fact generate-then-verify loop via the CLI, then merge.

    Without true model independence (one CLI on one subscription), the
    reverse-inferability gate is prompt-enforced rather than model-enforced.
    The verifier subprocess still receives only the fragments, not the hidden
    fact, so its inference is a genuine test of fragment sufficiency.
    """
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
            bundle = extract_json(call_claude(gen_prompt))
            fragments = bundle.get("fragments", [])

            ver_prompt = fill(ver_template, "{{INSERT_FRAGMENTS_JSON_HERE}}", fragments)
            ver_prompt = ver_prompt.replace("{{PERSONA_NAME}}", str(persona["name"]))
            ver_prompt = ver_prompt.replace("{{PERSONA_AGE}}", str(persona["age"]))
            ver_prompt = ver_prompt.replace(
                "{{PERSONA_OCCUPATION}}", str(persona["occupation"])
            )
            ver_prompt = ver_prompt.replace("{{PERSONA_LOCATION}}", str(persona["location"]))
            verification = extract_json(call_claude(ver_prompt))
            last_fragments = fragments
            last_verification = verification

            if verification.get("verdict") == "RECOVERED":
                recovered = (verification.get("candidate_label") or "").lower()
                target = (hf.get("ground_truth_label") or "").lower()
                rec_tokens = set(re.findall(r"[a-z0-9]{3,}", recovered))
                tgt_tokens = set(re.findall(r"[a-z0-9]{3,}", target))
                if tgt_tokens and len(rec_tokens & tgt_tokens) / len(tgt_tokens) >= 0.4:
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

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{base(name)}_app_logs_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))

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

    app_log = extract_json(call_claude(merge_prompt))
    out = output_dir / f"{base(name)}_app_logs.json"
    out.write_text(json.dumps(app_log, indent=2, ensure_ascii=False))
    return out


# ----- step 5 ---------------------------------------------------------------

def run_step5_gen(
    profile: dict,
    app_logs: dict,
    corrected_social_circle: dict,
    output_dir: Path,
    name: str,
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
    cases = extract_json(call_claude(template))
    out = output_dir / f"{base(name)}_test_cases.json"
    out.write_text(json.dumps(cases, indent=2, ensure_ascii=False))
    return out


def run_step5_verify(
    test_cases: dict,
    profile: dict,
    app_logs: dict,
    output_dir: Path,
    name: str,
) -> Path:
    template = load_prompt(STEP5_DIR / "verification_prompt.txt")
    template = fill(template, "{{INSERT_TEST_CASES_JSON_HERE}}", test_cases)
    template = fill(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", profile
    )
    template = fill(template, "{{INSERT_APP_LOGS_JSON_HERE}}", app_logs)
    verification = extract_json(call_claude(template))
    out = output_dir / f"{base(name)}_test_cases_verification.json"
    out.write_text(json.dumps(verification, indent=2, ensure_ascii=False))
    return out


# ----- step 6 ---------------------------------------------------------------

def run_step6(
    app_logs_path: Path, test_cases_path: Path, output_dir: Path
) -> int:
    """Delegate to step6_benchmark_runner/generator.py with --openclaw.

    Step 6 is already CLI-aware via its --openclaw flag (session-delete
    between test cases). Calling it as a subprocess preserves that behavior
    and keeps the two-pass split encapsulated.
    """
    cmd = [
        sys.executable,
        str(STEP6_DIR / "generator.py"),
        "--app-logs", str(app_logs_path),
        "--test-cases", str(test_cases_path),
        "--output", str(output_dir),
        "--openclaw",
        "--claude-cmd", CLAUDE_CMD,
    ]
    return subprocess.run(cmd, check=False).returncode


# ----- pipeline orchestration -----------------------------------------------

def run_pipeline(
    seed_path: Path,
    start: int = 2,
    stop: int = 6,
    log_start: str = "2026-02-20",
    log_end: str = "2026-04-20",
    per_fact_max_attempts: int = 3,
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
        interview_path = run_step2_gen(seed, step2_out)
        interview = json.loads(interview_path.read_text(encoding="utf-8"))
        verification_path = run_step2_verify(seed, interview, step2_out)
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    else:
        interview = json.loads((step2_out / f"{n}_interview.json").read_text(encoding="utf-8"))
        verification = json.loads((step2_out / f"{n}_verification.json").read_text(encoding="utf-8"))

    profile = verification.get("corrected_extracted_profile", verification)
    transcript = interview.get("transcript", interview)

    # Step 3
    if start <= 3 <= stop:
        circle_path = run_step3_gen(profile, transcript, step3_out, name)
        circle = json.loads(circle_path.read_text(encoding="utf-8"))
        run_step3_verify(profile, transcript, circle, step3_out, name)

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
        )

    app_logs_path = step4_out / f"{n}_app_logs.json"

    # Step 5
    if start <= 5 <= stop:
        if not app_logs_path.exists():
            print(f"[step5] skipping — {app_logs_path.name} not found")
        else:
            app_logs = json.loads(app_logs_path.read_text(encoding="utf-8"))
            test_cases_path = run_step5_gen(
                profile, app_logs, corrected_social_circle, step5_out, name
            )
            test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))
            run_step5_verify(test_cases, profile, app_logs, step5_out, name)

    test_cases_path = step5_out / f"{n}_test_cases.json"

    # Step 6
    if start <= 6 <= stop:
        if not app_logs_path.exists() or not test_cases_path.exists():
            print("[step6] skipping — required inputs missing")
        else:
            run_step6(app_logs_path, test_cases_path, step6_out)

    print(f"=== Pipeline complete for {name} ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=Path, help="Path to a seed JSON")
    parser.add_argument(
        "--all", action="store_true", help="Run against every seed JSON in step1_seed/"
    )
    parser.add_argument("--start", type=int, default=2, choices=[2, 3, 4, 5, 6])
    parser.add_argument("--stop", type=int, default=6, choices=[2, 3, 4, 5, 6])
    parser.add_argument("--log-start", default="2026-02-20")
    parser.add_argument("--log-end", default="2026-04-20")
    parser.add_argument("--per-fact-max-attempts", type=int, default=3)
    args = parser.parse_args()

    if args.start > args.stop:
        parser.error("--start must be <= --stop")

    if args.all:
        seed_dir = STEP1_DIR / "data_samples" / "output"
        for seed_file in sorted(seed_dir.glob("*_seed.json")):
            run_pipeline(
                seed_file, args.start, args.stop,
                args.log_start, args.log_end, args.per_fact_max_attempts,
            )
    elif args.seed:
        run_pipeline(
            args.seed, args.start, args.stop,
            args.log_start, args.log_end, args.per_fact_max_attempts,
        )
    else:
        parser.error("Provide --seed <path> or --all")


if __name__ == "__main__":
    main()
