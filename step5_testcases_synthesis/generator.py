"""Step 5 -- Test case generation and verification.

Two phases:

  1. Generation     -- fills prompt with the corrected extracted profile,
                      app logs, and corrected social circle. Calls the LLM
                      and writes the test case file.
  2. Verification   -- fills verification prompt with the test cases plus
                      the profile and app logs. Calls the LLM and writes the
                      verification report and corrected test cases.

Iterative refinement loop: if the verification report contains any REJECTED
case or a non-empty coverage_deficit, the generation is re-run up to
--max-iterations times. The last attempt's outputs are always retained.
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
VERIFICATION_V1_PATH = STEP_DIR / "verification_prompt.txt"

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_ITERATIONS = 3


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_json(template: str, placeholder: str, payload: Any) -> str:
    return template.replace(
        placeholder, json.dumps(payload, indent=2, ensure_ascii=False)
    )


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


def strip_filler_for_step5(app_logs: dict[str, Any]) -> dict[str, Any]:
    """Remove filler sessions from app logs before injecting into Step 5 prompt.

    Filler sessions exist to challenge the evaluator in Step 6 (they are noise
    the model must sift through). Step 5 only needs meaningful sessions and
    hidden_facts to build evidence anchors and test cases. Stripping filler
    reduces the prompt from ~826K tokens to ~185K tokens.
    """
    trimmed = dict(app_logs)
    messenger = dict(app_logs.get("messenger", {}))
    messenger.pop("filler_sessions", None)
    trimmed["messenger"] = messenger
    trimmed.pop("cross_app_index", None)
    trimmed.pop("token_stats", None)
    trimmed.pop("verified_fragment_registry", None)
    trimmed.pop("verified_fragment_check", None)
    trimmed.pop("decoy_registry", None)
    trimmed.pop("decoy_check", None)
    trimmed.pop("_validation", None)
    trimmed.pop("surprises_woven", None)
    trimmed.pop("metadata", None)
    return trimmed


def do_call(
    prompt: str,
    model: str,
    client: Any | None,
    backend: str,
    provider: str,
    claude_cmd: str,
    codex_cmd: str,
) -> str:
    if backend in SUBSCRIPTION_BACKENDS:
        return call_subscription_cli(
            prompt, model, backend, claude_cmd=claude_cmd, codex_cmd=codex_cmd
        )
    assert client is not None
    return call_llm(client, model, prompt, provider=provider)


def run_generation(
    corrected_profile: dict[str, Any],
    app_logs: dict[str, Any],
    corrected_social_circle: dict[str, Any],
    model: str,
    client: Any | None,
    backend: str,
    provider: str,
    claude_cmd: str,
    codex_cmd: str,
) -> dict[str, Any]:
    template = load_prompt(PROMPT_V1_PATH)
    logs_for_prompt = strip_filler_for_step5(app_logs)
    template = fill_json(template, "{{INSERT_APP_LOGS_JSON_HERE}}", logs_for_prompt)
    template = fill_json(
        template,
        "{{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}}",
        corrected_social_circle,
    )
    raw = do_call(template, model, client, backend, provider, claude_cmd, codex_cmd)
    return extract_json(raw)


def run_verification(
    test_cases: dict[str, Any],
    corrected_profile: dict[str, Any],
    app_logs: dict[str, Any],
    model: str,
    client: Any | None,
    backend: str,
    provider: str,
    claude_cmd: str,
    codex_cmd: str,
) -> dict[str, Any]:
    template = load_prompt(VERIFICATION_V1_PATH)
    template = fill_json(template, "{{INSERT_TEST_CASES_JSON_HERE}}", test_cases)
    logs_for_prompt = strip_filler_for_step5(app_logs)
    template = fill_json(template, "{{INSERT_APP_LOGS_JSON_HERE}}", logs_for_prompt)
    raw = do_call(template, model, client, backend, provider, claude_cmd, codex_cmd)
    return extract_json(raw)


def verification_passed(verification: dict[str, Any]) -> bool:
    summary = verification.get("verification_metadata", {}).get("summary", {})
    rejected = summary.get("rejected", 0)
    corrected = verification.get("corrected_test_cases", {})
    coverage_deficit = corrected.get("coverage_deficit", [])
    return rejected == 0 and not coverage_deficit


def extract_base_name(profile_path: Path) -> str:
    return profile_path.stem.replace("_verification", "")


def run_step(
    profile_path: Path,
    app_logs_path: Path,
    social_circle_path: Path,
    output_dir: Path,
    model: str,
    max_iterations: int,
    backend: str,
    provider: str,
    claude_cmd: str,
    codex_cmd: str,
) -> int:
    corrected_profile_file = json.loads(profile_path.read_text(encoding="utf-8"))
    corrected_profile = corrected_profile_file["corrected_extracted_profile"]
    app_logs = json.loads(app_logs_path.read_text(encoding="utf-8"))
    social_circle_file = json.loads(social_circle_path.read_text(encoding="utf-8"))
    corrected_social_circle = social_circle_file["corrected_social_circle"]

    client = None
    if backend in API_BACKENDS:
        client = make_client(provider)

    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = extract_base_name(profile_path)
    test_cases_out = output_dir / f"{base_name}_test_cases.json"
    verification_out = output_dir / f"{base_name}_test_cases_verification.json"

    for attempt in range(1, max_iterations + 1):
        print(f"[attempt {attempt}/{max_iterations}] generating test cases for {base_name}")
        test_cases = run_generation(
            corrected_profile, app_logs, corrected_social_circle,
            model, client, backend, provider, claude_cmd, codex_cmd,
        )
        test_cases_out.write_text(
            json.dumps(test_cases, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"[attempt {attempt}/{max_iterations}] verifying test cases for {base_name}")
        verification = run_verification(
            test_cases, corrected_profile, app_logs,
            model, client, backend, provider, claude_cmd, codex_cmd,
        )
        verification_out.write_text(
            json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if verification_passed(verification):
            print(f"[attempt {attempt}/{max_iterations}] passed")
            return 0
        summary = verification.get("verification_metadata", {}).get("summary", {})
        deficit = verification.get("corrected_test_cases", {}).get("coverage_deficit", [])
        print(
            f"[attempt {attempt}/{max_iterations}] rejected={summary.get('rejected', 0)} "
            f"coverage_deficit={len(deficit)}"
        )

    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Path to a corrected profile file ({persona}_verification.json)",
    )
    parser.add_argument(
        "--app-logs",
        type=Path,
        required=True,
        help="Path to the persona's app logs file ({persona}_app_logs.json)",
    )
    parser.add_argument(
        "--social-circle",
        type=Path,
        required=True,
        help="Path to the corrected social circle file "
        "({persona}_social_circle_verification.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STEP_DIR / "data_samples" / "output",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--backend",
        default="claude",
        choices=SUBSCRIPTION_BACKENDS + API_BACKENDS,
        help="Inference backend: claude/codex for subscription CLI, "
        "anthropic-api/openai-api for metered API.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=("anthropic", "openai"),
    )
    parser.add_argument(
        "--claude-cmd",
        default=os.environ.get(
            "CLAUDE_CMD", "claude.cmd" if sys.platform == "win32" else "claude"
        ),
    )
    parser.add_argument(
        "--codex-cmd",
        default=os.environ.get(
            "CODEX_CMD", "codex"
        ),
    )
    args = parser.parse_args()

    backend = args.backend
    provider = (
        provider_for_backend(backend, args.provider or "anthropic")
        if backend in API_BACKENDS
        else (args.provider or "anthropic")
    )

    if backend in API_BACKENDS and not check_api_key(provider):
        print(
            f"{provider.upper()}_API_KEY is not set and no subscription backend was selected.",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(
        run_step(
            args.profile,
            args.app_logs,
            args.social_circle,
            args.output,
            args.model,
            args.max_iterations,
            backend,
            provider,
            args.claude_cmd,
            args.codex_cmd,
        )
    )


if __name__ == "__main__":
    main()
