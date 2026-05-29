"""Step 3 — Social circle generation and verification.

Runs two phases:

  1. Generation     — fills prompt.txt with the corrected extracted profile and
                      interview transcript, calls the LLM, writes the draft
                      social circle.
  2. Verification   — fills verification_prompt.txt with the interview,
                      corrected profile, and draft social circle, calls the LLM,
                      writes the verification report + corrected social circle.

Iterative refinement loop: if verification returns a FAIL verdict, the
generation is re-run up to --max-iterations times.
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
    API_BACKENDS, SUBSCRIPTION_BACKENDS, SUPPORTED_BACKENDS,
    make_client, call_llm as _call_llm, call_subscription_cli,
    check_api_key, provider_for_backend, default_model_for_backend,
    SUPPORTED_PROVIDERS,
)

STEP_DIR = Path(__file__).parent
PROMPT_PATH = STEP_DIR / "prompt.txt"
VERIFICATION_PROMPT_PATH = STEP_DIR / "verification_prompt.txt"

DEFAULT_MODEL = None
DEFAULT_MAX_ITERATIONS = 3


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_json_placeholder(template: str, placeholder: str, payload: Any) -> str:
    return template.replace(placeholder, json.dumps(payload, indent=2, ensure_ascii=False))


def fill_identity_placeholders(template: str, profile: dict[str, Any]) -> str:
    return template.replace("{{participant_id}}", profile.get("participant_id", "")).replace(
        "{{name}}", profile.get("name", "")
    )


CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude.cmd" if sys.platform == "win32" else "claude")
CODEX_CMD = os.environ.get("CODEX_CMD", "codex")


def call_llm(client, model: str, prompt: str, gpt_reasoning: str, provider: str = "anthropic",
             backend: str = "anthropic-api") -> str:
    if backend in SUBSCRIPTION_BACKENDS:
        return call_subscription_cli(prompt, model, backend, gpt_reasoning,
                                     claude_cmd=CLAUDE_CMD, codex_cmd=CODEX_CMD)
    return _call_llm(client, model, prompt, gpt_reasoning, provider=provider)


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


def run_generation(
    client,
    model: str,
    profile: dict[str, Any],
    transcript: list[dict[str, Any]],
    gpt_reasoning: str,
    provider: str = "anthropic",
    backend: str = "anthropic-api",
) -> dict[str, Any]:
    template = load_prompt(PROMPT_PATH)
    template = fill_identity_placeholders(template, profile)
    template = fill_json_placeholder(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", profile
    )
    template = fill_json_placeholder(template, "{{INSERT_TRANSCRIPT_JSON_HERE}}", transcript)
    raw = call_llm(client, model, template, gpt_reasoning, provider=provider, backend=backend)
    return extract_json(raw)


def run_verification(
    client,
    model: str,
    profile: dict[str, Any],
    transcript: list[dict[str, Any]],
    social_circle: dict[str, Any],
    gpt_reasoning: str,
    provider: str = "anthropic",
    backend: str = "anthropic-api",
) -> dict[str, Any]:
    template = load_prompt(VERIFICATION_PROMPT_PATH)
    template = fill_identity_placeholders(template, profile)
    template = fill_json_placeholder(template, "{{INSERT_TRANSCRIPT_JSON_HERE}}", transcript)
    template = fill_json_placeholder(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", profile
    )
    template = fill_json_placeholder(
        template, "{{INSERT_SOCIAL_CIRCLE_JSON_HERE}}", social_circle
    )
    raw = call_llm(client, model, template, gpt_reasoning, provider=provider, backend=backend)
    return extract_json(raw)


def overall_verdict(verification: dict[str, Any]) -> str:
    return verification.get("verification", {}).get("overall_verdict", "UNKNOWN")


def run_step(
    profile_path: Path,
    transcript_path: Path,
    output_dir: Path,
    model: str,
    max_iterations: int,
    gpt_reasoning: str,
    provider: str = "anthropic",
    backend: str = "anthropic-api",
) -> int:
    verification_bundle = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = verification_bundle.get("corrected_extracted_profile", verification_bundle)

    interview = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript = interview.get("transcript", interview)

    client = make_client(provider) if backend in API_BACKENDS else None
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = profile_path.stem.replace("_verification", "")
    circle_out = output_dir / f"{base_name}_social_circle.json"
    verification_out = output_dir / f"{base_name}_social_circle_verification.json"

    for attempt in range(1, max_iterations + 1):
        print(f"[attempt {attempt}/{max_iterations}] generating social circle for {base_name}")
        social_circle = run_generation(client, model, profile, transcript, gpt_reasoning, provider=provider, backend=backend)
        circle_out.write_text(json.dumps(social_circle, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[attempt {attempt}/{max_iterations}] verifying social circle for {base_name}")
        verification = run_verification(client, model, profile, transcript, social_circle, gpt_reasoning, provider=provider, backend=backend)
        verification_out.write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")

        verdict = overall_verdict(verification)
        print(f"[attempt {attempt}/{max_iterations}] verdict: {verdict}")
        if verdict in {"PASS", "PASS_WITH_REVISIONS"}:
            return 0

    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Path to the Step 2 verification JSON (contains corrected_extracted_profile)",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="Path to the Step 2 interview JSON (contains transcript)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STEP_DIR / "data_samples" / "output",
        help="Directory to write social circle and verification outputs into",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--gpt-reasoning",
        default="high",
        help="OpenAI model (openai-api, codex) reasoning effort.",
    )
    parser.add_argument("--provider", default=None, choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--backend", default=None, choices=SUPPORTED_BACKENDS)
    args = parser.parse_args()

    if args.backend is None:
        args.backend = f"{args.provider or 'anthropic'}-api" if args.provider else os.environ.get("PERSONABENCH_BACKEND", "claude")
    provider = provider_for_backend(args.backend, args.provider or "anthropic") if args.backend in API_BACKENDS else (args.provider or "anthropic")
    if args.model is None:
        args.model = default_model_for_backend(args.backend, "generator")

    if args.backend in API_BACKENDS and not check_api_key(provider):
        print(f"API key for {provider} is not set.", file=sys.stderr)
        sys.exit(2)

    sys.exit(run_step(args.profile, args.transcript, args.output, args.model, args.max_iterations, args.gpt_reasoning, provider, args.backend))


if __name__ == "__main__":
    main()
