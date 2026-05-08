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
from llm import make_client, call_llm as _call_llm, check_api_key, SUPPORTED_PROVIDERS

STEP_DIR = Path(__file__).parent
PROMPT_PATH = STEP_DIR / "prompt.txt"
VERIFICATION_PROMPT_PATH = STEP_DIR / "verification_prompt.txt"

DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_MAX_ITERATIONS = 3


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_json_placeholder(template: str, placeholder: str, payload: Any) -> str:
    return template.replace(placeholder, json.dumps(payload, indent=2, ensure_ascii=False))


def fill_identity_placeholders(template: str, profile: dict[str, Any]) -> str:
    return template.replace("{{participant_id}}", profile.get("participant_id", "")).replace(
        "{{name}}", profile.get("name", "")
    )


def call_llm(client, model: str, prompt: str, provider: str = "anthropic") -> str:
    return _call_llm(client, model, prompt, provider=provider)


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    payload = fenced.group(1) if fenced else text
    start = payload.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(payload[start:])


def run_generation(
    client,
    model: str,
    profile: dict[str, Any],
    transcript: list[dict[str, Any]],
    provider: str = "anthropic",
) -> dict[str, Any]:
    template = load_prompt(PROMPT_PATH)
    template = fill_identity_placeholders(template, profile)
    template = fill_json_placeholder(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", profile
    )
    template = fill_json_placeholder(template, "{{INSERT_TRANSCRIPT_JSON_HERE}}", transcript)
    raw = call_llm(client, model, template, provider=provider)
    return extract_json(raw)


def run_verification(
    client,
    model: str,
    profile: dict[str, Any],
    transcript: list[dict[str, Any]],
    social_circle: dict[str, Any],
    provider: str = "anthropic",
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
    raw = call_llm(client, model, template, provider=provider)
    return extract_json(raw)


def overall_verdict(verification: dict[str, Any]) -> str:
    return verification.get("verification", {}).get("overall_verdict", "UNKNOWN")


def run_step(
    profile_path: Path,
    transcript_path: Path,
    output_dir: Path,
    model: str,
    max_iterations: int,
    provider: str = "anthropic",
) -> int:
    verification_bundle = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = verification_bundle.get("corrected_extracted_profile", verification_bundle)

    interview = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript = interview.get("transcript", interview)

    client = make_client(provider)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = profile_path.stem.replace("_verification", "")
    circle_out = output_dir / f"{base_name}_social_circle.json"
    verification_out = output_dir / f"{base_name}_social_circle_verification.json"

    for attempt in range(1, max_iterations + 1):
        print(f"[attempt {attempt}/{max_iterations}] generating social circle for {base_name}")
        social_circle = run_generation(client, model, profile, transcript, provider=provider)
        circle_out.write_text(json.dumps(social_circle, indent=2, ensure_ascii=False))

        print(f"[attempt {attempt}/{max_iterations}] verifying social circle for {base_name}")
        verification = run_verification(client, model, profile, transcript, social_circle, provider=provider)
        verification_out.write_text(json.dumps(verification, indent=2, ensure_ascii=False))

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
    parser.add_argument("--provider", default="anthropic", choices=SUPPORTED_PROVIDERS)
    args = parser.parse_args()

    if not check_api_key(args.provider):
        print(f"API key for {args.provider} is not set.", file=sys.stderr)
        sys.exit(2)

    sys.exit(run_step(args.profile, args.transcript, args.output, args.model, args.max_iterations, args.provider))


if __name__ == "__main__":
    main()
