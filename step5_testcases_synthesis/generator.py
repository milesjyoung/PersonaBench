"""Step 5 — Test case generation and verification.

Two phases:

  1. Generation     — fills prompt.txt with the corrected extracted profile,
                      app logs, and corrected social circle. Calls the LLM
                      and writes the test case file.
  2. Verification   — fills verification_prompt.txt with the test cases plus
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

import anthropic

STEP_DIR = Path(__file__).parent
PROMPT_PATH = STEP_DIR / "prompt.txt"
VERIFICATION_PROMPT_PATH = STEP_DIR / "verification_prompt.txt"

DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_MAX_ITERATIONS = 3


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_json(template: str, placeholder: str, payload: Any) -> str:
    return template.replace(
        placeholder, json.dumps(payload, indent=2, ensure_ascii=False)
    )


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


def run_generation(
    client: anthropic.Anthropic,
    model: str,
    corrected_profile: dict[str, Any],
    app_logs: dict[str, Any],
    corrected_social_circle: dict[str, Any],
) -> dict[str, Any]:
    template = load_prompt(PROMPT_PATH)
    template = fill_json(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", corrected_profile
    )
    template = fill_json(template, "{{INSERT_APP_LOGS_JSON_HERE}}", app_logs)
    template = fill_json(
        template,
        "{{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}}",
        corrected_social_circle,
    )
    raw = call_llm(client, model, template)
    return extract_json(raw)


def run_verification(
    client: anthropic.Anthropic,
    model: str,
    test_cases: dict[str, Any],
    corrected_profile: dict[str, Any],
    app_logs: dict[str, Any],
) -> dict[str, Any]:
    template = load_prompt(VERIFICATION_PROMPT_PATH)
    template = fill_json(template, "{{INSERT_TEST_CASES_JSON_HERE}}", test_cases)
    template = fill_json(
        template, "{{INSERT_CORRECTED_EXTRACTED_PROFILE_JSON_HERE}}", corrected_profile
    )
    template = fill_json(template, "{{INSERT_APP_LOGS_JSON_HERE}}", app_logs)
    raw = call_llm(client, model, template)
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
) -> int:
    corrected_profile_file = json.loads(profile_path.read_text(encoding="utf-8"))
    corrected_profile = corrected_profile_file["corrected_extracted_profile"]
    app_logs = json.loads(app_logs_path.read_text(encoding="utf-8"))
    social_circle_file = json.loads(social_circle_path.read_text(encoding="utf-8"))
    corrected_social_circle = social_circle_file["corrected_social_circle"]

    client = anthropic.Anthropic()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = extract_base_name(profile_path)
    test_cases_out = output_dir / f"{base_name}_test_cases.json"
    verification_out = output_dir / f"{base_name}_test_cases_verification.json"

    for attempt in range(1, max_iterations + 1):
        print(f"[attempt {attempt}/{max_iterations}] generating test cases for {base_name}")
        test_cases = run_generation(
            client, model, corrected_profile, app_logs, corrected_social_circle
        )
        test_cases_out.write_text(
            json.dumps(test_cases, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"[attempt {attempt}/{max_iterations}] verifying test cases for {base_name}")
        verification = run_verification(
            client, model, test_cases, corrected_profile, app_logs
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
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)

    sys.exit(
        run_step(
            args.profile,
            args.app_logs,
            args.social_circle,
            args.output,
            args.model,
            args.max_iterations,
        )
    )


if __name__ == "__main__":
    main()
