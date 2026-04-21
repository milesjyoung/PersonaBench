"""Step 2 — Interview generation and verification.

Runs two phases inside one step:

  1. Generation     — fills prompt.txt with seed data, calls the LLM, writes the
                      interview transcript + draft extracted profile.
  2. Verification   — fills verification_prompt.txt with seed + transcript +
                      draft profile, calls the LLM, writes the verification
                      report + corrected extracted profile + hidden_facts.

Iterative refinement loop: if verification returns a FAIL verdict, the
generation is re-run up to --max-iterations times. If every attempt still fails,
the last attempt's outputs are retained and the process exits with a non-zero
status.
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


def fill_seed_placeholders(template: str, seed: dict[str, Any]) -> str:
    mapping = {
        "{{participant_id}}": seed["uuid"],
        "{{name}}": extract_name(seed),
        "{{age}}": str(seed["age"]),
        "{{sex}}": seed["sex"],
        "{{marital_status}}": seed["marital_status"],
        "{{education_level}}": seed["education_level"],
        "{{occupation}}": seed["occupation"],
        "{{city}}": seed["city"],
        "{{state}}": seed["state"],
        "{{zipcode}}": seed["zipcode"],
        "{{country}}": seed["country"],
        "{{professional_persona}}": seed["professional_persona"],
        "{{sports_persona}}": seed["sports_persona"],
        "{{arts_persona}}": seed["arts_persona"],
        "{{travel_persona}}": seed["travel_persona"],
        "{{culinary_persona}}": seed["culinary_persona"],
        "{{persona}}": seed["persona"],
        "{{cultural_background}}": seed["cultural_background"],
        "{{skills_and_expertise}}": seed["skills_and_expertise"],
        "{{hobbies_and_interests}}": seed["hobbies_and_interests"],
        "{{career_goals_and_ambitions}}": seed["career_goals_and_ambitions"],
    }
    for key, value in mapping.items():
        template = template.replace(key, value)
    return template


def extract_name(seed: dict[str, Any]) -> str:
    persona_text = seed.get("persona", "") or seed.get("professional_persona", "")
    match = re.match(r"\s*([A-Z][a-z]+\s+[A-Z][a-z]+)", persona_text)
    return match.group(1) if match else seed["uuid"]


def fill_json_placeholder(template: str, placeholder: str, payload: Any) -> str:
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    return template.replace(placeholder, serialized)


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
    client: anthropic.Anthropic, model: str, seed: dict[str, Any]
) -> dict[str, Any]:
    prompt = fill_seed_placeholders(load_prompt(PROMPT_PATH), seed)
    raw = call_llm(client, model, prompt)
    return extract_json(raw)


def run_verification(
    client: anthropic.Anthropic,
    model: str,
    seed: dict[str, Any],
    interview: dict[str, Any],
) -> dict[str, Any]:
    template = fill_seed_placeholders(load_prompt(VERIFICATION_PROMPT_PATH), seed)
    template = fill_json_placeholder(template, "{{INSERT_SEED_JSON_HERE}}", seed)
    template = fill_json_placeholder(
        template, "{{INSERT_TRANSCRIPT_JSON_HERE}}", interview["transcript"]
    )
    template = fill_json_placeholder(
        template,
        "{{INSERT_EXTRACTED_PROFILE_JSON_HERE}}",
        interview["extracted_profile"],
    )
    raw = call_llm(client, model, template)
    return extract_json(raw)


def overall_verdict(verification: dict[str, Any]) -> str:
    return verification.get("verification", {}).get("overall_verdict", "UNKNOWN")


def run_step(
    seed_path: Path,
    output_dir: Path,
    model: str,
    max_iterations: int,
) -> int:
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    client = anthropic.Anthropic()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = seed_path.stem.replace("_seed", "")
    interview_out = output_dir / f"{base_name}_interview.json"
    verification_out = output_dir / f"{base_name}_verification.json"

    interview: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None

    for attempt in range(1, max_iterations + 1):
        print(f"[attempt {attempt}/{max_iterations}] generating interview for {base_name}")
        interview = run_generation(client, model, seed)
        interview_out.write_text(json.dumps(interview, indent=2, ensure_ascii=False))

        print(f"[attempt {attempt}/{max_iterations}] verifying interview for {base_name}")
        verification = run_verification(client, model, seed, interview)
        verification_out.write_text(json.dumps(verification, indent=2, ensure_ascii=False))

        verdict = overall_verdict(verification)
        print(f"[attempt {attempt}/{max_iterations}] verdict: {verdict}")
        if verdict in {"PASS", "PASS_WITH_REVISIONS"}:
            return 0

    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True, help="Path to a seed JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=STEP_DIR / "data_samples" / "output",
        help="Directory to write interview and verification outputs into",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)

    sys.exit(run_step(args.seed, args.output, args.model, args.max_iterations))


if __name__ == "__main__":
    main()
