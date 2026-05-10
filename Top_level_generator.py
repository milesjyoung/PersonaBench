"""Top-level orchestrator for the PersonaBench pipeline.

Chains all six steps for a single persona:

  Step 1: seed lookup               (step1_seed/)
  Step 2: interview + verification  (step2_interview/)
  Step 3: social circle + verification (step3_social_circle/)
  Step 4: app log synthesis         (step4_app_log_synthesizer/)
  Step 5: test case synthesis       (step5_testcases_synthesis/)
  Step 6: benchmark runner          (step6_benchmark_runner/)

Each step runs as a subprocess invoking its own generator.py, so individual
steps remain independently runnable. Use --start and --stop to run a subset.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
STEP1_DIR = REPO_ROOT / "step1_seed"
STEP2_DIR = REPO_ROOT / "step2_interview"
STEP3_DIR = REPO_ROOT / "step3_social_circle"
STEP4_DIR = REPO_ROOT / "step4_app_log_synthesizer"
STEP5_DIR = REPO_ROOT / "step5_testcases_synthesis"
STEP6_DIR = REPO_ROOT / "step6_benchmark_runner"


def run_subprocess(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def run_step1(uuid: str) -> int:
    return run_subprocess(
        [
            sys.executable,
            str(STEP1_DIR / "generator.py"),
            "--uuid", uuid,
            "--output", str(STEP1_DIR / "data_samples" / "output"),
        ]
    )


def run_step2(seed_path: Path, model: str, max_iterations: int) -> int:
    return run_subprocess(
        [
            sys.executable,
            str(STEP2_DIR / "generator.py"),
            "--seed", str(seed_path),
            "--output", str(STEP2_DIR / "data_samples" / "output"),
            "--model", model,
            "--max-iterations", str(max_iterations),
        ]
    )


def run_step3(
    profile_path: Path, transcript_path: Path, model: str, max_iterations: int
) -> int:
    return run_subprocess(
        [
            sys.executable,
            str(STEP3_DIR / "generator.py"),
            "--profile", str(profile_path),
            "--transcript", str(transcript_path),
            "--output", str(STEP3_DIR / "data_samples" / "output"),
            "--model", model,
            "--max-iterations", str(max_iterations),
        ]
    )


def run_step4(
    profile_path: Path,
    social_circle_path: Path,
    model: str,
    verifier_model: str,
    per_fact_max_attempts: int,
    log_start: str,
    log_end: str,
) -> int:
    return run_subprocess(
        [
            sys.executable,
            str(STEP4_DIR / "generator.py"),
            "--profile", str(profile_path),
            "--social-circle", str(social_circle_path),
            "--output", str(STEP4_DIR / "data_samples" / "output"),
            "--model", model,
            "--verifier-model", verifier_model,
            "--per-fact-max-attempts", str(per_fact_max_attempts),
            "--log-start", log_start,
            "--log-end", log_end,
        ]
    )


def run_step5(
    profile_path: Path,
    app_logs_path: Path,
    social_circle_path: Path,
    model: str,
    max_iterations: int,
) -> int:
    return run_subprocess(
        [
            sys.executable,
            str(STEP5_DIR / "generator.py"),
            "--profile", str(profile_path),
            "--app-logs", str(app_logs_path),
            "--social-circle", str(social_circle_path),
            "--output", str(STEP5_DIR / "data_samples" / "output"),
            "--model", model,
            "--max-iterations", str(max_iterations),
        ]
    )


def run_step6(
    app_logs_path: Path,
    test_cases_path: Path,
    model_pass1: str,
    model_pass2: str,
    openclaw: bool,
) -> int:
    cmd = [
        sys.executable,
        str(STEP6_DIR / "generator.py"),
        "--app-logs", str(app_logs_path),
        "--test-cases", str(test_cases_path),
        "--output", str(STEP6_DIR / "data_samples" / "output"),
        "--model-pass1", model_pass1,
        "--model-pass2", model_pass2,
    ]
    if openclaw:
        cmd.append("--openclaw")
    return run_subprocess(cmd)


def run_pipeline(
    seed_path: Path,
    model: str,
    max_iterations: int,
    start: int,
    stop: int,
    openclaw: bool,
    verifier_model: str = "claude-sonnet-4-6",
    per_fact_max_attempts: int = 3,
    log_start: str = "2026-03-01",
    log_end: str = "2026-03-31",
) -> int:
    base = seed_path.stem.replace("_seed", "")

    interview_path = STEP2_DIR / "data_samples" / "output" / f"{base}_interview.json"
    verification_path = STEP2_DIR / "data_samples" / "output" / f"{base}_verification.json"
    social_circle_verification_path = (
        STEP3_DIR / "data_samples" / "output" / f"{base}_social_circle_verification.json"
    )
    app_logs_path = STEP4_DIR / "data_samples" / "output" / f"{base}_app_logs.json"
    test_cases_path = STEP5_DIR / "data_samples" / "output" / f"{base}_test_cases.json"

    if start <= 2 <= stop:
        if (rc := run_step2(seed_path, model, max_iterations)) != 0:
            return rc
    if start <= 3 <= stop:
        if (rc := run_step3(verification_path, interview_path, model, max_iterations)) != 0:
            return rc
    if start <= 4 <= stop:
        if (rc := run_step4(
            verification_path, social_circle_verification_path, model,
            verifier_model, per_fact_max_attempts, log_start, log_end,
        )) != 0:
            print(
                "[step4] warning: some hidden facts did not pass the "
                "reverse-inferability gate. Continuing anyway — see "
                "data_samples/output/{persona}_app_logs_trace.json"
            )
    if start <= 5 <= stop:
        if (rc := run_step5(
            verification_path, app_logs_path, social_circle_verification_path,
            model, max_iterations,
        )) != 0:
            return rc
    if start <= 6 <= stop:
        if (rc := run_step6(app_logs_path, test_cases_path, model, model, openclaw)) != 0:
            return rc
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=Path, required=True,
        help="Path to the seed JSON produced by Step 1 (or supplied directly)",
    )
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--start", type=int, default=2, choices=[1, 2, 3, 4, 5, 6],
        help="First step to run",
    )
    parser.add_argument(
        "--stop", type=int, default=6, choices=[1, 2, 3, 4, 5, 6],
        help="Last step to run",
    )
    parser.add_argument(
        "--openclaw", action="store_true",
        help="Route Step 6 Pass 1 through the claude CLI (OpenClaw-style) with "
        "session isolation.",
    )
    parser.add_argument(
        "--verifier-model", default="claude-sonnet-4-6",
        help="Independent model for Step 4's reverse-inferability gate.",
    )
    parser.add_argument("--per-fact-max-attempts", type=int, default=3)
    parser.add_argument("--log-start", default="2026-03-01")
    parser.add_argument("--log-end", default="2026-03-31")
    args = parser.parse_args()

    if args.start > args.stop:
        print("--start must be <= --stop", file=sys.stderr)
        sys.exit(2)

    sys.exit(
        run_pipeline(
            args.seed, args.model, args.max_iterations,
            args.start, args.stop, args.openclaw,
            args.verifier_model, args.per_fact_max_attempts,
            args.log_start, args.log_end,
        )
    )


if __name__ == "__main__":
    main()
