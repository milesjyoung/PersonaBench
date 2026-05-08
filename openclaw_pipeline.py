"""Pipeline runner that delegates each step to its own generator.py.

Each step's generator.py is a standalone script with its own CLI. This
file is a thin orchestrator: it resolves file paths, chains the steps
in order, and passes the right arguments. All generation logic lives
in the per-step generators, so changes there apply to every backend
automatically.

Usage:
    python openclaw_pipeline.py --persona julio_simmons --start 2 --stop 6
    python openclaw_pipeline.py --all --start 4 --stop 6
    python openclaw_pipeline.py --seed step1_seed/data_samples/output/julio_simmons_seed.json
"""

from __future__ import annotations

import argparse
import json
import re
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


def extract_name(seed: dict) -> str:
    persona_text = seed.get("persona", "") or seed.get("professional_persona", "")
    match = re.match(r"\s*([A-Z][a-z]+\s+[A-Z][a-z]+)", persona_text)
    return match.group(1) if match else seed.get("uuid", "persona")


def base(name: str) -> str:
    return name.lower().replace(" ", "_")


def run_generator(step_dir: Path, args: list[str]) -> int:
    cmd = [sys.executable, str(step_dir / "generator.py")] + args
    print(f"  > {' '.join(cmd[:4])} ...")
    return subprocess.run(cmd, check=False).returncode


def run_step2(seed_path: Path, output_dir: Path, model: str | None) -> int:
    args = ["--seed", str(seed_path), "--output", str(output_dir)]
    if model:
        args += ["--model", model]
    return run_generator(STEP2_DIR, args)


def run_step3(
    profile_path: Path, transcript_path: Path, output_dir: Path,
    model: str | None,
) -> int:
    args = [
        "--profile", str(profile_path),
        "--transcript", str(transcript_path),
        "--output", str(output_dir),
    ]
    if model:
        args += ["--model", model]
    return run_generator(STEP3_DIR, args)


def run_step4(
    profile_path: Path, social_circle_path: Path, output_dir: Path,
    model: str | None, verifier_model: str | None,
    per_fact_max_attempts: int, log_start: str, log_end: str,
) -> int:
    args = [
        "--profile", str(profile_path),
        "--social-circle", str(social_circle_path),
        "--output", str(output_dir),
        "--per-fact-max-attempts", str(per_fact_max_attempts),
        "--log-start", log_start,
        "--log-end", log_end,
    ]
    if model:
        args += ["--model", model]
    if verifier_model:
        args += ["--verifier-model", verifier_model]
    return run_generator(STEP4_DIR, args)


def run_step5(
    profile_path: Path, app_logs_path: Path, social_circle_path: Path,
    output_dir: Path, model: str | None,
) -> int:
    args = [
        "--profile", str(profile_path),
        "--app-logs", str(app_logs_path),
        "--social-circle", str(social_circle_path),
        "--output", str(output_dir),
    ]
    if model:
        args += ["--model", model]
    return run_generator(STEP5_DIR, args)


def run_step6(
    app_logs_path: Path, test_cases_path: Path, output_dir: Path,
    model_pass1: str | None, model_pass2: str | None,
) -> int:
    args = [
        "--app-logs", str(app_logs_path),
        "--test-cases", str(test_cases_path),
        "--output", str(output_dir),
    ]
    if model_pass1:
        args += ["--model-pass1", model_pass1]
    if model_pass2:
        args += ["--model-pass2", model_pass2]
    return run_generator(STEP6_DIR, args)


def run_pipeline(
    seed_path: Path,
    start: int = 2,
    stop: int = 6,
    log_start: str = "2026-02-20",
    log_end: str = "2026-04-20",
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

    # Step 2
    if start <= 2 <= stop:
        print("[step2] generating interview + verification")
        rc = run_step2(seed_path, step2_out, model)
        if rc != 0:
            print(f"[step2] failed with exit code {rc}")
            return

    profile_path = step2_out / f"{n}_verification.json"
    interview_path = step2_out / f"{n}_interview.json"

    # Step 3
    if start <= 3 <= stop:
        print("[step3] generating social circle")
        rc = run_step3(profile_path, interview_path, step3_out, model)
        if rc != 0:
            print(f"[step3] failed with exit code {rc}")
            return

    social_circle_path = step3_out / f"{n}_social_circle_verification.json"

    # Step 4
    if start <= 4 <= stop:
        print("[step4] generating app logs")
        rc = run_step4(
            profile_path, social_circle_path, step4_out,
            model, verifier_model,
            per_fact_max_attempts, log_start, log_end,
        )
        if rc != 0:
            print(f"[step4] failed with exit code {rc}")

    app_logs_path = step4_out / f"{n}_app_logs.json"

    # Step 5
    if start <= 5 <= stop:
        if not app_logs_path.exists():
            print(f"[step5] skipping — {app_logs_path.name} not found")
        else:
            print("[step5] generating test cases")
            rc = run_step5(
                profile_path, app_logs_path, social_circle_path,
                step5_out, model,
            )
            if rc != 0:
                print(f"[step5] failed with exit code {rc}")

    test_cases_path = step5_out / f"{n}_test_cases.json"

    # Step 6
    if start <= 6 <= stop:
        if not app_logs_path.exists() or not test_cases_path.exists():
            print("[step6] skipping — required inputs missing")
        else:
            print("[step6] running benchmark")
            rc = run_step6(
                app_logs_path, test_cases_path, step6_out,
                model_pass1=model,
                model_pass2=judge_model or model,
            )
            if rc != 0:
                print(f"[step6] failed with exit code {rc}")

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
        "step1_seed/data_samples/output/<persona>_seed.json.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run against every seed JSON in step1_seed/"
    )
    parser.add_argument("--start", type=int, default=2, choices=[2, 3, 4, 5, 6])
    parser.add_argument("--stop", type=int, default=6, choices=[2, 3, 4, 5, 6])
    parser.add_argument("--log-start", default="2026-02-20")
    parser.add_argument("--log-end", default="2026-04-20")
    parser.add_argument("--per-fact-max-attempts", type=int, default=3)
    parser.add_argument("--model", default="claude-opus-4-7")
    parser.add_argument("--verifier-model", default="claude-sonnet-4-6")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    if args.start > args.stop:
        parser.error("--start must be <= --stop")

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
                seed_file, args.start, args.stop,
                args.log_start, args.log_end, args.per_fact_max_attempts,
                model=args.model,
                verifier_model=args.verifier_model,
                judge_model=args.judge_model,
            )
    elif args.seed:
        run_pipeline(
            args.seed, args.start, args.stop,
            args.log_start, args.log_end, args.per_fact_max_attempts,
            model=args.model,
            verifier_model=args.verifier_model,
            judge_model=args.judge_model,
        )
    else:
        parser.error("Provide --persona <name>, --seed <path>, or --all")


if __name__ == "__main__":
    main()
