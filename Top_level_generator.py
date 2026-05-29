"""Top-level orchestrator for the PersonaBench pipeline.

Chains all six steps for a single persona:

  Step 1: seed lookup                  (step1_seed/)
  Step 2: interview + verification     (step2_interview/)
  Step 3: social circle + verification (step3_social_circle/)
  Step 4: app log synthesis            (step4_app_log_synthesizer/)
  Step 5: test case synthesis          (step5_testcases_synthesis/)
  Step 6: benchmark runner             (step6_benchmark_runner/)

Each step runs as a subprocess invoking its own generator.py, so individual
steps remain independently runnable. Use --start and --stop to run a subset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from llm import default_model_for_backend

REPO_ROOT = Path(__file__).parent
STEP1_DIR = REPO_ROOT / "step1_seed"
STEP2_DIR = REPO_ROOT / "step2_interview"
STEP3_DIR = REPO_ROOT / "step3_social_circle"
STEP4_DIR = REPO_ROOT / "step4_app_log_synthesizer"
STEP5_DIR = REPO_ROOT / "step5_testcases_synthesis"
STEP6_DIR = REPO_ROOT / "step6_benchmark_runner"
STEP1_OUTPUT_DIR = STEP1_DIR / "data_samples" / "output"


def run_subprocess(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def run_step1(uuid: str) -> int:
    return run_subprocess(
        [
            sys.executable,
            str(STEP1_DIR / "generator.py"),
            "--uuid",
            uuid,
            "--output",
            str(STEP1_OUTPUT_DIR),
        ]
    )


def find_seed_by_uuid(uuid: str) -> Path | None:
    for seed_path in sorted(STEP1_OUTPUT_DIR.glob("*_seed.json")):
        try:
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if seed.get("uuid") == uuid:
            return seed_path
    return None


def add_backend_args(cmd: list[str], backend: str, provider: str | None) -> None:
    cmd += ["--backend", backend]
    if provider:
        cmd += ["--provider", provider]


def run_step2(
    seed_path: Path,
    model: str,
    max_iterations: int,
    backend: str,
    provider: str | None,
    gpt_reasoning: str,
) -> int:
    cmd = [
        sys.executable,
        str(STEP2_DIR / "generator.py"),
        "--seed",
        str(seed_path),
        "--output",
        str(STEP2_DIR / "data_samples" / "output"),
        "--model",
        model,
        "--max-iterations",
        str(max_iterations),
        "--gpt-reasoning",
        gpt_reasoning,
    ]
    add_backend_args(cmd, backend, provider)
    return run_subprocess(cmd)


def run_step3(
    profile_path: Path,
    transcript_path: Path,
    model: str,
    max_iterations: int,
    backend: str,
    provider: str | None,
    gpt_reasoning: str,
) -> int:
    cmd = [
        sys.executable,
        str(STEP3_DIR / "generator.py"),
        "--profile",
        str(profile_path),
        "--transcript",
        str(transcript_path),
        "--output",
        str(STEP3_DIR / "data_samples" / "output"),
        "--model",
        model,
        "--max-iterations",
        str(max_iterations),
        "--gpt-reasoning",
        gpt_reasoning,
    ]
    add_backend_args(cmd, backend, provider)
    return run_subprocess(cmd)


def run_step4(
    profile_path: Path,
    social_circle_path: Path,
    model: str,
    verifier_model: str,
    per_cluster_max_attempts: int,
    log_start: str,
    log_end: str,
    max_facts: int,
    filler_ratio: float,
    verified_decoys_path: Path | None,
    decoy_count: int,
    decoy_pool_size: int,
    backend: str,
    provider: str | None,
) -> int:
    cmd = [
        sys.executable,
        str(STEP4_DIR / "generator.py"),
        "--profile",
        str(profile_path),
        "--social-circle",
        str(social_circle_path),
        "--output",
        str(STEP4_DIR / "data_samples" / "output"),
        "--model",
        model,
        "--verifier-model",
        verifier_model,
        "--per-cluster-max-attempts",
        str(per_cluster_max_attempts),
        "--log-start",
        log_start,
        "--log-end",
        log_end,
        "--max-facts",
        str(max_facts),
        "--filler-ratio",
        str(filler_ratio),
        "--decoy-pool-size",
        str(decoy_pool_size),
    ]
    add_backend_args(cmd, backend, provider)
    if verified_decoys_path is not None:
        cmd += ["--verified-decoys", str(verified_decoys_path)]
    if decoy_count:
        cmd += ["--decoy-count", str(decoy_count)]
    return run_subprocess(cmd)


def run_step5(
    profile_path: Path,
    app_logs_path: Path,
    social_circle_path: Path,
    model: str,
    max_iterations: int,
    backend: str,
    provider: str | None,
) -> int:
    cmd = [
        sys.executable,
        str(STEP5_DIR / "generator.py"),
        "--profile",
        str(profile_path),
        "--app-logs",
        str(app_logs_path),
        "--social-circle",
        str(social_circle_path),
        "--output",
        str(STEP5_DIR / "data_samples" / "output"),
        "--model",
        model,
        "--max-iterations",
        str(max_iterations),
    ]
    add_backend_args(cmd, backend, provider)
    return run_subprocess(cmd)


def run_step6(
    app_logs_path: Path,
    test_cases_path: Path,
    model_pass1: str,
    model_pass2: str,
    openclaw: bool,
    backend: str,
    provider: str | None,
    reasoning_pass1: str,
    reasoning_pass2: str,
) -> int:
    if openclaw:
        backend = "claude"
    cmd = [
        sys.executable,
        str(STEP6_DIR / "generator.py"),
        "--app-logs",
        str(app_logs_path),
        "--test-cases",
        str(test_cases_path),
        "--output",
        str(STEP6_DIR / "data_samples" / "output"),
        "--model-pass1",
        model_pass1,
        "--model-pass2",
        model_pass2,
        "--backend",
        backend,
        "--reasoning-pass1",
        reasoning_pass1,
        "--reasoning-pass2",
        reasoning_pass2,
    ]
    if provider:
        cmd += ["--provider", provider]
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
    verifier_model: str,
    per_cluster_max_attempts: int = 3,
    log_start: str = "2026-03-01",
    log_end: str = "2026-03-31",
    max_facts: int = 100,
    filler_ratio: float = 2.5,
    verified_decoys_path: Path | None = None,
    decoy_count: int = 0,
    decoy_pool_size: int = 40,
    backend: str = "claude",
    provider: str | None = None,
    gpt_reasoning: str = "high",
    gpt_eval_reasoning: str = "low",
    model_pass1: str | None = None,
    model_pass2: str | None = None,
    step6_backend: str | None = None,
    step6_provider: str | None = None,
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
        rc = run_step2(
            seed_path, model, max_iterations, backend, provider, gpt_reasoning
        )
        if rc != 0:
            return rc
    if start <= 3 <= stop:
        rc = run_step3(
            verification_path,
            interview_path,
            model,
            max_iterations,
            backend,
            provider,
            gpt_reasoning,
        )
        if rc != 0:
            return rc
    if start <= 4 <= stop:
        rc = run_step4(
            verification_path,
            social_circle_verification_path,
            model,
            verifier_model,
            per_cluster_max_attempts,
            log_start,
            log_end,
            max_facts,
            filler_ratio,
            verified_decoys_path,
            decoy_count,
            decoy_pool_size,
            backend,
            provider,
        )
        if rc != 0:
            return rc
    if start <= 5 <= stop:
        rc = run_step5(
            verification_path,
            app_logs_path,
            social_circle_verification_path,
            model,
            max_iterations,
            backend,
            provider,
        )
        if rc != 0:
            return rc
    if start <= 6 <= stop:
        effective_step6_backend = step6_backend or backend
        rc = run_step6(
            app_logs_path,
            test_cases_path,
            model_pass1 or model,
            model_pass2 or model,
            openclaw,
            effective_step6_backend,
            step6_provider if step6_provider is not None else provider,
            gpt_eval_reasoning,
            gpt_reasoning,
        )
        if rc != 0:
            return rc
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=Path,
        help="Path to the seed JSON produced by Step 1 or supplied directly",
    )
    parser.add_argument(
        "--uuid",
        help="Persona UUID for Step 1 seed lookup. Required when --start is 1.",
    )
    parser.add_argument("--model", default=None)
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
        help="Deprecated alias for --backend claude in Step 6.",
    )
    parser.add_argument(
        "--verifier-model", default=None,
        help="Independent model for Step 4's reverse-inferability gate.",
    )
    parser.add_argument("--per-cluster-max-attempts", type=int, default=3)
    parser.add_argument(
        "--per-fact-max-attempts",
        type=int,
        dest="per_cluster_max_attempts",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--log-start", default="2026-03-01")
    parser.add_argument("--log-end", default="2026-03-31")
    parser.add_argument("--max-facts", type=int, default=100)
    parser.add_argument("--filler-ratio", type=float, default=2.5)
    parser.add_argument("--verified-decoys", type=Path, default=None)
    parser.add_argument("--decoy-count", type=int, default=0)
    parser.add_argument("--decoy-pool-size", type=int, default=40)
    parser.add_argument(
        "--backend",
        default="claude",
        choices=["claude", "codex", "anthropic-api", "openai-api"],
    )
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument(
        "--step6-backend",
        default=None,
        choices=["claude", "codex", "anthropic-api", "openai-api"],
        help="Optional backend override for Step 6 Pass 1/2 evaluation.",
    )
    parser.add_argument(
        "--step6-provider",
        choices=["anthropic", "openai"],
        default=None,
        help="Optional provider override for Step 6 when using an API backend.",
    )
    parser.add_argument("--gpt-reasoning", default="high")
    parser.add_argument("--gpt-eval-reasoning", default="low")
    parser.add_argument("--model-pass1", default=None)
    parser.add_argument("--model-pass2", default=None)
    args = parser.parse_args()

    if args.start > args.stop:
        print("--start must be <= --stop", file=sys.stderr)
        sys.exit(2)

    if args.openclaw:
        args.step6_backend = "claude"

    step6_backend = args.step6_backend or args.backend
    if args.model is None:
        args.model = default_model_for_backend(args.backend, "generator")
    if args.verifier_model is None:
        args.verifier_model = default_model_for_backend(args.backend, "verifier")
    if args.model_pass1 is None:
        args.model_pass1 = default_model_for_backend(step6_backend, "evaluator")
    if args.model_pass2 is None:
        args.model_pass2 = default_model_for_backend(step6_backend, "judge")

    if args.start <= 1:
        if not args.uuid:
            print("--uuid is required when --start is 1", file=sys.stderr)
            sys.exit(2)
        rc = run_step1(args.uuid)
        if rc != 0:
            sys.exit(rc)
        args.seed = find_seed_by_uuid(args.uuid)
        if args.seed is None:
            print(f"Step 1 finished but no seed was found for {args.uuid}", file=sys.stderr)
            sys.exit(1)
        if args.stop == 1:
            sys.exit(0)
        args.start = 2
    elif args.seed is None and args.uuid:
        args.seed = find_seed_by_uuid(args.uuid)

    if args.seed is None:
        print("Provide --seed <path>, or --uuid with --start 1", file=sys.stderr)
        sys.exit(2)

    sys.exit(
        run_pipeline(
            args.seed,
            args.model,
            args.max_iterations,
            args.start,
            args.stop,
            args.openclaw,
            args.verifier_model,
            args.per_cluster_max_attempts,
            args.log_start,
            args.log_end,
            args.max_facts,
            args.filler_ratio,
            args.verified_decoys,
            args.decoy_count,
            args.decoy_pool_size,
            args.backend,
            args.provider,
            args.gpt_reasoning,
            args.gpt_eval_reasoning,
            args.model_pass1,
            args.model_pass2,
            args.step6_backend,
            args.step6_provider,
        )
    )


if __name__ == "__main__":
    main()
