"""Batched-output Pass 1 for small models (Qwen3-8B via Ollama, GPT-4o-mini
via OpenAI). Fixes the truncation observed in the prior runs (54/101 for Qwen3,
11/25 chunk for GPT-4o-mini) by sending test cases in batches of N=5 with
ample max_tokens, then merging the chunked answer files.

Pass 2 is delegated to claude CLI with Sonnet (canonical scorer) so the small
model is never grading its own answers.

Outputs:
  step6_benchmark_runner/data_samples/output_small_models/
    {persona}_pass1_{provider}.json
    {persona}_benchmark_results_{provider}.json

Usage:
  python -m step6_benchmark_runner._scripts.run_small_model \\
    --persona julio_simmons --provider openai --model gpt-4o-mini --batch-size 5

  python -m step6_benchmark_runner._scripts.run_small_model \\
    --persona julio_simmons --provider ollama --model qwen3:8b --batch-size 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from _cli_helpers import (
    approx_token_count,
    call_and_parse_json,
    call_claude_cli,
    extract_json,
    fill,
    load_prompt_sections,
    prune_filler_with_diagnostics,
    strip_ground_truth,
    strip_to_raw_logs,
    strip_to_raw_logs_filler_pruned,
    truncate_to_tokens,
)
from run_pass2_batched import aggregate as aggregate_pass2, chunk as chunk_cases

REPO = Path(__file__).resolve().parents[2]
STEP4_OUT = REPO / "step4_app_log_synthesizer" / "data_samples" / "output"
STEP5_OUT = REPO / "step5_testcases_synthesis" / "data_samples" / "output"
STEP6 = REPO / "step6_benchmark_runner"
PROMPT = STEP6 / "prompt.txt"
OUT_DIR = STEP6 / "data_samples" / "output_small_models"

# Small models cannot fit the full ~220K-token Julio log. Truncate to fit
# their 128K context windows.
#
# With filler-pruned mode (default): all meaningful sessions + all calendar
# events are retained, filler is sampled to hit the budget. The prior
# MODEL_TIER_COMPARISON run used 110K input tokens (sampling 434 of 2,445
# filler sessions, ~18%) for the same purpose. We match that.
#
# Budget breakdown for gpt-4o-mini at 110K log target:
#   128K total - 8K output - 8K prompt template - 2K test batch - margin
#   = ~110K available for raw logs.
SMALL_MODEL_LOG_BUDGET_TOKENS = 110_000

# Per-provider hard ceilings on output tokens. Ollama models with reasoning
# (qwen3.5 thinking mode) can emit 20K+ tokens of scratchpad before the answer.
# At CoT scaffold + Type 5 cases, a single case can run 15-20K. 64K headroom
# protects against truncation even when multiple Type 5 cases land in the
# same batch.
OPENAI_MAX_OUTPUT = 8_000
OLLAMA_MAX_OUTPUT = 64_000

# OpenAI gpt-4o-mini: 200K TPM cap. Each batch ~95K tokens. Sleep between
# batches to stay under the cap with safety margin.
OPENAI_BATCH_DELAY_SEC = 35


def _load_openai_key() -> str:
    """Source key from VET_CLAIM_PROJECT/apps/api/.env if not in env."""
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    env_path = Path(r"C:/Users/eclip/Desktop/VET_CLAIM_PROJECT/apps/api/.env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    os.environ["OPENAI_API_KEY"] = val
                    return val
    raise RuntimeError("OPENAI_API_KEY not set and not found in VET_CLAIM_PROJECT")


def call_openai(prompt: str, model: str, max_tokens: int, max_retries: int = 4) -> str:
    api_key = _load_openai_key()
    # gpt-5* and reasoning models require max_completion_tokens; legacy gpt-4*
    # uses max_tokens. Send the right parameter for the model family.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0
    body = json.dumps(payload).encode("utf-8")
    last_err = ""
    for attempt in range(max_retries):
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_err = f"OpenAI HTTP {e.code}: {err_body[:500]}"
            if e.code == 429:
                # Parse "Please try again in Ns" if present
                import re as _re
                wait = 30
                m = _re.search(r"try again in ([\d.]+)s", err_body)
                if m:
                    wait = max(int(float(m.group(1))) + 2, 5)
                print(f"  [openai 429] sleeping {wait}s (attempt {attempt+1}/{max_retries})", flush=True)
                import time as _t
                _t.sleep(wait)
                continue
            if e.code in (500, 502, 503, 504):
                import time as _t
                _t.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(last_err) from e
    raise RuntimeError(last_err)


def call_ollama(
    prompt: str, model: str, max_tokens: int, num_ctx: int = 262144
) -> str:
    # qwen3.5:9b at Q4 fits a 16GB GPU at 262K context. num_gpu=99 forces every
    # model layer onto GPU; falling back to CPU offload makes Step 6 unusable.
    #
    # Qwen has a thinking mode that can emit 20K+ tokens of scratchpad before
    # the JSON answer. /api/chat with think:false reliably suppresses the
    # scratchpad; /api/generate ignores think:false in some Ollama versions.
    # Use /api/chat as the primary path.
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
                "num_gpu": 99,
                "temperature": 0,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        # Long timeout: 262K prefill on a 9B model is multi-minute per batch
        # even on GPU. 7200s gives margin without false-failing on heavy
        # cases.
        with urllib.request.urlopen(req, timeout=7200) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Ollama not reachable at http://127.0.0.1:11434. "
            "Ensure `ollama serve` is running and the model is pulled."
        ) from e
    return data.get("message", {}).get("content", "")


def call_provider(
    provider: str, prompt: str, model: str, max_tokens: int,
    num_ctx: int = 262144,
) -> str:
    if provider == "openai":
        return call_openai(prompt, model, max_tokens)
    if provider == "ollama":
        return call_ollama(prompt, model, max_tokens, num_ctx=num_ctx)
    if provider == "claude_cli":
        return call_claude_cli(prompt, model=model, timeout_sec=1500, max_retries=2)
    raise ValueError(f"unknown provider: {provider}")


def chunk(seq: list, n: int) -> list[list]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def run_pass1_batched(
    persona: str,
    provider: str,
    model: str,
    batch_size: int,
    log_budget: int | None = None,
    truncation: str = "meaningful_preserved",
    seed: int = 42,
    num_ctx: int = 262144,
    cot_prompt_path: Path | None = None,
) -> dict[str, Any]:
    app_logs = json.loads((STEP4_OUT / f"{persona}_app_logs.json").read_text(encoding="utf-8"))
    test_cases = json.loads((STEP5_OUT / f"{persona}_test_cases.json").read_text(encoding="utf-8"))
    budget = log_budget if log_budget is not None else SMALL_MODEL_LOG_BUDGET_TOKENS
    diagnostics: dict[str, Any] | None = None
    if truncation in ("filler_pruned", "meaningful_preserved"):
        raw_logs, diagnostics = prune_filler_with_diagnostics(
            app_logs, target_tokens=budget, seed=seed
        )
    elif truncation == "full":
        raw_logs = strip_to_raw_logs(app_logs)
    elif truncation == "chronological":
        raw_full = strip_to_raw_logs(app_logs)
        raw_logs = truncate_to_tokens(raw_full, budget)
    else:
        raise ValueError(f"unknown truncation mode: {truncation}")
    actual_tokens = approx_token_count(raw_logs)

    stripped = strip_ground_truth(test_cases)
    cases = stripped["test_cases"]

    prompt_source = cot_prompt_path if cot_prompt_path else PROMPT
    sections = load_prompt_sections(prompt_source)
    template = fill(sections["pass1"], "{{INSERT_RAW_LOGS_TEXT_HERE}}", raw_logs)

    max_out = OPENAI_MAX_OUTPUT if provider == "openai" else OLLAMA_MAX_OUTPUT

    import time as _t
    all_answers: list[dict[str, Any]] = []
    batches = chunk(cases, batch_size)
    for batch_idx, batch in enumerate(batches, start=1):
        batch_payload = {"metadata": stripped["metadata"], "test_cases": batch}
        prompt = fill(template, "{{INSERT_TEST_CASE_QUESTIONS_JSON_HERE}}", batch_payload)
        prompt += (
            f"\n\nIMPORTANT: Respond with ONE valid JSON object only, "
            f"matching the Pass 1 output format. Include exactly {len(batch)} "
            f"answers, one per test case in this batch. Do not include any "
            f"prose outside the JSON. Do not truncate."
        )
        print(
            f"[{provider}/{persona}] batch {batch_idx}/{len(batches)}: "
            f"{len(batch)} cases ({batch[0]['id']}..{batch[-1]['id']})",
            flush=True,
        )
        answers = []
        last_err: Exception | None = None
        for parse_attempt in range(1, 4):  # up to 3 attempts per batch
            try:
                raw = call_provider(provider, prompt, model, max_out, num_ctx=num_ctx)
                parsed = extract_json(raw)
                answers = parsed.get("answers", [])
                if answers:
                    break
                last_err = ValueError("empty answers list")
            except (RuntimeError, ValueError, json.JSONDecodeError) as e:
                last_err = e
                print(
                    f"  [retry] batch {batch_idx} attempt {parse_attempt}/3 failed: {e}",
                    flush=True,
                )
        if not answers:
            print(
                f"  [warn] batch {batch_idx} exhausted retries: {last_err}; backfilling INCORRECT",
                flush=True,
            )
        if provider == "openai" and batch_idx < len(batches):
            _t.sleep(OPENAI_BATCH_DELAY_SEC)
        # Backfill any missing test_case_id for this batch
        seen_ids = {a.get("test_case_id") for a in answers}
        for tc in batch:
            if tc["id"] not in seen_ids:
                answers.append(
                    {
                        "test_case_id": tc["id"],
                        "type": tc["type"],
                        "answer": "",
                        "cited_evidence": [],
                        "missing_due_to_batch_failure": True,
                    }
                )
        all_answers.extend(answers)

    truncation_meta = truncation
    metadata = {
        "persona": test_cases.get("metadata", {}).get("persona", persona),
        "participant_id": test_cases.get("metadata", {}).get("participant_id", ""),
        "truncation": truncation_meta,
        "pass_1_date": date.today().isoformat(),
        "model_used": f"{provider}:{model}",
        "raw_log_token_count": actual_tokens,
        "log_token_budget": budget,
        "cases_answered": sum(
            1 for a in all_answers if not a.get("missing_due_to_batch_failure")
        ),
        "cases_attempted": len(all_answers),
        "batch_size": batch_size,
    }
    if diagnostics is not None:
        # Pin the meaningful-preserved guarantee into the output JSON so
        # downstream readers can verify nothing meaningful was dropped.
        metadata["preservation"] = diagnostics
    if cot_prompt_path is not None:
        metadata["pass1_prompt_variant"] = cot_prompt_path.name
    if provider == "ollama":
        metadata["ollama_num_ctx"] = num_ctx
    pass1 = {"metadata": metadata, "answers": all_answers}
    return pass1


def run_pass2_via_claude(persona: str, pass1: dict, judge_model: str) -> dict[str, Any]:
    test_cases = json.loads((STEP5_OUT / f"{persona}_test_cases.json").read_text(encoding="utf-8"))
    sections = load_prompt_sections(PROMPT)

    cases = test_cases["test_cases"]
    answers_by_id = {a["test_case_id"]: a for a in pass1.get("answers", [])}
    persona_name = test_cases.get("metadata", {}).get("persona", persona)
    participant_id = test_cases.get("metadata", {}).get("participant_id", "")
    pass1_model = pass1.get("metadata", {}).get("model_used", "unknown")

    all_evals = []
    batch_size = 20
    batches = chunk_cases(cases, batch_size)
    for bi, batch in enumerate(batches, start=1):
        batch_payload = {"metadata": test_cases.get("metadata", {}), "test_cases": batch}
        batch_pass1 = {
            "metadata": pass1.get("metadata", {}),
            "answers": [answers_by_id[c["id"]] for c in batch if c["id"] in answers_by_id],
        }
        prompt = fill(sections["pass2"], "{{INSERT_FULL_TEST_CASES_JSON_HERE}}", batch_payload)
        prompt = fill(prompt, "{{INSERT_PASS_1_ANSWERS_JSON_HERE}}", batch_pass1)
        prompt += (
            f"\n\nIMPORTANT: This batch contains {len(batch)} cases. "
            f"Score every case in this batch. Return JSON with an `evaluations` "
            f"array containing exactly {len(batch)} entries."
        )
        print(f"[pass2/{persona}] batch {bi}/{len(batches)} judge={judge_model}", flush=True)
        parsed = call_and_parse_json(prompt, model=judge_model, timeout_sec=1500, max_retries=3)
        evals = parsed.get("evaluations", [])
        case_meta = {c["id"]: c for c in batch}
        for ev in evals:
            tc = case_meta.get(ev.get("test_case_id"), {})
            if "subtype" not in ev and tc.get("subtype"):
                ev["subtype"] = tc["subtype"]
            if "is_capstone" not in ev and tc.get("is_capstone") is not None:
                ev["is_capstone"] = tc["is_capstone"]
        all_evals.extend(evals)

    result = aggregate_pass2(all_evals, persona_name, participant_id, pass1_model, judge_model)
    return result


def run(
    persona: str,
    provider: str,
    model: str,
    batch_size: int,
    judges: list[str],
    log_budget: int | None = None,
    truncation: str = "meaningful_preserved",
    seed: int = 42,
    num_ctx: int = 262144,
    cot_prompt_path: Path | None = None,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pass1 = run_pass1_batched(
        persona, provider, model, batch_size, log_budget,
        truncation=truncation, seed=seed,
        num_ctx=num_ctx, cot_prompt_path=cot_prompt_path,
    )
    base_suffix = provider if provider == "ollama" else model.replace("-", "").replace(".", "")
    if cot_prompt_path is not None:
        base_suffix = f"{base_suffix}_cot"
    if truncation == "full":
        suffix = base_suffix
    elif truncation in ("filler_pruned", "meaningful_preserved"):
        suffix = f"{base_suffix}_preserved"
    else:
        suffix = base_suffix
    pass1_path = OUT_DIR / f"{persona}_pass1_{suffix}.json"
    pass1_path.write_text(json.dumps(pass1, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[done pass1] coverage="
        f"{pass1['metadata']['cases_answered']}/{pass1['metadata']['cases_attempted']}"
    )

    if len(judges) == 1:
        result = run_pass2_via_claude(persona, pass1, judges[0])
        out_path = OUT_DIR / f"{persona}_benchmark_results_{suffix}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] overall={result.get('overall_accuracy')} -> {out_path}")
    else:
        per_judge_results = {}
        for j in judges:
            print(f"[pass2/{persona}] judge={j}", flush=True)
            per_judge_results[j] = run_pass2_via_claude(persona, pass1, j)
            judge_path = OUT_DIR / f"{persona}_benchmark_results_{suffix}_{j}.json"
            judge_path.write_text(
                json.dumps(per_judge_results[j], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        majority = aggregate_majority(per_judge_results, judges)
        majority_path = OUT_DIR / f"{persona}_benchmark_results_{suffix}_majority.json"
        majority_path.write_text(
            json.dumps(majority, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"[done] majority overall={majority.get('overall_accuracy')} "
            f"({len(judges)} judges) -> {majority_path}"
        )


def aggregate_majority(
    per_judge: dict[str, dict[str, Any]],
    judges: list[str],
) -> dict[str, Any]:
    """Take per-judge Pass 2 reports and produce a majority verdict.

    For Types 1-4 (CORRECT/PARTIAL/INCORRECT): mode of the three verdicts.
    Tie-break favors the harsher verdict (INCORRECT > PARTIAL > CORRECT)
    so the panel is conservative under disagreement.

    For Type 5 (4-dim 0-3 rubric): median of each dimension across judges.
    Final Type 5 verdict is recomputed from the median dims using the
    standard rule (VERIFIED if all >=2; FLAGGED if any ==1; FAILED if any ==0).

    Returns a single benchmark_results-shaped dict with `judges_used` listed
    in metadata and per-case `judge_verdicts` showing each judge's call.
    """
    import statistics
    HARSHNESS = {"INCORRECT": 3, "FAILED": 3, "PARTIAL": 2, "FLAGGED": 2,
                 "CORRECT": 1, "VERIFIED": 1}

    def majority_verdict(verdicts: list[str]) -> str:
        if not verdicts:
            return "INCORRECT"
        counts: dict[str, int] = {}
        for v in verdicts:
            counts[v] = counts.get(v, 0) + 1
        max_count = max(counts.values())
        winners = [v for v, c in counts.items() if c == max_count]
        if len(winners) == 1:
            return winners[0]
        return max(winners, key=lambda v: HARSHNESS.get(v, 0))

    judge_results = [per_judge[j] for j in judges]
    base = json.loads(json.dumps(judge_results[0]))  # deep copy structure
    case_lists = [r.get("evaluations", []) for r in judge_results]
    by_id: dict[str, list[dict[str, Any]]] = {}
    for evals in case_lists:
        for ev in evals:
            by_id.setdefault(ev.get("test_case_id"), []).append(ev)

    merged_evals: list[dict[str, Any]] = []
    for tc_id, ev_list in by_id.items():
        merged: dict[str, Any] = {"test_case_id": tc_id, "judge_verdicts": {}}
        type_ = ev_list[0].get("type", "")
        merged["type"] = type_
        if "subtype" in ev_list[0]:
            merged["subtype"] = ev_list[0]["subtype"]
        if "is_capstone" in ev_list[0]:
            merged["is_capstone"] = ev_list[0]["is_capstone"]

        for j_idx, ev in enumerate(ev_list):
            merged["judge_verdicts"][judges[j_idx]] = {
                "verdict": ev.get("verdict"),
                "dimensions": ev.get("dimensions"),
            }

        if type_.startswith("type_5"):
            dim_keys = ["risk_surfacing", "appropriate_response",
                        "evidence_use", "hallucination_control"]
            median_dims: dict[str, float] = {}
            for k in dim_keys:
                vals = [
                    ev.get("dimensions", {}).get(k)
                    for ev in ev_list
                    if isinstance(ev.get("dimensions", {}).get(k), (int, float))
                ]
                if vals:
                    median_dims[k] = statistics.median(vals)
            if any(v == 0 for v in median_dims.values()):
                verdict = "FAILED"
            elif any(v == 1 for v in median_dims.values()):
                verdict = "FLAGGED"
            else:
                verdict = "VERIFIED"
            merged["dimensions"] = median_dims
            merged["verdict"] = verdict
        else:
            verdicts = [ev.get("verdict") for ev in ev_list]
            merged["verdict"] = majority_verdict([v for v in verdicts if v])
        merged_evals.append(merged)

    score_map = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0,
                 "VERIFIED": 1.0, "FLAGGED": 0.5, "FAILED": 0.0}
    scored = [score_map.get(ev.get("verdict"), 0.0) for ev in merged_evals]
    overall = sum(scored) / len(scored) if scored else 0.0

    base["evaluations"] = merged_evals
    base["overall_accuracy"] = round(overall, 4)
    base["aggregation"] = {
        "method": "majority_vote",
        "judges_used": judges,
        "tie_break": "harshest_verdict",
        "type5_method": "median_per_dimension",
    }
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--provider", required=True, choices=["openai", "ollama", "claude_cli"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument(
        "--judge-model", default="sonnet",
        help="Single Pass 2 judge model (legacy single-judge mode). "
        "Use --judges for the 3-judge majority panel.",
    )
    ap.add_argument(
        "--judges", default=None,
        help="Comma-separated list of judge models for majority-vote Pass 2 "
        "(e.g. 'haiku,sonnet,opus'). When set, --judge-model is ignored.",
    )
    ap.add_argument("--log-budget", type=int, default=None,
                    help="Raw log token budget for filler-sampling truncation modes. "
                    "Ignored when --truncation=full.")
    ap.add_argument(
        "--truncation",
        default="meaningful_preserved",
        choices=["meaningful_preserved", "filler_pruned", "full", "chronological"],
        help=(
            "meaningful_preserved (default, alias filler_pruned): keep ALL meaningful "
            "sessions + ALL calendar events; only filler is sampled to fit budget. "
            "full: render the entire log without any sampling (use when model context "
            "comfortably fits the persona's full log, e.g. Qwen at 262K). "
            "chronological: legacy. Naive cut at the budget boundary; can drop meaningful "
            "evidence in late-window dates. Not recommended."
        ),
    )
    ap.add_argument("--seed", type=int, default=42, help="Filler-sampling seed.")
    ap.add_argument(
        "--num-ctx", type=int, default=262144,
        help="Ollama num_ctx (context window). Default 262144 matches qwen3.5 native. "
        "Drop to 131072 if VRAM is tight. Ignored for non-ollama providers.",
    )
    ap.add_argument(
        "--cot", action="store_true",
        help="Use prompt_cot_v2.txt (chain-of-thought scaffold for Type 5) "
        "as the Pass 1 prompt instead of the canonical single-shot prompt.",
    )
    args = ap.parse_args()

    if args.judges:
        judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    else:
        judges = [args.judge_model]

    cot_path: Path | None = None
    if args.cot:
        cot_path = STEP6 / "prompt_cot_v2.txt"
        if not cot_path.exists():
            cot_path = STEP6 / "prompt_cot.txt"

    run(args.persona, args.provider, args.model, args.batch_size, judges,
        args.log_budget, truncation=args.truncation, seed=args.seed,
        num_ctx=args.num_ctx, cot_prompt_path=cot_path)


if __name__ == "__main__":
    main()
