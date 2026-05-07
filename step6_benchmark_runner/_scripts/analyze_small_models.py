"""Compare GPT-4o-mini batched and Qwen3-8B batched runs against the prior
truncated runs (julio_simmons_benchmark_results_gpt4omini_sonnet.json,
julio_simmons_benchmark_results_qwen3.json) to show the batching fix.

Outputs 4.28.26/small_model_batched_rerun.md.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJ = REPO.parent
NEW_DIR = REPO / "step6_benchmark_runner" / "data_samples" / "output_small_models"
OLD_DIR = REPO / "step6_benchmark_runner" / "data_samples" / "output"
OUT_PATH = PROJ / "4.28.26" / "small_model_batched_rerun.md"

TYPES = [
    "type_1_simple_fact_check",
    "type_2_cross_log_fact_check",
    "type_3_dynamic_tracking",
    "type_4_reasoning",
    "type_5_agent_behavior",
]


def pct(s):
    if s is None:
        return None
    if isinstance(s, dict):
        # Old format: {count, sum_score, accuracy: "X.X%", verdicts: {...}}
        s = s.get("accuracy")
        if s is None:
            return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def coverage(d):
    if not d:
        return None
    m = d.get("metadata", {})
    return m.get("cases_answered", m.get("cases_attempted"))


def main() -> None:
    persona = "julio_simmons"
    new_gpt = load(NEW_DIR / f"{persona}_benchmark_results_gpt4omini.json")
    new_qwen = load(NEW_DIR / f"{persona}_benchmark_results_ollama.json")
    new_gpt_p1 = load(NEW_DIR / f"{persona}_pass1_gpt4omini.json")
    new_qwen_p1 = load(NEW_DIR / f"{persona}_pass1_ollama.json")
    old_gpt = load(OLD_DIR / f"{persona}_benchmark_results_gpt4omini_sonnet.json")
    old_qwen = load(OLD_DIR / f"{persona}_benchmark_results_qwen3.json")

    md = ["# Small-model batched-output rerun — Julio Simmons", ""]
    md.append("**Question:** does the prior Qwen3-8B 0% on Types 1-3 reflect model capability or output truncation (54/101 answers in the original run)? Fix the truncation via batched output and re-score.")
    md.append("")
    md.append("**Method:** Pass 1 sends test cases in batches of 5, each with `max_tokens >= 8K` (Qwen3) / `max_tokens = 8K` (GPT-4o-mini). 35-second sleep between OpenAI batches to stay under the 200K TPM cap. Logs truncated to 80K tokens to fit each provider's 128K context. Pass 2 scoring uses Sonnet 4.6 (canonical scorer, not the small model itself — to avoid the 87.9% vs 36.6% self-grading bias documented earlier).")
    md.append("")

    md.append("## Output coverage (the actual fix)")
    md.append("")
    md.append("| Run | Pass 1 coverage | Backfilled INCORRECT |")
    md.append("|---|---:|---:|")
    if old_gpt:
        old_c = coverage(old_gpt) or "?"
        md.append(f"| GPT-4o-mini (prior, chunked) | reported truncation at chunk 2 (11/25); 14 backfilled | yes |")
    if new_gpt_p1:
        c = new_gpt_p1["metadata"].get("cases_answered", 0)
        a = new_gpt_p1["metadata"].get("cases_attempted", 101)
        md.append(f"| GPT-4o-mini (this rerun, batched) | {c}/{a} | {a - c} |")
    if old_qwen:
        md.append(f"| Qwen3-8B (prior, Ollama default num_predict) | 54/101 | 47 |")
    if new_qwen_p1:
        c = new_qwen_p1["metadata"].get("cases_answered", 0)
        a = new_qwen_p1["metadata"].get("cases_attempted", 101)
        md.append(f"| Qwen3-8B (this rerun, batched) | {c}/{a} | {a - c} |")
    md.append("")

    md.append("## Per-type accuracy (batched fix vs prior)")
    md.append("")
    md.append("| Type | GPT-4o-mini prior | GPT-4o-mini batched | Δ | Qwen3-8B prior | Qwen3-8B batched | Δ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for t in TYPES:
        og = pct((old_gpt or {}).get("accuracy_by_type", {}).get(t))
        ng = pct((new_gpt or {}).get("accuracy_by_type", {}).get(t))
        oq = pct((old_qwen or {}).get("accuracy_by_type", {}).get(t))
        nq = pct((new_qwen or {}).get("accuracy_by_type", {}).get(t))
        def fmt(x): return f"{x:.1f}%" if x is not None else "n/a"
        def delta(a, b): return f"{b-a:+.1f}" if a is not None and b is not None else "n/a"
        md.append(f"| {t} | {fmt(og)} | {fmt(ng)} | {delta(og,ng)} | {fmt(oq)} | {fmt(nq)} | {delta(oq,nq)} |")
    md.append("")

    md.append("## Headline overall")
    md.append("")
    md.append("| Run | Overall |")
    md.append("|---|---:|")
    for label, d in [
        ("GPT-4o-mini (prior, Sonnet-judged)", old_gpt),
        ("GPT-4o-mini (this rerun, batched + Sonnet-judged)", new_gpt),
        ("Qwen3-8B (prior, Sonnet-judged)", old_qwen),
        ("Qwen3-8B (this rerun, batched + Sonnet-judged)", new_qwen),
    ]:
        v = pct((d or {}).get("overall_accuracy"))
        md.append(f"| {label} | {v:.1f}% |" if v is not None else f"| {label} | n/a |")
    md.append("")

    md.append("## Interpretation (template — fill once results land)")
    md.append("")
    md.append("Targets:")
    md.append("- GPT-4o-mini: Pass 1 coverage moves to 101/101 (was 87/101). The accuracy gain is mostly mechanical: cases that were force-INCORRECT now have a real chance to score CORRECT or PARTIAL. Expect modest lift in Types 1-4, near-zero change in Type 5 (which is hard-floored by the model's safety reasoning).")
    md.append("- Qwen3-8B: Pass 1 coverage moves to 101/101 (was 54/101). The mechanical lift here is larger because more answers were missing. Expect Types 1-3 to come off the 0% floor. Type 5 likely stays near 7% — the open-weights 8B floor on safety-grounded persona inference.")
    md.append("- The cross-tier ordering (Opus > Haiku > GPT-4o-mini > Qwen3-8B) should hold even after the fix. If it does not, that's a finding.")
    md.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
