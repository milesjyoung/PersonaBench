"""Compare canonical single-shot Pass 1 vs CoT-scaffolded Pass 1 across all
personas. Outputs 4.28.26/cot_scaffold_results.md.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJ = REPO.parent
BASE_DIR = REPO / "step6_benchmark_runner" / "data_samples" / "output"
COT_DIR = REPO / "step6_benchmark_runner" / "data_samples" / "output_cot"
OUT_PATH = PROJ / "4.28.26" / "cot_scaffold_results.md"

PERSONAS = ["julio_simmons", "mary_alberti", "alicia_gonzalez", "deeva_cintron", "maria_buendia"]
TYPES = [
    "type_1_simple_fact_check",
    "type_2_cross_log_fact_check",
    "type_3_dynamic_tracking",
    "type_4_reasoning",
    "type_5_agent_behavior",
]
DIMS = ["risk_surfacing", "appropriate_response", "evidence_use", "hallucination_control"]


def pct(s):
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


def main() -> None:
    rows = []
    for p in PERSONAS:
        base = load(BASE_DIR / f"{p}_benchmark_results.json")
        cot = load(COT_DIR / f"{p}_benchmark_results_cot.json")
        if not base or not cot:
            print(f"[{p}] missing baseline={bool(base)} cot={bool(cot)}")
            continue
        rows.append(
            {
                "persona": p,
                "base_overall": pct(base.get("overall_accuracy")),
                "cot_overall": pct(cot.get("overall_accuracy")),
                "base_by_type": {t: pct(base.get("accuracy_by_type", {}).get(t)) for t in TYPES},
                "cot_by_type": {t: pct(cot.get("accuracy_by_type", {}).get(t)) for t in TYPES},
                "base_dims": base.get("type_5_breakdown", {}).get("by_dimension", {}),
                "cot_dims": cot.get("type_5_breakdown", {}).get("by_dimension", {}),
                "base_capstone": base.get("type_5_breakdown", {}).get("capstone_result"),
                "cot_capstone": cot.get("type_5_breakdown", {}).get("capstone_result"),
            }
        )

    md = ["# CoT scaffold A/B — Type 5 lift", ""]
    md.append("**Question:** does a five-phase chain-of-thought scaffold (PARSE / RECALL / RISK SCAN / DRAFT / CRITIQUE) lift Type 5 agent-behavior scores against the single-shot baseline?")
    md.append("")
    md.append("**Method:** Pass 1 prompt was rewritten as `step6_benchmark_runner/prompt_cot.txt`. The CoT scaffold applies to Type 5 cases only (Types 1-4 keep single-shot since they are direct retrieval). The model fills a `cot_scratchpad` per Type 5 case before committing to its `answer`. Pass 2 (Sonnet 4.6, canonical scoring prompt) sees both runs and scores them blind to the variant.")
    md.append("")
    md.append("Pass 1 model: Opus 4.7 via Claude CLI subscription. Pass 2 judge: Sonnet 4.6.")
    md.append("")

    md.append("## Overall accuracy")
    md.append("")
    md.append("| Persona | Single-shot | + CoT | Δ |")
    md.append("|---|---:|---:|---:|")
    for r in rows:
        delta = (r["cot_overall"] - r["base_overall"]) if r["cot_overall"] and r["base_overall"] else None
        md.append(
            f"| {r['persona']} | {r['base_overall']:.1f}% | {r['cot_overall']:.1f}% | "
            f"{delta:+.1f} |" if delta is not None else f"| {r['persona']} | n/a | n/a | n/a |"
        )
    md.append("")

    md.append("## Type 5 lift (the target)")
    md.append("")
    md.append("| Persona | Single-shot Type 5 | + CoT Type 5 | Δ |")
    md.append("|---|---:|---:|---:|")
    for r in rows:
        b = r["base_by_type"].get("type_5_agent_behavior")
        c = r["cot_by_type"].get("type_5_agent_behavior")
        if b is None or c is None:
            continue
        md.append(f"| {r['persona']} | {b:.1f}% | {c:.1f}% | {c-b:+.1f} |")
    md.append("")

    md.append("## Type 5 dimensions (averaged across personas, 0-3 scale)")
    md.append("")
    md.append("| Dimension | Single-shot | + CoT | Δ |")
    md.append("|---|---:|---:|---:|")
    for d in DIMS:
        b_vals, c_vals = [], []
        for r in rows:
            try:
                bv = float(str(r["base_dims"].get(d, "")).rstrip("%"))
                b_vals.append(bv)
            except ValueError:
                pass
            try:
                cv = float(str(r["cot_dims"].get(d, "")).rstrip("%"))
                c_vals.append(cv)
            except ValueError:
                pass
        if b_vals and c_vals:
            bm = sum(b_vals) / len(b_vals)
            cm = sum(c_vals) / len(c_vals)
            md.append(f"| {d} | {bm:.2f} | {cm:.2f} | {cm-bm:+.2f} |")
    md.append("")

    md.append("## Capstone verdict change")
    md.append("")
    md.append("| Persona | Single-shot | + CoT |")
    md.append("|---|---|---|")
    for r in rows:
        md.append(f"| {r['persona']} | {r['base_capstone']} | {r['cot_capstone']} |")
    md.append("")

    md.append("## How to read the lift")
    md.append("")
    md.append("Target effect: Type 5 dimensions — especially `risk_surfacing` and `evidence_use` — lift by at least +0.2 on the 0-3 scale. A lift below that suggests the scaffold is not earning its output budget. A lift above +0.4 suggests the canonical single-shot prompt was leaving real reasoning on the table.")
    md.append("")
    md.append("Watch for regression on Types 1-4. The CoT prompt explicitly tells the model to use single-shot for those, but if the structured-output discipline bleeds across, retrieval accuracy could drop. If Type 1-4 hold within ±2 percentage points, the scaffold is clean.")
    md.append("")
    md.append("If `hallucination_control` rises significantly, the Phase 5 critique step is doing useful work suppressing assumed-but-unstated facts. If it falls, the scratchpad is encouraging the model to commit to inferences it would otherwise hedge.")
    md.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
