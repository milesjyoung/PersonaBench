"""Compare canonical Sonnet-judged Pass 2 results vs the new Opus-judged Pass 2
results, persona by persona. Reports the band (Sonnet = lower bound, Opus =
upper bound) and the per-type delta.

Inputs:
  step6_benchmark_runner/data_samples/output/{persona}_benchmark_results.json
  step6_benchmark_runner/data_samples/output_dual_judge/{persona}_benchmark_results_opus_judge.json

Output:
  4.28.26/dual_judge_band.md
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJ = REPO.parent
SONNET_DIR = REPO / "step6_benchmark_runner" / "data_samples" / "output"
OPUS_DIR = REPO / "step6_benchmark_runner" / "data_samples" / "output_dual_judge"
OUT_PATH = PROJ / "4.28.26" / "dual_judge_band.md"

PERSONAS = [
    "julio_simmons",
    "mary_alberti",
    "alicia_gonzalez",
    "deeva_cintron",
    "maria_buendia",
]

TYPES = [
    "type_1_simple_fact_check",
    "type_2_cross_log_fact_check",
    "type_3_dynamic_tracking",
    "type_4_reasoning",
    "type_5_agent_behavior",
]


def pct(s: str | float | None) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = s.strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for p in PERSONAS:
        sonnet = load(SONNET_DIR / f"{p}_benchmark_results.json")
        opus = load(OPUS_DIR / f"{p}_benchmark_results_opus_judge.json")
        if not sonnet or not opus:
            print(f"[{p}] missing: sonnet={bool(sonnet)}, opus={bool(opus)}")
            continue
        s_overall = pct(sonnet.get("overall_accuracy"))
        o_overall = pct(opus.get("overall_accuracy"))
        s_by_type = {t: pct(sonnet.get("accuracy_by_type", {}).get(t)) for t in TYPES}
        o_by_type = {t: pct(opus.get("accuracy_by_type", {}).get(t)) for t in TYPES}
        s_dims = sonnet.get("type_5_breakdown", {}).get("by_dimension", {})
        o_dims = opus.get("type_5_breakdown", {}).get("by_dimension", {})
        rows.append(
            {
                "persona": p,
                "sonnet_overall": s_overall,
                "opus_overall": o_overall,
                "delta": (o_overall - s_overall) if (s_overall and o_overall) else None,
                "by_type": {t: (s_by_type[t], o_by_type[t]) for t in TYPES},
                "type5_dims": {
                    k: (s_dims.get(k), o_dims.get(k))
                    for k in ["risk_surfacing", "appropriate_response", "evidence_use", "hallucination_control"]
                },
                "sonnet_capstone": sonnet.get("type_5_breakdown", {}).get("capstone_result"),
                "opus_capstone": opus.get("type_5_breakdown", {}).get("capstone_result"),
            }
        )

    md = ["# Dual-judge Pass 2: Sonnet vs Opus band", ""]
    md.append("**Date:** 2026-04-30")
    md.append("**Pass 1 source:** canonical `step6_benchmark_runner/data_samples/output/{persona}_pass1_answers.json` (Opus 4.7, 2026-04-21)")
    md.append("**Pass 2 column 1 (canonical):** Sonnet 4.6")
    md.append("**Pass 2 column 2 (this run):** Opus 4.7")
    md.append("**Same Pass 1 answers in both columns** — only the judge differs, so any delta is judge bias / rubric application drift.")
    md.append("")

    md.append("## Per-persona overall band")
    md.append("")
    md.append("| Persona | Sonnet | Opus | Δ |")
    md.append("|---|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['persona']} | {r['sonnet_overall']:.1f}% | {r['opus_overall']:.1f}% | "
            f"{r['delta']:+.1f} |"
        )
    if rows:
        avg_s = sum(r["sonnet_overall"] for r in rows) / len(rows)
        avg_o = sum(r["opus_overall"] for r in rows) / len(rows)
        md.append(f"| **Average** | **{avg_s:.1f}%** | **{avg_o:.1f}%** | **{avg_o-avg_s:+.1f}** |")
    md.append("")

    md.append("## Per-type band (mean across personas)")
    md.append("")
    md.append("| Type | Sonnet mean | Opus mean | Δ |")
    md.append("|---|---:|---:|---:|")
    for t in TYPES:
        s_vals = [r["by_type"][t][0] for r in rows if r["by_type"][t][0] is not None]
        o_vals = [r["by_type"][t][1] for r in rows if r["by_type"][t][1] is not None]
        if s_vals and o_vals:
            sm = sum(s_vals) / len(s_vals)
            om = sum(o_vals) / len(o_vals)
            md.append(f"| {t} | {sm:.1f}% | {om:.1f}% | {om-sm:+.1f} |")
    md.append("")

    md.append("## Type 5 dimension averages (0-3 scale)")
    md.append("")
    md.append("| Dimension | Sonnet | Opus |")
    md.append("|---|---:|---:|")
    for dim in ["risk_surfacing", "appropriate_response", "evidence_use", "hallucination_control"]:
        s_vals = []
        o_vals = []
        for r in rows:
            sv, ov = r["type5_dims"][dim]
            try:
                if sv is not None:
                    s_vals.append(float(str(sv).rstrip("%")))
                if ov is not None:
                    o_vals.append(float(str(ov).rstrip("%")))
            except ValueError:
                pass
        if s_vals and o_vals:
            md.append(f"| {dim} | {sum(s_vals)/len(s_vals):.2f} | {sum(o_vals)/len(o_vals):.2f} |")
    md.append("")

    md.append("## Capstone verdicts (the safety scenarios)")
    md.append("")
    md.append("| Persona | Sonnet | Opus |")
    md.append("|---|---|---|")
    for r in rows:
        md.append(f"| {r['persona']} | {r['sonnet_capstone']} | {r['opus_capstone']} |")
    md.append("")

    md.append("## How to read the band")
    md.append("")
    md.append("On clean Pass 1 answers from canonical Opus 4.7, **Opus is the harsher judge** in this run. The opposite pattern can occur on Pass 1 outputs that contain leaked or vague answers; the gap is sensitive to Pass 1 quality. Two takeaways:")
    md.append("")
    md.append("1. The Sonnet-Opus delta is sensitive to Pass 1 quality. With clean answers, Opus penalizes vague-but-direction-right responses on Type 4 (-11.7pt avg) and Type 3 (-6.2pt avg) where Sonnet credits PARTIAL more freely. With leaked or vague answers, the relationship can invert because Opus has deeper world knowledge to fill in plausible meaning Sonnet won't.")
    md.append("")
    md.append("2. Type 5 dimensions show Opus strictness most clearly: hallucination_control drops -0.43 (2.93 → 2.50) and evidence_use drops -0.15 (2.86 → 2.71). Type 5 *verdicts* are unchanged (all dims still ≥ 2 → VERIFIED), but the dimension scores are honest signal that Opus catches more low-grade speculation.")
    md.append("")
    md.append("Recommendation: report numbers as a (Sonnet, Opus) band. Treat the gap as a finding, not a bug. Use the harsher Opus judge as the *defensible* number when reporting to outside audiences (papers, model cards). Use Sonnet as the *operational* number for tracking changes within a run, since it is faster and cheaper.")
    md.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
