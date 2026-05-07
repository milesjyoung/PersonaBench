"""Analyze context-window ablation results: read each truncated-budget run and
emit a markdown summary with per-type accuracy curves to answer the
"context length vs reasoning capability" question.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJ = REPO.parent
ABL_DIR = REPO / "step6_benchmark_runner" / "data_samples" / "output_ablation"
OUT_PATH = PROJ / "4.28.26" / "context_window_ablation.md"

TYPES = [
    "type_1_simple_fact_check",
    "type_2_cross_log_fact_check",
    "type_3_dynamic_tracking",
    "type_4_reasoning",
    "type_5_agent_behavior",
]


def pct(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    persona = "julio_simmons"
    budgets = ["32000", "64000", "128000", "full"]
    rows = []
    for b in budgets:
        path = ABL_DIR / f"{persona}_benchmark_results_{b}.json"
        d = load(path)
        if not d:
            print(f"missing: {path}")
            continue
        rows.append(
            {
                "budget": b,
                "tokens": d.get("metadata", {}).get("raw_log_token_count"),
                "overall": pct(d.get("overall_accuracy")),
                "by_type": {t: pct(d.get("accuracy_by_type", {}).get(t)) for t in TYPES},
                "type5_dims": d.get("type_5_breakdown", {}).get("by_dimension", {}),
                "capstone": d.get("type_5_breakdown", {}).get("capstone_result"),
            }
        )

    md = ["# Context-window ablation — Julio Simmons", ""]
    md.append("**Question:** is Opus 4.7's 86.6% on Julio driven by 1M context length, or by reasoning capability that holds up under smaller context?")
    md.append("")
    md.append("**Method:** rerun Pass 1 with raw logs truncated to {32K, 64K, 128K, full ~220K} input tokens. Same test cases, same Pass 2 judge (Sonnet 4.6). Truncation cuts from the END of the log stream, so the EARLIEST signals stay intact (they tend to be higher-density meaningful sessions).")
    md.append("")
    md.append("## Overall accuracy by context budget")
    md.append("")
    md.append("| Budget | Tokens (actual) | Overall |")
    md.append("|---|---:|---:|")
    for r in rows:
        tok = r["tokens"]
        ov = r["overall"]
        md.append(f"| {r['budget']} | {tok if tok else '—'} | {ov:.1f}% |" if ov is not None else f"| {r['budget']} | {tok} | n/a |")
    md.append("")

    md.append("## Per-type accuracy curve")
    md.append("")
    md.append("| Type | " + " | ".join(r["budget"] for r in rows) + " |")
    md.append("|---|" + "---:|" * len(rows))
    for t in TYPES:
        cells = []
        for r in rows:
            v = r["by_type"].get(t)
            cells.append(f"{v:.1f}%" if v is not None else "n/a")
        md.append(f"| {t} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Type 5 dimensions across budgets (0-3 scale)")
    md.append("")
    md.append("| Dimension | " + " | ".join(r["budget"] for r in rows) + " |")
    md.append("|---|" + "---:|" * len(rows))
    for dim in ["risk_surfacing", "appropriate_response", "evidence_use", "hallucination_control"]:
        cells = []
        for r in rows:
            v = r["type5_dims"].get(dim)
            try:
                cells.append(f"{float(str(v).rstrip('%')):.2f}" if v is not None else "n/a")
            except ValueError:
                cells.append("n/a")
        md.append(f"| {dim} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Interpretation (template — fill once results land)")
    md.append("")
    md.append("Hypothesis going in:")
    md.append("- Type 2 (cross-log) and Type 3 (dynamic tracking) drop with smaller context — they need retrieval across the full window.")
    md.append("- Type 5 (agent behavior + safety) holds up — it is reasoning-bound, not retrieval-bound.")
    md.append("- Type 1 sits in the middle: single-source retrieval is robust until truncation drops the source.")
    md.append("")
    md.append("If Type 5 dimensions hold roughly flat across {64K, 128K, full} but Types 2/3 drop, the answer is reasoning > context past ~64K. That is the deployment-relevant finding: most production deployments do not have 1M-context budgets, and PersonaBench's safety probe survives the cut.")
    md.append("")
    md.append("If Type 5 drops sharply at 32K or 64K, then context is structurally load-bearing for safety-grounded persona inference, and the 1M frontier really is the relevant architecture.")
    md.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
