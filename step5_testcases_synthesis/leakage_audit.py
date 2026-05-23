from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


LEAKAGE_MAX_OVERALL_PCT_DIRECTNESS_3_OR_4 = 0.50
LEAKAGE_MAX_TYPE4_PCT_DIRECTNESS_3_OR_4 = 0.4
LEAKAGE_MAX_TYPE5_PCT_DIRECTNESS_3_OR_4 = 0.4

LEAKAGE_FAIL_TYPES = {
    "type_4_reasoning",
    "type_5_agent_behavior",
}


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "to", "in", "on", "for", "from", "with", "without", "by", "as", "at",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "has", "have", "had", "can", "could", "would", "should", "will", "may",
    "might", "must", "this", "that", "these", "those", "it", "its", "into",
    "about", "over", "under", "between", "through", "during", "before", "after",
    "what", "why", "how", "who", "whom", "whose", "which", "where", "there",
    "their", "them", "they", "he", "she", "his", "her", "hers", "him", "person",
    "persona", "user", "logs", "evidence", "suggests", "show", "shows",
    "indicates", "indicate", "based", "given", "full", "history", "explain",
    "describe", "support", "supports", "interpretation", "better", "plausible",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\$?\d+(?:[.,]\d+)?%?")
DATEISH_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"
    r"|\b20\d{2}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)


@dataclass
class LogEntry:
    entry_id: str
    source_type: str
    date: str
    text: str


@dataclass
class LeakageTaskMetric:
    task_id: str
    task_type: str
    subtype: str
    ground_truth: str
    evidence_directness_level: int
    evidence_directness_label: str
    max_log_similarity: float
    max_log_gt_containment: float
    max_rare_term_containment: float
    direct_match_count: int
    moderate_match_count: int
    failed_leakage_gate: bool
    best_log_entry_id: str
    best_log_source_type: str
    best_log_date: str
    best_log_excerpt: str


def run_leakage_audit(test_cases: dict[str, Any], app_logs: dict[str, Any]) -> dict[str, Any]:
    log_entries = flatten_log_entries(app_logs)
    tasks = test_cases.get("test_cases", [])

    task_metrics = [
        analyze_task_directness(task, log_entries)
        for task in tasks
    ]

    by_type: dict[str, list[LeakageTaskMetric]] = defaultdict(list)
    for row in task_metrics:
        by_type[row.task_type].append(row)

    summary_by_type = {
        task_type: _summarize_directness(rows)
        for task_type, rows in sorted(by_type.items())
    }

    overall = _summarize_directness(task_metrics)

    type4_pct = summary_by_type.get("type_4_reasoning", {}).get("pct_directness_3_or_4", 0.0)
    type5_pct = summary_by_type.get("type_5_agent_behavior", {}).get("pct_directness_3_or_4", 0.0)
    overall_pct = overall.get("pct_directness_3_or_4", 0.0)

    failed_tasks = [
        asdict(row)
        for row in task_metrics
        if row.failed_leakage_gate
    ]

    passed = (
        overall_pct <= LEAKAGE_MAX_OVERALL_PCT_DIRECTNESS_3_OR_4
        and type4_pct <= LEAKAGE_MAX_TYPE4_PCT_DIRECTNESS_3_OR_4
        and type5_pct <= LEAKAGE_MAX_TYPE5_PCT_DIRECTNESS_3_OR_4
        and not failed_tasks
    )

    return {
        "passed": passed,
        "policy": {
            "max_overall_pct_directness_3_or_4": LEAKAGE_MAX_OVERALL_PCT_DIRECTNESS_3_OR_4,
            "max_type4_pct_directness_3_or_4": LEAKAGE_MAX_TYPE4_PCT_DIRECTNESS_3_OR_4,
            "max_type5_pct_directness_3_or_4": LEAKAGE_MAX_TYPE5_PCT_DIRECTNESS_3_OR_4,
            "task_level_fail_types": sorted(LEAKAGE_FAIL_TYPES),
            "task_level_fail_rule": "Type 4/5 fail if evidence_directness_level >= 3",
        },
        "overall": overall,
        "summary_by_type": summary_by_type,
        "failed_tasks": failed_tasks,
        "task_metrics": [asdict(row) for row in task_metrics],
        "log_entry_count": len(log_entries),
    }


def write_leakage_audit_outputs(
    audit: dict[str, Any],
    audit_json_path: Path,
    audit_csv_path: Path,
) -> None:
    audit_json_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = audit.get("task_metrics", [])
    if not rows:
        audit_csv_path.write_text("", encoding="utf-8")
        return

    with audit_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_leakage_audit_output(audit: dict[str, Any], audit_json_path: Path) -> None:
    audit_json_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def analyze_task_directness(task: dict[str, Any], log_entries: list[LogEntry]) -> LeakageTaskMetric:
    ground_truth = task.get("ground_truth", "") or ""
    task_type = str(task.get("type", ""))
    subtype = str(task.get("subtype", ""))

    best_entry = LogEntry("", "", "", "")
    best_scores = {
        "similarity": 0.0,
        "gt_containment": 0.0,
        "rare_containment": 0.0,
    }

    direct_count = 0
    moderate_count = 0

    for entry in log_entries:
        scores = _score_log_against_ground_truth(entry.text, ground_truth)

        is_direct = (
            scores["gt_containment"] >= 0.60
            or scores["rare_containment"] >= 0.75
            or scores["similarity"] >= 0.45
        )
        is_moderate = (
            scores["gt_containment"] >= 0.35
            or scores["rare_containment"] >= 0.50
            or scores["similarity"] >= 0.28
        )

        if is_direct:
            direct_count += 1
        elif is_moderate:
            moderate_count += 1

        if scores["similarity"] > best_scores["similarity"]:
            best_scores = scores
            best_entry = entry

    level, label = _classify_directness(
        max_sim=best_scores["similarity"],
        max_gt_containment=best_scores["gt_containment"],
        max_rare_containment=best_scores["rare_containment"],
        direct_count=direct_count,
        moderate_count=moderate_count,
    )

    excerpt = best_entry.text
    if len(excerpt) > 500:
        excerpt = excerpt[:500] + "..."

    return LeakageTaskMetric(
        task_id=str(task.get("id", "")),
        task_type=task_type,
        subtype=subtype,
        ground_truth=ground_truth,
        evidence_directness_level=level,
        evidence_directness_label=label,
        max_log_similarity=round(best_scores["similarity"], 4),
        max_log_gt_containment=round(best_scores["gt_containment"], 4),
        max_rare_term_containment=round(best_scores["rare_containment"], 4),
        direct_match_count=direct_count,
        moderate_match_count=moderate_count,
        failed_leakage_gate=task_type in LEAKAGE_FAIL_TYPES and level >= 3,
        best_log_entry_id=best_entry.entry_id,
        best_log_source_type=best_entry.source_type,
        best_log_date=best_entry.date,
        best_log_excerpt=excerpt,
    )


def flatten_log_entries(app_logs: Any) -> list[LogEntry]:
    entries: list[LogEntry] = []

    def walk(obj: Any, path: list[str]) -> None:
        if isinstance(obj, dict):
            if path and path[-1] == "hidden_facts":
                return

            strings = _flatten_strings(obj)
            text = " | ".join(strings)
            if len(text) >= 30:
                entry_id = (
                    str(obj.get("id"))
                    or str(obj.get("event_id"))
                    or str(obj.get("message_id"))
                    or f"entry_{len(entries)+1:05d}"
                )
                entries.append(
                    LogEntry(
                        entry_id=entry_id,
                        source_type=_infer_source_type(path, obj),
                        date=_get_dateish_from_obj(obj),
                        text=text,
                    )
                )

            for k, v in obj.items():
                if k == "hidden_facts":
                    continue
                walk(v, path + [str(k)])

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, path + [str(i)])

    walk(app_logs, [])

    seen = set()
    deduped = []
    for e in entries:
        key = _normalize(e.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    return deduped


def _summarize_directness(rows: list[LeakageTaskMetric]) -> dict[str, Any]:
    if not rows:
        return {
            "n_tasks": 0,
            "avg_evidence_directness_level": 0.0,
            "pct_directness_3_or_4": 0.0,
            "directness_level_counts": {},
            "failed_task_count": 0,
        }

    counts = Counter(r.evidence_directness_level for r in rows)
    avg = sum(r.evidence_directness_level for r in rows) / len(rows)
    pct_3_or_4 = sum(1 for r in rows if r.evidence_directness_level >= 3) / len(rows)

    return {
        "n_tasks": len(rows),
        "avg_evidence_directness_level": round(avg, 4),
        "pct_directness_3_or_4": round(pct_3_or_4, 4),
        "directness_level_counts": dict(sorted(counts.items())),
        "failed_task_count": sum(1 for r in rows if r.failed_leakage_gate),
    }


def _score_log_against_ground_truth(log_text: str, ground_truth: str) -> dict[str, float]:
    log_set = _token_set(log_text)
    gt_set = _token_set(ground_truth)

    rare_gt = _extract_rareish_terms(ground_truth)
    rare_log = _extract_rareish_terms(log_text)

    gt_cont = _containment(log_set, gt_set)
    rare_cont = len(rare_gt & rare_log) / len(rare_gt) if rare_gt else 0.0
    jac = _jaccard(log_set, gt_set)
    seq = _sequence_ratio(log_text[:1000], ground_truth)

    return {
        "similarity": max(jac, 0.65 * gt_cont + 0.35 * rare_cont, seq),
        "gt_containment": gt_cont,
        "rare_containment": rare_cont,
        "jaccard": jac,
        "sequence_ratio": seq,
    }


def _classify_directness(
    max_sim: float,
    max_gt_containment: float,
    max_rare_containment: float,
    direct_count: int,
    moderate_count: int,
) -> tuple[int, str]:
    if direct_count >= 2:
        return 4, "direct_or_answer_shaped_repeated"
    if direct_count == 1:
        return 3, "direct_or_answer_shaped_once"
    if moderate_count >= 2:
        return 2, "moderate_support_multiple_entries"
    if max_sim >= 0.28 or max_gt_containment >= 0.40 or max_rare_containment >= 0.50:
        return 2, "moderate_support_one_entry"
    if max_sim >= 0.16 or max_gt_containment >= 0.20 or max_rare_containment >= 0.25:
        return 1, "weak_or_indirect_lexical_support"
    return 0, "no_obvious_lexical_support"


def _normalize(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    if keep_stopwords:
        return toks
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(source: set[str], target: set[str]) -> float:
    if not target:
        return 0.0
    return len(source & target) / len(target)


def _sequence_ratio(a: str, b: str) -> float:
    a = _normalize(a)
    b = _normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _extract_rareish_terms(text: str) -> set[str]:
    raw = TOKEN_RE.findall(text or "")
    out = set()

    for t in raw:
        low = t.lower()
        if low in STOPWORDS:
            continue
        if re.search(r"\d", low):
            out.add(low)
        elif len(low) >= 6:
            out.add(low)

    for m in DATEISH_RE.findall(text or ""):
        out.add(m.lower())

    return out


def _flatten_strings(obj: Any, parent_key: str = "") -> list[str]:
    skip_keys = {
        "hidden_facts",
        "source_hidden_fact_ids",
        "ground_truth",
        "ground_truth_label",
        "fact_id",
        "source_subcategories",
    }

    strings: list[str] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            strings.extend(_flatten_strings(v, k))
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(_flatten_strings(item, parent_key))
    elif isinstance(obj, str):
        s = obj.strip()
        if s:
            strings.append(s)
    elif isinstance(obj, (int, float)) and parent_key.lower() in {
        "amount", "price", "cost", "date", "time", "start", "end"
    }:
        strings.append(str(obj))

    return strings


def _get_dateish_from_obj(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""

    for k in [
        "date", "datetime", "timestamp", "created_at", "start", "start_time",
        "end", "end_time", "time", "event_date",
    ]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    text = " ".join(_flatten_strings(obj))
    m = DATEISH_RE.search(text)
    return m.group(0) if m else ""


def _infer_source_type(path_parts: list[str], obj: Any) -> str:
    joined = " ".join(path_parts).lower()
    if "calendar" in joined or "event" in joined:
        return "calendar"
    if "messenger" in joined or "message" in joined or "conversation" in joined or "session" in joined:
        return "messenger"

    if isinstance(obj, dict):
        keys = {str(k).lower() for k in obj.keys()}
        if {"start", "end"} & keys or "attendees" in keys:
            return "calendar"
        if {"sender", "recipient", "message", "messages"} & keys:
            return "messenger"

    return "unknown"


def run_and_write_leakage_audit(
    test_cases: dict[str, Any],
    app_logs: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    audit = run_leakage_audit(test_cases, app_logs)
    write_leakage_audit_output(audit, output_path)
    return audit

def format_leakage_audit_summary(audit: dict[str, Any]) -> str:
    overall = audit.get("overall", {})
    by_type = audit.get("summary_by_type", {})

    type4 = by_type.get("type_4_reasoning", {})
    type5 = by_type.get("type_5_agent_behavior", {})

    return (
        f"leakage_passed={audit.get('passed', False)} "
        f"overall_pct_3_or_4={overall.get('pct_directness_3_or_4', 0.0):.4f} "
        f"type4_pct_3_or_4={type4.get('pct_directness_3_or_4', 0.0):.4f} "
        f"type5_pct_3_or_4={type5.get('pct_directness_3_or_4', 0.0):.4f} "
        f"failed_tasks={len(audit.get('failed_tasks', []))}"
    )