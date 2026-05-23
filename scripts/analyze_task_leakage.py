#!/usr/bin/env python3
"""
Analyze PersonaBench task leakage and app-log directness.

Experiments:
D. Question/ground-truth leakage:
   - Do task questions contain answer-shaped terms from ground_truth?
   - Are questions lexically too similar to ground_truth?

E. Evidence directness audit:
   - Are hidden facts / ground truths directly stated in app logs?
   - Are answers repeated explicitly across multiple log entries?

Outputs:
  - all_task_metrics.csv
  - flagged_question_leakage.csv
  - evidence_directness.csv
  - persona_summary.csv
  - summary.json

Usage:
  python scripts/analyze_task_leakage.py \
    --tasks-glob "data_samples/output/*_test_cases.json" \
    --out-dir analysis_outputs/leakage_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# -----------------------------
# Basic text utilities
# -----------------------------

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


def normalize(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    if keep_stopwords:
        return toks
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(source: set[str], target: set[str]) -> float:
    """
    How much of target appears in source.
    For leakage: how much of ground_truth appears in question.
    """
    if not target:
        return 0.0
    return len(source & target) / len(target)


def sequence_ratio(a: str, b: str) -> float:
    a = normalize(a)
    b = normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def extract_rareish_terms(text: str) -> set[str]:
    """
    Cheap proxy for important answer-bearing terms:
    - non-stopword tokens length >= 6
    - numbers / money / percentages
    - date-ish strings
    """
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


def longest_common_ngram_len(a: str, b: str, max_n: int = 6) -> int:
    """
    Longest shared token n-gram length, excluding stopword-only ngrams.
    """
    at = tokens(a, keep_stopwords=True)
    bt = tokens(b, keep_stopwords=True)
    if not at or not bt:
        return 0

    b_ngrams_by_n: dict[int, set[tuple[str, ...]]] = {}
    for n in range(1, min(max_n, len(bt)) + 1):
        b_ngrams_by_n[n] = {
            tuple(bt[i:i+n])
            for i in range(len(bt) - n + 1)
        }

    best = 0
    for n in range(1, min(max_n, len(at)) + 1):
        for i in range(len(at) - n + 1):
            ng = tuple(at[i:i+n])
            if all(tok in STOPWORDS for tok in ng):
                continue
            if ng in b_ngrams_by_n.get(n, set()):
                best = max(best, n)
    return best


# -----------------------------
# JSON loading / flattening logs
# -----------------------------

@dataclass
class LogEntry:
    entry_id: str
    source_type: str
    date: str
    text: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_app_logs_path(task_file: Path, task_data: dict[str, Any]) -> Path | None:
    """
    Resolve app log path for this repo structure.

    Expected task files:
      step5_testcases_synthesis/data_samples/output/{persona}_test_cases.json

    Expected app log files:
      step4_app_log_synthesizer/data_samples/output/{persona}_app_logs.json
      step4_app_log_synthesizer/data_samples/output/{persona}_YYYYMMDD_YYYYMMDD_app_logs.json

    Also supports metadata.input_sources.app_logs if present.
    """
    candidates: list[Path] = []

    # 1. Try metadata path first, if generated tasks include it.
    src = (
        task_data.get("metadata", {})
        .get("input_sources", {})
        .get("app_logs")
    )
    if src:
        src_path = Path(src)
        candidates.extend([
            src_path,
            Path.cwd() / src_path,
            task_file.parent / src_path,
            task_file.parent.parent / src_path,
            task_file.parent.parent.parent / src_path,
        ])

    # 2. Infer persona slug from task filename.
    #    e.g. alicia_gonzalez_test_cases.json -> alicia_gonzalez
    persona_slug = (
        task_file.stem
        .replace("_test_cases_verification", "")
        .replace("_test_cases", "")
        .replace("-test-cases", "")
    )

    app_log_dir_candidates = [
        Path("step4_app_log_synthesizer/data_samples/output"),
        Path.cwd() / "step4_app_log_synthesizer/data_samples/output",
        task_file.parents[2] / "step4_app_log_synthesizer/data_samples/output"
        if len(task_file.parents) >= 3 else None,
    ]

    for app_log_dir in app_log_dir_candidates:
        if app_log_dir is None:
            continue

        # Prefer canonical non-date file.
        candidates.append(app_log_dir / f"{persona_slug}_app_logs.json")

        # Then date-windowed outputs.
        candidates.extend(sorted(app_log_dir.glob(f"{persona_slug}_*_app_logs.json")))

    # 3. Fallback for your alicia_condensed folder.
    candidates.extend(sorted(Path.cwd().glob(f"{persona_slug}*/**/*_app_logs.json")))

    # Exclude trace files.
    for c in candidates:
        if c and c.exists() and not c.name.endswith("_app_logs_trace.json"):
            return c.resolve()

    return None


def flatten_strings(obj: Any, parent_key: str = "") -> list[str]:
    """
    Recursively collect human-readable string fields.
    Skip hidden_facts because those are metadata/answer-key-ish, not raw logs.
    """
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
            strings.extend(flatten_strings(v, k))
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(flatten_strings(item, parent_key))
    elif isinstance(obj, str):
        s = obj.strip()
        if s:
            strings.append(s)
    elif isinstance(obj, (int, float)) and parent_key.lower() in {
        "amount", "price", "cost", "date", "time", "start", "end"
    }:
        strings.append(str(obj))

    return strings


def get_dateish_from_obj(obj: Any) -> str:
    """
    Best-effort extraction of a date/time field from a log object.
    """
    if not isinstance(obj, dict):
        return ""

    date_keys = [
        "date", "datetime", "timestamp", "created_at", "start", "start_time",
        "end", "end_time", "time", "event_date"
    ]
    for k in date_keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # fallback: find date-ish in any string field
    text = " ".join(flatten_strings(obj))
    m = DATEISH_RE.search(text)
    return m.group(0) if m else ""


def infer_source_type(path_parts: list[str], obj: Any) -> str:
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


def flatten_log_entries(app_logs: Any) -> list[LogEntry]:
    """
    Convert arbitrary app_logs JSON into entry-level text chunks.

    This is intentionally schema-tolerant:
    - each dict with several strings becomes one entry
    - list elements under messages/events/conversations become separate entries
    - hidden_facts are skipped
    """
    entries: list[LogEntry] = []

    def walk(obj: Any, path: list[str]) -> None:
        if isinstance(obj, dict):
            if path and path[-1] == "hidden_facts":
                return

            # If this dict looks like a meaningful log object, make an entry.
            strings = flatten_strings(obj)
            text = " | ".join(strings)
            if len(text) >= 30:
                entry_id = (
                    str(obj.get("id"))
                    or str(obj.get("event_id"))
                    or str(obj.get("message_id"))
                    or f"entry_{len(entries)+1:05d}"
                )
                source_type = infer_source_type(path, obj)
                date = get_dateish_from_obj(obj)
                entries.append(LogEntry(entry_id=entry_id, source_type=source_type, date=date, text=text))

                # Still walk children if there are nested messages/events.
                # This may create duplicates, but duplicate directness is informative.
                # We dedupe later by exact text.
            for k, v in obj.items():
                if k == "hidden_facts":
                    continue
                walk(v, path + [str(k)])

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, path + [str(i)])

    walk(app_logs, [])

    # Dedupe exact text while preserving first metadata.
    seen = set()
    deduped = []
    for e in entries:
        key = normalize(e.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    return deduped


# -----------------------------
# Experiment D: question leakage
# -----------------------------

@dataclass
class QuestionLeakageMetrics:
    task_file: str
    persona: str
    task_id: str
    task_type: str
    subtype: str
    question: str
    ground_truth: str
    question_gt_jaccard: float
    gt_token_containment_in_question: float
    rare_gt_terms: str
    rare_gt_terms_in_question: str
    rare_gt_term_containment: float
    longest_shared_ngram: int
    sequence_ratio: float
    leakage_flag: bool
    leakage_reasons: str


def analyze_question_leakage(task_file: Path, persona: str, task: dict[str, Any]) -> QuestionLeakageMetrics:
    question = task.get("question", "") or ""
    ground_truth = task.get("ground_truth", "") or ""

    q_set = token_set(question)
    gt_set = token_set(ground_truth)

    rare_gt = extract_rareish_terms(ground_truth)
    rare_q = extract_rareish_terms(question)
    rare_overlap = rare_gt & rare_q

    jac = jaccard(q_set, gt_set)
    cont = containment(q_set, gt_set)
    rare_cont = len(rare_overlap) / len(rare_gt) if rare_gt else 0.0
    lcn = longest_common_ngram_len(question, ground_truth)
    seq = sequence_ratio(question, ground_truth)

    reasons = []
    if cont >= 0.50 and len(gt_set) >= 4:
        reasons.append("high_gt_token_containment")
    if rare_cont >= 0.50 and len(rare_gt) >= 2:
        reasons.append("high_rare_term_overlap")
    if lcn >= 3:
        reasons.append("shared_3plus_token_phrase")
    if seq >= 0.45:
        reasons.append("high_sequence_similarity")
    if jac >= 0.25 and len(gt_set) >= 4:
        reasons.append("high_jaccard")

    return QuestionLeakageMetrics(
        task_file=str(task_file),
        persona=persona,
        task_id=str(task.get("id", "")),
        task_type=str(task.get("type", "")),
        subtype=str(task.get("subtype", "")),
        question=question,
        ground_truth=ground_truth,
        question_gt_jaccard=round(jac, 4),
        gt_token_containment_in_question=round(cont, 4),
        rare_gt_terms=" | ".join(sorted(rare_gt)),
        rare_gt_terms_in_question=" | ".join(sorted(rare_overlap)),
        rare_gt_term_containment=round(rare_cont, 4),
        longest_shared_ngram=lcn,
        sequence_ratio=round(seq, 4),
        leakage_flag=bool(reasons),
        leakage_reasons=";".join(reasons),
    )


# -----------------------------
# Experiment E: evidence directness
# -----------------------------

@dataclass
class EvidenceDirectnessMetrics:
    task_file: str
    app_logs_file: str
    persona: str
    task_id: str
    task_type: str
    subtype: str
    source_hidden_fact_ids: str
    ground_truth: str
    max_log_similarity: float
    max_log_gt_containment: float
    max_rare_term_containment: float
    direct_match_count: int
    moderate_match_count: int
    evidence_directness_level: int
    evidence_directness_label: str
    best_log_entry_id: str
    best_log_source_type: str
    best_log_date: str
    best_log_excerpt: str


def score_log_against_ground_truth(log_text: str, ground_truth: str) -> dict[str, float]:
    log_set = token_set(log_text)
    gt_set = token_set(ground_truth)

    rare_gt = extract_rareish_terms(ground_truth)
    rare_log = extract_rareish_terms(log_text)

    gt_cont = containment(log_set, gt_set)
    rare_cont = len(rare_gt & rare_log) / len(rare_gt) if rare_gt else 0.0
    jac = jaccard(log_set, gt_set)
    seq = sequence_ratio(log_text[:1000], ground_truth)

    # Weighted directness-ish score.
    sim = max(
        jac,
        0.65 * gt_cont + 0.35 * rare_cont,
        seq,
    )

    return {
        "similarity": sim,
        "gt_containment": gt_cont,
        "rare_containment": rare_cont,
        "jaccard": jac,
        "sequence_ratio": seq,
    }


def classify_directness(
    max_sim: float,
    max_gt_containment: float,
    max_rare_containment: float,
    direct_count: int,
    moderate_count: int,
) -> tuple[int, str]:
    """
    Heuristic 0-4 scale:
      0 = no obvious support
      1 = weak/indirect lexical support
      2 = moderate implied support
      3 = directly stated once
      4 = directly stated/repeated multiple times
    """
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


def analyze_evidence_directness(
    task_file: Path,
    app_logs_file: Path,
    persona: str,
    task: dict[str, Any],
    log_entries: list[LogEntry],
) -> EvidenceDirectnessMetrics:
    gt = task.get("ground_truth", "") or ""

    best_entry = LogEntry("", "", "", "")
    best_scores = {
        "similarity": 0.0,
        "gt_containment": 0.0,
        "rare_containment": 0.0,
        "jaccard": 0.0,
        "sequence_ratio": 0.0,
    }

    direct_count = 0
    moderate_count = 0

    for entry in log_entries:
        scores = score_log_against_ground_truth(entry.text, gt)

        # Direct-ish means much of the answer-bearing text appears in one entry.
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

    level, label = classify_directness(
        max_sim=best_scores["similarity"],
        max_gt_containment=best_scores["gt_containment"],
        max_rare_containment=best_scores["rare_containment"],
        direct_count=direct_count,
        moderate_count=moderate_count,
    )

    excerpt = best_entry.text
    if len(excerpt) > 500:
        excerpt = excerpt[:500] + "..."

    return EvidenceDirectnessMetrics(
        task_file=str(task_file),
        app_logs_file=str(app_logs_file),
        persona=persona,
        task_id=str(task.get("id", "")),
        task_type=str(task.get("type", "")),
        subtype=str(task.get("subtype", "")),
        source_hidden_fact_ids=" | ".join(task.get("source_hidden_fact_ids", []) or []),
        ground_truth=gt,
        max_log_similarity=round(best_scores["similarity"], 4),
        max_log_gt_containment=round(best_scores["gt_containment"], 4),
        max_rare_term_containment=round(best_scores["rare_containment"], 4),
        direct_match_count=direct_count,
        moderate_match_count=moderate_count,
        evidence_directness_level=level,
        evidence_directness_label=label,
        best_log_entry_id=best_entry.entry_id,
        best_log_source_type=best_entry.source_type,
        best_log_date=best_entry.date,
        best_log_excerpt=excerpt,
    )


# -----------------------------
# Summaries / CSV writing
# -----------------------------

def write_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    dict_rows = [asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r) for r in rows]
    fieldnames = list(dict_rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize(
    leakage_rows: list[QuestionLeakageMetrics],
    directness_rows: list[EvidenceDirectnessMetrics],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    leakage_by_key: dict[tuple[str, str], list[bool]] = defaultdict(list)

    for r in directness_rows:
        by_key[(r.persona, r.task_type)].append(r.evidence_directness_level)

    for r in leakage_rows:
        leakage_by_key[(r.persona, r.task_type)].append(r.leakage_flag)

    persona_task_keys = sorted(set(by_key) | set(leakage_by_key))

    summary_rows = []
    for persona, task_type in persona_task_keys:
        levels = by_key.get((persona, task_type), [])
        flags = leakage_by_key.get((persona, task_type), [])

        n = max(len(levels), len(flags))
        level_counts = Counter(levels)

        summary_rows.append({
            "persona": persona,
            "task_type": task_type,
            "n_tasks": n,
            "question_leakage_flag_rate": round(sum(flags) / len(flags), 4) if flags else 0.0,
            "avg_evidence_directness_level": round(mean([float(x) for x in levels]), 4) if levels else 0.0,
            "pct_directness_3_or_4": round(sum(1 for x in levels if x >= 3) / len(levels), 4) if levels else 0.0,
            "directness_0": level_counts.get(0, 0),
            "directness_1": level_counts.get(1, 0),
            "directness_2": level_counts.get(2, 0),
            "directness_3": level_counts.get(3, 0),
            "directness_4": level_counts.get(4, 0),
        })

    overall = {
        "n_tasks": len(leakage_rows),
        "question_leakage_flag_rate": round(
            sum(1 for r in leakage_rows if r.leakage_flag) / len(leakage_rows), 4
        ) if leakage_rows else 0.0,
        "avg_evidence_directness_level": round(
            mean([float(r.evidence_directness_level) for r in directness_rows]), 4
        ) if directness_rows else 0.0,
        "pct_directness_3_or_4": round(
            sum(1 for r in directness_rows if r.evidence_directness_level >= 3) / len(directness_rows), 4
        ) if directness_rows else 0.0,
        "directness_level_counts": dict(Counter(r.evidence_directness_level for r in directness_rows)),
    }

    return summary_rows, overall


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks-glob",
        default="data_samples/output/*_test_cases.json",
        help="Glob for generated task files.",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis_outputs/leakage_audit",
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--fail-on-missing-logs",
        action="store_true",
        help="Raise if app_logs file cannot be found for a task file.",
    )
    args = parser.parse_args()

    task_files = sorted(Path().glob(args.tasks_glob))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not task_files:
        raise FileNotFoundError(f"No task files matched glob: {args.tasks_glob}")

    leakage_rows: list[QuestionLeakageMetrics] = []
    directness_rows: list[EvidenceDirectnessMetrics] = []
    missing_logs: list[str] = []

    for task_file in task_files:
        task_data = load_json(task_file)
        metadata = task_data.get("metadata", {})
        persona = metadata.get("persona") or task_file.stem.replace("_test_cases", "")

        tasks = task_data.get("test_cases", [])
        if not isinstance(tasks, list):
            print(f"Skipping {task_file}: no test_cases list")
            continue

        app_logs_path = find_app_logs_path(task_file, task_data)

        log_entries: list[LogEntry] = []
        if app_logs_path is None:
            missing_logs.append(str(task_file))
            if args.fail_on_missing_logs:
                raise FileNotFoundError(f"Could not find app logs for {task_file}")
            print(f"Warning: could not find app logs for {task_file}; skipping Experiment E")
        else:
            app_logs = load_json(app_logs_path)
            log_entries = flatten_log_entries(app_logs)
            print(f"{task_file}: loaded {len(tasks)} tasks, {len(log_entries)} log entries from {app_logs_path}")

        for task in tasks:
            leakage_rows.append(analyze_question_leakage(task_file, persona, task))

            if app_logs_path is not None:
                directness_rows.append(
                    analyze_evidence_directness(
                        task_file=task_file,
                        app_logs_file=app_logs_path,
                        persona=persona,
                        task=task,
                        log_entries=log_entries,
                    )
                )

    flagged = [r for r in leakage_rows if r.leakage_flag]
    summary_rows, overall = summarize(leakage_rows, directness_rows)

    write_csv(out_dir / "all_task_question_leakage_metrics.csv", leakage_rows)
    write_csv(out_dir / "flagged_question_leakage.csv", flagged)
    write_csv(out_dir / "evidence_directness.csv", directness_rows)
    write_csv(out_dir / "persona_summary.csv", summary_rows)

    summary = {
        "tasks_glob": args.tasks_glob,
        "n_task_files": len(task_files),
        "missing_logs_for_task_files": missing_logs,
        "overall": overall,
        "notes": {
            "experiment_d": "Question-ground_truth leakage via token overlap, rare-term overlap, shared ngrams, and sequence similarity.",
            "experiment_e": "Evidence directness via lexical similarity between ground_truth and flattened raw app log entries.",
            "directness_scale": {
                "0": "no obvious lexical support",
                "1": "weak or indirect lexical support",
                "2": "moderate support",
                "3": "direct or answer-shaped once",
                "4": "direct or answer-shaped repeated",
            },
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Output directory: {out_dir}")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()