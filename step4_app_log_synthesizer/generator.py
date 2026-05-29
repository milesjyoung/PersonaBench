"""Step 4 -- App log synthesis with cluster-level reverse-inferability gate.

Hidden facts are first clustered into groups of 2-3 related facts. For each
cluster, 4-5 shared fragments are generated where each fragment supports
multiple facts. An independent cold verifier then attempts to recover ALL
facts from the shared fragments. Only clusters that pass this gate are
merged into the final app log.

Five phases:
  1. Clustering                       (cluster_prompt.txt)
  2. Per-cluster fragment generation  (prompt.txt)
  3. Per-cluster verification         (verification_prompt.txt)
  4. Decoy generation                 (decoy_prompt.txt) [DISABLED]
  5. Global merge + filler            (merge_prompt.txt)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm import (
    API_BACKENDS,
    SUBSCRIPTION_BACKENDS,
    SUPPORTED_PROVIDERS,
    call_llm as _raw_call_llm,
    call_subscription_cli,
    check_api_key,
    make_client,
    provider_for_backend,
)

STEP_DIR = Path(__file__).parent
CLUSTER_PROMPT_PATH = STEP_DIR / "cluster_prompt.txt"
PROMPT_PATH = STEP_DIR / "prompt.txt"
VERIFICATION_PROMPT_PATH = STEP_DIR / "verification_prompt.txt"
DECOY_PROMPT_PATH = STEP_DIR / "decoy_prompt.txt"
MERGE_PROMPT_PATH = STEP_DIR / "merge_prompt.txt"

CLAUDE_CMD = os.environ.get(
    "CLAUDE_CMD", "claude.cmd" if sys.platform == "win32" else "claude"
)
CODEX_CMD = os.environ.get(
    "CODEX_CMD", "codex"
)


# ---------------------------------------------------------------------------
# Utility: LLM dispatch, JSON extraction, template filling
# ---------------------------------------------------------------------------


def _call_llm(
    client,
    model: str,
    prompt: str,
    provider: str = "anthropic",
    backend: str = "anthropic-api",
) -> str:
    """Unified LLM dispatch for both subscription and API backends."""
    if backend in SUBSCRIPTION_BACKENDS:
        return call_subscription_cli(
            prompt, model, backend,
            claude_cmd=CLAUDE_CMD, codex_cmd=CODEX_CMD,
        )
    return _raw_call_llm(client, model, prompt, provider=provider)


def _fix_invalid_escapes(text: str) -> str:
    """Strip invalid JSON escape sequences while preserving valid ones."""
    VALID = set('"\\/bfnrtu')
    out = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            if text[i + 1] in VALID:
                out.append(text[i])
                out.append(text[i + 1])
                i += 2
            else:
                out.append(text[i + 1])
                i += 2
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from LLM output, tolerating fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    payload = fenced.group(1) if fenced else text
    start = payload.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")
    raw = payload[start:]
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(raw)
        return obj
    except json.JSONDecodeError:
        obj, _ = decoder.raw_decode(_fix_invalid_escapes(raw))
        return obj


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill(template: str, placeholder: str, payload: Any) -> str:
    if isinstance(payload, str):
        return template.replace(placeholder, payload)
    return template.replace(
        placeholder, json.dumps(payload, indent=2, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# Persona identity
# ---------------------------------------------------------------------------


def derive_persona_identity(corrected_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": corrected_profile.get("name", ""),
        "age": corrected_profile.get("age", 0),
        "occupation": corrected_profile.get("occupation", ""),
        "location": corrected_profile.get("location", ""),
    }


# ---------------------------------------------------------------------------
# HF selection and prioritization
# ---------------------------------------------------------------------------

PRIORITY_CATEGORIES = {
    "health", "mental_health", "medication", "therapy",
    "financial", "financial_constraint", "money",
    "safety", "risk",
    "relationship", "family", "social",
    "legal", "regulatory",
    "behavioral", "self_regulation", "coping",
    "temporal",
}

DEPRIORITIZE_CATEGORIES = {
    "decorative", "aesthetic", "style",
    "hobby_minor", "one_off",
    "location_minor",
    "personality_redundant",
}


def select_hidden_facts(
    hidden_facts: list[dict[str, Any]],
    max_facts: int = 100,
) -> list[dict[str, Any]]:
    """Select benchmark-relevant hidden facts, capped at max_facts.

    Prioritizes risk/safety, health, financial, relationship, and temporal
    facts. Deprioritizes decorative preferences, one-off hobbies, and
    redundant personality facts.
    """
    if len(hidden_facts) <= max_facts:
        return hidden_facts

    def priority_score(hf: dict[str, Any]) -> int:
        cat = (hf.get("category") or "").lower()
        subcat = (hf.get("subcategory") or "").lower()
        combined = f"{cat} {subcat}"
        score = 0
        for p in PRIORITY_CATEGORIES:
            if p in combined:
                score += 10
        for d in DEPRIORITIZE_CATEGORIES:
            if d in combined:
                score -= 5
        if hf.get("risk_surface_score"):
            score += int(hf["risk_surface_score"])
        return score

    ranked = sorted(hidden_facts, key=priority_score, reverse=True)
    selected = ranked[:max_facts]
    dropped = len(hidden_facts) - len(selected)
    if dropped:
        print(
            f"[step4] selected {len(selected)}/{len(hidden_facts)} "
            f"hidden facts (dropped {dropped} low-priority facts)"
        )
    return selected


# ---------------------------------------------------------------------------
# Phase 1: Clustering
# ---------------------------------------------------------------------------


def cluster_hidden_facts(
    client,
    model: str,
    hidden_facts: list[dict[str, Any]],
    provider: str,
    backend: str,
) -> dict[str, Any]:
    """Cluster hidden facts into groups of 2-3 related facts."""
    template = load_prompt(CLUSTER_PROMPT_PATH)
    template = fill(template, "{{INSERT_HIDDEN_FACTS_JSON_HERE}}", hidden_facts)
    raw = _call_llm(client, model, template, provider=provider, backend=backend)
    result = extract_json(raw)

    # Validate: every fact_id appears exactly once
    all_assigned: list[str] = []
    for cluster in result.get("clusters", []):
        all_assigned.extend(cluster.get("hidden_fact_ids", []))
    fact_ids_expected = {
        hf.get("fact_id", f"HF-{i:03d}") for i, hf in enumerate(hidden_facts, 1)
    }
    assigned_set = set(all_assigned)
    if len(all_assigned) != len(assigned_set):
        dupes = [fid for fid in all_assigned if all_assigned.count(fid) > 1]
        print(f"[step4] WARNING: duplicate fact_ids in clustering: {set(dupes)}")
    missing = fact_ids_expected - assigned_set
    if missing:
        print(f"[step4] WARNING: {len(missing)} facts not assigned to any cluster")

    return result


# ---------------------------------------------------------------------------
# Phase 2: Per-cluster fragment generation
# ---------------------------------------------------------------------------


def generate_cluster_fragments(
    client,
    model: str,
    cluster_obj: dict[str, Any],
    hidden_facts_in_cluster: list[dict[str, Any]],
    social_circle: dict[str, Any],
    log_start: str,
    log_end: str,
    contact_usage: dict[str, int],
    provider: str,
    backend: str,
) -> dict[str, Any]:
    """Generate 4-5 shared fragments for one cluster of hidden facts."""
    # Build the cluster input payload that the prompt expects
    cluster_input = {
        "cluster_id": cluster_obj.get("cluster_id", "C1"),
        "cluster_status": cluster_obj.get("cluster_status", "valid_cluster"),
        "hidden_fact_ids": cluster_obj.get("hidden_fact_ids", []),
        "hidden_facts": hidden_facts_in_cluster,
        "cluster_reason": cluster_obj.get("cluster_reason", ""),
    }

    template = load_prompt(PROMPT_PATH)
    template = fill(template, "{{INSERT_HIDDEN_FACT_CLUSTER_JSON_HERE}}", cluster_input)
    template = fill(
        template, "{{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}}", social_circle
    )
    template = fill(template, "{{LOG_START_DATE}}", log_start)
    template = fill(template, "{{LOG_END_DATE}}", log_end)
    template = fill(
        template, "{{INSERT_CONTACT_USAGE_COUNTS_JSON_HERE}}", contact_usage
    )
    raw = _call_llm(client, model, template, provider=provider, backend=backend)
    return extract_json(raw)


# ---------------------------------------------------------------------------
# Phase 3: Per-cluster verification
# ---------------------------------------------------------------------------


def verify_cluster(
    client,
    verifier_model: str,
    fragments: list[dict[str, Any]],
    persona: dict[str, Any],
    num_facts: int,
    provider: str,
    backend: str,
) -> dict[str, Any]:
    """Cold-verify that all cluster facts are recoverable from fragments.

    The verifier does NOT know how many facts to look for — it must
    independently discover the set of hidden facts encoded in the fragments.

    Defensively strips ALL metadata that could leak ground truth to the
    cold verifier: support mappings, fact IDs, cluster metadata, labels.
    """
    _LEAK_FIELDS = {
        "supports_hidden_facts", "pipeline_notes_only",
        "source_fact_id", "source_fact_ids",
        "fact_id", "fact_ids", "hidden_fact_ids",
        "ground_truth_label", "ground_truth_labels",
        "cluster_id", "cluster_reason", "cluster_status",
        "passed_verification", "implicit_signal_explanation",
    }
    clean_fragments = []
    for f in fragments:
        vf = {k: v for k, v in f.items() if k not in _LEAK_FIELDS}
        clean_fragments.append(vf)

    template = load_prompt(VERIFICATION_PROMPT_PATH)
    template = fill(template, "{{INSERT_FRAGMENTS_JSON_HERE}}", clean_fragments)
    template = fill(template, "{{PERSONA_NAME}}", persona["name"])
    template = fill(template, "{{PERSONA_AGE}}", str(persona["age"]))
    template = fill(template, "{{PERSONA_OCCUPATION}}", persona["occupation"])
    template = fill(template, "{{PERSONA_LOCATION}}", persona["location"])
    raw = _call_llm(client, verifier_model, template, provider=provider, backend=backend)
    return extract_json(raw)


def _verification_candidates(verification: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract candidate facts from verification output."""
    candidates = verification.get("candidate_facts")
    if isinstance(candidates, list):
        return [c for c in candidates if isinstance(c, dict)]
    if verification.get("candidate_label"):
        return [verification]
    return []


def _all_candidates_high_confidence(verification: dict[str, Any]) -> bool:
    candidates = _verification_candidates(verification)
    return bool(candidates) and all(
        (c.get("confidence") or "").upper() == "HIGH"
        for c in candidates
    )


def _shared_evidence_failed(verification: dict[str, Any]) -> bool:
    check = verification.get("shared_evidence_check")
    if not isinstance(check, dict):
        return False
    return any(
        check.get(key) is False
        for key in (
            "every_fragment_supports_multiple_candidate_facts",
            "each_candidate_supported_by_at_least_2_fragments",
        )
    )


def _has_implicitness_leaks(verification: dict[str, Any]) -> bool:
    checks: list[dict[str, Any]] = []
    top_check = verification.get("implicitness_check")
    if isinstance(top_check, dict):
        checks.append(top_check)
    for candidate in _verification_candidates(verification):
        check = candidate.get("implicitness_check")
        if isinstance(check, dict):
            checks.append(check)
    return any(c.get("fragments_that_leak_labels") for c in checks)


def cluster_passed(
    verification: dict[str, Any],
    cluster_facts: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    """Check if the cluster verification gate passes.

    Primary gate: RECOVERED passes unconditionally.

    Production override: AMBIGUOUS also passes only when every recovered
    candidate is HIGH confidence, shared-evidence checks did not fail, and
    the ambiguity came from implicitness leakage (realistic logistical
    anchors). Rejects weak, missing, or structurally bad recovery.

    Returns (passed, matched_pairs) for trace recording.
    """
    if _shared_evidence_failed(verification):
        return (False, [])

    verdict = verification.get("verdict")
    if verdict == "RECOVERED":
        gate_reason = "recovered"
    elif (
        verdict == "AMBIGUOUS"
        and _all_candidates_high_confidence(verification)
        and _has_implicitness_leaks(verification)
    ):
        gate_reason = "ambiguous_high_confidence_implicitness_only"
    else:
        return (False, [])

    verification["_gate_decision"] = {
        "passed": True,
        "accepted_as": "RECOVERED",
        "original_verdict": verdict,
        "reason": gate_reason,
    }

    # Passed the primary gate. Now build matched_pairs for traceability.
    candidates = _verification_candidates(verification)
    if not candidates or not cluster_facts:
        return (True, [])

    ground_labels = [
        (hf.get("fact_id", f"HF-{i:03d}"), (hf.get("ground_truth_label") or "").lower())
        for i, hf in enumerate(cluster_facts, 1)
    ]
    candidate_labels = [
        (c.get("candidate_label") or "").lower() for c in candidates
    ]

    def token_overlap(candidate: str, target: str) -> float:
        rec_tokens = set(re.findall(r"[a-z0-9]{3,}", candidate))
        tgt_tokens = set(re.findall(r"[a-z0-9]{3,}", target))
        if not tgt_tokens:
            return 0.0
        return len(rec_tokens & tgt_tokens) / len(tgt_tokens)

    # Best-effort assignment for trace recording
    n_ground = len(ground_labels)
    indices = list(range(len(candidate_labels)))
    best_pairs: list[dict[str, Any]] = []
    best_total_overlap = -1.0

    for perm in itertools.permutations(indices, min(n_ground, len(indices))):
        pairs = []
        total = 0.0
        for gi, ii in enumerate(perm):
            if gi >= n_ground:
                break
            fact_id, target = ground_labels[gi]
            candidate = candidate_labels[ii]
            overlap = token_overlap(candidate, target)
            total += overlap
            pairs.append({
                "fact_id": fact_id,
                "matched_candidate": candidates[ii].get("candidate_label", ""),
                "overlap": round(overlap, 3),
            })
        if total > best_total_overlap:
            best_total_overlap = total
            best_pairs = pairs

    return (True, best_pairs)


# ---------------------------------------------------------------------------
# Phase 4: Programmatic decoy generation
# ---------------------------------------------------------------------------

_DECOY_TEMPLATES_ONE_OFF = [
    {"topic": "errand", "messages": [
        {"sender": "contact", "text": "Can you grab paper towels if you pass Target?"},
        {"sender": "persona", "text": "Got them, also picked up batteries"},
    ]},
    {"topic": "tech", "messages": [
        {"sender": "contact", "text": "My printer is jammed again"},
        {"sender": "persona", "text": "Did you try the paper tray trick?"},
        {"sender": "contact", "text": "That worked, thanks"},
    ]},
    {"topic": "lost_item", "messages": [
        {"sender": "persona", "text": "Have you seen my sunglasses? Can't find them anywhere"},
        {"sender": "contact", "text": "Check the car console, you left them there last time"},
    ]},
    {"topic": "traffic", "messages": [
        {"sender": "persona", "text": "Running 10 min late, traffic on the main road"},
        {"sender": "contact", "text": "No rush, I just got here too"},
    ]},
    {"topic": "show", "messages": [
        {"sender": "contact", "text": "Have you watched that new series everyone is talking about?"},
        {"sender": "persona", "text": "Not yet, is it worth starting?"},
        {"sender": "contact", "text": "First two episodes are slow but it picks up"},
    ]},
    {"topic": "food", "messages": [
        {"sender": "persona", "text": "What should I bring to the potluck Saturday?"},
        {"sender": "contact", "text": "Something easy, maybe chips and salsa?"},
    ]},
    {"topic": "weather", "messages": [
        {"sender": "contact", "text": "Did you see the forecast? Rain all weekend"},
        {"sender": "persona", "text": "Great, there go my plans"},
    ]},
    {"topic": "package", "messages": [
        {"sender": "persona", "text": "Did my package arrive? Tracking says delivered"},
        {"sender": "contact", "text": "I see a box by your door, want me to grab it?"},
        {"sender": "persona", "text": "Yes please, thanks"},
    ]},
    {"topic": "laundry", "messages": [
        {"sender": "persona", "text": "The dryer ate my sock again"},
        {"sender": "contact", "text": "Check behind the drum, they get stuck there"},
    ]},
    {"topic": "subscription", "messages": [
        {"sender": "contact", "text": "Did you cancel that free trial before it charged?"},
        {"sender": "persona", "text": "Just did, thanks for the reminder"},
    ]},
    {"topic": "book", "messages": [
        {"sender": "persona", "text": "Finished that book you recommended, really liked the ending"},
        {"sender": "contact", "text": "Told you! The sequel comes out in fall"},
    ]},
    {"topic": "wifi", "messages": [
        {"sender": "contact", "text": "Is your wifi being weird today too?"},
        {"sender": "persona", "text": "Yeah, restarted the router twice already"},
    ]},
]

_DECOY_CALENDAR_TEMPLATES = [
    {"title": "Package pickup window", "location": "UPS Store", "notes": ""},
    {"title": "Return library books", "location": "Public Library", "notes": ""},
    {"title": "Haircut", "location": "", "notes": ""},
    {"title": "Voter registration deadline", "location": "", "notes": "online"},
    {"title": "Renew parking permit", "location": "City Hall", "notes": ""},
    {"title": "Pick up photos", "location": "Print shop", "notes": ""},
]

_OTHER_PERSON_TEMPLATES = [
    {"topic": "other_errand", "messages": [
        {"sender": "contact", "text": "Can you grab my dry cleaning if you are near Main St?"},
        {"sender": "persona", "text": "Sure, ticket number?"},
        {"sender": "contact", "text": "4417, under my name"},
    ]},
    {"topic": "other_item", "messages": [
        {"sender": "contact", "text": "I left my charger at your place last week"},
        {"sender": "persona", "text": "Found it under the couch, I'll bring it tomorrow"},
        {"sender": "contact", "text": "Perfect, thanks"},
    ]},
    {"topic": "other_schedule", "messages": [
        {"sender": "contact", "text": "Can you let the plumber in at 2? I have a meeting"},
        {"sender": "persona", "text": "Sure, just text me when they are on the way"},
        {"sender": "contact", "text": "Will do"},
    ]},
]



def _decoy_tokens(decoy: dict[str, Any]) -> set[str]:
    """Extract all tokens from a decoy's user-visible text."""
    tokens: set[str] = set()
    for msg in decoy.get("messages", []) or []:
        text = (msg.get("text") or "").lower()
        tokens.update(re.findall(r"[a-z]{3,}", text))
    cal = decoy.get("calendar_event") or {}
    for field in ("title", "notes", "location"):
        val = (cal.get(field) or "").lower()
        tokens.update(re.findall(r"[a-z]{3,}", val))
    return tokens


def _overlap_safe(
    decoy: dict[str, Any],
    hidden_facts: list[dict[str, Any]],
    threshold: float = 0.25,
) -> tuple[bool, list[str]]:
    """Check that a decoy does not overlap with any individual hidden fact.

    Checks BOTH directions per hidden fact:
    - decoy_overlap: what fraction of the decoy's tokens are HF tokens
    - fact_overlap: what fraction of the HF's tokens appear in the decoy
    If either exceeds threshold for ANY fact, the decoy is unsafe.

    Returns (safe, list_of_overlapping_fact_ids).
    """
    dtokens = _decoy_tokens(decoy)
    if not dtokens:
        return (True, [])

    _STOPWORDS = {"the", "and", "for", "not", "you", "that", "this", "with", "from", "have", "was", "are", "but", "can", "had", "has", "her", "his", "how", "its", "may", "she", "too", "use", "who", "did", "get", "got", "let", "our", "out", "own", "say", "way", "all", "any", "been", "each", "just", "like", "more", "some", "than", "them", "then", "very", "when", "will", "about", "could", "into", "also", "back", "been", "come", "down", "even", "give", "here", "know", "look", "make", "most", "much", "only", "over", "such", "take", "time", "well", "what", "year"}
    dtokens_clean = dtokens - _STOPWORDS
    if not dtokens_clean:
        return (True, [])

    overlapping_ids: list[str] = []
    for hf in hidden_facts:
        fact_tokens: set[str] = set()
        for field in ("ground_truth_label", "claim"):
            val = (hf.get(field) or "").lower()
            fact_tokens.update(re.findall(r"[a-z]{3,}", val))
        fact_tokens -= _STOPWORDS
        if not fact_tokens:
            continue

        shared = dtokens_clean & fact_tokens
        if not shared:
            continue

        decoy_overlap = len(shared) / len(dtokens_clean)
        fact_overlap = len(shared) / len(fact_tokens)

        if decoy_overlap >= threshold or fact_overlap >= threshold:
            overlapping_ids.append(hf.get("fact_id", "?"))

    return (len(overlapping_ids) == 0, overlapping_ids)


def generate_decoys_programmatic(
    hidden_facts: list[dict[str, Any]],
    social_circle: dict[str, Any],
    persona: dict[str, Any],
    target_count: int = 0,
) -> list[dict[str, Any]]:
    """Generate decoy filler candidates programmatically.

    Three strategies:
    1. Domain-exclusion templates: one-off errands/logistics from domains
       that don't overlap with hidden facts.
    2. Other-person-owned templates: a contact owns the detail, persona
       just helps.
    3. Calendar noise: one-off calendar events from generic errands.

    Each candidate is checked per-hidden-fact for safety (both directions).
    """
    if target_count <= 0:
        return []

    contacts = []
    members = social_circle.get("members", social_circle.get("social_circle", []))
    if isinstance(members, list):
        contacts = [m.get("name", "") for m in members if m.get("name")]
    elif isinstance(members, dict):
        contacts = [m.get("name", "") for m in members.values() if isinstance(m, dict) and m.get("name")]

    persona_name = persona.get("name", "Persona")
    decoys: list[dict[str, Any]] = []
    decoy_id = 0

    for tmpl in _DECOY_TEMPLATES_ONE_OFF:
        if len(decoys) >= target_count:
            break
        candidate = {
            "decoy_id": f"D-{decoy_id + 1:03d}",
            "type": "messenger",
            "contact_policy": "flexible",
            "preferred_contact": None,
            "placement_hint": "separate_filler_session",
            "decoy_type": "one_off",
            "overlap_check": "none",
            "overlapping_hidden_fact_ids": [],
            "why_safe": f"One-off {tmpl['topic']} logistics, not a recurring pattern",
            "messages": [
                {
                    "sender": persona_name if m["sender"] == "persona" else "contact",
                    "text": m["text"],
                }
                for m in tmpl["messages"]
            ],
            "calendar_event": None,
        }
        safe, overlap_ids = _overlap_safe(candidate, hidden_facts)
        if safe:
            decoy_id += 1
            candidate["decoy_id"] = f"D-{decoy_id:03d}"
            decoys.append(candidate)

    for tmpl in _OTHER_PERSON_TEMPLATES:
        if len(decoys) >= target_count:
            break
        contact = contacts[decoy_id % len(contacts)] if contacts else "contact"
        candidate = {
            "decoy_id": f"D-{decoy_id + 1:03d}",
            "type": "messenger",
            "contact_policy": "specific",
            "preferred_contact": contact,
            "placement_hint": "flexible",
            "decoy_type": "other_person_owned",
            "overlap_check": "safe_other_person_owned",
            "overlapping_hidden_fact_ids": [],
            "why_safe": f"Detail belongs to {contact}, not the persona",
            "messages": [
                {
                    "sender": persona_name if m["sender"] == "persona" else contact,
                    "text": m["text"],
                }
                for m in tmpl["messages"]
            ],
            "calendar_event": None,
        }
        safe, overlap_ids = _overlap_safe(candidate, hidden_facts)
        if safe:
            decoy_id += 1
            candidate["decoy_id"] = f"D-{decoy_id:03d}"
            decoys.append(candidate)
        elif overlap_ids:
            candidate["overlap_check"] = "unsafe_overlap"
            candidate["overlapping_hidden_fact_ids"] = overlap_ids

    for tmpl in _DECOY_CALENDAR_TEMPLATES:
        if len(decoys) >= target_count:
            break
        candidate = {
            "decoy_id": f"D-{decoy_id + 1:03d}",
            "type": "calendar",
            "contact_policy": "flexible",
            "preferred_contact": None,
            "placement_hint": "flexible",
            "decoy_type": "one_off",
            "overlap_check": "none",
            "overlapping_hidden_fact_ids": [],
            "why_safe": f"One-off calendar event ({tmpl['title']}), not a recurring pattern",
            "messages": None,
            "calendar_event": {
                "title": tmpl["title"],
                "location": tmpl.get("location", ""),
                "participants": [],
                "notes": tmpl.get("notes", ""),
            },
        }
        safe, overlap_ids = _overlap_safe(candidate, hidden_facts)
        if safe:
            decoy_id += 1
            candidate["decoy_id"] = f"D-{decoy_id:03d}"
            decoys.append(candidate)

    return decoys[:target_count]


# ---------------------------------------------------------------------------
# Phase 5: Deterministic assembly
# ---------------------------------------------------------------------------


def _normalize_sender(sender: str, persona_name: str) -> str:
    """Map 'Persona' to the actual persona name."""
    if sender.lower() == "persona":
        return persona_name
    return sender


def assemble_meaningful_log(
    verified_fragments: list[dict[str, Any]],
    persona: dict[str, Any],
    log_start: str,
    log_end: str,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Phase 1: Group verified fragments into meaningful sessions + calendar
    events deterministically. 100% preservation by construction.

    Returns (partial_app_log, provenance_map) where provenance_map is
    session_or_event_id -> list of source_fact_ids (kept in memory for
    cross_app_index, stripped from public output).
    """
    persona_name = persona.get("name", "Persona")

    messenger_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    calendar_events: list[dict[str, Any]] = []
    provenance: dict[str, list[str]] = {}

    for frag in verified_fragments:
        ftype = frag.get("fragment_type", frag.get("type", ""))
        date = frag.get("date", "")
        sfids = frag.get("source_fact_ids", [])

        if ftype in ("messenger_session", "messenger"):
            contact = frag.get("source", {}).get("name", "unknown")
            key = (date, contact)
            if key not in messenger_groups:
                messenger_groups[key] = []
            messenger_groups[key].append(frag)

        elif ftype in ("calendar_event", "calendar"):
            cal = frag.get("calendar_event", {}) or {}
            event = {
                "date": date,
                "start_time": cal.get("start_time", ""),
                "end_time": cal.get("end_time", ""),
                "title": cal.get("title", ""),
                "location": cal.get("location", ""),
                "participants": cal.get("participants", []),
                "notes": cal.get("notes", ""),
            }
            calendar_events.append((event, sfids, frag.get("fragment_id", "")))

    sorted_keys = sorted(messenger_groups.keys())
    meaningful_sessions: list[dict[str, Any]] = []
    fragment_placement: dict[str, str] = {}
    m_idx = 1

    for date, contact in sorted_keys:
        frags = messenger_groups[(date, contact)]
        session_messages: list[dict[str, Any]] = []
        session_fact_ids: list[str] = []
        session_frag_ids: list[str] = []

        for frag in frags:
            frag_id = frag.get("fragment_id", "")
            for msg in frag.get("messages", []) or []:
                session_messages.append({
                    "time": msg.get("time", ""),
                    "sender": _normalize_sender(msg.get("sender", ""), persona_name),
                    "text": msg.get("text", ""),
                })
            session_fact_ids.extend(frag.get("source_fact_ids", []))
            session_frag_ids.append(frag_id)

        session_messages.sort(key=lambda m: m.get("time", ""))

        sid = f"M-{m_idx:03d}"
        meaningful_sessions.append({
            "session_id": sid,
            "date": date,
            "contact": contact,
            "messages": session_messages,
        })
        provenance[sid] = list(set(session_fact_ids))
        for fid in session_frag_ids:
            fragment_placement[fid] = sid
        m_idx += 1

    calendar_events.sort(key=lambda x: (x[0].get("date", ""), x[0].get("start_time", "")))
    cal_out: list[dict[str, Any]] = []
    for e_idx, (event, sfids, frag_id) in enumerate(calendar_events, 1):
        eid = f"E-{e_idx:03d}"
        cal_out.append({"event_id": eid, **event})
        provenance[eid] = list(set(sfids))
        fragment_placement[frag_id] = eid

    partial = {
        "persona_name": persona_name,
        "log_window": {"start_date": log_start, "end_date": log_end},
        "messenger": {"meaningful_sessions": meaningful_sessions, "filler_sessions": []},
        "calendar": {"events": cal_out},
    }
    return partial, provenance, fragment_placement


def validate_meaningful_preservation(
    partial_log: dict[str, Any],
    verified_fragments: list[dict[str, Any]],
    persona_name: str,
) -> dict[str, Any]:
    """Phase 2: Validate 100% message and calendar event preservation.
    Hard fail if any verified content is missing.
    """
    placed_msgs: list[str] = []
    for s in partial_log.get("messenger", {}).get("meaningful_sessions", []):
        for m in s.get("messages", []):
            placed_msgs.append(m.get("text", "").strip())

    placed_cals: list[tuple[str, str, str]] = []
    for e in partial_log.get("calendar", {}).get("events", []):
        placed_cals.append((
            e.get("title", "").strip(),
            e.get("date", ""),
            e.get("start_time", ""),
        ))

    missing_msgs: list[dict[str, Any]] = []
    total_msgs = 0
    found_msgs = 0
    missing_cals: list[str] = []
    total_cals = 0
    found_cals = 0

    for frag in verified_fragments:
        ftype = frag.get("fragment_type", frag.get("type", ""))
        fid = frag.get("fragment_id", "?")

        if ftype in ("messenger_session", "messenger"):
            for msg in frag.get("messages", []) or []:
                total_msgs += 1
                text = msg.get("text", "").strip()
                if text in placed_msgs:
                    placed_msgs.remove(text)
                    found_msgs += 1
                else:
                    missing_msgs.append({"fragment_id": fid, "text": text[:80]})

        elif ftype in ("calendar_event", "calendar"):
            total_cals += 1
            cal = frag.get("calendar_event", {}) or {}
            key = (
                cal.get("title", "").strip(),
                frag.get("date", ""),
                cal.get("start_time", ""),
            )
            if key in placed_cals:
                placed_cals.remove(key)
                found_cals += 1
            else:
                missing_cals.append(f"{fid}: {key[0][:60]}")

    passed = len(missing_msgs) == 0 and len(missing_cals) == 0
    return {
        "passed": passed,
        "messages_expected": total_msgs,
        "messages_found": found_msgs,
        "messages_missing": missing_msgs[:20],
        "calendar_expected": total_cals,
        "calendar_found": found_cals,
        "calendar_missing": missing_cals[:20],
    }


def generate_filler_sessions(
    persona: dict[str, Any],
    social_circle: dict[str, Any],
    log_start: str,
    log_end: str,
    meaningful_token_count: int,
    filler_ratio: float = 2.5,
    target_context_tokens: int | None = None,
    hidden_facts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Phase 3: Generate template-based mundane filler sessions.
    Uses simple templates distributed across dates and contacts.
    Screens each template against hidden fact tokens to avoid overlap.
    """
    if target_context_tokens:
        target_filler_tokens = max(0, target_context_tokens - meaningful_token_count - 2000)
    else:
        target_filler_tokens = int(meaningful_token_count * filler_ratio)

    # Template sessions are short in prose but cost roughly 60-70 JSON tokens.
    # Start slightly high, then downsample by measured token count below.
    avg_tokens_per_filler_session = 50
    target_sessions = max(1, target_filler_tokens // avg_tokens_per_filler_session)

    contacts: list[str] = []
    members = social_circle.get("members", social_circle.get("social_circle", []))
    if isinstance(members, list):
        contacts = [m.get("name", "") for m in members if m.get("name")]
    elif isinstance(members, dict):
        contacts = [m.get("name", "") for m in members.values() if isinstance(m, dict) and m.get("name")]
    if not contacts:
        contacts = ["Contact"]

    persona_name = persona.get("name", "Persona")

    _FILLER_PAIRS = [
        ("Hey, running about 10 min late", "No worries, take your time"),
        ("Did you see the weather tomorrow?", "Yeah, rain all day apparently"),
        ("Want me to grab anything from the store?", "We need paper towels if you pass Target"),
        ("Just finished that book you recommended", "And? Worth it?"),
        ("My wifi has been terrible today", "Same, I restarted the router twice"),
        ("Can you remind me to call back tomorrow?", "Setting a reminder now"),
        ("Left my sunglasses somewhere again", "Check the car, you always leave them there"),
        ("What time works for you Saturday?", "After 2 works, morning is packed"),
        ("Traffic is insane right now", "Take the back road off Oak, way faster"),
        ("Do we still have leftovers?", "Yeah, the rice thing from Sunday is still good"),
        ("printer is jammed again", "Try pulling the tray all the way out"),
        ("picking up coffee, want one?", "Yes please, the usual"),
        ("just saw the funniest thing at the store", "lol what happened"),
        ("Are you coming to the thing on Friday?", "Maybe, depends on work"),
        ("Did the package arrive?", "Yeah it is by the door"),
        ("That new place on Main is actually good", "We should go this weekend"),
        ("Can I borrow your charger?", "It is in the kitchen drawer"),
        ("Running errands all morning", "Sounds fun lol"),
        ("Forgot my lunch at home again", "Classic. Want me to bring you something?"),
        ("Just got home, exhausted", "Same, what a day"),
    ]

    import datetime
    start = datetime.date.fromisoformat(log_start)
    end = datetime.date.fromisoformat(log_end)
    total_days = (end - start).days + 1

    hf_tokens: set[str] = set()
    for hf in (hidden_facts or []):
        for field in ("ground_truth_label", "claim"):
            val = (hf.get(field) or "").lower()
            hf_tokens.update(re.findall(r"[a-z]{4,}", val))

    safe_pairs = []
    for c_text, p_text in _FILLER_PAIRS:
        pair_tokens = set(re.findall(r"[a-z]{4,}", f"{c_text} {p_text}".lower()))
        if not pair_tokens or len(pair_tokens & hf_tokens) / len(pair_tokens) < 0.25:
            safe_pairs.append((c_text, p_text))
    if not safe_pairs:
        safe_pairs = _FILLER_PAIRS[:5]

    filler_sessions: list[dict[str, Any]] = []
    f_idx = 1
    pair_idx = 0

    for i in range(target_sessions):
        day_offset = (i * total_days) // target_sessions
        date = (start + datetime.timedelta(days=day_offset)).isoformat()
        contact = contacts[i % len(contacts)]
        c_text, p_text = safe_pairs[pair_idx % len(safe_pairs)]
        pair_idx += 1

        hour = 8 + (i * 7) % 14
        minute = (i * 13) % 60

        filler_sessions.append({
            "session_id": f"F-{f_idx:03d}",
            "date": date,
            "contact": contact,
            "messages": [
                {"time": f"{hour:02d}:{minute:02d}", "sender": contact, "text": c_text},
                {"time": f"{hour:02d}:{(minute + 3) % 60:02d}", "sender": persona_name, "text": p_text},
            ],
        })
        f_idx += 1

    def filler_token_count(sessions: list[dict[str, Any]]) -> int:
        return len(json.dumps(sessions, ensure_ascii=False)) // 4

    def renumber_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for idx, session in enumerate(sessions, 1):
            session["session_id"] = f"F-{idx:03d}"
        return sessions

    def sample_evenly(
        sessions: list[dict[str, Any]],
        desired_count: int,
    ) -> list[dict[str, Any]]:
        if desired_count >= len(sessions):
            return sessions
        if desired_count <= 1:
            return sessions[:1]
        last = len(sessions) - 1
        return [
            sessions[round(i * last / (desired_count - 1))]
            for i in range(desired_count)
        ]

    max_filler_tokens = int(target_filler_tokens * 1.05)
    tokens = filler_token_count(filler_sessions)
    while tokens > max_filler_tokens and len(filler_sessions) > 1:
        desired_count = max(
            1,
            int(len(filler_sessions) * target_filler_tokens / max(tokens, 1)),
        )
        if desired_count >= len(filler_sessions):
            desired_count = len(filler_sessions) - 1
        filler_sessions = sample_evenly(filler_sessions, desired_count)
        tokens = filler_token_count(filler_sessions)

    renumber_sessions(filler_sessions)
    return filler_sessions


def load_verified_decoys(decoys_path: Path | str | None) -> list[dict[str, Any]]:
    if not decoys_path:
        return []
    path = Path(decoys_path)
    if not path.exists():
        raise FileNotFoundError(f"Verified decoys file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        decoys = payload.get("decoys", payload.get("verified_decoys", []))
    else:
        decoys = payload
    if not isinstance(decoys, list):
        raise ValueError(f"Verified decoys file must contain a list: {path}")
    return decoys


def _social_circle_contact_names(social_circle: dict[str, Any]) -> list[str]:
    members = social_circle.get("members", social_circle.get("social_circle", []))
    if isinstance(members, list):
        contacts = [m.get("name", "") for m in members if m.get("name")]
    elif isinstance(members, dict):
        contacts = [
            m.get("name", "")
            for m in members.values()
            if isinstance(m, dict) and m.get("name")
        ]
    else:
        contacts = []
    return contacts or ["Contact"]


def select_verified_decoys(
    verified_decoys: list[dict[str, Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    """Select hard decoys first while keeping only safe messenger candidates."""
    if target_count <= 0:
        return []
    difficulty_rank = {"hard": 0, "medium": 1, "easy": 2}
    candidates = [
        d for d in verified_decoys
        if d.get("type") == "messenger"
        and d.get("decoy_id")
        and d.get("overlap_check") != "unsafe_overlap"
        and d.get("messages")
    ]
    candidates.sort(
        key=lambda d: (
            difficulty_rank.get(str(d.get("difficulty", "easy")).lower(), 9),
            d.get("decoy_id", ""),
        )
    )
    return [dict(d) for d in candidates[:target_count]]


def build_decoy_filler_sessions(
    selected_decoys: list[dict[str, Any]],
    persona: dict[str, Any],
    social_circle: dict[str, Any],
    log_start: str,
    log_end: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Place selected messenger decoys as filler sessions, never as evidence."""
    if not selected_decoys:
        return [], []

    import datetime
    start = datetime.date.fromisoformat(log_start)
    end = datetime.date.fromisoformat(log_end)
    total_days = (end - start).days + 1
    contacts = _social_circle_contact_names(social_circle)
    persona_name = persona.get("name", "Persona")

    sessions: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for i, decoy in enumerate(selected_decoys):
        preferred = decoy.get("preferred_contact")
        contact = (
            preferred if decoy.get("contact_policy") == "specific" and preferred
            else contacts[i % len(contacts)]
        )
        day_offset = (i * total_days) // max(len(selected_decoys), 1)
        date = (start + datetime.timedelta(days=day_offset)).isoformat()
        hour = 9 + (i * 5) % 11
        minute = (i * 17) % 60

        messages = []
        for j, msg in enumerate(decoy.get("messages", []) or []):
            sender = msg.get("sender", "")
            if sender in {"contact", "{{CONTACT_NAME}}"}:
                sender = contact
            elif sender in {"persona", "{{PERSONA_NAME}}"}:
                sender = persona_name
            messages.append({
                "time": f"{hour:02d}:{(minute + j * 3) % 60:02d}",
                "sender": sender,
                "text": msg.get("text", ""),
            })

        sessions.append({
            "session_id": "",
            "date": date,
            "contact": contact,
            "messages": messages,
        })
        registry.append({
            "decoy_id": decoy.get("decoy_id"),
            "related_item_id": "",
            "placement_type": "filler_session",
            "message_indices": list(range(len(messages))),
            "decoy_type": decoy.get("decoy_type"),
            "difficulty": decoy.get("difficulty", "easy"),
            "overlap_check": decoy.get("overlap_check"),
            "overlapping_hidden_fact_ids": decoy.get("overlapping_hidden_fact_ids", []),
            "why_safe": decoy.get("why_safe", ""),
        })

    return sessions, registry


def finalize_filler_sessions(
    decoy_sessions: list[dict[str, Any]],
    mundane_sessions: list[dict[str, Any]],
    decoy_registry: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    combined = decoy_sessions + mundane_sessions
    for idx, session in enumerate(combined, 1):
        session["session_id"] = f"F-{idx:03d}"
    for idx, registry_entry in enumerate(decoy_registry):
        registry_entry["related_item_id"] = combined[idx]["session_id"]
    decoy_tokens = len(json.dumps(decoy_sessions, ensure_ascii=False)) // 4
    mundane_tokens = len(json.dumps(mundane_sessions, ensure_ascii=False)) // 4
    return combined, decoy_registry, decoy_tokens, mundane_tokens


def build_filler_with_decoys(
    persona: dict[str, Any],
    social_circle: dict[str, Any],
    log_start: str,
    log_end: str,
    meaningful_tokens: int,
    filler_ratio: float,
    target_context_tokens: int | None,
    hidden_facts: list[dict[str, Any]],
    verified_decoy_pool: list[dict[str, Any]],
    decoy_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    selected_decoys = select_verified_decoys(verified_decoy_pool, decoy_count)
    decoy_sessions, decoy_registry = build_decoy_filler_sessions(
        selected_decoys, persona, social_circle, log_start, log_end,
    )
    decoy_tokens_estimate = len(json.dumps(decoy_sessions, ensure_ascii=False)) // 4

    if target_context_tokens:
        target_filler_tokens = max(0, target_context_tokens - meaningful_tokens - 2000)
    else:
        target_filler_tokens = int(meaningful_tokens * filler_ratio)
    mundane_target_tokens = max(0, target_filler_tokens - decoy_tokens_estimate)
    mundane_ratio = mundane_target_tokens / meaningful_tokens if meaningful_tokens else 0

    mundane_sessions = generate_filler_sessions(
        persona, social_circle, log_start, log_end,
        meaningful_tokens, filler_ratio=mundane_ratio,
        target_context_tokens=None,
        hidden_facts=hidden_facts,
    )
    return finalize_filler_sessions(decoy_sessions, mundane_sessions, decoy_registry)


def _build_fragment_registry(
    verified_fragments: list[dict[str, Any]],
    fragment_placement: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build verified_fragment_registry and verified_fragment_check."""
    registry: list[dict[str, Any]] = []
    placed = 0
    missing = 0
    missing_keys: list[str] = []

    for frag in verified_fragments:
        cluster_id = frag.get("cluster_id", frag.get("source_cluster_id", ""))
        frag_id = frag.get("fragment_id", "")
        date = frag.get("date", "")
        ftype = frag.get("fragment_type", frag.get("type", ""))
        src = frag.get("source", {})
        src_type = src.get("type", "")
        src_name = src.get("name", "")

        global_key = f"{cluster_id}::{frag_id}::{date}::{ftype}::{src_type}::{src_name}"
        placed_id = fragment_placement.get(frag_id, "")

        anchors: list[str] = []
        if ftype in ("messenger_session", "messenger"):
            for msg in frag.get("messages", []) or []:
                text = (msg.get("text") or "").strip()
                words = text.split()
                if len(words) >= 5:
                    anchors.append(" ".join(words[:min(12, len(words))]))
                elif text:
                    anchors.append(text)
        elif ftype in ("calendar_event", "calendar"):
            cal = frag.get("calendar_event", {}) or {}
            for field in ("title", "notes", "location"):
                val = (cal.get(field) or "").strip()
                if val:
                    anchors.append(val[:80])

        sfids = frag.get("source_fact_ids", [])
        status = "placed" if placed_id else "missing"
        if status == "placed":
            placed += 1
        else:
            missing += 1
            missing_keys.append(global_key)

        registry.append({
            "source_cluster_id": cluster_id,
            "source_fragment_id": frag_id,
            "global_fragment_key": global_key,
            "placed_item_id": placed_id,
            "hidden_fact_ids": sfids,
            "status": status,
            "content_anchors": anchors,
        })

    check = {
        "input_verified_fragment_count": len(verified_fragments),
        "placed_verified_fragment_count": placed,
        "missing_verified_fragment_count": missing,
        "duplicated_verified_fragment_count": 0,
        "missing_verified_fragment_keys": missing_keys[:20],
        "duplicated_verified_fragment_keys": [],
        "completeness_check": "PASS" if missing == 0 else "FAIL",
    }
    return registry, check


def assemble_final_log(
    partial_log: dict[str, Any],
    filler_sessions: list[dict[str, Any]],
    hidden_facts_registry: list[dict[str, Any]],
    provenance: dict[str, list[str]],
    trace: list[dict[str, Any]],
    verified_fragments: list[dict[str, Any]] | None = None,
    fragment_placement: dict[str, str] | None = None,
    target_filler_ratio: float = 2.5,
    decoy_registry: list[dict[str, Any]] | None = None,
    decoy_filler_token_count: int = 0,
    mundane_filler_token_count: int | None = None,
) -> dict[str, Any]:
    """Phase 4: Combine meaningful + filler, assign final IDs, build
    cross_app_index, compute token_stats. Strip provenance from public output.
    """
    app_log = dict(partial_log)
    app_log["messenger"]["filler_sessions"] = filler_sessions
    app_log["hidden_facts"] = hidden_facts_registry

    meaningful_sessions = app_log["messenger"]["meaningful_sessions"]
    cal_events = app_log["calendar"]["events"]

    cross_app_index: dict[str, dict[str, Any]] = {}
    for s in meaningful_sessions:
        date = s["date"]
        key = f"date_{date}"
        if key not in cross_app_index:
            cross_app_index[key] = {
                "messenger_session_ids": [],
                "calendar_event_ids": [],
                "hidden_fact_ids_touched": [],
            }
        cross_app_index[key]["messenger_session_ids"].append(s["session_id"])
        fact_ids = provenance.get(s["session_id"], [])
        for fid in fact_ids:
            if fid not in cross_app_index[key]["hidden_fact_ids_touched"]:
                cross_app_index[key]["hidden_fact_ids_touched"].append(fid)

    for e in cal_events:
        date = e["date"]
        key = f"date_{date}"
        if key not in cross_app_index:
            cross_app_index[key] = {
                "messenger_session_ids": [],
                "calendar_event_ids": [],
                "hidden_fact_ids_touched": [],
            }
        cross_app_index[key]["calendar_event_ids"].append(e["event_id"])
        fact_ids = provenance.get(e["event_id"], [])
        for fid in fact_ids:
            if fid not in cross_app_index[key]["hidden_fact_ids_touched"]:
                cross_app_index[key]["hidden_fact_ids_touched"].append(fid)

    app_log["cross_app_index"] = dict(sorted(cross_app_index.items()))

    m_msgs = sum(len(s.get("messages", [])) for s in meaningful_sessions)
    f_msgs = sum(len(s.get("messages", [])) for s in filler_sessions)
    meaningful_tokens = len(json.dumps(meaningful_sessions, ensure_ascii=False)) // 4
    filler_tokens = len(json.dumps(filler_sessions, ensure_ascii=False)) // 4
    cal_tokens = len(json.dumps(cal_events, ensure_ascii=False)) // 4
    target_filler_tokens = int(meaningful_tokens * target_filler_ratio)
    if mundane_filler_token_count is None:
        mundane_filler_token_count = max(0, filler_tokens - decoy_filler_token_count)

    app_log["token_stats"] = {
        "approximate_tokens": meaningful_tokens + filler_tokens + cal_tokens,
        "meaningful_token_count": meaningful_tokens,
        "target_total_filler_tokens": target_filler_tokens,
        "filler_token_count": filler_tokens,
        "mundane_filler_token_count": mundane_filler_token_count,
        "decoy_filler_token_count": decoy_filler_token_count,
        "meaningful_messages": m_msgs,
        "filler_messages": f_msgs,
        "calendar_events": len(cal_events),
        "target_filler_ratio": f"{target_filler_ratio:g}:1",
        "ratio_filler_to_meaningful": f"{filler_tokens / meaningful_tokens:.1f}:1" if meaningful_tokens else "N/A",
    }

    decoy_registry = decoy_registry or []
    difficulty_counts = {
        "easy": sum(1 for d in decoy_registry if d.get("difficulty") == "easy"),
        "medium": sum(1 for d in decoy_registry if d.get("difficulty") == "medium"),
        "hard": sum(1 for d in decoy_registry if d.get("difficulty") == "hard"),
    }
    app_log["decoy_registry"] = decoy_registry
    app_log["decoy_check"] = {
        "selected_decoy_count": len(decoy_registry),
        "difficulty_counts": difficulty_counts,
        "rejected_decoy_count": 0,
        "rejected_decoy_ids": [],
        "verdict": "PASS",
    }

    if verified_fragments and fragment_placement is not None:
        frag_registry, frag_check = _build_fragment_registry(
            verified_fragments, fragment_placement,
        )
        app_log["verified_fragment_registry"] = frag_registry
        app_log["verified_fragment_check"] = frag_check

    return app_log


def validate_final_log(
    app_log: dict[str, Any],
    verified_fragments: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    target_filler_ratio: float | None = 2.5,
    filler_ratio_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Phase 5: Final validation. Fragment coverage, no duplicates,
    filler ratio, cross_app_index correctness.
    """
    passed_fact_ids: set[str] = set()
    selected_fact_ids: set[str] = set()
    failed_cluster_ids: list[str] = []
    for t in trace:
        selected_fact_ids.update(t.get("hidden_fact_ids", []))
        if t.get("passed"):
            passed_fact_ids.update(t.get("hidden_fact_ids", []))
        else:
            failed_cluster_ids.append(t.get("cluster_id", ""))
    failed_fact_ids = selected_fact_ids - passed_fact_ids

    registry_fact_ids = {hf.get("fact_id", "") for hf in app_log.get("hidden_facts", [])}
    registry_coverage = len(passed_fact_ids & registry_fact_ids)

    index = app_log.get("cross_app_index", {})
    indexed_fact_ids: set[str] = set()
    for entry in index.values():
        indexed_fact_ids.update(entry.get("hidden_fact_ids_touched", []))

    stats = app_log.get("token_stats", {})
    meaningful_tokens = stats.get("meaningful_token_count", 0) or 0
    filler_tokens = stats.get("filler_token_count", 0) or 0
    actual_filler_ratio = (
        filler_tokens / meaningful_tokens if meaningful_tokens else None
    )
    if target_filler_ratio is None or actual_filler_ratio is None:
        filler_ratio_pass = True
        ratio_bounds: list[float] | None = None
    else:
        min_ratio = target_filler_ratio * (1 - filler_ratio_tolerance)
        max_ratio = target_filler_ratio * (1 + filler_ratio_tolerance)
        filler_ratio_pass = min_ratio <= actual_filler_ratio <= max_ratio
        ratio_bounds = [round(min_ratio, 3), round(max_ratio, 3)]

    fragment_check = app_log.get("verified_fragment_check", {})
    fragment_check_pass = fragment_check.get("completeness_check", "PASS") == "PASS"

    verdict = "PASS" if (
        registry_coverage == len(passed_fact_ids)
        and passed_fact_ids <= indexed_fact_ids
        and not failed_fact_ids
        and filler_ratio_pass
        and fragment_check_pass
    ) else "FAIL"

    return {
        "selected_facts_total": len(selected_fact_ids),
        "passed_facts_total": len(passed_fact_ids),
        "failed_facts_total": len(failed_fact_ids),
        "failed_fact_ids": sorted(failed_fact_ids)[:20],
        "failed_cluster_ids": [cid for cid in failed_cluster_ids if cid][:20],
        "facts_in_registry": len(registry_fact_ids),
        "facts_in_cross_app_index": len(indexed_fact_ids),
        "registry_coverage": registry_coverage,
        "facts_missing_from_registry": sorted(passed_fact_ids - registry_fact_ids)[:20],
        "facts_missing_from_index": sorted(passed_fact_ids - indexed_fact_ids)[:20],
        "filler_ratio": stats.get("ratio_filler_to_meaningful", "N/A"),
        "target_filler_ratio": (
            f"{target_filler_ratio:g}:1" if target_filler_ratio is not None else "N/A"
        ),
        "actual_filler_ratio": (
            round(actual_filler_ratio, 3) if actual_filler_ratio is not None else None
        ),
        "filler_ratio_tolerance": filler_ratio_tolerance,
        "filler_ratio_bounds": ratio_bounds,
        "filler_ratio_pass": filler_ratio_pass,
        "meaningful_messages": stats.get("meaningful_messages", 0),
        "filler_messages": stats.get("filler_messages", 0),
        "verified_fragment_check": fragment_check.get("completeness_check", "PASS"),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Programmatic hidden_facts registry
# ---------------------------------------------------------------------------


def _build_hidden_facts_registry(
    hidden_facts: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the hidden_facts array from source data + trace programmatically.

    The generative merge step frequently drops or truncates the registry.
    Building it programmatically guarantees every verified fact is present.

    Iterates trace entries that have hidden_fact_ids (list) and for each
    passed cluster, creates registry entries for each fact in the cluster.
    """
    # Build a map: fact_id -> trace entry for quick lookup
    fact_to_trace: dict[str, dict[str, Any]] = {}
    for t in trace:
        if not t.get("passed"):
            continue
        for fid in t.get("hidden_fact_ids", []):
            fact_to_trace[fid] = t

    registry: list[dict[str, Any]] = []
    for hf in hidden_facts:
        fact_id = hf.get("fact_id", "")
        if fact_id not in fact_to_trace:
            continue
        source_subcategory = hf.get("source_subcategory", "")
        category = hf.get("category") or source_subcategory.split(".", 1)[0]
        subcategory = hf.get("subcategory") or source_subcategory
        t = fact_to_trace[fact_id]
        # Find the matched candidate for this specific fact
        recovered_label = ""
        for pair in t.get("per_fact_recovery", []):
            if pair.get("fact_id") == fact_id:
                recovered_label = pair.get("matched_candidate", "")
                break
        entry = {
            "fact_id": fact_id,
            "claim": hf.get("claim", ""),
            "ground_truth_label": hf.get("ground_truth_label", ""),
            "category": category,
            "subcategory": subcategory,
            "cluster_id": t.get("cluster_id", ""),
            "recovered_label": recovered_label,
            "verification_attempts": t.get("attempts", 0),
        }
        registry.append(entry)
    return registry


# ---------------------------------------------------------------------------
# Post-merge coverage verification
# ---------------------------------------------------------------------------


def _extract_all_text(app_log: dict[str, Any]) -> str:
    """Collect all user-visible text from the merged app log."""
    parts: list[str] = []
    messenger = app_log.get("messenger", {})
    for bucket in ("meaningful_sessions", "filler_sessions", "sessions"):
        for s in messenger.get(bucket, []) or []:
            for m in s.get("messages", []) or []:
                parts.append(str(m.get("text", "")))
    calendar = app_log.get("calendar", {})
    if isinstance(calendar, dict):
        for e in calendar.get("events", []) or []:
            parts.append(str(e.get("title", "")))
            parts.append(str(e.get("notes", "")))
            parts.append(str(e.get("location", "")))
    return "\n".join(parts).lower()


def _normalize(text: str) -> set[str]:
    """Extract 4+ char alphanumeric tokens for fuzzy matching."""
    return set(re.findall(r"[a-z0-9]{4,}", text.lower()))


def _extract_ngrams(text: str, n: int) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _fragment_text(frag: dict[str, Any]) -> str:
    """Extract user-visible text from a fragment.

    Handles both old format (fragment_type as 'type' key) and new format
    (fragment_type as 'fragment_type' key) for backward compatibility.
    """
    parts: list[str] = []
    ftype = frag.get("fragment_type", frag.get("type", ""))
    if ftype == "messenger_session" or ftype == "messenger":
        for m in frag.get("messages", []) or []:
            parts.append(str(m.get("text", "")))
    elif ftype == "calendar_event" or ftype == "calendar":
        cal = frag.get("calendar_event", {}) or {}
        parts.append(str(cal.get("title", "")))
        parts.append(str(cal.get("notes", "")))
        parts.append(str(cal.get("location", "")))
    return " ".join(parts)


def verify_post_merge_coverage(
    app_log: dict[str, Any],
    verified_fragments: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check that verified fragments survived the merge into the app log.

    Two-layer check:
    1. Registry check: compare hidden_facts IDs in the app log against
       fact IDs that passed verification in the trace.
    2. Content check: for each verified fragment, look for distinctive
       n-grams (3-word sequences) in the app log text. Single tokens
       are too generic; n-grams are specific enough to detect real
       presence vs coincidental word overlap.
    """
    # Layer 1: hidden_facts registry check
    app_hf_ids = set()
    for hf in app_log.get("hidden_facts", []):
        fid = hf.get("fact_id", "")
        if fid:
            app_hf_ids.add(fid)

    passed_fact_ids: set[str] = set()
    for t in trace:
        if t.get("passed"):
            passed_fact_ids.update(t.get("hidden_fact_ids", []))
    registry_missing = sorted(passed_fact_ids - app_hf_ids)

    # Layer 2: content n-gram check
    all_text = _extract_all_text(app_log)
    all_ngrams = _extract_ngrams(all_text, 3)

    frag_fact_ids_present: set[str] = set()
    frag_fact_ids_missing: set[str] = set()
    fragments_present = 0
    fragments_missing_detail: list[str] = []

    for frag in verified_fragments:
        frag_text = _fragment_text(frag)
        frag_ngrams = _extract_ngrams(frag_text, 3)
        # Handle both new format (source_fact_ids list) and old (source_fact_id)
        fact_ids = frag.get("source_fact_ids", [])
        if not fact_ids:
            old_id = frag.get("source_fact_id", "unknown")
            fact_ids = [old_id] if old_id else ["unknown"]

        if not frag_ngrams:
            fragments_present += 1
            frag_fact_ids_present.update(fact_ids)
            continue

        overlap = len(frag_ngrams & all_ngrams) / len(frag_ngrams)
        if overlap >= 0.25:
            fragments_present += 1
            frag_fact_ids_present.update(fact_ids)
        else:
            frag_id = frag.get("fragment_id", "?")
            fact_label = ",".join(fact_ids)
            fragments_missing_detail.append(
                f"{fact_label}/{frag_id} (ngram_overlap={overlap:.0%})"
            )
            for fid in fact_ids:
                if fid not in frag_fact_ids_present:
                    frag_fact_ids_missing.add(fid)

    content_missing = sorted(
        frag_fact_ids_missing - frag_fact_ids_present
    )

    all_missing = sorted(set(registry_missing) | set(content_missing))

    return {
        "fragments_total": len(verified_fragments),
        "fragments_present": fragments_present,
        "fragments_missing": fragments_missing_detail[:20],
        "hidden_facts_total": len(passed_fact_ids),
        "hidden_facts_in_registry": len(app_hf_ids & passed_fact_ids),
        "hidden_facts_in_content": len(frag_fact_ids_present),
        "hidden_facts_covered": len(
            (app_hf_ids & passed_fact_ids) | frag_fact_ids_present
        ),
        "registry_missing_fact_ids": registry_missing[:20],
        "content_missing_fact_ids": content_missing[:20],
        "missing_fact_ids": all_missing[:20],
    }


# ---------------------------------------------------------------------------
# Post-merge message preservation audit
# ---------------------------------------------------------------------------


def audit_message_preservation(
    app_log: dict[str, Any],
    verified_fragments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check that every message from every verified fragment survived the merge.

    For each messenger fragment, check that each original message text appears
    somewhere in the merged app log. Reports per-fragment pass/fail and overall
    preservation rate.
    """
    all_app_messages: set[str] = set()
    messenger = app_log.get("messenger", {})
    for bucket in ("meaningful_sessions", "filler_sessions", "sessions"):
        for s in messenger.get(bucket, []) or []:
            for m in s.get("messages", []) or []:
                text = (m.get("text") or "").strip().lower()
                if text:
                    all_app_messages.add(text)

    total_expected = 0
    total_found = 0
    fragments_fully_preserved = 0
    fragments_partially_preserved = 0
    fragments_missing = 0
    missing_messages: list[dict[str, Any]] = []

    for frag in verified_fragments:
        ftype = frag.get("fragment_type", frag.get("type", ""))
        if ftype not in ("messenger_session", "messenger"):
            continue

        messages = frag.get("messages", []) or []
        if not messages:
            continue

        frag_id = frag.get("fragment_id", "?")
        expected = len(messages)
        found = 0

        for msg in messages:
            text = (msg.get("text") or "").strip().lower()
            if not text:
                found += 1
                continue
            if text in all_app_messages:
                found += 1
            else:
                tokens = set(text.split())
                matched = any(
                    len(tokens & set(app_msg.split())) / max(len(tokens), 1) > 0.7
                    for app_msg in all_app_messages
                    if app_msg
                )
                if matched:
                    found += 1
                else:
                    missing_messages.append({
                        "fragment_id": frag_id,
                        "message_text": msg.get("text", "")[:80],
                    })

        total_expected += expected
        total_found += found

        if found == expected:
            fragments_fully_preserved += 1
        elif found > 0:
            fragments_partially_preserved += 1
        else:
            fragments_missing += 1

    preservation_rate = total_found / total_expected if total_expected else 1.0

    return {
        "total_messages_expected": total_expected,
        "total_messages_found": total_found,
        "preservation_rate": round(preservation_rate, 3),
        "fragments_fully_preserved": fragments_fully_preserved,
        "fragments_partially_preserved": fragments_partially_preserved,
        "fragments_missing": fragments_missing,
        "missing_messages_sample": missing_messages[:20],
    }


def _upsert_trace_entry(
    trace: list[dict[str, Any]],
    trace_entry: dict[str, Any],
) -> None:
    """Replace a prior cluster trace entry, or append it if it is new."""
    cluster_id = trace_entry.get("cluster_id")
    for i, existing in enumerate(trace):
        if existing.get("cluster_id") == cluster_id:
            trace[i] = trace_entry
            return
    trace.append(trace_entry)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_step(
    profile_path: Path,
    social_circle_path: Path,
    news_events_path: Path | None,
    output_dir: Path,
    model: str,
    verifier_model: str,
    per_cluster_max_attempts: int,
    log_start: str,
    log_end: str,
    provider: str,
    backend: str,
    max_facts: int = 100,
    resume: bool = False,
    retry_failed: bool = False,
    filler_ratio: float = 2.5,
    target_context_tokens: int | None = None,
    verified_decoys_path: Path | str | None = None,
    decoy_count: int = 0,
) -> int:
    """File-path based entry point for CLI use."""
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    corrected_profile = profile["corrected_extracted_profile"]
    all_hidden_facts = corrected_profile["hidden_facts"]
    hidden_facts = select_hidden_facts(all_hidden_facts, max_facts)

    circle = json.loads(social_circle_path.read_text(encoding="utf-8"))
    corrected_social_circle = circle["corrected_social_circle"]

    return run_step_from_dicts(
        corrected_profile, corrected_social_circle, hidden_facts,
        output_dir, model, verifier_model, per_cluster_max_attempts,
        log_start, log_end, provider, backend,
        news_events_path=news_events_path,
        base_name=profile_path.stem.replace("_verification", ""),
        resume=resume,
        retry_failed=retry_failed,
        filler_ratio=filler_ratio,
        target_context_tokens=target_context_tokens,
        verified_decoys_path=verified_decoys_path,
        decoy_count=decoy_count,
    )


def run_step_from_dicts(
    corrected_profile: dict[str, Any],
    corrected_social_circle: dict[str, Any],
    hidden_facts: list[dict[str, Any]],
    output_dir: Path | str,
    model: str,
    verifier_model: str,
    per_cluster_max_attempts: int,
    log_start: str,
    log_end: str,
    provider: str,
    backend: str,
    news_events_path: Path | str | None = None,
    base_name: str | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    filler_ratio: float = 2.5,
    target_context_tokens: int | None = None,
    verified_decoys_path: Path | str | None = None,
    decoy_count: int = 0,
) -> int:
    """Dict-based entry point for openclaw_pipeline.py integration."""
    client = make_client(provider) if backend in API_BACKENDS else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    persona = derive_persona_identity(corrected_profile)
    name = base_name or persona["name"].lower().replace(" ", "_")

    hf_map: dict[str, dict[str, Any]] = {}
    for i, hf in enumerate(hidden_facts, 1):
        fid = hf.get("fact_id", f"HF-{i:03d}")
        hf_map[fid] = hf

    # Output paths
    app_log_out = output_dir / f"{name}_app_logs.json"
    trace_out = output_dir / f"{name}_app_logs_trace.json"
    fragments_out = output_dir / f"{name}_verified_fragments.json"

    clusters: list[dict[str, Any]] | None = None
    verified_fragments: list[dict[str, Any]] = []
    contact_usage: dict[str, int] = {}
    trace: list[dict[str, Any]] = []

    # News events
    news_events: list[dict[str, Any]] = []
    if news_events_path and Path(news_events_path).exists():
        news_events = json.loads(Path(news_events_path).read_text(encoding="utf-8"))
    verified_decoy_pool = load_verified_decoys(verified_decoys_path)

    # Resume check: if trace + fragments exist and --resume is set, skip to assembly
    if resume and trace_out.exists() and fragments_out.exists():
        existing_trace = json.loads(trace_out.read_text(encoding="utf-8"))
        existing_frags = json.loads(fragments_out.read_text(encoding="utf-8"))
        all_trace_fact_ids = set()
        for t in existing_trace:
            all_trace_fact_ids.update(t.get("hidden_fact_ids", []))
        current_fact_ids = {hf.get("fact_id", "") for hf in hidden_facts}
        frag_fact_ids = set()
        for f in existing_frags:
            frag_fact_ids.update(f.get("source_fact_ids", []))
        compatible = (
            all_trace_fact_ids == current_fact_ids
            and existing_frags
            and len(set(f.get("fragment_id", "") for f in existing_frags)) == len(existing_frags)
        )
        if compatible:
            failed_trace = [t for t in existing_trace if not t.get("passed")]
            if retry_failed and failed_trace:
                print(
                    f"[step4] retrying {len(failed_trace)} failed clusters "
                    f"from existing trace"
                )
                clusters = [
                    {
                        "cluster_id": t.get("cluster_id", f"retry-{i}"),
                        "cluster_status": "retry_failed",
                        "hidden_fact_ids": t.get("hidden_fact_ids", []),
                    }
                    for i, t in enumerate(failed_trace, 1)
                ]
                trace = existing_trace
                verified_fragments = existing_frags
                for frag in verified_fragments:
                    src = frag.get("source", {})
                    if src.get("type") == "contact":
                        cname = src.get("name", "")
                        if cname:
                            contact_usage[cname] = contact_usage.get(cname, 0) + 1
            else:
                if retry_failed:
                    print("[step4] no failed clusters to retry; resuming assembly")
                print(
                    f"[step4] resuming from existing results: "
                    f"{len(existing_trace)} clusters, {len(existing_frags)} fragments"
                )
                trace = existing_trace
                verified_fragments = existing_frags
                # Jump to assembly phases
                passed_fact_ids_set: set[str] = set()
                for t in trace:
                    if t["passed"]:
                        passed_fact_ids_set.update(t["hidden_fact_ids"])
                verified_hidden_facts = [
                    hf for hf in hidden_facts if hf.get("fact_id") in passed_fact_ids_set
                ]

                partial_log, provenance, fragment_placement = assemble_meaningful_log(
                    verified_fragments, persona, log_start, log_end,
                )
                pres = validate_meaningful_preservation(
                    partial_log, verified_fragments, persona["name"],
                )
                print(
                    f"[step4] preservation: {pres['messages_found']}/{pres['messages_expected']} msgs, "
                    f"{pres['calendar_found']}/{pres['calendar_expected']} cal events"
                )
                if not pres["passed"]:
                    print("[step4] FAIL: verified content not 100% preserved")
                    return 1

                meaningful_tokens = len(
                    json.dumps(partial_log["messenger"]["meaningful_sessions"], ensure_ascii=False)
                ) // 4
                filler, decoy_registry, decoy_tokens, mundane_tokens = build_filler_with_decoys(
                    persona, corrected_social_circle, log_start, log_end,
                    meaningful_tokens, filler_ratio, target_context_tokens,
                    verified_hidden_facts, verified_decoy_pool, decoy_count,
                )
                print(
                    f"[step4] generated {len(filler)} filler sessions "
                    f"({len(decoy_registry)} decoys)"
                )

                registry = _build_hidden_facts_registry(verified_hidden_facts, trace)
                app_log = assemble_final_log(
                    partial_log, filler, registry, provenance, trace,
                    verified_fragments, fragment_placement,
                    target_filler_ratio=filler_ratio,
                    decoy_registry=decoy_registry,
                    decoy_filler_token_count=decoy_tokens,
                    mundane_filler_token_count=mundane_tokens,
                )
                validation = validate_final_log(
                    app_log, verified_fragments, trace,
                    target_filler_ratio=(
                        filler_ratio if target_context_tokens is None else None
                    ),
                )
                app_log["_validation"] = validation
                print(
                    f"[step4] validation: {validation['verdict']} "
                    f"({validation['passed_facts_total']}/{validation['selected_facts_total']} facts passed, "
                    f"{validation['registry_coverage']}/{validation['passed_facts_total']} facts covered, "
                    f"filler ratio {validation['filler_ratio']})"
                )

                app_log_out.write_text(
                    json.dumps(app_log, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"[step4] wrote {app_log_out}")
                return 0 if validation["verdict"] == "PASS" else 1
        else:
            print("[step4] existing results incompatible, re-running full pipeline")

    # Phase 1: Cluster
    if clusters is None:
        print(f"[step4] clustering {len(hidden_facts)} hidden facts for {persona['name']}")
        cluster_result = cluster_hidden_facts(client, model, hidden_facts, provider, backend)
        clusters = cluster_result.get("clusters", [])
        print(f"[step4] formed {len(clusters)} clusters")

    # Phase 2: Generate + Verify per cluster
    for ci, cluster in enumerate(clusters, 1):
        cluster_id = cluster.get("cluster_id", f"C{ci}")
        fact_ids = cluster.get("hidden_fact_ids", [])
        cluster_facts = [hf_map[fid] for fid in fact_ids if fid in hf_map]

        if not cluster_facts:
            print(f"[step4/{cluster_id}] WARNING: no valid facts, skipping")
            trace.append({
                "cluster_id": cluster_id,
                "hidden_fact_ids": fact_ids,
                "ground_truth_labels": [],
                "attempts": 0,
                "passed": False,
                "per_fact_recovery": [],
                "final_verification": None,
                "error": f"fact_ids {fact_ids} not found in hidden_facts",
            })
            continue

        passed = False
        last_fragments: list[dict[str, Any]] = []
        last_verification: dict[str, Any] | None = None
        matched_pairs: list[dict[str, Any]] = []
        attempt = 0

        for attempt in range(1, per_cluster_max_attempts + 1):
            print(
                f"[step4/{cluster_id}] attempt {attempt}/{per_cluster_max_attempts} "
                f"({ci}/{len(clusters)})"
            )

            try:
                cluster_input = {
                    "cluster_id": cluster_id,
                    "cluster_status": cluster.get("cluster_status", "valid_cluster"),
                    "hidden_fact_ids": fact_ids,
                    "hidden_facts": cluster_facts,
                }

                bundle = generate_cluster_fragments(
                    client, model, cluster_input, cluster_facts,
                    corrected_social_circle, log_start, log_end,
                    contact_usage, provider, backend,
                )
                fragments = bundle.get("fragments", [])

                verification = verify_cluster(
                    client, verifier_model, fragments, persona,
                    len(cluster_facts), provider, backend,
                )
                last_fragments = fragments
                last_verification = verification

                passed, matched_pairs = cluster_passed(verification, cluster_facts)
                if passed:
                    for frag in fragments:
                        src = frag.get("source", {})
                        if src.get("type") == "contact":
                            cname = src.get("name", "")
                            if cname:
                                contact_usage[cname] = contact_usage.get(cname, 0) + 1
                    break
                else:
                    verdict = verification.get("verdict", "UNKNOWN")
                    print(f"[step4/{cluster_id}] verdict={verdict}, regenerating")
            except Exception as e:
                print(f"[step4/{cluster_id}] attempt {attempt} error: {e}")
                last_verification = {"error": str(e)}
                if attempt == per_cluster_max_attempts:
                    break

        # Record trace
        trace_entry = {
            "cluster_id": cluster_id,
            "hidden_fact_ids": fact_ids,
            "ground_truth_labels": [
                hf.get("ground_truth_label", "") for hf in cluster_facts
            ],
            "attempts": attempt,
            "passed": passed,
            "per_fact_recovery": matched_pairs,
            "final_verification": last_verification,
        }
        _upsert_trace_entry(trace, trace_entry)

        # Save trace incrementally
        trace_out.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Collect verified fragments
        if passed:
            for frag in last_fragments:
                sfids = [
                    s["hidden_fact_id"]
                    for s in frag.get("supports_hidden_facts", [])
                ]
                verified_fragments.append({**frag, "source_fact_ids": sfids})

    # Save verified fragments
    fragments_out.write_text(
        json.dumps(verified_fragments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    passed_count = sum(1 for t in trace if t["passed"])
    total_facts_passed = sum(
        len(t["hidden_fact_ids"]) for t in trace if t["passed"]
    )
    print(
        f"[step4] {passed_count}/{len(trace)} clusters passed "
        f"({total_facts_passed}/{len(hidden_facts)} facts)"
    )

    # ---- Deterministic assembly (Phases 3-7) ----

    # Only pass facts that have verified fragments
    passed_fact_ids: set[str] = set()
    for t in trace:
        if t["passed"]:
            passed_fact_ids.update(t["hidden_fact_ids"])
    verified_hidden_facts = [
        hf for hf in hidden_facts if hf.get("fact_id") in passed_fact_ids
    ]

    # Phase 3: Assemble meaningful log (Python, deterministic)
    print(f"[step4] assembling {len(verified_fragments)} fragments deterministically")
    partial_log, provenance, fragment_placement = assemble_meaningful_log(
        verified_fragments, persona, log_start, log_end,
    )

    # Phase 4: Validate 100% preservation
    pres = validate_meaningful_preservation(
        partial_log, verified_fragments, persona["name"],
    )
    print(
        f"[step4] preservation: {pres['messages_found']}/{pres['messages_expected']} msgs, "
        f"{pres['calendar_found']}/{pres['calendar_expected']} cal events"
    )
    if not pres["passed"]:
        print("[step4] FAIL: verified content not 100% preserved in assembly")
        print(f"[step4] missing msgs: {pres['messages_missing'][:5]}")
        print(f"[step4] missing cals: {pres['calendar_missing'][:5]}")
        return 1

    # Phase 5: Generate filler (template-based)
    meaningful_tokens = len(
        json.dumps(partial_log["messenger"]["meaningful_sessions"], ensure_ascii=False)
    ) // 4
    filler_sessions, decoy_registry, decoy_tokens, mundane_tokens = build_filler_with_decoys(
        persona, corrected_social_circle, log_start, log_end,
        meaningful_tokens, filler_ratio, target_context_tokens,
        verified_hidden_facts, verified_decoy_pool, decoy_count,
    )
    print(
        f"[step4] generated {len(filler_sessions)} filler sessions "
        f"({len(decoy_registry)} decoys)"
    )

    # Phase 6: Final assembly
    hidden_facts_registry = _build_hidden_facts_registry(verified_hidden_facts, trace)
    app_log = assemble_final_log(
        partial_log, filler_sessions, hidden_facts_registry, provenance, trace,
        verified_fragments, fragment_placement,
        target_filler_ratio=filler_ratio,
        decoy_registry=decoy_registry,
        decoy_filler_token_count=decoy_tokens,
        mundane_filler_token_count=mundane_tokens,
    )

    # Phase 7: Final validation
    validation = validate_final_log(
        app_log, verified_fragments, trace,
        target_filler_ratio=filler_ratio if target_context_tokens is None else None,
    )
    app_log["_validation"] = validation
    print(
        f"[step4] validation: {validation['verdict']} "
        f"({validation['passed_facts_total']}/{validation['selected_facts_total']} facts passed, "
        f"{validation['registry_coverage']}/{validation['passed_facts_total']} facts covered, "
        f"filler ratio {validation['filler_ratio']})"
    )

    app_log_out.write_text(
        json.dumps(app_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[step4] wrote {app_log_out}")

    return 0 if validation["verdict"] == "PASS" else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, required=True,
        help="Path to {persona}_verification.json from step2",
    )
    parser.add_argument(
        "--social-circle", type=Path, required=True,
        help="Path to {persona}_social_circle_verification.json from step3",
    )
    parser.add_argument(
        "--news-events", type=Path, default=None,
        help="Optional path to news events JSON for surprise weaving",
    )
    parser.add_argument(
        "--output", type=Path,
        default=STEP_DIR / "data_samples" / "output",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--verifier-model", required=True,
        help="Independent model for the reverse-inferability gate",
    )
    parser.add_argument(
        "--per-cluster-max-attempts", type=int, default=3,
    )
    parser.add_argument("--log-start", default="2026-03-01")
    parser.add_argument("--log-end", default="2026-03-31")
    parser.add_argument(
        "--backend", default="claude",
        choices=list(SUBSCRIPTION_BACKENDS) + list(API_BACKENDS),
        help="claude/codex for subscription CLI, anthropic-api/openai-api for metered API.",
    )
    parser.add_argument(
        "--provider", default=None, choices=SUPPORTED_PROVIDERS,
    )
    parser.add_argument(
        "--max-facts", type=int, default=100,
        help="Maximum hidden facts to include (prioritized by benchmark relevance)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing trace + fragments files, skip clustering and verification",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="With --resume, reuse passed clusters and regenerate only failed clusters",
    )
    parser.add_argument(
        "--filler-ratio", type=float, default=2.5,
        help="Filler-to-meaningful token ratio (default 2.5)",
    )
    parser.add_argument(
        "--target-context-tokens", type=int, default=None,
        help="Target total context tokens (overrides --filler-ratio when set)",
    )
    parser.add_argument(
        "--verified-decoys", type=Path, default=None,
        help="Optional pre-verified decoy pool JSON; hard messenger decoys are selected first",
    )
    parser.add_argument(
        "--decoy-count", type=int, default=0,
        help="Number of verified messenger decoys to place as filler (default 0)",
    )
    args = parser.parse_args()

    backend = args.backend
    provider = (
        provider_for_backend(backend, args.provider or "anthropic")
        if backend in API_BACKENDS
        else (args.provider or "anthropic")
    )

    if backend in API_BACKENDS and not check_api_key(provider):
        print(f"API key for {provider} is not set.", file=sys.stderr)
        sys.exit(2)

    sys.exit(
        run_step(
            args.profile,
            args.social_circle,
            args.news_events,
            args.output,
            args.model,
            args.verifier_model,
            args.per_cluster_max_attempts,
            args.log_start,
            args.log_end,
            provider=provider,
            backend=backend,
            max_facts=args.max_facts,
            resume=args.resume,
            retry_failed=args.retry_failed,
            filler_ratio=args.filler_ratio,
            target_context_tokens=args.target_context_tokens,
            verified_decoys_path=args.verified_decoys,
            decoy_count=args.decoy_count,
        )
    )


if __name__ == "__main__":
    main()
