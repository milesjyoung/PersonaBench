"""Decoy generation and placement helpers for Step 4."""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any, Callable


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


def _decoy_tokens(decoy: dict[str, Any]) -> set[str]:
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
    dtokens = _decoy_tokens(decoy)
    if not dtokens:
        return (True, [])

    stopwords = {
        "the", "and", "for", "not", "you", "that", "this", "with", "from",
        "have", "was", "are", "but", "can", "had", "has", "her", "his",
        "how", "its", "may", "she", "too", "use", "who", "did", "get",
        "got", "let", "our", "out", "own", "say", "way", "all", "any",
        "been", "each", "just", "like", "more", "some", "than", "them",
        "then", "very", "when", "will", "about", "could", "into", "also",
        "back", "come", "down", "even", "give", "here", "know", "look",
        "make", "most", "much", "only", "over", "such", "take", "time",
        "well", "what", "year",
    }
    dtokens_clean = dtokens - stopwords
    if not dtokens_clean:
        return (True, [])

    overlapping_ids: list[str] = []
    for hf in hidden_facts:
        fact_tokens: set[str] = set()
        for field in ("ground_truth_label", "claim"):
            val = (hf.get(field) or "").lower()
            fact_tokens.update(re.findall(r"[a-z]{3,}", val))
        fact_tokens -= stopwords
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


def _fill(template: str, placeholder: str, payload: Any) -> str:
    if isinstance(payload, str):
        return template.replace(placeholder, payload)
    return template.replace(
        placeholder, json.dumps(payload, indent=2, ensure_ascii=False)
    )


def _decoys_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        decoys = payload.get("decoys", payload.get("verified_decoys", []))
    else:
        decoys = payload
    if not isinstance(decoys, list):
        raise ValueError("Decoy generation output must contain a decoys list")
    return [d for d in decoys if isinstance(d, dict)]


def normalize_verified_decoys(
    payload: Any,
    persona_name: str,
    hidden_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, raw in enumerate(_decoys_from_payload(payload), 1):
        decoy = dict(raw)
        if decoy.get("overlap_check") == "unsafe_overlap":
            continue
        decoy.setdefault("decoy_id", f"D-{idx:03d}")
        decoy.setdefault("difficulty", "easy")
        decoy.setdefault("overlapping_hidden_fact_ids", [])
        decoy.setdefault("why_safe", "")

        messages = decoy.get("messages")
        if messages:
            clean_messages = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                sender = msg.get("sender", "")
                if sender in {"{{PERSONA_NAME}}", "persona"}:
                    sender = persona_name
                clean_messages.append({
                    "sender": sender,
                    "text": msg.get("text", ""),
                })
            decoy["messages"] = clean_messages

        if decoy.get("type") == "messenger" and not decoy.get("messages"):
            continue
        if decoy.get("type") == "calendar" and not decoy.get("calendar_event"):
            continue

        safe, overlap_ids = _overlap_safe(decoy, hidden_facts)
        if not safe and decoy.get("overlap_check") != "safe_other_person_owned":
            continue
        if overlap_ids:
            decoy["overlapping_hidden_fact_ids"] = overlap_ids

        normalized.append(decoy)
    return normalized


def generate_verified_decoys(
    call_model: Callable[[str], str],
    decoy_prompt: str,
    extract_json: Callable[[str], Any],
    persona: dict[str, Any],
    hidden_facts: list[dict[str, Any]],
    social_circle: dict[str, Any],
    log_start: str,
    log_end: str,
    contact_usage: dict[str, int],
    target_count: int,
) -> list[dict[str, Any]]:
    prompt = decoy_prompt
    prompt = _fill(prompt, "{{PERSONA_NAME}}", persona["name"])
    prompt = _fill(prompt, "{{PERSONA_AGE}}", str(persona["age"]))
    prompt = _fill(prompt, "{{PERSONA_OCCUPATION}}", persona["occupation"])
    prompt = _fill(prompt, "{{PERSONA_LOCATION}}", persona["location"])
    prompt = _fill(prompt, "{{INSERT_HIDDEN_FACTS_JSON_HERE}}", hidden_facts)
    prompt = _fill(
        prompt, "{{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}}", social_circle
    )
    prompt = _fill(prompt, "{{LOG_START_DATE}}", log_start)
    prompt = _fill(prompt, "{{LOG_END_DATE}}", log_end)
    prompt = _fill(prompt, "{{TARGET_DECOY_COUNT}}", str(target_count))
    prompt = _fill(
        prompt, "{{INSERT_CONTACT_USAGE_COUNTS_JSON_HERE}}", contact_usage
    )
    return normalize_verified_decoys(
        extract_json(call_model(prompt)), persona["name"], hidden_facts
    )


def resolve_verified_decoy_pool(
    call_model: Callable[[str], str],
    decoy_prompt: str,
    extract_json: Callable[[str], Any],
    persona: dict[str, Any],
    hidden_facts: list[dict[str, Any]],
    social_circle: dict[str, Any],
    log_start: str,
    log_end: str,
    contact_usage: dict[str, int],
    explicit_pool: list[dict[str, Any]],
    decoys_out: Path,
    decoy_count: int,
    decoy_pool_size: int,
    resume: bool,
) -> list[dict[str, Any]]:
    if explicit_pool or decoy_count <= 0:
        return explicit_pool
    if resume and decoys_out.exists():
        return load_verified_decoys(decoys_out)

    target_count = max(decoy_count, decoy_pool_size)
    print(f"[step4] generating {target_count} verified decoy candidates")
    decoys = generate_verified_decoys(
        call_model,
        decoy_prompt,
        extract_json,
        persona,
        hidden_facts,
        social_circle,
        log_start,
        log_end,
        contact_usage,
        target_count,
    )
    decoys_out.write_text(
        json.dumps(decoys, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[step4] wrote {len(decoys)} verified decoys to {decoys_out}")
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
    if not selected_decoys:
        return [], []

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
            "overlapping_hidden_fact_ids": decoy.get(
                "overlapping_hidden_fact_ids", []
            ),
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
