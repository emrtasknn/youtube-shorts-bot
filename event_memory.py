"""Event Memory — Canonical Historical Event Identity (V2).

This module implements:
- Persistent event-level memory (event_memory.json)
- Canonical event identity (event_id, aliases, date, location, entities)
- Event identity matching: NEW_EVENT / KNOWN_EVENT / UNCERTAIN
- Batch-level deduplication
- Migration from content_memory.json + topics_history.json
- Broad historical event discovery (15-30 candidates)
- Research/verification dossier generation
- Safe fallback: novelty failure → UNCERTAIN (never ACCEPT)
- Visual source resolution: Wikimedia → Web → AI
"""

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from google import genai
import gemini_config

log = logging.getLogger("shorts-bot")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVENT_MEMORY_FILE = Path(os.getenv("EVENT_MEMORY_FILE", "event_memory.json"))
CONTENT_MEMORY_FILE = Path(os.getenv("CONTENT_MEMORY_FILE", "content_memory.json"))
TOPICS_HISTORY_FILE = Path(os.getenv("TOPICS_HISTORY_FILE", "topics_history.json"))

# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------
NEW_EVENT = "NEW_EVENT"
KNOWN_EVENT = "KNOWN_EVENT"
UNCERTAIN = "UNCERTAIN"

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _call_gemini_json(prompt: str, label: str = "gemini call") -> Optional[dict | list]:
    """Call Gemini with JSON output mode using centralized retry logic."""
    res = gemini_config.call_gemini_with_retry(
        prompt=prompt,
        label=label,
        response_mime_type="application/json"
    )
    if res and res.text:
        try:
            return json.loads(res.text.strip())
        except Exception as exc:
            log.warning("%s: Failed to parse JSON response: %s", label, exc)
    return None


# ---------------------------------------------------------------------------
# Event Memory Persistence
# ---------------------------------------------------------------------------


def load_event_memory(memory_path: Optional[Path] = None) -> list[dict]:
    """Load all previously used historical events from persistent storage."""
    target = memory_path or EVENT_MEMORY_FILE
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as exc:
        log.warning("Could not read event memory: %s", exc)
        return []


def save_event_to_memory(event: dict, memory_path: Optional[Path] = None):
    """Append or update a historical event in persistent memory."""
    target = memory_path or EVENT_MEMORY_FILE
    events = load_event_memory(target)

    # Overwrite if same event_id already exists
    event_id = event.get("event_id")
    if event_id:
        for i, e in enumerate(events):
            if e.get("event_id") == event_id:
                events[i] = event
                break
        else:
            events.append(event)
    else:
        events.append(event)

    try:
        target.write_text(
            json.dumps(events, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Event memory updated: %d events total", len(events))
    except Exception as exc:
        log.error("Could not save event memory: %s", exc)
        raise RuntimeError(f"Event memory persistence failed: {exc}") from exc


def build_event_record(
    canonical_title: str,
    aliases: list[str],
    date: str,
    date_normalized: str,
    location: str,
    entities: list[str],
    event_summary: str,
    core_facts: list[str],
    claims: list[str],
    sources: list[str],
    first_video_id: str = "",
    status: str = "used",
    event_id: str = "",
) -> dict:
    """Build a canonical event memory record."""
    return {
        "event_id": event_id or f"evt_{uuid.uuid4().hex[:12]}",
        "canonical_title": canonical_title,
        "aliases": aliases,
        "date": date,
        "date_normalized": date_normalized,
        "location": location,
        "entities": entities,
        "event_summary": event_summary,
        "core_facts": core_facts,
        "claims": claims,
        "sources": sources,
        "first_video_id": first_video_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Phase 1 — Migration from existing content_memory / topics_history
# ---------------------------------------------------------------------------


def migrate_existing_memory(
    content_memory_path: Optional[Path] = None,
    topics_history_path: Optional[Path] = None,
    event_memory_path: Optional[Path] = None,
    dry_run: bool = False,
) -> list[dict]:
    """Migrate existing content_memory.json and topics_history.json entries
    into the new event_memory.json format.

    Uses Gemini to extract canonical event identity from each existing entry.
    Returns the list of migrated event records.
    """
    cm_path = content_memory_path or CONTENT_MEMORY_FILE
    th_path = topics_history_path or TOPICS_HISTORY_FILE
    em_path = event_memory_path or EVENT_MEMORY_FILE

    # Load existing data
    content_entries = []
    if cm_path.exists():
        try:
            data = json.loads(cm_path.read_text(encoding="utf-8"))
            content_entries = data if isinstance(data, list) else []
        except Exception as exc:
            log.warning("Could not read content_memory for migration: %s", exc)

    topic_entries = []
    if th_path.exists():
        try:
            data = json.loads(th_path.read_text(encoding="utf-8"))
            topic_entries = data if isinstance(data, list) else []
        except Exception as exc:
            log.warning("Could not read topics_history for migration: %s", exc)

    if not content_entries and not topic_entries:
        log.info("Migration: No existing entries found to migrate.")
        return []

    # Build combined unique titles for migration
    titles_seen: set[str] = set()
    entries_to_migrate = []

    for entry in content_entries:
        title = entry.get("title") or entry.get("topic", "")
        if title and title.lower() not in titles_seen:
            titles_seen.add(title.lower())
            entries_to_migrate.append({
                "title": title,
                "topic": entry.get("topic", title),
                "angle": entry.get("angle", ""),
                "summary": entry.get("summary", ""),
                "key_facts": entry.get("key_facts", []),
                "entities": entry.get("entities", []),
                "script": entry.get("script", ""),
                "content_id": entry.get("content_id", ""),
            })

    for entry in topic_entries:
        title = entry.get("title", "")
        if title and title.lower() not in titles_seen:
            titles_seen.add(title.lower())
            entries_to_migrate.append({
                "title": title,
                "topic": title,
                "angle": "",
                "summary": entry.get("hook_question", ""),
                "key_facts": [],
                "entities": [],
                "script": "",
                "content_id": "",
            })

    log.info("Migration: Found %d unique entries to process.", len(entries_to_migrate))

    migrated_events = []
    existing_events = load_event_memory(em_path)
    existing_canonical_titles = {e.get("canonical_title", "").lower() for e in existing_events}

    for entry in entries_to_migrate:
        title = entry["title"]
        log.info("Migration: Processing '%s'", title)

        # Ask Gemini to extract canonical event identity
        prompt = f"""Sen bir tarih uzmanısın. Aşağıdaki YouTube Shorts içerik girişini inceleyip altındaki gerçek tarihsel olayın kanonik kimliğini çıkar.

Başlık: {title}
Konu: {entry.get('topic', '')}
Açı: {entry.get('angle', '')}
Özet: {entry.get('summary', '')}
Temel Gerçekler: {', '.join(entry.get('key_facts', [])[:5])}
Varlıklar: {', '.join(entry.get('entities', [])[:5])}

Lütfen aşağıdaki JSON formatında SADECE geçerli JSON olarak yanıt ver:
{{
  "canonical_title": "Olayın kanonik İngilizce adı (ör: Mary Celeste)",
  "aliases": ["diğer bilinen isimler", "Türkçe adı", "alternatif ifadeler"],
  "date": "Olayın tarihi (ör: 1872, 1590-1600, 64 AD)",
  "date_normalized": "Normalize tarih (ör: 1872, 1590)",
  "location": "Olayın yaşandığı yer",
  "entities": ["İlgili kişi veya varlık 1", "İlgili kişi veya varlık 2"],
  "event_summary": "Olayın 1-2 cümlelik özeti",
  "core_facts": ["Temel doğrulanmış gerçek 1", "Temel doğrulanmış gerçek 2"]
}}
"""

        result = _call_gemini_json(prompt, label=f"migration: {title}")
        if not result or not isinstance(result, dict):
            log.warning("Migration: Could not extract event identity for '%s'; skipping.", title)
            continue

        canonical_title = result.get("canonical_title", title)
        if canonical_title.lower() in existing_canonical_titles:
            log.info("Migration: '%s' → '%s' already in event memory; skipping.", title, canonical_title)
            continue

        event_record = build_event_record(
            canonical_title=canonical_title,
            aliases=result.get("aliases", [title]),
            date=result.get("date", ""),
            date_normalized=result.get("date_normalized", ""),
            location=result.get("location", ""),
            entities=result.get("entities", entry.get("entities", [])),
            event_summary=result.get("event_summary", entry.get("summary", "")),
            core_facts=result.get("core_facts", entry.get("key_facts", [])),
            claims=[],
            sources=[],
            first_video_id=entry.get("content_id", ""),
            status="used",
        )

        migrated_events.append(event_record)
        existing_canonical_titles.add(canonical_title.lower())
        log.info("Migration: Added event '%s' (id: %s)", canonical_title, event_record["event_id"])

        if not dry_run:
            save_event_to_memory(event_record, em_path)

        time.sleep(0.5)  # Avoid rate limiting

    log.info("Migration complete: %d new events added to event memory.", len(migrated_events))
    return migrated_events


# ---------------------------------------------------------------------------
# Phase 2 — Broad Historical Event Discovery (15–30 candidates)
# ---------------------------------------------------------------------------

DISCOVERY_CATEGORIES = [
    "unexplained disappearances",
    "strange deaths",
    "historical mysteries",
    "lost expeditions",
    "bizarre accidents",
    "unusual disasters",
    "forgotten events",
    "archaeological mysteries",
    "historical coincidences",
    "unusual inventions",
    "strange mass phenomena",
    "lost places and civilizations",
    "mysterious artifacts",
    "unusual battles and incidents",
    "unexpected survival stories",
    "historical frauds and hoaxes",
    "unusual scientific events",
    "strange natural phenomena",
    "cryptic historical documents",
    "unexplained medical events",
]

OVERSATURATED_TOPICS = [
    "Mary Celeste",
    "Roanoke",
    "Göbeklitepe",
    "Titanic",
    "Bermuda Triangle",
    "Amelia Earhart",
    "Jack the Ripper",
    "Atlantis",
    "Nazca Lines",
    "Easter Island",
    "Black Death",
    "Stonehenge",
]


def discover_historical_events(
    used_event_titles: list[str],
    used_aliases: list[list[str]],
    count: int = 20,
) -> list[dict]:
    """Discover 15–30 candidate historical events for consideration.

    Returns a list of structured candidate objects. Does NOT filter against
    event memory here — that is done by check_event_identity() separately.
    """
    used_titles_str = ", ".join(used_event_titles[-50:]) if used_event_titles else "none"

    # Pick 3 random discovery categories to inject variety
    import random
    chosen_categories = random.sample(DISCOVERY_CATEGORIES, min(5, len(DISCOVERY_CATEGORIES)))
    category_str = "\n".join(f"- {c}" for c in chosen_categories)

    oversaturated_str = ", ".join(OVERSATURATED_TOPICS)

    prompt = f"""Sen derin tarih araştırmacısısın. Aşağıdaki kurallara göre {count} farklı tarihi olay keşfet.

KULLANILMIŞ OLAYLAR (bunları ve bunların hiçbir versiyonunu önerme):
{used_titles_str}

AŞIRI DOYGUN KONULAR (ünlü ama aşırı kullanılmış — sadece olağanüstü neden olmadıkça önerme):
{oversaturated_str}

BUGÜN KEŞFEDİLECEK KATEGORİLER:
{category_str}

HEDEF: Gerçekten belgelenmiş, az bilinen, şaşırtıcı veya merak uyandıran tarihi olaylar bul.
Her olay:
- Tarihsel olarak gerçek ve belgelenmiş olmalı
- Kısa bir anlatıya uygun olmalı
- Görsel açıdan temsil edilebilir olmalı
- Araştırılabilir olmalı

SADECE şu JSON formatında yanıt ver — dizi içinde {count} olay:
{{
  "candidates": [
    {{
      "canonical_title": "Olayın İngilizce kanonik adı",
      "aliases": ["Türkçe adı", "alternatif ifadeler"],
      "date": "Tarihi (ör: 1816, 64 AD, 13th century)",
      "date_normalized": "Normalize yıl (ör: 1816)",
      "location": "Yaşandığı yer",
      "entities": ["İlgili kişi/varlık 1", "İlgili kişi/varlık 2"],
      "event_summary": "Olayın 2-3 cümlelik özeti",
      "why_interesting": "Neden ilginç/merak uyandırıcı — spesifik ol",
      "known_facts": ["Doğrulanmış gerçek 1", "Doğrulanmış gerçek 2", "Doğrulanmış gerçek 3"],
      "source_urls": [],
      "visual_search_terms": ["görsel arama terimi 1", "görsel arama terimi 2"],
      "discovery_category": "Bu olayın düştüğü kategori"
    }}
  ]
}}
"""

    result = _call_gemini_json(prompt, label="historical event discovery")
    if not result or not isinstance(result, dict):
        log.warning("DISCOVERY: Gemini returned no candidates.")
        return []

    candidates = result.get("candidates", [])
    if not isinstance(candidates, list):
        log.warning("DISCOVERY: Unexpected format from Gemini.")
        return []

    log.info("DISCOVERY: Found %d candidate historical events.", len(candidates))
    return candidates


def discover_historical_events_with_retry(
    used_event_titles: list[str],
    used_aliases: list[list[str]],
    target_count: int = 20,
    max_attempts: int = 3,
) -> list[dict]:
    """Try discovery with retry and alternate strategy on failure.

    Returns candidates or empty list if all attempts fail.
    Does NOT fall back to FALLBACK_STORIES.
    """
    for attempt in range(1, max_attempts + 1):
        log.info("DISCOVERY: Attempt %d/%d", attempt, max_attempts)
        candidates = discover_historical_events(
            used_event_titles=used_event_titles,
            used_aliases=used_aliases,
            count=target_count,
        )
        if candidates:
            return candidates

        log.warning("DISCOVERY: Attempt %d failed. Waiting before retry...", attempt)
        time.sleep((attempt) * 5)

    log.error(
        "DISCOVERY: All %d discovery attempts failed. "
        "Returning empty list — pipeline will abort safely.",
        max_attempts,
    )
    return []


# ---------------------------------------------------------------------------
# Phase 3 — Event Identity Matching
# ---------------------------------------------------------------------------


def _build_event_memory_summary(events: list[dict], max_events: int = 100) -> str:
    """Build a compact text representation of used events for identity matching."""
    if not events:
        return "No previous events."
    recent = events[-max_events:]
    lines = []
    for i, ev in enumerate(recent, 1):
        aliases_str = ", ".join(ev.get("aliases", [])[:4])
        lines.append(
            f"{i}. canonical_title={ev.get('canonical_title', 'N/A')} | "
            f"date={ev.get('date', 'N/A')} | "
            f"location={ev.get('location', 'N/A')} | "
            f"entities={', '.join(ev.get('entities', [])[:3])} | "
            f"aliases=[{aliases_str}] | "
            f"summary={ev.get('event_summary', '')[:100]}"
        )
    return "\n".join(lines)


def check_event_identity(
    candidate: dict,
    event_memory: list[dict],
    batch_memory: list[dict] | None = None,
) -> dict:
    """Check if a candidate is a NEW_EVENT, KNOWN_EVENT, or UNCERTAIN.

    Compares against:
    - Full event_memory (all previously used events)
    - batch_memory (events already accepted in the current discovery batch)

    Returns:
        {
          "decision": "NEW_EVENT" | "KNOWN_EVENT" | "UNCERTAIN",
          "reason": str,
          "confidence": float,
          "matched_event": str | None,
          "matched_event_id": str | None,
        }
    """
    all_previous = (event_memory or []) + (batch_memory or [])

    if not all_previous:
        return {
            "decision": NEW_EVENT,
            "reason": "Event memory is empty. This is a new event.",
            "confidence": 1.0,
            "matched_event": None,
            "matched_event_id": None,
        }

    previous_summary = _build_event_memory_summary(all_previous)

    candidate_info = (
        f"canonical_title={candidate.get('canonical_title', '')} | "
        f"aliases={candidate.get('aliases', [])} | "
        f"date={candidate.get('date', '')} | "
        f"location={candidate.get('location', '')} | "
        f"entities={candidate.get('entities', [])} | "
        f"summary={candidate.get('event_summary', '')} | "
        f"known_facts={candidate.get('known_facts', [])[:5]}"
    )

    prompt = f"""Sen tarih olayı kimlik eşleştirme uzmanısın.

Aday tarihi olay:
{candidate_info}

Daha önce kullanılmış tüm olaylar:
{previous_summary}

GÖREV: Bu adayın, daha önce kullanılmış olaylardan herhangi biriyle AYNI TARİHİ OLAYI temsil edip etmediğini belirle.

ÖNEMLI KURALLAR:
- Başlık farklı olsa da aynı tarihsel olay ise KNOWN_EVENT
- Farklı bir açı veya odak bile olsa aynı temel olay ise KNOWN_EVENT
- Tamamen farklı bir olay ise NEW_EVENT
- Emin değilsen veya yeterli bilgi yoksa UNCERTAIN

Örnekler:
- "Mary Celeste mürettebatının kaybı" → Eğer "Mary Celeste" kayıtlıysa → KNOWN_EVENT
- "1872'de Atlas Okyanusunda terk edilmiş gemi" → Mary Celeste ise → KNOWN_EVENT
- "London Beer Flood" → Hiç kayıtlı değilse → NEW_EVENT

SADECE şu JSON formatında yanıt ver:
{{
  "decision": "NEW_EVENT veya KNOWN_EVENT veya UNCERTAIN",
  "reason": "Kararın gerekçesi — spesifik ve kısa",
  "confidence": 0.95,
  "matched_event": "Eşleşen olayın kanonik adı (yoksa null)",
  "matched_event_id": "Eşleşen olayın event_id'si (yoksa null)"
}}
"""

    result = _call_gemini_json(prompt, label=f"event identity check: {candidate.get('canonical_title', '')}")
    if not result or not isinstance(result, dict):
        # Safe failure: UNCERTAIN (never ACCEPT/NEW_EVENT on failure)
        log.warning(
            "EVENT IDENTITY: Check failed for '%s' — returning UNCERTAIN",
            candidate.get("canonical_title", ""),
        )
        return {
            "decision": UNCERTAIN,
            "reason": "Identity check unavailable — API failure. Treating as UNCERTAIN.",
            "confidence": 0.0,
            "matched_event": None,
            "matched_event_id": None,
        }

    raw_decision = result.get("decision", UNCERTAIN).upper()
    if raw_decision not in (NEW_EVENT, KNOWN_EVENT, UNCERTAIN):
        raw_decision = UNCERTAIN

    log.info(
        "EVENT IDENTITY: candidate='%s' → %s (confidence=%.2f, matched='%s')",
        candidate.get("canonical_title", ""),
        raw_decision,
        result.get("confidence", 0),
        result.get("matched_event", "none"),
    )

    return {
        "decision": raw_decision,
        "reason": result.get("reason", ""),
        "confidence": float(result.get("confidence", 0.0)),
        "matched_event": result.get("matched_event"),
        "matched_event_id": result.get("matched_event_id"),
    }


# ---------------------------------------------------------------------------
# Phase 4 — Batch Deduplication + Full Filtering
# ---------------------------------------------------------------------------


def check_event_identity_batch(candidates: list[dict], event_memory: list[dict]) -> list[dict]:
    """Check a discovery batch in ONE Gemini call to reduce quota usage."""
    if not candidates:
        return []

    previous_summary = _build_event_memory_summary(event_memory)
    payload = []
    for idx, c in enumerate(candidates):
        payload.append({
            "index": idx,
            "canonical_title": c.get("canonical_title", ""),
            "aliases": c.get("aliases", []),
            "date": c.get("date", ""),
            "location": c.get("location", ""),
            "entities": c.get("entities", []),
            "event_summary": c.get("event_summary", ""),
            "known_facts": c.get("known_facts", [])[:5],
        })

    prompt = f"""Sen tarih olayı kimlik eşleştirme uzmanısın.

Daha önce kullanılmış olaylar:
{previous_summary}

Yeni adaylar:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Her aday için karar ver:
- KNOWN_EVENT: Daha önce kullanılan olayla aynı temel tarihsel olay.
- NEW_EVENT: Tamamen farklı tarihsel olay.
- UNCERTAIN: Emin değilsen veya bilgi yetersizse.
- Farklı başlık/açı aynı olayı değiştirmez.
- Adayların kendi aralarında aynı olay olması halinde yalnızca ilkini NEW_EVENT, diğerini KNOWN_EVENT yap.
- API hatası veya yetersiz bilgi varsa tahmin etme; UNCERTAIN kullan.

SADECE JSON:
{{
  "results": [
    {{
      "index": 0,
      "decision": "NEW_EVENT",
      "reason": "kısa gerekçe",
      "confidence": 0.95,
      "matched_event": null,
      "matched_event_id": null
    }}
  ]
}}
"""
    result = _call_gemini_json(prompt, label="batch event identity check")
    if not result or not isinstance(result, dict):
        log.warning("EVENT IDENTITY BATCH: API failed — rejecting all candidates.")
        return [
            {"candidate": c, "identity": {"decision": UNCERTAIN, "reason": "Batch identity check unavailable.", "confidence": 0.0,
                                          "matched_event": None, "matched_event_id": None}}
            for c in candidates
        ]

    raw = result.get("results", [])
    by_index = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[idx] = item

    output = []
    for idx, candidate in enumerate(candidates):
        item = by_index.get(idx)
        if not item:
            identity = {"decision": UNCERTAIN, "reason": "Candidate missing from batch response.", "confidence": 0.0,
                        "matched_event": None, "matched_event_id": None}
        else:
            decision = str(item.get("decision", UNCERTAIN)).upper()
            if decision not in (NEW_EVENT, KNOWN_EVENT, UNCERTAIN):
                decision = UNCERTAIN
            identity = {"decision": decision, "reason": item.get("reason", ""),
                        "confidence": float(item.get("confidence", 0.0)),
                        "matched_event": item.get("matched_event"),
                        "matched_event_id": item.get("matched_event_id")}
        output.append({"candidate": candidate, "identity": identity})
    return output


def _event_tokens(value: str) -> set[str]:
    """Normalize a title/location/entity string for deterministic matching."""
    stop = {
        "the", "of", "and", "on", "in", "from", "de", "da", "do",
        "bir", "ve", "ile", "olayı", "olay", "case", "event", "incident",
    }
    return {
        token.lower()
        for token in re.findall(r"[a-zA-ZÀ-ž0-9]+", str(value))
        if len(token) >= 4 and token.lower() not in stop
    }


def _candidate_matches_used_event(candidate: dict, event: dict) -> tuple[bool, str]:
    """Conservative local same-event check.

    We reject only when there is strong identity evidence. This avoids spending
    Gemini requests on a task that can be safely handled from canonical event
    metadata.
    """
    cand_titles = _event_tokens(" ".join([
        candidate.get("canonical_title", ""),
        *candidate.get("aliases", []),
    ]))
    event_titles = _event_tokens(" ".join([
        event.get("canonical_title", ""),
        *event.get("aliases", []),
    ]))

    title_overlap = len(cand_titles & event_titles)
    date_c = str(candidate.get("date_normalized") or candidate.get("date") or "").strip().lower()
    date_e = str(event.get("date_normalized") or event.get("date") or "").strip().lower()
    loc_c = _event_tokens(candidate.get("location", ""))
    loc_e = _event_tokens(event.get("location", ""))
    ent_c = _event_tokens(" ".join(candidate.get("entities", [])))
    ent_e = _event_tokens(" ".join(event.get("entities", [])))

    same_date = bool(date_c and date_e and date_c == date_e)
    location_overlap = len(loc_c & loc_e)
    entity_overlap = len(ent_c & ent_e)

    # Strongest signal: canonical/alias title overlap plus another identity field.
    if title_overlap >= 1 and (same_date or location_overlap >= 1 or entity_overlap >= 1):
        return True, "title/alias overlap plus matching date, location, or entity"

    # Same event under a completely different title often retains date + place
    # and at least one named entity.
    if same_date and location_overlap >= 1 and entity_overlap >= 1:
        return True, "matching date + location + entity"

    return False, ""


def filter_candidates(candidates: list[dict], event_memory: list[dict]) -> list[dict]:
    """Deterministically filter same-event candidates without Gemini requests.

    This is deliberately conservative: only strong identity matches are
    rejected. Ambiguous candidates remain available for selection/research.
    """
    log.info(
        "\n--- EVENT FILTER ---\nDiscovered: %d\nChecking against %d known events locally",
        len(candidates), len(event_memory),
    )

    accepted = []
    seen_batch = []

    for candidate in candidates:
        title = candidate.get("canonical_title", "unknown")
        matched = False

        for event in event_memory:
            is_match, reason = _candidate_matches_used_event(candidate, event)
            if is_match:
                log.info(
                    "REJECTED KNOWN EVENT | Candidate=%s | Matched=%s | Reason=%s",
                    title, event.get("canonical_title", "N/A"), reason,
                )
                matched = True
                break

        if matched:
            continue

        for previous in seen_batch:
            is_match, reason = _candidate_matches_used_event(candidate, previous)
            if is_match:
                log.info(
                    "REJECTED BATCH DUPLICATE | Candidate=%s | Matched=%s | Reason=%s",
                    title, previous.get("canonical_title", "N/A"), reason,
                )
                matched = True
                break

        if not matched:
            accepted.append(candidate)
            seen_batch.append(candidate)
            log.info("ACCEPTED: '%s'", title)

    log.info(
        "\n--- EVENT FILTER RESULT ---\n%d discovered\n%d rejected\n%d remaining",
        len(candidates), len(candidates) - len(accepted), len(accepted),
    )
    return accepted


def score_and_select_event(candidates: list[dict]) -> Optional[dict]:
    """Score remaining candidates and select the best one.

    Scoring dimensions:
    - Historical interest
    - Curiosity / surprising factor
    - Story potential
    - Visual availability estimate

    Returns the best candidate, or None if the list is empty.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    prompt = f"""Sen YouTube Shorts için tarihsel içerik seçimi yapan bir uzmansın.

Aşağıdaki tarihsel olay adaylarını değerlendir ve en iyi birini seç.

Adaylar:
{json.dumps(candidates, ensure_ascii=False, indent=2)[:4000]}

Değerlendirme kriterleri:
1. Tarihsel ilgi düzeyi (1-10)
2. Merak/şaşırtma potansiyeli (1-10)
3. Hikaye anlatım potansiyeli (1-10)
4. Görsel temsil edilebilirliği (1-10)
5. Araştırma kalitesi — mevcut bilgi yeterliliği (1-10)

Popülerlik/viral potansiyel, olay yeniliğini ve olgusal güvenilirliği geçersiz KILMAMALI.

SADECE şu JSON formatında yanıt ver:
{{
  "selected_index": 0,
  "selected_title": "Seçilen olayın canonical_title'ı",
  "scores": {{
    "historical_interest": 8,
    "curiosity": 9,
    "story_potential": 8,
    "visual_availability": 7,
    "research_quality": 7
  }},
  "selection_reason": "Neden seçildi — kısa"
}}
"""

    result = _call_gemini_json(prompt, label="candidate scoring")
    if result and isinstance(result, dict):
        idx = result.get("selected_index", 0)
        if isinstance(idx, int) and 0 <= idx < len(candidates):
            selected = candidates[idx]
            log.info(
                "SELECTED EVENT: '%s'\nReason: %s\nScores: %s",
                result.get("selected_title", ""),
                result.get("selection_reason", ""),
                result.get("scores", {}),
            )
            return selected

    # Fallback: return first candidate
    log.warning("Candidate scoring failed; using first available candidate.")
    return candidates[0]


# ---------------------------------------------------------------------------
# Phase 5 — Research / Verification Dossier
# ---------------------------------------------------------------------------


def research_historical_event(candidate: dict) -> dict:
    """Conduct a research step for the selected candidate before script generation.

    Returns a structured research dossier with verified facts, disputed claims,
    possible myths, and visual source hints.
    """
    title = candidate.get("canonical_title", "")
    log.info("RESEARCH: Beginning research for '%s'", title)

    prompt = f"""Sen titiz bir tarih araştırmacısısın. Aşağıdaki tarihi olay hakkında kapsamlı bir araştırma dosyası oluştur.

Olay: {title}
Tarih: {candidate.get('date', 'bilinmiyor')}
Yer: {candidate.get('location', 'bilinmiyor')}
Varlıklar: {', '.join(candidate.get('entities', []))}
Özet: {candidate.get('event_summary', '')}
Bilinen gerçekler: {', '.join(candidate.get('known_facts', []))}

Araştırma soruları:
1. Ne oldu?
2. Ne zaman oldu?
3. Nerede oldu?
4. Kimler dahildi?
5. Hangi gerçekler iyi belgelenmiş?
6. Hangi ayrıntılar tartışmalı?
7. Hangi ayrıntılar efsane/spekülasyon?
8. Bu olayı gerçekten ilginç yapan ne?
9. Hangi birincil/ikincil kaynaklar mevcut?
10. Hangi gerçek görseller mevcut olabilir?

SADECE şu JSON formatında yanıt ver:
{{
  "canonical_title": "{title}",
  "research_confidence": "high/medium/low",
  "verified_facts": [
    "Birincil kaynaklarla doğrulanmış gerçek 1",
    "Birincil kaynaklarla doğrulanmış gerçek 2",
    "Birincil kaynaklarla doğrulanmış gerçek 3",
    "Birincil kaynaklarla doğrulanmış gerçek 4",
    "Birincil kaynaklarla doğrulanmış gerçek 5"
  ],
  "disputed_claims": [
    "Tarihçiler arasında tartışmalı iddia 1",
    "Tarihçiler arasında tartışmalı iddia 2"
  ],
  "possible_myths": [
    "Efsane veya spekülasyon olabilecek iddia 1"
  ],
  "what_makes_it_interesting": "Bu olayı genuinely ilginç yapan şey",
  "story_hook": "Kısa bir videoda kullanılabilecek en güçlü açılış kancası",
  "sources": [
    "Akademik kaynak veya belge adı 1",
    "Akademik kaynak veya belge adı 2"
  ],
  "visual_sources": [
    {{
      "description": "Mevcut görsel türü (ör: tarihsel fotoğraf, harita, belge)",
      "wikimedia_search_term": "Wikimedia'da aranacak terim",
      "likely_available": true
    }}
  ]
}}
"""

    result = _call_gemini_json(prompt, label=f"research: {title}")
    if not result or not isinstance(result, dict):
        log.warning("RESEARCH: Failed for '%s' — returning minimal dossier.", title)
        return {
            "canonical_title": title,
            "research_confidence": "low",
            "verified_facts": candidate.get("known_facts", []),
            "disputed_claims": [],
            "possible_myths": [],
            "what_makes_it_interesting": candidate.get("why_interesting", ""),
            "story_hook": "",
            "sources": candidate.get("source_urls", []),
            "visual_sources": [],
        }

    confidence = result.get("research_confidence", "low")
    log.info("RESEARCH: '%s' — confidence=%s, verified_facts=%d",
             title, confidence, len(result.get("verified_facts", [])))
    return result


# ---------------------------------------------------------------------------
# Phase 6 — Wikimedia Visual Source Resolution
# ---------------------------------------------------------------------------


def _visual_tokens(text: str) -> set[str]:
    """Normalize visual search text into meaningful tokens."""
    stop = {"historical", "history", "photo", "photograph", "image", "documentary",
            "scene", "cinematic", "vertical", "realistic", "reconstruction",
            "the", "and", "with", "from", "this", "that", "about"}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-zÀ-ž0-9]+", str(text))
        if len(token) >= 4 and token.lower() not in stop
    }


def _visual_relevance(query: str, title: str, description: str = "") -> float:
    """Deterministic relevance score; weakly related images are rejected."""
    q = _visual_tokens(query)
    if not q:
        return 0.0
    hay = _visual_tokens(f"{title} {description}")
    overlap = len(q & hay)
    coverage = overlap / max(len(q), 1)
    precision = overlap / max(len(hay), 1)
    return round(min(1.0, 0.78 * coverage + 0.22 * precision), 3)


def search_wikimedia_image(
    search_term: str,
    event_title: str = "",
    excluded_urls: Optional[list[str]] = None,
    min_relevance: float = 0.55,
) -> Optional[dict]:
    """Search Wikimedia Commons and reject weakly related results."""
    excluded = set(excluded_urls or [])
    try:
        import requests
        params = {
            "action": "query", "list": "search", "srsearch": search_term,
            "srnamespace": "6", "srlimit": "8", "format": "json", "origin": "*",
        }
        resp = requests.get(
            "https://commons.wikimedia.org/w/api.php", params=params, timeout=10,
            headers={"User-Agent": "YouTubeShortsBot/2.2 (historical-content-bot)"},
        )
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        ranked = []
        for item in results:
            title = item.get("title", "")
            snippet = re.sub(r"<[^>]+>", " ", item.get("snippet", ""))
            score = _visual_relevance(search_term, title, snippet)
            ranked.append((score, item))
        ranked.sort(key=lambda x: x[0], reverse=True)
        for score, item in ranked[:5]:
            file_title = item.get("title", "")
            if not file_title:
                continue
            info_params = {
                "action": "query", "titles": file_title, "prop": "imageinfo",
                "iiprop": "url|extmetadata", "format": "json", "origin": "*",
            }
            info_resp = requests.get(
                "https://commons.wikimedia.org/w/api.php", params=info_params, timeout=10,
                headers={"User-Agent": "YouTubeShortsBot/2.2 (historical-content-bot)"},
            )
            info_resp.raise_for_status()
            pages = info_resp.json().get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    continue
                infos = page.get("imageinfo", [])
                if not infos:
                    continue
                info = infos[0]
                url = info.get("url", "")
                if not url or url in excluded or not url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                    continue
                if score < min_relevance:
                    continue
                meta = info.get("extmetadata", {})
                author = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
                return {
                    "source_type": "wikimedia",
                    "source_url": f"https://commons.wikimedia.org/wiki/{file_title.replace(" ", "_")}",
                    "image_url": url, "title": file_title,
                    "license": meta.get("LicenseShortName", {}).get("value", ""),
                    "attribution": author, "relevance_score": score, "search_query": search_term,
                }
    except Exception as exc:
        log.debug("Wikimedia search failed for %r: %s", search_term, exc)
    return None


def search_openverse_image(
    search_terms: list[str], excluded_urls: Optional[list[str]] = None, min_relevance: float = 0.55
) -> Optional[dict]:
    """Find an openly licensed image and reject weakly related results."""
    excluded = set(excluded_urls or [])
    queries = [str(x).strip() for x in search_terms if str(x).strip()]
    if not queries:
        return None
    try:
        import requests
        best = None
        for query in queries[:5]:
            response = requests.get(
                "https://api.openverse.org/v1/images/",
                params={"q": query, "page_size": 10}, timeout=12,
                headers={"User-Agent": "YouTubeShortsBot/2.2"},
            )
            response.raise_for_status()
            for item in response.json().get("results", []):
                url = item.get("url", "")
                if not url or url in excluded:
                    continue
                haystack = " ".join([str(item.get("title", "")), str(item.get("description", "")),
                    " ".join(str(t.get("name", "")) if isinstance(t, dict) else str(t) for t in item.get("tags", []))])
                score = _visual_relevance(query, haystack)
                if score < min_relevance:
                    continue
                if best is None or score > best[0]:
                    best = (score, item, query)
        if best:
            score, item, query = best
            return {
                "source_type": "openverse",
                "source_url": item.get("foreign_landing_url") or item.get("detail_url"),
                "image_url": item.get("url", ""), "title": item.get("title", ""),
                "license": item.get("license", ""),
                "attribution": item.get("attribution") or item.get("creator", ""),
                "creator": item.get("creator", ""), "provider": item.get("provider", ""),
                "relevance_score": round(score, 3), "search_query": query,
            }
    except Exception as exc:
        log.debug("Openverse search failed: %s", exc)
    return None


def build_visual_search_queries(scene_description: str, visual_intent: dict, event_title: str = "") -> list[str]:
    """Build precise scene-specific queries instead of searching raw narration."""
    explicit = visual_intent.get("search_queries", []) if isinstance(visual_intent, dict) else []
    must_show = visual_intent.get("must_show", []) if isinstance(visual_intent, dict) else []
    primary = visual_intent.get("primary_subject", "") if isinstance(visual_intent, dict) else ""
    queries = [str(q).strip() for q in explicit if str(q).strip()]
    if primary:
        queries.append(f"{event_title} {primary}".strip())
    if must_show:
        queries.append(f"{event_title} {" ".join(str(x) for x in must_show[:3])}".strip())
    if not queries:
        queries.append(f"{event_title} {scene_description[:100]}".strip())
    seen = set()
    unique = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key); unique.append(q)
    return unique[:6]


def resolve_visual_source(
    scene_description: str, visual_search_terms: list[str], event_title: str = "",
    excluded_urls: Optional[list[str]] = None, visual_intent: Optional[dict] = None,
) -> dict:
    """Resolve visual: relevant real source first, otherwise scene-specific AI."""
    excluded = set(excluded_urls or [])
    intent = visual_intent or {}
    queries = build_visual_search_queries(scene_description, intent, event_title)
    for query in queries:
        result = search_wikimedia_image(query, event_title, list(excluded), min_relevance=0.55)
        if result:
            log.info("VISUAL: %s → Wikimedia %s relevance=%.2f", scene_description[:40], result.get("title", ""), result.get("relevance_score", 0))
            return result
    result = search_openverse_image(queries, list(excluded), min_relevance=0.55)
    if result:
        log.info("VISUAL: %s → Openverse %s relevance=%.2f", scene_description[:40], result.get("title", ""), result.get("relevance_score", 0))
        return result
    log.info("VISUAL: %s → AI reconstruction; no relevant real visual found", scene_description[:40])
    return {"source_type": "ai_reconstruction", "note": "No sufficiently relevant real visual found.", "search_queries": queries, "relevance_threshold": 0.55}
}

# ---------------------------------------------------------------------------
# Phase 7 — Full Discovery Pipeline Entry Point
# ---------------------------------------------------------------------------


def run_discovery_pipeline(
    event_memory_path: Optional[Path] = None,
    target_candidates: int = 20,
) -> Optional[tuple[dict, dict]]:
    """Run the full event discovery pipeline.

    Steps:
    1. Load event memory
    2. Discover 15–30 candidate events
    3. Filter against event memory + batch dedup
    4. Score and select best event
    5. Research selected event

    Returns:
        (selected_candidate, research_dossier) or None if discovery fails.
    """
    em_path = event_memory_path or EVENT_MEMORY_FILE
    event_memory = load_event_memory(em_path)

    used_titles = [e.get("canonical_title", "") for e in event_memory]
    used_aliases = [e.get("aliases", []) for e in event_memory]

    log.info(
        "\n=== DISCOVERY ===\n"
        "Event memory: %d previously used events\n"
        "Starting candidate discovery...",
        len(event_memory),
    )

    # Discover candidates
    candidates = discover_historical_events_with_retry(
        used_event_titles=used_titles,
        used_aliases=used_aliases,
        target_count=target_candidates,
    )

    if not candidates:
        log.error(
            "DISCOVERY FAILED: No candidates generated after all retries.\n"
            "Pipeline will abort safely. Do NOT fall back to hardcoded stories."
        )
        return None

    log.info("DISCOVERY: Found %d raw candidates.", len(candidates))

    # Filter candidates
    accepted = filter_candidates(candidates, event_memory)

    if not accepted:
        log.error(
            "DISCOVERY: All %d candidates were rejected (known events or uncertain).\n"
            "No suitable new historical event found. Pipeline will abort safely.",
            len(candidates),
        )
        return None

    # Score and select
    selected = score_and_select_event(accepted)
    if not selected:
        log.error("DISCOVERY: Score/select step returned nothing. Aborting.")
        return None

    log.info(
        "\n=== RESEARCH ===\n"
        "Selected: %s\nStarting research...",
        selected.get("canonical_title", ""),
    )

    # Research the selected event
    dossier = research_historical_event(selected)
    log.info(
        "Research confidence: %s",
        dossier.get("research_confidence", "unknown"),
    )

    return selected, dossier


# ---------------------------------------------------------------------------
# Expose summary for pipeline.py use
# ---------------------------------------------------------------------------


def get_used_event_titles(event_memory_path: Optional[Path] = None) -> list[str]:
    """Return list of canonical titles of all used events."""
    events = load_event_memory(event_memory_path)
    return [e.get("canonical_title", "") for e in events]


def mark_event_as_used(
    candidate: dict,
    research_dossier: dict,
    video_id: str = "",
    event_memory_path: Optional[Path] = None,
) -> dict:
    """Create and save an event record after successful video generation."""
    record = build_event_record(
        canonical_title=candidate.get("canonical_title", ""),
        aliases=candidate.get("aliases", []),
        date=candidate.get("date", ""),
        date_normalized=candidate.get("date_normalized", ""),
        location=candidate.get("location", ""),
        entities=candidate.get("entities", []),
        event_summary=candidate.get("event_summary", ""),
        core_facts=research_dossier.get("verified_facts", candidate.get("known_facts", [])),
        claims=research_dossier.get("disputed_claims", []),
        sources=research_dossier.get("sources", []),
        first_video_id=video_id,
        status="used",
    )
    save_event_to_memory(record, event_memory_path)
    log.info(
        "\n=== SAVE EVENT MEMORY ===\n"
        "Event ID: %s\n"
        "Status: completed\n"
        "NEVER USE THIS EVENT AGAIN: %s",
        record["event_id"],
        record["canonical_title"],
    )
    return record
