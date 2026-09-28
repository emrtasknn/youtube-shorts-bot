"""
prompt_engine.py -- Visual Engine V2: Advanced Prompt Construction (Phase E)

Builds structured, historically-enriched AI image prompts from scene metadata.
Replaces the inline f-string prompt construction in pipeline.run().

Design principles:
- Pure function: no I/O, no LLM calls, fully deterministic and testable
- Only non-empty fields are included (no blank "Avoid: ." noise)
- Historical context (era, culture, architecture) is derived from event metadata
- Prompt is structured in priority order: action > context > constraints
- Backward compatible: if event_context is None, falls back to V1 behavior

Public API:
    build_ai_prompt(scene, topic_title, event_context=None) -> str
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ai_prompt(
    scene: dict,
    topic_title: str,
    event_context: dict | None = None,
) -> str:
    """Build a structured, context-enriched AI image prompt from scene metadata.

    Args:
        scene:         Scene dict with visual_intent, visual_fact, visual_role, etc.
        topic_title:   Canonical title of the historical event (e.g. "Fatih Sultan
                       Mehmed'in Istanbul'u Fethi").
        event_context: Optional dict with keys: date, location, entities, aliases.
                       Used to derive historical era and cultural constraints.

    Returns:
        A single English prompt string for AI image providers.
    """
    ctx = event_context or {}
    intent = scene.get("visual_intent", {})

    parts: list[str] = []

    # ── 1. Base prompt (Gemini-authored, already in English) ─────────────────
    base = str(scene.get("image_prompt", "")).strip()
    if base:
        parts.append(base)

    # ── 2. Visual fact (the concrete visual truth to depict) ─────────────────
    visual_fact = str(scene.get("visual_fact", "")).strip()
    if visual_fact:
        parts.append(f"Visual fact to depict: {visual_fact}.")

    # ── 3. Visual action (the motion/relationship in the frame) ──────────────
    visual_action = str(intent.get("visual_action", "")).strip()
    if visual_action:
        parts.append(f"Visual action: {visual_action}.")

    # ── 4. Scene context + shot type ─────────────────────────────────────────
    scene_context = str(intent.get("scene_context", "")).strip()
    shot_type = str(intent.get("shot_type", "")).strip()
    if scene_context and shot_type:
        parts.append(f"Scene context: {scene_context}. Shot type: {shot_type}.")
    elif scene_context:
        parts.append(f"Scene context: {scene_context}.")
    elif shot_type:
        parts.append(f"Shot type: {shot_type}.")

    # ── 5. Composition ───────────────────────────────────────────────────────
    composition = str(intent.get("composition", "")).strip()
    if composition:
        parts.append(f"Composition: {composition}.")

    # ── 6. Primary subject + visual entities + visual role ───────────────────
    primary = str(intent.get("primary_subject", "")).strip()
    entities = [str(e).strip() for e in intent.get("visual_entities", []) if str(e).strip()]
    visual_role = str(scene.get("visual_role", "")).strip()

    if primary:
        parts.append(f"Primary subject: {primary}.")
    if entities:
        parts.append(f"Visual entities: {', '.join(entities)}.")
    if visual_role:
        parts.append(f"Visual role: {visual_role}.")

    # ── 7. Must-show / avoid constraints ─────────────────────────────────────
    must_show = [str(m).strip() for m in intent.get("must_show", []) if str(m).strip()]
    avoid = [str(a).strip() for a in intent.get("avoid", []) if str(a).strip()]
    if must_show:
        parts.append(f"Must visibly include: {', '.join(must_show)}.")
    if avoid:
        parts.append(f"Avoid: {', '.join(avoid)}.")

    # ── 8. Historical event label ─────────────────────────────────────────────
    if topic_title:
        parts.append(f"Historical event context: {topic_title}.")

    # ── 9. Historical accuracy constraints (V2 addition) ─────────────────────
    historical = _build_historical_context(ctx)
    if historical:
        parts.append(historical)

    # ── 10. Directorial instructions ──────────────────────────────────────────
    parts.append(
        "Depict the historical ACTION and relationship first, not an isolated object. "
        "The frame must communicate who/what is doing what, where, and in what historical context. "
        "Do not substitute a generic landscape, generic portrait, keyword collage, isolated prop, "
        "or stock-photo composition. Respect the requested shot type and composition."
    )

    # ── 11. Technical + negative constraints ──────────────────────────────────
    parts.append(
        "Vertical 9:16 composition. "
        "No modern objects, no contemporary clothing, no anachronisms. "
        "No text overlays, no watermarks, no borders. "
        "Documentary historical reconstruction style. Photorealistic, cinematic."
    )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Historical context helpers
# ---------------------------------------------------------------------------

def _build_historical_context(event_context: dict) -> str:
    """Build a historical accuracy constraint string from event context metadata.

    Derives era and cultural/architectural constraints that help AI providers
    generate period-accurate images without anachronisms.
    """
    ctx_parts: list[str] = []

    date = str(event_context.get("date", "")).strip()
    location = str(event_context.get("location", "")).strip()

    era = era_from_date(date)
    if era:
        ctx_parts.append(f"Historical era: {era}.")

    geo_ctx = geographic_context(location)
    if geo_ctx:
        ctx_parts.append(geo_ctx)

    # Only add the no-anachronism warning when we have enough context
    if era or geo_ctx:
        ctx_parts.append(
            "Clothing, architecture, objects, and technology must be strictly "
            "period-appropriate. No anachronisms."
        )

    return " ".join(ctx_parts)


def era_from_date(date_str: str) -> str:
    """Derive a rough historical era label from a date string.

    Supports:
    - Plain years: "1453", "BCE 44", "MO 44"
    - ISO dates: "1453-05-29"
    - Turkish: "1453 yili", "M.S. 1453", "M.O. 44"
    - Approximate: "yaklasik 1200"
    """
    if not date_str:
        return ""

    text = date_str.lower()

    # Check for BC/BCE / MO (Milattan Once)
    is_bce = any(m in text for m in ("bce", "bc", "b.c", "m.o.", "mo ", "milattan once"))

    match = re.search(r"\b(\d{1,4})\b", date_str)
    if not match:
        return ""

    year = int(match.group(1))
    if is_bce:
        year = -year

    if year < -500:
        return "ancient era (before 500 BCE)"
    elif year < 0:
        return "late ancient era (500 BCE – 0)"
    elif year < 500:
        return "ancient era (0 – 500 CE)"
    elif year < 1000:
        return "early medieval era (500–1000 CE)"
    elif year < 1300:
        return "medieval era (1000–1300 CE)"
    elif year < 1500:
        return "late medieval era (1300–1500 CE)"
    elif year < 1600:
        return "early modern era (16th century)"
    elif year < 1700:
        return "early modern era (17th century)"
    elif year < 1800:
        return "18th century"
    elif year < 1850:
        return "early 19th century"
    elif year < 1900:
        return "late 19th century"
    elif year < 1920:
        return "early 20th century (1900–1920)"
    elif year < 1950:
        return "mid 20th century (1920–1950)"
    elif year < 1980:
        return "post-war era (1950–1980)"
    elif year < 2000:
        return "late 20th century (1980–2000)"
    else:
        return "contemporary"


def geographic_context(location: str) -> str:
    """Build a geographic/cultural context hint from a location string.

    Returns a short English sentence describing period-appropriate
    architecture and attire for the most common historical regions.
    Returns empty string for unknown or missing locations.
    """
    if not location:
        return ""

    loc = location.lower()

    # Ottoman / Turkish
    if any(k in loc for k in (
        "osmanlı", "ottoman", "istanbul", "konstantinopolis",
        "türkiye", "turkey", "anadolu", "anatolia",
        "edirne", "bursa", "trabzon",
    )):
        return (
            "Cultural context: Ottoman/Turkish. Architecture: Ottoman mosques, minarets, "
            "bazaars, hammams, Topkapi Palace. Attire: period-appropriate Ottoman robes, "
            "turbans, janissary uniforms as applicable."
        )

    # Ancient Rome / Roman Empire
    if any(k in loc for k in (
        "rome", "roma", "roman empire", "roma imparatorluğu",
        "italya", "italy", "latium", "lazio",
    )):
        return (
            "Cultural context: Roman/Italian. Architecture: Roman columns, aqueducts, "
            "forums, Colosseum. Attire: period-appropriate Roman togas, armor, or medieval "
            "Italian dress."
        )

    # Ancient Egypt
    if any(k in loc for k in (
        "egypt", "mısır", "cairo", "kahire", "giza",
        "nil", "nile", "alexandria", "iskenderiye",
    )):
        return (
            "Cultural context: Ancient Egyptian. Architecture: pyramids, obelisks, "
            "hieroglyph-covered temples, Nile boats. Attire: period-appropriate Egyptian "
            "linen, headdresses."
        )

    # Ancient Greece
    if any(k in loc for k in (
        "greece", "yunanistan", "athens", "atina", "sparta",
        "macedon", "makedonya",
    )):
        return (
            "Cultural context: Ancient Greek. Architecture: marble temples, Doric/Ionic "
            "columns, agora. Attire: period-appropriate Greek chitons, sandals."
        )

    # Mesopotamia / Middle East ancient
    if any(k in loc for k in (
        "mezopotamya", "mesopotamia", "babylon", "babil",
        "sumer", "sümer", "iraq", "ırak", "nineveh", "ninive",
    )):
        return (
            "Cultural context: Mesopotamian. Architecture: ziggurats, mud-brick structures, "
            "Hanging Gardens. Attire: period-appropriate Mesopotamian robes, headdresses."
        )

    # Mongol / Central Asian
    if any(k in loc for k in (
        "mongol", "central asia", "orta asya", "steppe", "bozkır",
    )):
        return (
            "Cultural context: Mongol/Central Asian. Architecture: yurts, tent cities, "
            "steppe landscapes. Attire: Mongol military dress, fur-trimmed robes."
        )

    # Chinese / East Asian
    if any(k in loc for k in (
        "china", "çin", "beijing", "pekin", "shanghai", "han dynasty",
        "tang dynasty", "ming dynasty", "qing dynasty",
    )):
        return (
            "Cultural context: Chinese/East Asian. Architecture: pagodas, imperial palaces, "
            "Great Wall. Attire: period-appropriate Chinese imperial or military dress."
        )

    # Western / Central Europe (generic)
    if any(k in loc for k in (
        "france", "fransa", "england", "ingiltere", "germany", "almanya",
        "europe", "avrupa", "austria", "avusturya", "spain", "ispanya",
    )):
        return (
            "Cultural context: European. Architecture and attire must be period-appropriate "
            "for the specified era (e.g., gothic cathedrals for medieval, baroque palaces "
            "for 17th–18th century)."
        )

    # Generic fallback: just note the location without specific constraints
    return f"Geographic setting: {location}."
