"""Content Memory & Novelty Engine for YouTube Shorts Bot.

This module implements:
- Persistent content memory (content_memory.json)
- Topic / Angle separation
- Semantic novelty checks via Gemini
- Three-way decision: ACCEPT / REVISE / REJECT
- Content-aware audio selection
- Visual reuse guard within a single video
- Structured novelty logging
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
CONTENT_MEMORY_FILE = Path(os.getenv("CONTENT_MEMORY_FILE", "content_memory.json"))

# ---------------------------------------------------------------------------
# Content Memory Persistence
# ---------------------------------------------------------------------------


def load_content_memory(memory_path: Optional[Path] = None) -> list[dict]:
    """Load all previously generated content entries from persistent storage."""
    target = memory_path or CONTENT_MEMORY_FILE
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as exc:
        log.warning("Could not read content memory: %s", exc)
        return []


def save_content_entry(entry: dict, memory_path: Optional[Path] = None):
    """Append a new content entry to persistent memory."""
    target = memory_path or CONTENT_MEMORY_FILE
    memory = load_content_memory(target)
    memory.append(entry)
    try:
        target.write_text(
            json.dumps(memory, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Content memory updated: %d entries total", len(memory))
    except Exception as exc:
        log.warning("Could not save content memory: %s", exc)


def build_content_entry(
    topic: dict,
    scenes: list[dict],
    analysis: dict,
    full_text: str,
) -> dict:
    """Build a structured content entry for memory storage."""
    return {
        "content_id": str(uuid.uuid4())[:12],
        "title": topic.get("title", ""),
        "topic": analysis.get("topic", topic.get("title", "")),
        "angle": analysis.get("angle", ""),
        "summary": analysis.get("summary", ""),
        "main_claim": analysis.get("main_claim", ""),
        "key_facts": analysis.get("key_facts", []),
        "entities": analysis.get("entities", []),
        "mood": analysis.get("mood", "mysterious"),
        "energy": analysis.get("energy", 0.5),
        "tension": analysis.get("tension", 0.3),
        "script": full_text,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "visual_facts": [scene.get("visual_fact", "") for scene in scenes if scene.get("visual_fact")],
        "visual_roles": [scene.get("visual_role", "") for scene in scenes if scene.get("visual_role")],
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Content Analysis (extract topic, angle, facts from a script)
# ---------------------------------------------------------------------------

def analyze_script_content(scenes: list[dict], topic_title: str = "") -> dict:
    """Use Gemini to extract structured content metadata from a script.

    Returns a dict with: topic, angle, summary, main_claim, key_facts,
    entities, mood, energy, tension.
    """
    full_text = " ".join(s.get("narration", "") for s in scenes)

    prompt = f"""Aşağıdaki YouTube Shorts senaryosunu analiz et ve yapılandırılmış metadata çıkar.

Konu başlığı: {topic_title}

Senaryo metni:
{full_text}

Aşağıdaki JSON şemasında SADECE geçerli JSON olarak yanıt ver:
{{
  "topic": "Ana konu (ör: Göbeklitepe, Mary Celeste, vb.)",
  "angle": "Bu videonun ele aldığı spesifik bakış açısı/alt konu (ör: T-biçimli sütunların anlamı, yapının gömülmesi, vb.)",
  "summary": "Videonun 1-2 cümlelik özeti",
  "main_claim": "Videonun ana iddiası veya tezi",
  "key_facts": ["Fakt 1", "Fakt 2", "Fakt 3", "Fakt 4", "Fakt 5"],
  "entities": ["Varlık1", "Varlık2", "Varlık3"],
  "mood": "mysterious/dramatic/suspenseful/cinematic/investigative/epic",
  "energy": 0.5,
  "tension": 0.3
}}

Kurallar:
- "angle" alanı konunun genel tekrarı DEĞİL, spesifik bilgisel odak olmalı.
- "key_facts" en az 3, en fazla 7 madde olsun.
- "energy" ve "tension" 0.0-1.0 arası ondalık sayı olsun.
- "mood" yukarıdaki seçeneklerden biri olsun.
"""

    res = gemini_config.call_gemini_with_retry(
        prompt=prompt,
        label="script analysis",
        response_mime_type="application/json"
    )

    if res and res.text:
        try:
            parsed = json.loads(res.text.strip())
            if isinstance(parsed, dict) and parsed.get("topic"):
                log.info(
                    "Content analysis complete — topic=%s, angle=%s, mood=%s",
                    parsed.get("topic"),
                    parsed.get("angle"),
                    parsed.get("mood"),
                )
                return parsed
        except Exception as exc:
            log.warning("Script analysis JSON parse failed: %s", exc)

    # Fallback: return minimal metadata derived from the text
    log.warning("All analysis models failed; using text-derived fallback metadata")
    return {
        "topic": topic_title,
        "angle": "genel bakış",
        "summary": full_text[:200],
        "main_claim": "",
        "key_facts": [],
        "entities": [],
        "mood": "mysterious",
        "energy": 0.5,
        "tension": 0.3,
    }


# ---------------------------------------------------------------------------
# Semantic Novelty Check — ACCEPT / REVISE / REJECT
# ---------------------------------------------------------------------------

def format_previous_content_summary(memory: list[dict], max_entries: int = 15) -> str:
    """Build a compact text representation of previous content for the LLM."""
    if not memory:
        return "Henüz üretilmiş içerik yok."

    recent = memory[-max_entries:]
    lines = []
    for i, entry in enumerate(recent, 1):
        facts_str = ", ".join(entry.get("key_facts", [])[:5])
        lines.append(
            f"{i}.\n"
            f"  Konu: {entry.get('topic', 'N/A')}\n"
            f"  Açı: {entry.get('angle', 'N/A')}\n"
            f"  Ana İddia: {entry.get('main_claim', 'N/A')}\n"
            f"  Temel Bilgiler: {facts_str}\n"
            f"  Özet: {entry.get('summary', '')[:150]}"
        )
    return "\n".join(lines)


def check_content_novelty(
    candidate_analysis: dict,
    memory: list[dict],
) -> dict:
    """Evaluate a candidate script against content memory.

    Returns a dict with:
      decision: "ACCEPT" | "REVISE" | "REJECT"
      reason: human-readable explanation
      similarity_score: 0.0-1.0
      fact_overlap_pct: 0-100
      nearest_content: title of nearest previous content (if any)
      revision_hint: suggested angle change (if REVISE)
    """
    if not memory:
        log.info("Content memory is empty — ACCEPT by default")
        return {
            "decision": "ACCEPT",
            "reason": "İçerik belleği boş, ilk video.",
            "similarity_score": 0.0,
            "fact_overlap_pct": 0,
            "nearest_content": None,
            "revision_hint": None,
        }

    previous_summary = format_previous_content_summary(memory)
    candidate_facts = ", ".join(candidate_analysis.get("key_facts", []))

    prompt = f"""Sen bir içerik yenilik değerlendirme uzmanısın. Aday bir YouTube Shorts videosunun daha önce üretilen videolara göre yeterince farklı olup olmadığını değerlendir.

ÖNCEKİ İÇERİKLER:
{previous_summary}

ADAY VİDEO:
  Konu: {candidate_analysis.get('topic', '')}
  Açı: {candidate_analysis.get('angle', '')}
  Ana İddia: {candidate_analysis.get('main_claim', '')}
  Temel Bilgiler: {candidate_facts}
  Özet: {candidate_analysis.get('summary', '')}

DEĞERLENDİRME KRİTERLERİ:
1. Konu benzerliği — Aynı konu olabilir ama açı farklı olmalı
2. Açı benzerliği — Aynı açıdan ele alınmışsa REJECT
3. Ana iddia benzerliği — Aynı iddiayı tekrarlıyorsa REJECT
4. Bilgi örtüşmesi — Aynı bilgi koleksiyonuyla aynı anlatı kuruluyorsa REJECT
5. Anlatı yeniliği — Farklı bilgisel yük taşıyorsa ACCEPT

ÖNEMLİ KURALLAR:
- Bireysel bilginin tekrar kullanımı kabul edilebilir (ör: "Göbeklitepe 12 bin yıllıktır" birden fazla videoda geçebilir)
- SORUN olan şey: aynı bilgi koleksiyonu + aynı anlatı açısıyla esasen aynı hikayeyi yeniden anlatmak
- Aynı geniş konu, farklı bilgisel açı = ACCEPT
- Aynı konu, aynı açı, farklı kelimeler = REJECT

SADECE şu JSON şemasında yanıt ver:
{{
  "decision": "ACCEPT veya REVISE veya REJECT",
  "reason": "Kararın açıklaması",
  "similarity_score": 0.75,
  "fact_overlap_pct": 45,
  "nearest_content": "En benzer önceki videonun konusu/açısı",
  "revision_hint": "REVISE ise: önerilen alternatif açı (yoksa null)"
}}
"""

    client = _get_genai_client()
    models = _get_analysis_models()

    for model in models:
        try:
            res = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            if res and res.text:
                parsed = json.loads(res.text.strip())
                decision = parsed.get("decision", "ACCEPT").upper()
                if decision not in ("ACCEPT", "REVISE", "REJECT"):
                    decision = "ACCEPT"
                parsed["decision"] = decision

                log.info(
                    "Novelty check — decision=%s, similarity=%.2f, fact_overlap=%d%%, nearest=%s",
                    decision,
                    parsed.get("similarity_score", 0),
                    parsed.get("fact_overlap_pct", 0),
                    parsed.get("nearest_content", "N/A"),
                )
                return parsed
        except Exception as exc:
            log.warning("Novelty check failed with %s: %s", model, exc)
            time.sleep(1)

    # If all models fail, ACCEPT to avoid blocking pipeline
    log.warning("All novelty check models failed; defaulting to ACCEPT")
    return {
        "decision": "ACCEPT",
        "reason": "Yenilik kontrolü yapılamadı — API hatası nedeniyle kabul edildi.",
        "similarity_score": 0.0,
        "fact_overlap_pct": 0,
        "nearest_content": None,
        "revision_hint": None,
    }


# ---------------------------------------------------------------------------
# Content-aware script generation prompt enhancement
# ---------------------------------------------------------------------------

def build_content_aware_negative_prompt(memory: list[dict], max_entries: int = 10) -> str:
    """Build a negative prompt section that informs the script generator about
    previously covered content so it generates genuinely new angles."""
    if not memory:
        return ""

    recent = memory[-max_entries:]
    lines = ["ÖNCEKİ İÇERİKLER (bunları tekrar ETME, farklı bir bilgisel açı seç):"]
    for i, entry in enumerate(recent, 1):
        facts = ", ".join(entry.get("key_facts", [])[:4])
        lines.append(
            f"{i}. Konu: {entry.get('topic', 'N/A')} | "
            f"Açı: {entry.get('angle', 'N/A')} | "
            f"Ana İddia: {entry.get('main_claim', 'N/A')} | "
            f"Bilgiler: {facts}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Content-aware Audio Selection
# ---------------------------------------------------------------------------

# Map mood/energy/tension profiles to preferred audio tracks
AUDIO_MOOD_MAP = {
    "mysterious": {
        "preferred": ["ambient_mystery.mp3", "mystery_investigation.mp3", "mystery_suspense.mp3"],
        "fallback": ["investigation_tension.mp3"],
    },
    "dramatic": {
        "preferred": ["suspense_thriller.mp3", "mystery_suspense.mp3"],
        "fallback": ["investigation_tension.mp3", "ambient_mystery.mp3"],
    },
    "suspenseful": {
        "preferred": ["suspense_thriller.mp3", "investigation_tension.mp3", "mystery_suspense.mp3"],
        "fallback": ["ambient_mystery.mp3"],
    },
    "cinematic": {
        "preferred": ["suspense_thriller.mp3", "mystery_investigation.mp3"],
        "fallback": ["ambient_mystery.mp3", "mystery_suspense.mp3"],
    },
    "investigative": {
        "preferred": ["investigation_tension.mp3", "mystery_investigation.mp3"],
        "fallback": ["mystery_suspense.mp3", "ambient_mystery.mp3"],
    },
    "epic": {
        "preferred": ["suspense_thriller.mp3", "investigation_tension.mp3"],
        "fallback": ["mystery_suspense.mp3"],
    },
}


def select_audio_for_content(
    analysis: dict,
    audio_dir: Path,
    recently_used: Optional[list[str]] = None,
) -> Optional[Path]:
    """Select the most appropriate background audio based on content analysis.

    Args:
        analysis: Content analysis dict with mood, energy, tension.
        audio_dir: Path to assets/audio/ directory.
        recently_used: List of recently used track filenames to avoid repetition.

    Returns:
        Path to the selected audio file, or None if no suitable file found.
    """
    if not audio_dir.exists():
        return None

    available = {p.name: p for p in audio_dir.glob("*.mp3") if p.stat().st_size > 10_000}
    if not available:
        return None

    mood = analysis.get("mood", "mysterious").lower()
    energy = float(analysis.get("energy", 0.5))
    tension = float(analysis.get("tension", 0.3))
    recently_used = recently_used or []

    # Get mood-based candidates
    mood_config = AUDIO_MOOD_MAP.get(mood, AUDIO_MOOD_MAP["mysterious"])
    candidates = mood_config["preferred"] + mood_config["fallback"]

    # Filter to actually available tracks
    candidates = [name for name in candidates if name in available]

    # Deprioritize recently used tracks (but don't exclude entirely)
    if recently_used and len(candidates) > 1:
        fresh = [c for c in candidates if c not in recently_used]
        if fresh:
            candidates = fresh + [c for c in candidates if c in recently_used]

    # High tension content prefers more intense tracks
    if tension > 0.6 and "suspense_thriller.mp3" in available:
        if "suspense_thriller.mp3" in candidates:
            candidates.remove("suspense_thriller.mp3")
            candidates.insert(0, "suspense_thriller.mp3")

    # Low energy content prefers ambient tracks
    if energy < 0.3 and "ambient_mystery.mp3" in available:
        if "ambient_mystery.mp3" in candidates:
            candidates.remove("ambient_mystery.mp3")
            candidates.insert(0, "ambient_mystery.mp3")

    if candidates:
        chosen = candidates[0]
        log.info(
            "Audio selection: mood=%s, energy=%.2f, tension=%.2f → %s",
            mood, energy, tension, chosen,
        )
        return available[chosen]

    # Absolute fallback: any available track
    import random
    chosen_path = random.choice(list(available.values()))
    log.info("Audio selection fallback: %s", chosen_path.name)
    return chosen_path


def get_recently_used_audio(memory: list[dict], lookback: int = 3) -> list[str]:
    """Return filenames of audio tracks used in recent videos."""
    recent = memory[-lookback:] if memory else []
    return [entry.get("audio_track", "") for entry in recent if entry.get("audio_track")]


# ---------------------------------------------------------------------------
# Visual Reuse & Diversity Guard (Phase H)
# ---------------------------------------------------------------------------

def _get_scene_field(scene: dict, field_name: str, default: str = "") -> str:
    """Retrieve a visual field from scene dict or nested visual_intent dict."""
    val = scene.get(field_name)
    if not val and isinstance(scene.get("visual_intent"), dict):
        val = scene.get("visual_intent", {}).get(field_name)
    return str(val or default).strip()


def _normalize_prompt(prompt: str) -> str:
    """Normalize an image prompt for comparison."""
    # Remove common filler words and normalize
    prompt = prompt.lower().strip()
    prompt = re.sub(r"\b(vertical|9:16|8k|photorealistic|cinematic|render|shot)\b", "", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


def check_visual_reuse(scenes: list[dict]) -> list[dict]:
    """Check for visual/prompt reuse and structural monotony within a video's scenes.

    Returns a list of warning dicts:
    - prompt_similarity: high prompt word Jaccard similarity (> 0.65)
    - consecutive_visual_type: 3+ consecutive scenes with identical visual_type
    - consecutive_shot_type: 3+ consecutive scenes with identical shot_type
    - visual_type_overload: single visual_type accounts for > 60% of all scenes
    - primary_subject_overload: same primary_subject in > 50% of scenes
    """
    warnings = []
    if not scenes:
        return warnings

    # 1. Prompt similarity check
    prompts = [s.get("image_prompt", "") for s in scenes]
    normalized = [_normalize_prompt(p) for p in prompts]

    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            words_i = set(normalized[i].split())
            words_j = set(normalized[j].split())
            if not words_i or not words_j:
                continue
            intersection = words_i & words_j
            union = words_i | words_j
            jaccard = len(intersection) / len(union) if union else 0

            if jaccard > 0.65:
                warnings.append({
                    "type": "prompt_similarity",
                    "scene_a": i + 1,
                    "scene_b": j + 1,
                    "similarity": round(jaccard, 2),
                    "message": f"Scene {i+1} and Scene {j+1} have very similar image prompts (similarity: {jaccard:.0%})",
                })
                log.warning(
                    "Visual reuse detected: Scene %d ↔ Scene %d (Jaccard=%.2f)",
                    i + 1, j + 1, jaccard,
                )

    # 2. Consecutive visual_type repetition (3 or more)
    v_types = [_get_scene_field(s, "visual_type").lower() for s in scenes]
    run_len = 1
    for i in range(1, len(v_types)):
        if v_types[i] and v_types[i] == v_types[i - 1]:
            run_len += 1
            if run_len >= 3 and (i == len(v_types) - 1 or (i < len(v_types) - 1 and v_types[i + 1] != v_types[i])):
                scene_indices = list(range(i - run_len + 2, i + 2))
                warnings.append({
                    "type": "consecutive_visual_type",
                    "visual_type": v_types[i],
                    "scenes": scene_indices,
                    "scene_a": scene_indices[0],
                    "scene_b": scene_indices[-1],
                    "count": run_len,
                    "message": f"Visual type '{v_types[i]}' repeated in {run_len} consecutive scenes ({scene_indices})",
                })
                log.warning("Consecutive visual_type repetition: '%s' in scenes %s", v_types[i], scene_indices)
        else:
            run_len = 1

    # 3. Consecutive shot_type repetition (3 or more)
    s_types = [_get_scene_field(s, "shot_type").lower() for s in scenes]
    run_len = 1
    for i in range(1, len(s_types)):
        if s_types[i] and s_types[i] == s_types[i - 1]:
            run_len += 1
            if run_len >= 3 and (i == len(s_types) - 1 or (i < len(s_types) - 1 and s_types[i + 1] != s_types[i])):
                scene_indices = list(range(i - run_len + 2, i + 2))
                warnings.append({
                    "type": "consecutive_shot_type",
                    "shot_type": s_types[i],
                    "scenes": scene_indices,
                    "scene_a": scene_indices[0],
                    "scene_b": scene_indices[-1],
                    "count": run_len,
                    "message": f"Shot type '{s_types[i]}' repeated in {run_len} consecutive scenes ({scene_indices})",
                })
                log.warning("Consecutive shot_type repetition: '%s' in scenes %s", s_types[i], scene_indices)
        else:
            run_len = 1

    # 4. Global visual_type overload (> 60% of all scenes when N >= 4)
    if len(scenes) >= 4:
        from collections import Counter
        non_empty_types = [vt for vt in v_types if vt]
        if non_empty_types:
            type_counts = Counter(non_empty_types)
            for vt, count in type_counts.items():
                pct = count / len(scenes)
                if pct > 0.60:
                    warnings.append({
                        "type": "visual_type_overload",
                        "visual_type": vt,
                        "count": count,
                        "percentage": round(pct, 2),
                        "message": f"Visual type '{vt}' dominates video ({count}/{len(scenes)} scenes, {pct:.0%})",
                    })
                    log.warning("Visual type overload: '%s' constitutes %.0f%% of scenes", vt, pct * 100)

    # 5. Global primary_subject overload (> 50% of scenes when N >= 4)
    subjects = [_get_scene_field(s, "primary_subject").lower() for s in scenes]
    non_empty_subs = [sb for sb in subjects if sb and len(sb) > 2]
    if len(scenes) >= 4 and non_empty_subs:
        from collections import Counter
        sub_counts = Counter(non_empty_subs)
        for sb, count in sub_counts.items():
            pct = count / len(scenes)
            if pct > 0.50:
                warnings.append({
                    "type": "primary_subject_overload",
                    "primary_subject": sb,
                    "count": count,
                    "percentage": round(pct, 2),
                    "message": f"Primary subject '{sb}' repeated in {count}/{len(scenes)} scenes ({pct:.0%})",
                })
                log.warning("Primary subject overload: '%s' in %.0f%% of scenes", sb, pct * 100)

    return warnings


def evaluate_visual_diversity(scenes: list[dict]) -> dict:
    """Evaluate visual diversity across all scenes of a video.

    Returns a dict with:
    - diversity_score: float 0.0-1.0
    - type_distribution: dict of visual_type counts
    - shot_distribution: dict of shot_type counts
    - unique_visual_types_count: int
    - unique_shot_types_count: int
    - warnings: list of warnings from check_visual_reuse()
    - is_acceptable: bool
    """
    if not scenes:
        return {
            "diversity_score": 1.0,
            "type_distribution": {},
            "shot_distribution": {},
            "unique_visual_types_count": 0,
            "unique_shot_types_count": 0,
            "warnings": [],
            "is_acceptable": True,
        }

    from collections import Counter
    v_types = [_get_scene_field(s, "visual_type").lower() for s in scenes]
    s_types = [_get_scene_field(s, "shot_type").lower() for s in scenes]

    type_counts = dict(Counter([vt for vt in v_types if vt]))
    shot_counts = dict(Counter([st for st in s_types if st]))

    warnings = check_visual_reuse(scenes)

    score = 1.0

    for w in warnings:
        w_type = w.get("type", "prompt_similarity")
        if w_type == "prompt_similarity":
            score -= 0.15
        elif w_type == "consecutive_visual_type":
            score -= 0.20
        elif w_type == "consecutive_shot_type":
            score -= 0.10
        elif w_type == "visual_type_overload":
            score -= 0.20
        elif w_type == "primary_subject_overload":
            score -= 0.15

    if len(type_counts) >= 3:
        score += 0.05
    if len(shot_counts) >= 3:
        score += 0.05

    score = max(0.0, min(1.0, round(score, 2)))

    is_acceptable = score >= 0.50 and not any(w.get("type") == "visual_type_overload" for w in warnings)

    return {
        "diversity_score": score,
        "type_distribution": type_counts,
        "shot_distribution": shot_counts,
        "unique_visual_types_count": len(type_counts),
        "unique_shot_types_count": len(shot_counts),
        "warnings": warnings,
        "is_acceptable": is_acceptable,
    }


def diversify_image_prompts(scenes: list[dict]) -> list[dict]:
    """Add variation cues to scenes with prompt similarity or framing monotony.

    Modifies scenes in-place and returns the modified list.
    """
    warnings = check_visual_reuse(scenes)
    if not warnings:
        return scenes

    variation_suffixes = [
        ", different camera angle, unique composition",
        ", close-up detail shot, different perspective",
        ", wide establishing shot, different lighting",
        ", dramatic low angle, contrasting color palette",
        ", aerial view, atmospheric fog",
        ", golden hour lighting, silhouette composition",
        ", candlelit interior, intimate framing",
    ]

    modified_indices = set()
    for w in warnings:
        w_type = w.get("type", "prompt_similarity")
        
        idx_b = w.get("scene_b", 2) - 1
        if idx_b in modified_indices or idx_b >= len(scenes):
            continue

        scene = scenes[idx_b]
        suffix = variation_suffixes[idx_b % len(variation_suffixes)]

        if w_type == "prompt_similarity":
            scene["image_prompt"] = scene.get("image_prompt", "").rstrip(",. ") + suffix
            modified_indices.add(idx_b)
            log.info("Diversified image prompt for Scene %d (prompt similarity)", idx_b + 1)

        elif w_type == "consecutive_visual_type":
            scene["image_prompt"] = scene.get("image_prompt", "").rstrip(",. ") + f", distinct visual framing style ({suffix.lstrip(', ')})"
            modified_indices.add(idx_b)
            log.info("Diversified visual_type repetition for Scene %d", idx_b + 1)

        elif w_type == "consecutive_shot_type":
            alt_shots = ["wide establishing shot", "macro detail shot", "high overhead angle", "low tilt-up shot"]
            alt = alt_shots[idx_b % len(alt_shots)]
            scene["image_prompt"] = scene.get("image_prompt", "").rstrip(",. ") + f", {alt}"
            modified_indices.add(idx_b)
            log.info("Diversified shot_type repetition for Scene %d with '%s'", idx_b + 1, alt)

        elif w_type in ("visual_type_overload", "primary_subject_overload"):
            scene["image_prompt"] = scene.get("image_prompt", "").rstrip(",. ") + suffix
            modified_indices.add(idx_b)

    return scenes


# ---------------------------------------------------------------------------
# Audio Ducking Helper
# ---------------------------------------------------------------------------

def compute_ducking_volume(
    narration_words: list[dict],
    total_duration: float,
    base_volume: float = 0.18,
    ducked_volume: float = 0.10,
    pause_volume: float = 0.22,
) -> callable:
    """Create a volume function that ducks music during narration and raises it during pauses.

    Returns a function(t) -> volume_multiplier suitable for use with moviepy.
    """
    if not narration_words or total_duration <= 0:
        return lambda t: base_volume

    # Build intervals of narration activity
    # Merge words into continuous narration blocks with small gaps
    blocks = []
    current_start = narration_words[0]["start"]
    current_end = narration_words[0]["end"]

    for w in narration_words[1:]:
        if w["start"] - current_end < 0.3:  # Merge if gap < 300ms
            current_end = w["end"]
        else:
            blocks.append((current_start, current_end))
            current_start = w["start"]
            current_end = w["end"]
    blocks.append((current_start, current_end))

    def volume_at_time(t):
        """Return ducking volume for scalar or NumPy-array time inputs.

        MoviePy may evaluate audio frame functions with a scalar timestamp
        or with a NumPy array of timestamps. Array inputs must be handled
        element-wise because chained comparisons on an array are ambiguous.
        """
        import numpy as np

        t_array = np.asarray(t)

        # Fast path for the normal scalar timestamp case.
        if t_array.ndim == 0:
            value = float(t_array)
            for start, end in blocks:
                if start - 0.1 <= value <= end + 0.1:
                    return ducked_volume
            return pause_volume

        # Vectorized path for MoviePy/NumPy audio batches.
        volume = np.full(t_array.shape, pause_volume, dtype=float)
        for start, end in blocks:
            mask = (t_array >= start - 0.1) & (t_array <= end + 0.1)
            volume[mask] = ducked_volume
        return volume

    return volume_at_time
