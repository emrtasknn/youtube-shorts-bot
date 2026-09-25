"""Tests for the content_memory module — Content Novelty Engine.

These tests validate persistence, visual reuse detection, audio selection,
and ducking computation without requiring external API calls.
"""

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path for pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST")
os.environ.setdefault("TELEGRAM_CHAT_ID", "1")

import content_memory


# ---------------------------------------------------------------------------
# Content Memory Persistence
# ---------------------------------------------------------------------------

def test_load_empty_content_memory(tmp_path):
    """Loading from a non-existent file returns an empty list."""
    result = content_memory.load_content_memory(tmp_path / "nonexistent.json")
    assert result == []


def test_save_and_load_content_entry(tmp_path):
    """Saving an entry and reloading it returns the same data."""
    memory_file = tmp_path / "content_memory.json"
    entry = {
        "content_id": "test-001",
        "title": "Göbeklitepe",
        "topic": "Göbeklitepe",
        "angle": "T-biçimli sütunlar",
        "summary": "Test özet",
        "main_claim": "Test iddia",
        "key_facts": ["fakt1", "fakt2"],
        "entities": ["Göbeklitepe"],
        "script": "Test script text",
        "created_at": "2026-01-01T00:00:00",
        "status": "completed",
    }
    content_memory.save_content_entry(entry, memory_file)
    loaded = content_memory.load_content_memory(memory_file)
    assert len(loaded) == 1
    assert loaded[0]["content_id"] == "test-001"
    assert loaded[0]["angle"] == "T-biçimli sütunlar"


def test_multiple_entries_accumulate(tmp_path):
    """Multiple saves accumulate entries rather than overwriting."""
    memory_file = tmp_path / "content_memory.json"
    for i in range(5):
        content_memory.save_content_entry(
            {"content_id": f"entry-{i}", "title": f"Topic {i}"},
            memory_file,
        )
    loaded = content_memory.load_content_memory(memory_file)
    assert len(loaded) == 5


def test_build_content_entry_structure():
    """build_content_entry produces all required fields."""
    entry = content_memory.build_content_entry(
        topic={"title": "Test Topic", "hook_question": "Question?"},
        scenes=[{"narration": "Test narration", "image_prompt": "Test prompt"}],
        analysis={
            "topic": "Test Topic",
            "angle": "test angle",
            "summary": "summary",
            "main_claim": "claim",
            "key_facts": ["f1"],
            "entities": ["e1"],
            "mood": "mysterious",
            "energy": 0.5,
            "tension": 0.3,
        },
        full_text="Test narration",
    )
    assert "content_id" in entry
    assert entry["topic"] == "Test Topic"
    assert entry["angle"] == "test angle"
    assert entry["mood"] == "mysterious"
    assert entry["status"] == "completed"


# ---------------------------------------------------------------------------
# Content-Aware Negative Prompt
# ---------------------------------------------------------------------------

def test_build_content_aware_negative_prompt_empty():
    """Empty memory yields empty prompt."""
    assert content_memory.build_content_aware_negative_prompt([]) == ""


def test_build_content_aware_negative_prompt_with_entries():
    """Non-empty memory yields a formatted prompt."""
    memory = [
        {
            "topic": "Göbeklitepe",
            "angle": "T-biçimli sütunlar",
            "main_claim": "Test claim",
            "key_facts": ["f1", "f2"],
            "summary": "Test summary",
        }
    ]
    result = content_memory.build_content_aware_negative_prompt(memory)
    assert "Göbeklitepe" in result
    assert "T-biçimli sütunlar" in result


# ---------------------------------------------------------------------------
# Visual Reuse Guard
# ---------------------------------------------------------------------------

def test_visual_reuse_detection_with_similar_prompts():
    """Detects similar image prompts within a video."""
    scenes = [
        {"narration": "n1", "image_prompt": "Ancient stone pillars in desert, cinematic lighting, vertical 9:16, photorealistic 8k"},
        {"narration": "n2", "image_prompt": "Dark cave with mysterious artifacts, moody lighting, vertical 9:16"},
        {"narration": "n3", "image_prompt": "Ancient stone pillars in desert, dramatic lighting, vertical 9:16, photorealistic 8k render"},
    ]
    warnings = content_memory.check_visual_reuse(scenes)
    # Scenes 1 and 3 should have high similarity
    assert len(warnings) >= 1
    assert any(w["scene_a"] == 1 and w["scene_b"] == 3 for w in warnings)


def test_visual_reuse_no_false_positives():
    """Different prompts should not trigger reuse warnings."""
    scenes = [
        {"narration": "n1", "image_prompt": "Ancient Roman forum with marble columns and senators"},
        {"narration": "n2", "image_prompt": "Dark ocean with moonlight reflecting on waves"},
        {"narration": "n3", "image_prompt": "Medieval castle on hilltop during sunset"},
    ]
    warnings = content_memory.check_visual_reuse(scenes)
    assert len(warnings) == 0


def test_diversify_image_prompts():
    """Diversification modifies similar scene prompts."""
    scenes = [
        {"narration": "n1", "image_prompt": "Ancient pillars in desert landscape with dramatic lighting"},
        {"narration": "n2", "image_prompt": "Unique ocean scene with mysterious fog"},
        {"narration": "n3", "image_prompt": "Ancient pillars in desert landscape with dramatic lighting effect"},
    ]
    original_prompt_3 = scenes[2]["image_prompt"]
    content_memory.diversify_image_prompts(scenes)
    # Scene 3's prompt should have been modified
    assert scenes[2]["image_prompt"] != original_prompt_3


# ---------------------------------------------------------------------------
# Content-Aware Audio Selection
# ---------------------------------------------------------------------------

def test_select_audio_for_mysterious_mood(tmp_path):
    """Mysterious mood should prefer ambient/mystery tracks."""
    # Create fake audio files
    for name in ["ambient_mystery.mp3", "suspense_thriller.mp3", "electronic_chill.mp3"]:
        (tmp_path / name).write_bytes(b"fake audio content" * 1000)

    analysis = {"mood": "mysterious", "energy": 0.3, "tension": 0.2}
    result = content_memory.select_audio_for_content(analysis, tmp_path)
    assert result is not None
    assert result.name == "ambient_mystery.mp3"


def test_select_audio_for_suspenseful_mood(tmp_path):
    """Suspenseful mood should prefer suspense/investigation tracks."""
    for name in ["ambient_mystery.mp3", "suspense_thriller.mp3", "investigation_tension.mp3"]:
        (tmp_path / name).write_bytes(b"fake audio content" * 1000)

    analysis = {"mood": "suspenseful", "energy": 0.7, "tension": 0.8}
    result = content_memory.select_audio_for_content(analysis, tmp_path)
    assert result is not None
    assert result.name == "suspense_thriller.mp3"


def test_select_audio_avoids_recently_used(tmp_path):
    """Recently used tracks should be deprioritized."""
    for name in ["ambient_mystery.mp3", "suspense_thriller.mp3", "mystery_investigation.mp3"]:
        (tmp_path / name).write_bytes(b"fake audio content" * 1000)

    analysis = {"mood": "mysterious", "energy": 0.3, "tension": 0.2}
    result = content_memory.select_audio_for_content(
        analysis, tmp_path, recently_used=["ambient_mystery.mp3"]
    )
    assert result is not None
    # Should pick mystery_investigation.mp3 instead of ambient_mystery.mp3
    assert result.name != "ambient_mystery.mp3"


def test_select_audio_empty_directory(tmp_path):
    """Empty audio directory returns None."""
    result = content_memory.select_audio_for_content(
        {"mood": "mysterious"}, tmp_path
    )
    assert result is None


def test_get_recently_used_audio():
    """Extracts recent audio track names from memory."""
    memory = [
        {"audio_track": "ambient_mystery.mp3"},
        {"audio_track": "suspense_thriller.mp3"},
        {"audio_track": "mystery_investigation.mp3"},
    ]
    recent = content_memory.get_recently_used_audio(memory, lookback=2)
    assert len(recent) == 2
    assert "suspense_thriller.mp3" in recent
    assert "mystery_investigation.mp3" in recent


# ---------------------------------------------------------------------------
# Audio Ducking
# ---------------------------------------------------------------------------

def test_compute_ducking_volume_returns_callable():
    """Ducking function should return different volumes during/between narration."""
    words = [
        {"word": "test", "start": 0.5, "end": 1.0},
        {"word": "word", "start": 1.1, "end": 1.5},
    ]
    fn = content_memory.compute_ducking_volume(words, total_duration=5.0)
    # During narration (t=0.7) — should be ducked
    vol_during = fn(0.7)
    # During pause (t=3.0) — should be louder
    vol_pause = fn(3.0)
    assert vol_during < vol_pause


def test_compute_ducking_volume_empty_words():
    """Empty words data should return constant base volume."""
    fn = content_memory.compute_ducking_volume([], total_duration=5.0)
    assert fn(0.0) == 0.18
    assert fn(2.5) == 0.18


# ---------------------------------------------------------------------------
# Novelty Check Format
# ---------------------------------------------------------------------------

def test_format_previous_content_summary_empty():
    """Empty memory returns a no-content message."""
    result = content_memory.format_previous_content_summary([])
    assert "Henüz" in result


def test_format_previous_content_summary_with_data():
    """Formats memory entries into structured text."""
    memory = [
        {
            "topic": "Roanoke",
            "angle": "Kaybolma",
            "main_claim": "115 insan kayboldu",
            "key_facts": ["fakt1"],
            "summary": "Roanoke hakkında özet",
        }
    ]
    result = content_memory.format_previous_content_summary(memory)
    assert "Roanoke" in result
    assert "Kaybolma" in result


# ---------------------------------------------------------------------------
# Novelty Check — Default Behavior (empty memory)
# ---------------------------------------------------------------------------

def test_check_novelty_empty_memory_accepts():
    """Empty content memory should always ACCEPT."""
    result = content_memory.check_content_novelty(
        candidate_analysis={"topic": "Test", "angle": "angle"},
        memory=[],
    )
    assert result["decision"] == "ACCEPT"
