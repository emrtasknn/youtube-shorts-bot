"""Regression tests for the Visual Engine V2 guardrails added after output review."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import event_memory
import prompt_engine
import scene_motion


def test_text_heavy_editorial_assets_are_flagged():
    risk, terms = event_memory._visual_text_risk(
        "The Death of Aeschylus - Poster",
        "illustrated title card with text overlay",
    )
    assert risk >= 0.6
    assert "poster" in terms


def test_clean_historical_photo_has_low_text_risk():
    risk, terms = event_memory._visual_text_risk(
        "Aeschylus marble bust",
        "Ancient Greek sculpture photograph",
    )
    assert risk == 0.0
    assert terms == []


def test_final_atmosphere_resolution_falls_back_to_ai_reconstruction():
    result = event_memory.resolve_visual_source(
        scene_description="A mystery remains unresolved",
        visual_search_terms=[],
        event_title="Death of Aeschylus",
        visual_intent={
            "visual_role": "atmosphere",
            "visual_fact": "The unresolved historical mystery",
        },
        event_context={"title": "Death of Aeschylus"},
        final_scene=True,
    )
    assert result["source_type"] == "ai_reconstruction"
    assert result["visual_role"] == "aftermath"


def test_event_reconstruction_prompt_prioritizes_the_event():
    scene = {
        "image_prompt": "Ancient Greek historical scene",
        "visual_fact": "An eagle drops a tortoise onto Aeschylus",
        "visual_role": "event_reconstruction",
        "visual_intent": {
            "visual_action": "An eagle releases a tortoise directly above Aeschylus",
            "scene_context": "Ancient Greece",
            "shot_type": "medium-wide action shot",
            "composition": "Aeschylus below, eagle and falling tortoise clearly visible",
            "primary_subject": "Aeschylus and the eagle-tortoise event",
            "visual_entities": ["Aeschylus", "eagle", "tortoise"],
            "must_show": ["eagle", "falling tortoise", "Aeschylus"],
            "avoid": ["sunset landscape", "generic eagle portrait"],
        },
    }
    prompt = prompt_engine.build_ai_prompt(scene, "The Death of Aeschylus")
    assert "EVENT RECONSTRUCTION" in prompt
    assert "event itself is the subject" in prompt
    assert "NO visible text" in prompt
    assert "title cards" in prompt


def test_event_reconstruction_uses_action_motion():
    scene = {
        "visual_role": "event_reconstruction",
        "visual_intent": {
            "visual_type": "event_reconstruction",
        },
    }
    assert scene_motion.derive_scene_type(scene) == "action"
