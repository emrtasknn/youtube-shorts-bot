"""Tests for prompt_engine.py -- Visual Engine V2 Prompt Construction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompt_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_scene(
    image_prompt="cinematic wide shot ancient battle",
    visual_fact="Army crossed the river at dawn",
    visual_role="reconstruction",
    visual_action="Soldiers charging across the river",
    scene_context="River bank in ancient battlefield",
    shot_type="wide",
    composition="Soldiers fill lower 2/3 of frame",
    primary_subject="Army of soldiers crossing river",
    visual_entities=None,
    must_show=None,
    avoid=None,
):
    return {
        "image_prompt": image_prompt,
        "visual_fact": visual_fact,
        "visual_role": visual_role,
        "visual_intent": {
            "visual_action": visual_action,
            "scene_context": scene_context,
            "shot_type": shot_type,
            "composition": composition,
            "primary_subject": primary_subject,
            "visual_entities": visual_entities or ["army", "river"],
            "must_show": must_show or ["soldiers", "river"],
            "avoid": avoid or ["modern objects", "text"],
        },
    }


def _ctx(date="1453", location="Istanbul", title="Istanbul Fethinin Anlami"):
    return {"date": date, "location": location, "title": title}


# ---------------------------------------------------------------------------
# build_ai_prompt: structural tests
# ---------------------------------------------------------------------------

def test_prompt_includes_base_image_prompt():
    scene = _full_scene(image_prompt="Majestic ancient citadel at sunset")
    result = prompt_engine.build_ai_prompt(scene, "Test Event")
    assert "Majestic ancient citadel at sunset" in result


def test_prompt_includes_visual_fact():
    scene = _full_scene(visual_fact="The wall was 12 meters tall")
    result = prompt_engine.build_ai_prompt(scene, "Test Event")
    assert "The wall was 12 meters tall" in result


def test_prompt_includes_visual_action():
    scene = _full_scene(visual_action="Soldiers storming the gate")
    result = prompt_engine.build_ai_prompt(scene, "Test Event")
    assert "Soldiers storming the gate" in result


def test_prompt_includes_shot_type():
    scene = _full_scene(shot_type="close-up")
    result = prompt_engine.build_ai_prompt(scene, "Test Event")
    assert "close-up" in result.lower()


def test_prompt_includes_topic_title():
    scene = _full_scene()
    result = prompt_engine.build_ai_prompt(scene, "Fatih Sultan Mehmet'in Fethi")
    assert "Fatih Sultan Mehmet" in result


def test_prompt_includes_must_show():
    scene = _full_scene(must_show=["Ottoman banner", "cannon"])
    result = prompt_engine.build_ai_prompt(scene, "Test Event")
    assert "Ottoman banner" in result
    assert "cannon" in result


def test_prompt_includes_avoid():
    scene = _full_scene(avoid=["modern clothing", "cars"])
    result = prompt_engine.build_ai_prompt(scene, "Test Event")
    assert "modern clothing" in result
    assert "cars" in result


def test_prompt_includes_directorial_instruction():
    scene = _full_scene()
    result = prompt_engine.build_ai_prompt(scene, "Test Event")
    assert "historical ACTION" in result


def test_prompt_includes_technical_constraint():
    scene = _full_scene()
    result = prompt_engine.build_ai_prompt(scene, "Test Event")
    assert "9:16" in result
    assert "watermarks" in result


def test_prompt_returns_string():
    scene = _full_scene()
    result = prompt_engine.build_ai_prompt(scene, "Test Event")
    assert isinstance(result, str)
    assert len(result) > 100


def test_prompt_no_blank_fields():
    # Fields with no data should not produce "Avoid: ." or "Visual action: ."
    scene = {
        "image_prompt": "ancient scene",
        "visual_fact": "A wall stood",
        "visual_role": "evidence",
        "visual_intent": {
            "visual_action": "",
            "scene_context": "",
            "shot_type": "",
            "composition": "",
            "primary_subject": "",
            "visual_entities": [],
            "must_show": [],
            "avoid": [],
        },
    }
    result = prompt_engine.build_ai_prompt(scene, "Test Event")
    assert "Avoid: ." not in result
    assert "Visual action: ." not in result
    assert "Must visibly include: ." not in result


def test_prompt_without_event_context_still_works():
    scene = _full_scene()
    result = prompt_engine.build_ai_prompt(scene, "Test Event", event_context=None)
    assert len(result) > 50


def test_prompt_with_event_context_adds_era():
    scene = _full_scene()
    ctx = {"date": "1453", "location": "Istanbul"}
    result = prompt_engine.build_ai_prompt(scene, "Fetih", event_context=ctx)
    assert "Ottoman" in result or "late medieval" in result


# ---------------------------------------------------------------------------
# era_from_date tests
# ---------------------------------------------------------------------------

def test_era_ancient_bc():
    assert "ancient" in prompt_engine.era_from_date("44 BCE").lower()
    assert "ancient" in prompt_engine.era_from_date("M.O. 44").lower()


def test_era_medieval():
    assert "medieval" in prompt_engine.era_from_date("1095").lower()


def test_era_late_medieval_ottoman():
    result = prompt_engine.era_from_date("1453")
    assert "medieval" in result.lower() or "modern" in result.lower()


def test_era_early_modern_16th():
    assert "16th" in prompt_engine.era_from_date("1520")


def test_era_19th_century():
    assert "19th" in prompt_engine.era_from_date("1876")


def test_era_early_20th():
    assert "20th" in prompt_engine.era_from_date("1915")


def test_era_contemporary():
    assert "contemporary" in prompt_engine.era_from_date("2024")


def test_era_empty_returns_empty():
    assert prompt_engine.era_from_date("") == ""
    assert prompt_engine.era_from_date("no year here") == ""


def test_era_iso_date():
    result = prompt_engine.era_from_date("1453-05-29")
    assert result != ""
    assert "medieval" in result.lower() or "modern" in result.lower()


# ---------------------------------------------------------------------------
# geographic_context tests
# ---------------------------------------------------------------------------

def test_geo_ottoman():
    result = prompt_engine.geographic_context("Istanbul, Osmanlı İmparatorluğu")
    assert "Ottoman" in result


def test_geo_roman():
    result = prompt_engine.geographic_context("Rome, Italy")
    assert "Roman" in result


def test_geo_egypt():
    result = prompt_engine.geographic_context("Kahire, Mısır")
    assert "Egyptian" in result


def test_geo_greece():
    result = prompt_engine.geographic_context("Atina, Yunanistan")
    assert "Greek" in result


def test_geo_mesopotamia():
    result = prompt_engine.geographic_context("Babil, Mezopotamya")
    assert "Mesopotamian" in result


def test_geo_europe():
    result = prompt_engine.geographic_context("Paris, Fransa")
    assert "European" in result


def test_geo_unknown_returns_generic():
    result = prompt_engine.geographic_context("Unknown Planet X")
    assert "Unknown Planet X" in result


def test_geo_empty_returns_empty():
    assert prompt_engine.geographic_context("") == ""


def test_build_ai_video_prompt_is_single_shot_and_uses_event_context():
    prompt = prompt_engine.build_ai_video_prompt(
        {
            "image_prompt": "A 1908 Siberian forest after the blast.",
            "visual_fact": "The blast wave knocked down trees.",
            "visual_role": "aftermath",
            "visual_intent": {
                "primary_subject": "fallen trees",
                "visual_action": "trees visibly knocked down by a blast wave",
                "scene_context": "remote Siberian forest",
                "shot_type": "wide establishing shot",
                "composition": "fallen trunks radiating away from the blast center",
                "must_show": ["flattened trees", "forest floor"],
                "avoid": ["modern machinery"],
            },
        },
        "The Tunguska Event",
        {"date": "1908", "location": "Siberia, Russia"},
    )
    assert "single continuous unbroken historical documentary shot" in prompt.lower()
    assert "trees visibly knocked down by a blast wave" in prompt
    assert "historical event: the tunguska event" in prompt.lower()
    assert "1908" in prompt
