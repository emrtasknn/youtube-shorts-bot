"""Tests for scene_motion.py — Visual Engine V2 motion strategy selection."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST")
os.environ.setdefault("TELEGRAM_CHAT_ID", "1")

import scene_motion


# ---------------------------------------------------------------------------
# derive_scene_type tests
# ---------------------------------------------------------------------------

def _scene(role: str, vtype: str = "") -> dict:
    return {
        "visual_role": role,
        "visual_intent": {"visual_type": vtype},
    }


def test_derive_context_map_gives_map():
    assert scene_motion.derive_scene_type(_scene("context_map", "map")) == "map"
    assert scene_motion.derive_scene_type(_scene("evidence", "map")) == "map"
    assert scene_motion.derive_scene_type(_scene("mechanism", "diagram")) == "map"


def test_derive_person_or_entity_gives_portrait():
    assert scene_motion.derive_scene_type(_scene("person_or_entity", "person")) == "portrait"
    assert scene_motion.derive_scene_type(_scene("person_or_entity", "historical_photo")) == "portrait"


def test_derive_person_or_entity_crowd_gives_crowd():
    assert scene_motion.derive_scene_type(_scene("person_or_entity", "crowd")) == "crowd"
    assert scene_motion.derive_scene_type(_scene("person_or_entity", "action")) == "crowd"


def test_derive_evidence_document_gives_document():
    assert scene_motion.derive_scene_type(_scene("evidence", "document")) == "document"
    assert scene_motion.derive_scene_type(_scene("evidence", "letter")) == "document"


def test_derive_evidence_photo_gives_archival():
    assert scene_motion.derive_scene_type(_scene("evidence", "historical_photo")) == "archival"
    assert scene_motion.derive_scene_type(_scene("evidence", "photo")) == "archival"


def test_derive_evidence_other_gives_artifact():
    assert scene_motion.derive_scene_type(_scene("evidence", "artifact")) == "artifact"
    assert scene_motion.derive_scene_type(_scene("evidence", "")) == "artifact"


def test_derive_reconstruction_landscape_gives_landscape():
    assert scene_motion.derive_scene_type(_scene("reconstruction", "landscape")) == "landscape"
    assert scene_motion.derive_scene_type(_scene("reconstruction", "aerial")) == "landscape"


def test_derive_reconstruction_architecture_gives_architecture():
    assert scene_motion.derive_scene_type(_scene("reconstruction", "architecture")) == "architecture"
    assert scene_motion.derive_scene_type(_scene("reconstruction", "palace")) == "architecture"


def test_derive_reconstruction_default_gives_reconstruction():
    assert scene_motion.derive_scene_type(_scene("reconstruction", "")) == "reconstruction"
    assert scene_motion.derive_scene_type(_scene("reconstruction", "cinematic")) == "reconstruction"


def test_derive_mechanism_action_gives_action():
    assert scene_motion.derive_scene_type(_scene("mechanism", "action")) == "action"
    assert scene_motion.derive_scene_type(_scene("mechanism", "battle")) == "action"


def test_derive_mechanism_other_gives_reconstruction():
    assert scene_motion.derive_scene_type(_scene("mechanism", "")) == "reconstruction"


def test_derive_aftermath_gives_aftermath():
    assert scene_motion.derive_scene_type(_scene("aftermath", "")) == "aftermath"
    assert scene_motion.derive_scene_type(_scene("aftermath", "landscape")) == "aftermath"


def test_derive_atmosphere_landscape_gives_landscape():
    assert scene_motion.derive_scene_type(_scene("atmosphere", "landscape")) == "landscape"


def test_derive_atmosphere_default_gives_atmosphere():
    assert scene_motion.derive_scene_type(_scene("atmosphere", "fog")) == "atmosphere"


def test_derive_unknown_gives_default():
    assert scene_motion.derive_scene_type(_scene("", "")) == "default"
    assert scene_motion.derive_scene_type({}) == "default"


# ---------------------------------------------------------------------------
# select_motion tests
# ---------------------------------------------------------------------------

def test_select_motion_portrait_gives_slow_push_in():
    s = _scene("person_or_entity", "person")
    assert scene_motion.select_motion(s, 1) == "slow_push_in"


def test_select_motion_aftermath_gives_zoom_out():
    s = _scene("aftermath", "")
    assert scene_motion.select_motion(s, 3) == "zoom_out"


def test_select_motion_document_gives_drift():
    s = _scene("evidence", "document")
    assert scene_motion.select_motion(s, 2) == "drift"


def test_select_motion_archival_gives_drift():
    s = _scene("evidence", "historical_photo")
    assert scene_motion.select_motion(s, 4) == "drift"


def test_select_motion_architecture_gives_vertical_tilt():
    s = _scene("reconstruction", "architecture")
    assert scene_motion.select_motion(s, 5) == "vertical_tilt"


def test_select_motion_map_gives_zoom_in():
    s = _scene("context_map", "map")
    assert scene_motion.select_motion(s, 1) == "zoom_in"


def test_select_motion_crowd_alternates_pan_direction():
    s = _scene("person_or_entity", "crowd")
    motions = {scene_motion.select_motion(s, i) for i in range(1, 9)}
    # Both directions should appear
    assert "pan_left_right" in motions
    assert "pan_right_left" in motions
    # Only valid pan directions
    assert motions <= {"pan_left_right", "pan_right_left"}


def test_select_motion_landscape_alternates():
    s = _scene("atmosphere", "landscape")
    results = [scene_motion.select_motion(s, i) for i in range(1, 9)]
    assert set(results) <= {"pan_left_right", "pan_right_left"}


def test_select_motion_reconstruction_gives_slow_push_in():
    s = _scene("reconstruction", "battle")
    assert scene_motion.select_motion(s, 2) == "slow_push_in"


def test_select_motion_legacy_fallback_when_disabled(monkeypatch):
    monkeypatch.setattr(scene_motion, "SCENE_AWARE_MOTION_ENABLED", False)
    s = _scene("person_or_entity", "person")
    results = [scene_motion.select_motion(s, i) for i in range(1, 5)]
    assert results == ["zoom_in", "pan_left_right", "zoom_out", "pan_right_left"]


def test_select_motion_returns_string_for_all_known_roles():
    roles = [
        "evidence", "mechanism", "reconstruction",
        "context_map", "person_or_entity", "aftermath", "atmosphere",
    ]
    vtypes = ["", "document", "person", "crowd", "map", "landscape", "architecture"]
    for role in roles:
        for vtype in vtypes:
            result = scene_motion.select_motion(_scene(role, vtype), 1)
            assert isinstance(result, str)
            assert len(result) > 0
