"""Tests for scene_transitions.py -- Visual Engine V2 Scene Transitions (Phase 15)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scene_transitions


def test_select_scene_transition_initial_scene():
    """First scene returns cut transition."""
    trans = scene_transitions.select_scene_transition(
        scene_a={}, scene_b={"visual_type": "location"}, idx_b=0, total_scenes=5
    )
    assert trans["transition_type"] == "cut"


def test_select_scene_transition_final_scene():
    """Final scene returns fade_black transition."""
    scene_a = {"visual_type": "person"}
    scene_b = {"visual_type": "location"}
    trans = scene_transitions.select_scene_transition(
        scene_a=scene_a, scene_b=scene_b, idx_b=4, total_scenes=5
    )
    assert trans["transition_type"] == "fade_black"
    assert trans["duration_sec"] > 0


def test_select_scene_transition_map_or_graphic():
    """Transitioning to a map or document yields crisp cut."""
    scene_a = {"visual_type": "location"}
    scene_b = {"visual_type": "map"}
    trans = scene_transitions.select_scene_transition(
        scene_a=scene_a, scene_b=scene_b, idx_b=2, total_scenes=5
    )
    assert trans["transition_type"] == "cut"
    assert trans["duration_sec"] == 0.0


def test_select_scene_transition_zoom():
    """Wide establishing shot to close-up yields zoom_transition."""
    scene_a = {"shot_type": "wide_establishing_shot", "visual_type": "location"}
    scene_b = {"shot_type": "close_up_detail", "visual_type": "location"}
    trans = scene_transitions.select_scene_transition(
        scene_a=scene_a, scene_b=scene_b, idx_b=1, total_scenes=5
    )
    assert trans["transition_type"] == "zoom_transition"


def test_select_scene_transition_crossfade_type_shift():
    """Visual type shift between narrative scenes yields crossfade."""
    scene_a = {"visual_type": "person"}
    scene_b = {"visual_type": "artifact"}
    trans = scene_transitions.select_scene_transition(
        scene_a=scene_a, scene_b=scene_b, idx_b=1, total_scenes=5
    )
    assert trans["transition_type"] == "crossfade"
    assert trans["duration_sec"] >= 0.25
