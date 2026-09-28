"""Visual Engine V2 Scene Transitions Engine (Phase 15).

This module determines scene-aware transitions (crossfade, cut, zoom_transition,
fade_black) based on visual_type, shot_type, and narrative context.
"""

import logging

log = logging.getLogger("shorts-bot.scene_transitions")


def _get_scene_field(scene: dict, field_name: str, default: str = "") -> str:
    """Retrieve a visual field from scene dict or nested visual_intent dict."""
    if not isinstance(scene, dict):
        return default
    val = scene.get(field_name)
    if not val and isinstance(scene.get("visual_intent"), dict):
        val = scene.get("visual_intent", {}).get(field_name)
    return str(val or default).strip().lower()


def select_scene_transition(
    scene_a: dict,
    scene_b: dict,
    idx_b: int,
    total_scenes: int,
) -> dict:
    """Determine the optimal transition between scene_a and scene_b.

    Args:
        scene_a: Previous scene dictionary.
        scene_b: Current scene dictionary.
        idx_b: 0-indexed position of scene_b (1 for second scene).
        total_scenes: Total number of scenes in the video.

    Returns:
        dict with transition_type, duration_sec, and reason.
    """
    if not scene_a:
        return {
            "transition_type": "cut",
            "duration_sec": 0.0,
            "reason": "Initial scene",
        }

    vtype_a = _get_scene_field(scene_a, "visual_type", "historical_photo")
    vtype_b = _get_scene_field(scene_b, "visual_type", "historical_photo")

    shot_a = _get_scene_field(scene_a, "shot_type", "medium_shot")
    shot_b = _get_scene_field(scene_b, "shot_type", "medium_shot")

    # 1. Final scene: fade black transition
    if idx_b == total_scenes - 1:
        return {
            "transition_type": "fade_black",
            "duration_sec": 0.35,
            "reason": "Final scene transition to black",
        }

    # 2. Informational graphics / maps / documents prefer crisp cut
    if any(k in vtype_b or k in vtype_a for k in ("map", "document", "diagram", "infographic", "text")):
        return {
            "transition_type": "cut",
            "duration_sec": 0.0,
            "reason": "Graphic/map scene crisp cut",
        }

    # 3. Macro / Close-up push: zoom transition
    if ("wide" in shot_a or "establishing" in shot_a) and ("close" in shot_b or "macro" in shot_b):
        return {
            "transition_type": "zoom_transition",
            "duration_sec": 0.25,
            "reason": "Wide shot to close-up zoom transition",
        }

    # 4. Cinematic / Narrative shift between distinct subjects: crossfade
    if vtype_a != vtype_b or "person" in vtype_b or "dramatic" in _get_scene_field(scene_b, "visual_role"):
        return {
            "transition_type": "crossfade",
            "duration_sec": 0.30,
            "reason": "Cinematic visual type shift crossfade",
        }

    # Default fallback: subtle crossfade
    return {
        "transition_type": "crossfade",
        "duration_sec": 0.25,
        "reason": "Default smooth crossfade",
    }
