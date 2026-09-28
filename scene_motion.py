"""
scene_motion.py — Visual Engine V2: Scene-Aware Motion Strategy

Replaces the previous round-robin motion selection with content-driven
motion decisions derived from each scene's structured metadata.

Feature flag: SCENE_AWARE_MOTION_ENABLED (env var, default "true")
When disabled, falls back to the original V1 round-robin behavior.

Motion types produced:
  zoom_in        - standard Ken Burns zoom in (legacy, used for maps/default)
  zoom_out       - standard Ken Burns zoom out (aftermath)
  pan_left_right - horizontal pan, left to right (crowd, action)
  pan_right_left - horizontal pan, right to left (alternated)
  slow_push_in   - gentle cinematic push in (portrait, reconstruction)
  vertical_tilt  - bottom-to-top reveal (architecture, establishing)
  drift          - subtle diagonal drift (archival, document, artifact)

All motion types are implemented in pipeline.build_scene_clip().
This module is purely a routing/selection layer.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------
SCENE_AWARE_MOTION_ENABLED: bool = os.getenv("SCENE_AWARE_MOTION_ENABLED", "true").lower() in (
    "1", "true", "yes"
)

# ---------------------------------------------------------------------------
# Legacy fallback (V1 round-robin)
# ---------------------------------------------------------------------------
_LEGACY_MOTION_TYPES: list[str] = ["zoom_in", "pan_left_right", "zoom_out", "pan_right_left"]

# ---------------------------------------------------------------------------
# Scene type -> base motion mapping
# ---------------------------------------------------------------------------
_SCENE_TYPE_TO_MOTION: dict[str, str] = {
    "portrait": "slow_push_in",
    "crowd": "pan_left_right",
    "map": "zoom_in",
    "landscape": "pan_left_right",
    "atmosphere": "pan_left_right",
    "architecture": "vertical_tilt",
    "document": "drift",
    "archival": "drift",
    "artifact": "drift",
    "action": "pan_left_right",
    "reconstruction": "slow_push_in",
    "mechanism": "zoom_in",
    "aftermath": "zoom_out",
    "default": "zoom_in",
}

# Motions that should alternate direction to prevent monotony
_ALTERNATING_PAIRS: dict[str, list[str]] = {
    "pan_left_right": ["pan_left_right", "pan_right_left"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_scene_type(scene: dict) -> str:
    """Return a canonical scene type string from structured scene metadata."""
    visual_role: str = str(scene.get("visual_role", "")).lower().strip()
    visual_type: str = str(
        scene.get("visual_intent", {}).get("visual_type", "")
    ).lower().strip()

    if visual_role == "context_map" or "map" in visual_type or "diagram" in visual_type:
        return "map"

    if visual_role == "person_or_entity":
        if "crowd" in visual_type or "action" in visual_type or "battle" in visual_type:
            return "crowd"
        return "portrait"

    if visual_role == "evidence":
        if "document" in visual_type or "letter" in visual_type or "text" in visual_type:
            return "document"
        if (
            "historical_photo" in visual_type
            or "photo" in visual_type
            or "archive" in visual_type
        ):
            return "archival"
        return "artifact"

    if visual_role == "reconstruction":
        if "landscape" in visual_type or "aerial" in visual_type or "wide" in visual_type:
            return "landscape"
        if (
            "architecture" in visual_type
            or "building" in visual_type
            or "city" in visual_type
            or "castle" in visual_type
            or "palace" in visual_type
        ):
            return "architecture"
        return "reconstruction"

    if visual_role == "mechanism":
        if "action" in visual_type or "crowd" in visual_type or "battle" in visual_type:
            return "action"
        return "reconstruction"

    if visual_role == "aftermath":
        return "aftermath"

    if visual_role == "atmosphere":
        if "landscape" in visual_type or "location" in visual_type or "wide" in visual_type:
            return "landscape"
        return "atmosphere"

    return "default"


def select_motion(scene: dict, scene_index: int) -> str:
    """Return the motion type string for a given scene.

    Args:
        scene:       The scene dict from the validated script.
        scene_index: 1-based scene position in the video (1 = first scene).

    Returns:
        A motion type string consumed by pipeline.build_scene_clip().
    """
    if not SCENE_AWARE_MOTION_ENABLED:
        return _LEGACY_MOTION_TYPES[(scene_index - 1) % len(_LEGACY_MOTION_TYPES)]

    scene_type = derive_scene_type(scene)
    base_motion = _SCENE_TYPE_TO_MOTION.get(scene_type, "zoom_in")

    alternates = _ALTERNATING_PAIRS.get(base_motion)
    if alternates:
        return alternates[scene_index % len(alternates)]

    return base_motion
