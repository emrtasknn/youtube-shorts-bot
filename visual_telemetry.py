"""Visual Engine V2 Telemetry & Storyboard Reporting (Phase 19).

This module records, aggregates, and exports structured visual telemetry
and storyboard metadata for YouTube Shorts generation runs.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("shorts-bot.visual_telemetry")


def _get_scene_field(scene: dict, field_name: str, default: str = "") -> str:
    """Retrieve a visual field from scene dict or nested visual_intent dict."""
    val = scene.get(field_name)
    if not val and isinstance(scene.get("visual_intent"), dict):
        val = scene.get("visual_intent", {}).get(field_name)
    return str(val or default).strip()


def create_scene_telemetry(
    scene: dict,
    scene_index: int,
    start_time: float,
    end_time: float,
    motion_type: str = "none",
    provider: str = "pollinations",
) -> dict:
    """Build a standardized telemetry record for a single scene."""
    qc = scene.get("image_qc", {})
    enhancement = scene.get("image_enhancement", {})

    duration = round(max(0.0, end_time - start_time), 2)

    return {
        "scene_index": scene_index,
        "start_time": round(start_time, 2),
        "end_time": round(end_time, 2),
        "duration_sec": duration,
        "visual_type": _get_scene_field(scene, "visual_type", "historical_photo"),
        "shot_type": _get_scene_field(scene, "shot_type", "medium_shot"),
        "primary_subject": _get_scene_field(scene, "primary_subject", ""),
        "visual_role": scene.get("visual_role", ""),
        "provider": provider,
        "image_prompt": scene.get("image_prompt", ""),
        "generation_attempts": qc.get("attempts", 1),
        "qc_valid": qc.get("valid", True),
        "qc_reason": qc.get("reason", ""),
        "qc_width": qc.get("width", 0),
        "qc_height": qc.get("height", 0),
        "qc_unique_colors": qc.get("unique_colors", -1),
        "qc_file_size_bytes": qc.get("file_size_bytes", 0),
        "enhancement_enabled": enhancement.get("enabled", False),
        "enhancement_profile": enhancement.get("profile", "none"),
        "motion_type": motion_type,
    }


def build_video_telemetry_report(
    scenes: list[dict],
    candidate: dict,
    diversity_report: Optional[dict] = None,
    run_dir: Optional[Path] = None,
) -> dict:
    """Aggregate scene telemetry records into a video-level telemetry report.

    Optionally saves the report to run_dir / 'visual_telemetry.json'.
    """
    scene_records = []
    total_attempts = 0
    qc_passed_count = 0

    for i, s in enumerate(scenes, 1):
        record = s.get("telemetry")
        if not record:
            record = create_scene_telemetry(
                scene=s,
                scene_index=i,
                start_time=s.get("start_time", 0.0),
                end_time=s.get("end_time", 0.0),
                motion_type=s.get("motion_type", "none"),
            )
        scene_records.append(record)

        attempts = record.get("generation_attempts", 1)
        total_attempts += attempts
        if record.get("qc_valid", True):
            qc_passed_count += 1

    total_scenes = len(scenes)
    qc_pass_rate = round(qc_passed_count / total_scenes, 2) if total_scenes > 0 else 1.0

    div = diversity_report or {}

    report = {
        "video_id": candidate.get("id", "unknown"),
        "title": candidate.get("title", ""),
        "topic": candidate.get("topic", ""),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_scenes": total_scenes,
        "total_duration_sec": round(sum(r["duration_sec"] for r in scene_records), 2),
        "diversity_score": div.get("diversity_score", 1.0),
        "diversity_acceptable": div.get("is_acceptable", True),
        "visual_type_distribution": div.get("type_distribution", {}),
        "shot_type_distribution": div.get("shot_distribution", {}),
        "qc_summary": {
            "total_scenes": total_scenes,
            "qc_passed_scenes": qc_passed_count,
            "qc_pass_rate": qc_pass_rate,
            "total_attempts": total_attempts,
        },
        "scenes": scene_records,
    }

    if run_dir:
        try:
            target_path = run_dir / "visual_telemetry.json"
            target_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info("Visual telemetry report saved to %s", target_path)
        except Exception as exc:
            log.warning("Could not save visual_telemetry.json: %s", exc)

    return report
