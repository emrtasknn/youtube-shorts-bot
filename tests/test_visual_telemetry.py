"""Tests for visual_telemetry.py -- Visual Engine V2 Telemetry & Storyboard Export (Phase 19)."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import visual_telemetry


def test_create_scene_telemetry_fields():
    """create_scene_telemetry includes all expected fields."""
    scene = {
        "visual_type": "historical_photo",
        "shot_type": "medium_shot",
        "primary_subject": "Test Subject",
        "image_prompt": "Prompt text",
        "image_qc": {
            "attempts": 2,
            "valid": True,
            "reason": "",
            "width": 1080,
            "height": 1920,
            "unique_colors": 3000,
            "file_size_bytes": 120000,
        },
        "image_enhancement": {
            "enabled": True,
            "profile": "vintage_archive",
        },
    }

    record = visual_telemetry.create_scene_telemetry(
        scene=scene,
        scene_index=1,
        start_time=0.0,
        end_time=4.5,
        motion_type="slow_push_in",
        provider="pollinations",
    )

    assert record["scene_index"] == 1
    assert record["duration_sec"] == 4.5
    assert record["visual_type"] == "historical_photo"
    assert record["shot_type"] == "medium_shot"
    assert record["primary_subject"] == "Test Subject"
    assert record["generation_attempts"] == 2
    assert record["qc_valid"] is True
    assert record["qc_unique_colors"] == 3000
    assert record["enhancement_profile"] == "vintage_archive"
    assert record["motion_type"] == "slow_push_in"


def test_build_video_telemetry_report_saves_file(tmp_path):
    """build_video_telemetry_report aggregates data and writes visual_telemetry.json."""
    scenes = [
        {
            "start_time": 0.0,
            "end_time": 3.0,
            "visual_type": "location",
            "shot_type": "wide_shot",
            "telemetry": {
                "scene_index": 1,
                "duration_sec": 3.0,
                "generation_attempts": 1,
                "qc_valid": True,
            },
        },
        {
            "start_time": 3.0,
            "end_time": 7.0,
            "visual_type": "person",
            "shot_type": "close_up",
            "telemetry": {
                "scene_index": 2,
                "duration_sec": 4.0,
                "generation_attempts": 2,
                "qc_valid": True,
            },
        },
    ]

    candidate = {"id": "test-vid-101", "topic": "Test Topic", "title": "Test Title"}
    diversity_report = {
        "diversity_score": 0.90,
        "is_acceptable": True,
        "type_distribution": {"location": 1, "person": 1},
        "shot_distribution": {"wide_shot": 1, "close_up": 1},
    }

    report = visual_telemetry.build_video_telemetry_report(
        scenes=scenes,
        candidate=candidate,
        diversity_report=diversity_report,
        run_dir=tmp_path,
    )

    assert report["video_id"] == "test-vid-101"
    assert report["total_scenes"] == 2
    assert report["total_duration_sec"] == 7.0
    assert report["diversity_score"] == 0.90
    assert report["qc_summary"]["qc_passed_scenes"] == 2
    assert report["qc_summary"]["total_attempts"] == 3

    saved_file = tmp_path / "visual_telemetry.json"
    assert saved_file.exists()
    saved_data = json.loads(saved_file.read_text(encoding="utf-8"))
    assert saved_data["video_id"] == "test-vid-101"
