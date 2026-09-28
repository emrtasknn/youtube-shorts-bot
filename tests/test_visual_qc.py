"""Tests for visual_qc.py -- Visual Engine V2 Image Quality Control."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import visual_qc


# ---------------------------------------------------------------------------
# Helpers to create test images
# ---------------------------------------------------------------------------

def _write_tiny_file(path: Path, size_bytes: int = 10) -> Path:
    """Write a small dummy file (not a valid image)."""
    path.write_bytes(b"x" * size_bytes)
    return path


def _write_valid_jpeg(path: Path, width: int = 400, height: int = 700, mode: str = "RGB") -> Path:
    """Write a real JPEG using Pillow."""
    from PIL import Image as PILImage
    import random

    img = PILImage.new(mode, (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            if mode == "RGB":
                pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                pixels[x, y] = random.randint(0, 255)
    img.save(path, format="JPEG", quality=85)
    return path


def _write_blank_jpeg(path: Path, width: int = 400, height: int = 700, color: tuple = (128, 128, 128)) -> Path:
    """Write a solid-colour JPEG (simulates a blank AI render failure)."""
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (width, height), color)
    img.save(path, format="JPEG")
    return path


# ---------------------------------------------------------------------------
# validate_image: file-level checks
# ---------------------------------------------------------------------------

def test_validate_missing_file_returns_invalid():
    result = visual_qc.validate_image(Path("/nonexistent/path/scene_01.jpg"))
    assert result["valid"] is False
    assert "not_found" in result["reason"]


def test_validate_too_small_file_returns_invalid(tmp_path):
    small = tmp_path / "small.jpg"
    _write_tiny_file(small, size_bytes=1000)
    result = visual_qc.validate_image(small)
    assert result["valid"] is False
    assert "too_small" in result["reason"]


def test_validate_corrupt_file_returns_invalid(tmp_path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"\xff" * 60_000)  # 60 KB of garbage
    result = visual_qc.validate_image(corrupt)
    assert result["valid"] is False


def test_validate_valid_image_passes(tmp_path):
    path = tmp_path / "scene_01.jpg"
    _write_valid_jpeg(path, width=400, height=700)
    result = visual_qc.validate_image(path)
    assert result["valid"] is True
    assert result["reason"] == "ok"
    assert result["width"] == 400
    assert result["height"] == 700
    assert result["unique_colors"] >= visual_qc.MIN_UNIQUE_COLORS
    assert result["file_size_bytes"] > visual_qc.MIN_FILE_SIZE_BYTES


def test_validate_blank_image_returns_invalid(tmp_path):
    path = tmp_path / "blank.jpg"
    _write_blank_jpeg(path, color=(200, 200, 200))
    result = visual_qc.validate_image(path)
    assert result["valid"] is False
    assert "monochrome" in result["reason"] or "unique_colors" in result["reason"] or "too_small" in result["reason"]


def test_validate_tiny_dimensions_returns_invalid(tmp_path):
    path = tmp_path / "tiny.jpg"
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (50, 50), (100, 150, 200))
    img.save(path, format="JPEG")
    # Make it big enough in bytes by padding, but it will fail dimension check
    result = visual_qc.validate_image(path)
    # Either too small in bytes OR dimensions too small
    assert result["valid"] is False


def test_validate_returns_dict_always(tmp_path):
    path = tmp_path / "none.jpg"
    result = visual_qc.validate_image(path)
    for key in ("valid", "reason", "file_size_bytes", "width", "height", "unique_colors"):
        assert key in result


# ---------------------------------------------------------------------------
# download_with_qc: download function integration
# ---------------------------------------------------------------------------

def _good_download_fn(path: Path):
    """Simulate a successful AI image download."""
    from PIL import Image as PILImage
    import random
    img = PILImage.new("RGB", (400, 700))
    px = img.load()
    for y in range(700):
        for x in range(400):
            px[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img.save(path, format="JPEG", quality=85)


def _bad_download_fn(path: Path):
    """Simulate a blank image failure (provider returns solid colour)."""
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (400, 700), (128, 128, 128))
    img.save(path, format="JPEG")


def _raising_download_fn(path: Path):
    """Simulate a provider that raises an exception."""
    raise RuntimeError("Provider timeout")


# Wrappers to match signature (prompt, path)
def _make_fn(inner):
    def wrapped(prompt: str, path: Path):
        inner(path)
    return wrapped


def test_download_with_qc_success(tmp_path):
    path = tmp_path / "scene.jpg"
    result = visual_qc.download_with_qc(
        prompt="test prompt",
        image_path=path,
        download_fn=_make_fn(_good_download_fn),
        max_retries=1,
    )
    assert result["valid"] is True
    assert result["attempts"] == 1
    assert result["qc_enabled"] is True


def test_download_with_qc_retries_on_failure(tmp_path):
    path = tmp_path / "scene.jpg"
    call_count = {"n": 0}

    def flaky(prompt: str, p: Path):
        call_count["n"] += 1
        if call_count["n"] < 2:
            _bad_download_fn(p)  # First attempt: blank image
        else:
            _good_download_fn(p)  # Second attempt: good image

    result = visual_qc.download_with_qc(
        prompt="test",
        image_path=path,
        download_fn=flaky,
        max_retries=2,
    )
    assert result["valid"] is True
    assert result["attempts"] == 2
    assert call_count["n"] == 2


def test_download_with_qc_exhausts_retries(tmp_path):
    path = tmp_path / "scene.jpg"
    result = visual_qc.download_with_qc(
        prompt="test",
        image_path=path,
        download_fn=_make_fn(_bad_download_fn),
        max_retries=2,
    )
    assert result["valid"] is False
    assert result["attempts"] == 3  # initial + 2 retries
    assert result["qc_enabled"] is True


def test_download_with_qc_handles_raising_fn(tmp_path):
    path = tmp_path / "scene.jpg"
    result = visual_qc.download_with_qc(
        prompt="test",
        image_path=path,
        download_fn=_make_fn(_raising_download_fn),
        max_retries=1,
    )
    assert result["valid"] is False
    assert "download_fn_raised" in result.get("reason", "")


def test_download_with_qc_disabled_still_returns_dict(monkeypatch, tmp_path):
    monkeypatch.setattr(visual_qc, "VISUAL_QC_ENABLED", False)
    path = tmp_path / "scene.jpg"
    result = visual_qc.download_with_qc(
        prompt="test",
        image_path=path,
        download_fn=_make_fn(_good_download_fn),
        max_retries=1,
    )
    assert result["qc_enabled"] is False
    assert "valid" in result
    assert "attempts" in result


def test_download_with_qc_zero_retries_tries_once(tmp_path):
    path = tmp_path / "scene.jpg"
    call_count = {"n": 0}

    def counting_bad(prompt: str, p: Path):
        call_count["n"] += 1
        _bad_download_fn(p)

    result = visual_qc.download_with_qc(
        prompt="test",
        image_path=path,
        download_fn=counting_bad,
        max_retries=0,
    )
    assert call_count["n"] == 1
    assert result["attempts"] == 1
    assert result["valid"] is False
