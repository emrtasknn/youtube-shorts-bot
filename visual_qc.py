"""
visual_qc.py -- Visual Engine V2: Image Quality Control (Phase G)

Validates AI-generated and downloaded images before they enter the
enhancement and composition pipeline.

Design principles:
- Pure validation layer: no prompt calls, no LLM, no side effects
- Returns rich metadata for observability rather than raising on soft failures
- Hard failures (file missing, corrupt PIL data) raise ImageQCError
- Retry wrapper: download_with_qc() calls any download_fn and retries on QC fail
- Feature flag: VISUAL_QC_ENABLED (env, default true)
- Retry count: MAX_VISUAL_RETRIES (env, default 2)

Public API:
    validate_image(image_path) -> dict
    download_with_qc(prompt, image_path, download_fn, max_retries=None) -> dict
    ImageQCError
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

log = logging.getLogger("shorts-bot.visual_qc")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VISUAL_QC_ENABLED: bool = os.getenv("VISUAL_QC_ENABLED", "true").lower() in (
    "1", "true", "yes"
)
MAX_VISUAL_RETRIES: int = max(0, int(os.getenv("MAX_VISUAL_RETRIES", "2")))

# Minimum acceptable file size for a generated image.
MIN_FILE_SIZE_BYTES: int = 50_000          # 50 KB

# Minimum number of unique colours in a 64x64 sample of the image.
# Values below this indicate a blank/solid-colour render failure.
MIN_UNIQUE_COLORS: int = 10

# Minimum image dimension (width or height) in pixels.
MIN_DIMENSION_PX: int = 100


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------
class ImageQCError(RuntimeError):
    """Raised when a critical image validation failure cannot be recovered from."""


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

def validate_image(image_path: Path) -> dict:
    """Validate a downloaded or generated image file.

    Checks (in order):
    1. File exists on disk
    2. File size >= MIN_FILE_SIZE_BYTES (catches empty / truncated downloads)
    3. PIL can open and decode the file (catches corrupt / wrong-format files)
    4. Image dimensions >= MIN_DIMENSION_PX (catches 1x1 error thumbnails)
    5. Colour diversity >= MIN_UNIQUE_COLORS in a 64x64 sample
       (catches solid-colour / blank render failures from AI providers)

    Returns:
        dict with:
            valid           bool   -- True if all checks pass
            reason          str    -- "ok" or short failure description
            file_size_bytes int    -- raw file size (0 if missing)
            width           int    -- image width (0 if unreadable)
            height          int    -- image height (0 if unreadable)
            unique_colors   int    -- colour diversity sample (-1 if unreadable)

    Never raises -- always returns a dict so the caller can decide how
    to handle degraded images without crashing the entire pipeline.
    """
    result: dict = {
        "valid": False,
        "reason": "unknown",
        "file_size_bytes": 0,
        "width": 0,
        "height": 0,
        "unique_colors": -1,
    }

    # ── 1. File existence ────────────────────────────────────────────────────
    if not image_path.exists():
        result["reason"] = "file_not_found"
        return result

    # ── 2. File size ─────────────────────────────────────────────────────────
    file_size = image_path.stat().st_size
    result["file_size_bytes"] = file_size
    if file_size < MIN_FILE_SIZE_BYTES:
        result["reason"] = f"file_too_small:{file_size}b<{MIN_FILE_SIZE_BYTES}b"
        return result

    # ── 3. PIL decodable ─────────────────────────────────────────────────────
    try:
        from PIL import Image  # imported lazily; Pillow is always available
        with Image.open(image_path) as img:
            img.verify()
    except Exception as exc:
        result["reason"] = f"pil_verify_failed:{exc}"
        return result

    # ── 4. Dimensions + 5. Colour diversity ──────────────────────────────────
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            result["width"] = w
            result["height"] = h

            if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
                result["reason"] = f"dimensions_too_small:{w}x{h}"
                return result

            sample = rgb.resize((64, 64), Image.Resampling.LANCZOS)
            unique_colors = len(set(sample.getdata()))
            result["unique_colors"] = unique_colors

            if unique_colors < MIN_UNIQUE_COLORS:
                result["reason"] = f"blank_or_monochrome:{unique_colors}_unique_colors"
                return result

    except Exception as exc:
        result["reason"] = f"image_analysis_error:{exc}"
        return result

    result["valid"] = True
    result["reason"] = "ok"
    return result


# ---------------------------------------------------------------------------
# Download + QC wrapper
# ---------------------------------------------------------------------------

def download_with_qc(
    prompt: str,
    image_path: Path,
    download_fn: Callable[[str, Path], None],
    max_retries: int | None = None,
) -> dict:
    """Download an image and run QC, retrying on failure.

    Args:
        prompt:       The image generation prompt string.
        image_path:   Target path where the image should be saved.
        download_fn:  Callable(prompt, path) -> None that performs the actual
                      download/generation. Called again on each retry.
        max_retries:  Number of additional attempts after the first failure.
                      Defaults to MAX_VISUAL_RETRIES (env configurable).

    Returns:
        dict with:
            qc_enabled      bool   -- whether QC was active for this call
            valid           bool   -- True if final image passed QC
            attempts        int    -- how many download attempts were made
            + all fields from validate_image() for the final attempt

    Never raises -- soft failures are logged and returned for the caller
    to handle. The pipeline should log a warning and proceed with the
    potentially degraded image rather than aborting the entire video.
    """
    retries = MAX_VISUAL_RETRIES if max_retries is None else max(0, max_retries)
    total_attempts = retries + 1  # initial attempt + N retries

    if not VISUAL_QC_ENABLED:
        download_fn(prompt, image_path)
        qc = validate_image(image_path)  # Still capture metrics for observability
        return {
            "qc_enabled": False,
            "valid": qc.get("valid", True),
            "attempts": 1,
            **qc,
        }

    last_qc: dict = {}

    for attempt in range(1, total_attempts + 1):
        try:
            download_fn(prompt, image_path)
        except Exception as exc:
            log.warning(
                "Visual QC: download_fn raised on attempt %d/%d: %s",
                attempt, total_attempts, exc,
            )
            last_qc = {
                "valid": False,
                "reason": f"download_fn_raised:{exc}",
                "file_size_bytes": 0,
                "width": 0,
                "height": 0,
                "unique_colors": -1,
            }
            if attempt < total_attempts:
                continue
            break

        qc = validate_image(image_path)
        last_qc = qc

        if qc["valid"]:
            log.debug(
                "Visual QC passed on attempt %d/%d: %s (%dx%d, %d colours, %d bytes)",
                attempt, total_attempts,
                image_path.name,
                qc["width"], qc["height"],
                qc["unique_colors"], qc["file_size_bytes"],
            )
            return {
                "qc_enabled": True,
                "valid": True,
                "attempts": attempt,
                **qc,
            }

        log.warning(
            "Visual QC failed (attempt %d/%d) for %s: %s",
            attempt, total_attempts, image_path.name, qc["reason"],
        )

    log.error(
        "Visual QC: all %d attempt(s) failed for %s. Reason: %s. "
        "Proceeding with potentially degraded image.",
        total_attempts, image_path.name, last_qc.get("reason", "unknown"),
    )

    return {
        "qc_enabled": True,
        "valid": False,
        "attempts": total_attempts,
        **last_qc,
    }
