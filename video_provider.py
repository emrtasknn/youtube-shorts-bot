"""Centralized Google video generation routing for the Shorts pipeline.

The provider layer supports the current Google generative-video models:
- Gemini Omni Flash (gemini-omni-1.1-flash)
- Veo 3.1 (veo-3.1-generate-preview)
- Veo 3.1 Fast (veo-3.1-fast-generate-preview)
- Veo 3.1 Lite (veo-3.1-lite-generate-preview)

The public API is intentionally small so pipeline.py can later switch between
static-image animation and video-first generation without knowing provider
specific SDK details.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("shorts-bot.video")


DEFAULT_PROVIDER_ORDER = ["omni", "veo31_fast", "veo31_lite", "veo31"]


class VideoProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        cooldown_seconds: float | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.cooldown_seconds = cooldown_seconds


@dataclass
class VideoResponse:
    provider: str
    model: str
    path: Path
    duration_seconds: float | None = None


_PROVIDER_COOLDOWN_UNTIL: dict[str, float] = {}


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _provider_order() -> list[str]:
    raw = _env("VIDEO_PROVIDER_ORDER", ",".join(DEFAULT_PROVIDER_ORDER))
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def _configured(provider: str) -> bool:
    # All current providers use the same Gemini API key.
    return bool(_env("GEMINI_API_KEY"))


def _cooled(provider: str) -> bool:
    return time.monotonic() < _PROVIDER_COOLDOWN_UNTIL.get(provider, 0.0)


def _cooldown(provider: str, seconds: float) -> None:
    _PROVIDER_COOLDOWN_UNTIL[provider] = time.monotonic() + seconds


def _save_bytes(content: bytes, filename: Path) -> Path:
    if not content or len(content) < 50_000:
        raise VideoProviderError("video response is empty or unexpectedly small")
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_bytes(content)
    return filename


def _read_image_input(image_path: Path) -> tuple[str, str]:
    if not image_path.exists():
        raise VideoProviderError(f"reference image does not exist: {image_path}")
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    if not data:
        raise VideoProviderError(f"reference image is empty: {image_path}")
    return data, mime


def _omni(prompt: str, filename: Path, image_path: Path | None = None) -> VideoResponse:
    from google import genai

    client = genai.Client(api_key=_env("GEMINI_API_KEY"))
    model = _env("GEMINI_OMNI_MODEL", "gemini-omni-1.1-flash")

    if image_path:
        image_data, mime_type = _read_image_input(image_path)
        inputs = [
            {
                "type": "image",
                "data": image_data,
                "mime_type": mime_type,
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
        generation_config = {
            "video_config": {
                "task": "image_to_video",
            }
        }
    else:
        inputs = prompt
        generation_config = None

    response_format = {
        "type": "video",
        "aspect_ratio": _env("VIDEO_ASPECT_RATIO", "9:16"),
        "resolution": _env("GEMINI_OMNI_RESOLUTION", "720p"),
        "delivery": "uri",
    }

    try:
        kwargs = {
            "model": model,
            "input": inputs,
            "response_format": response_format,
        }
        if generation_config:
            kwargs["generation_config"] = generation_config
        interaction = client.interactions.create(**kwargs)
    except Exception as exc:
        text = str(exc)
        lower = text.lower()
        raise VideoProviderError(
            f"{model} request failed: {text}",
            retryable=any(
                marker in lower
                for marker in ("429", "500", "502", "503", "504", "timeout", "timed out", "unavailable", "resource_exhausted")
            ),
            cooldown_seconds=3600 if any(
                marker in lower for marker in ("quota", "402", "403", "resource_exhausted")
            ) else None,
        ) from exc

    output_video = getattr(interaction, "output_video", None)
    if output_video is None:
        raise VideoProviderError(f"{model} returned no video output")

    data = getattr(output_video, "data", None)
    if data:
        _save_bytes(base64.b64decode(data), filename)
    else:
        uri = getattr(output_video, "uri", None)
        if not uri:
            raise VideoProviderError(f"{model} returned neither inline video data nor a URI")
        file_name = str(uri).rstrip("/").split("/")[-1]
        deadline = time.time() + float(_env("GEMINI_VIDEO_URI_TIMEOUT_SECONDS", "600"))
        while time.time() < deadline:
            info = client.files.get(name=f"files/{file_name}")
            state = getattr(getattr(info, "state", None), "name", str(getattr(info, "state", "")))
            if str(state).upper() == "ACTIVE":
                break
            if str(state).upper() == "FAILED":
                raise VideoProviderError(f"{model} video file processing failed")
            time.sleep(5)
        else:
            raise VideoProviderError(f"{model} video URI did not become ACTIVE in time", retryable=True)
        client.files.download(file=uri, destination=str(filename))

    return VideoResponse("omni", model, filename)


def _veo(prompt: str, filename: Path, *, model: str, image_path: Path | None = None) -> VideoResponse:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_env("GEMINI_API_KEY"))
    duration = _env("GEMINI_VEO_DURATION_SECONDS", "8")
    resolution = _env("GEMINI_VEO_RESOLUTION", "720p")
    aspect_ratio = _env("VIDEO_ASPECT_RATIO", "9:16")

    config_kwargs = {
        "number_of_videos": 1,
        "aspect_ratio": aspect_ratio,
        "duration_seconds": duration,
        "resolution": resolution,
    }

    image = None
    if image_path:
        image_data = image_path.read_bytes()
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        image = types.Image(image_bytes=image_data, mime_type=mime_type)

    try:
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=image,
            config=types.GenerateVideosConfig(**config_kwargs),
        )
    except Exception as exc:
        text = str(exc)
        lower = text.lower()
        raise VideoProviderError(
            f"{model} request failed: {text}",
            retryable=any(
                marker in lower
                for marker in ("429", "500", "502", "503", "504", "timeout", "timed out", "unavailable", "resource_exhausted")
            ),
            cooldown_seconds=3600 if any(
                marker in lower for marker in ("quota", "402", "403", "resource_exhausted")
            ) else None,
        ) from exc

    deadline = time.time() + float(_env("GEMINI_VIDEO_OPERATION_TIMEOUT_SECONDS", "900"))
    while not getattr(operation, "done", False):
        if time.time() >= deadline:
            raise VideoProviderError(f"{model} generation timed out", retryable=True)
        time.sleep(float(_env("GEMINI_VIDEO_POLL_SECONDS", "10")))
        operation = client.operations.get(operation)

    error = getattr(operation, "error", None)
    if error:
        raise VideoProviderError(f"{model} generation failed: {error}")

    response = getattr(operation, "response", None)
    generated = getattr(response, "generated_videos", None) if response else None
    generated = generated or []
    if not generated:
        raise VideoProviderError(f"{model} returned no generated video")

    video = getattr(generated[0], "video", None)
    if video is None:
        raise VideoProviderError(f"{model} returned an empty video object")

    video_bytes = getattr(video, "video_bytes", None)
    if video_bytes:
        _save_bytes(video_bytes, filename)
    else:
        client.files.download(file=video, destination=str(filename))

    provider_name = "veo31"\n    if "fast" in model:\n        provider_name = "veo31_fast"\n    elif "lite" in model:\n        provider_name = "veo31_lite"\n    return VideoResponse(provider_name, model, filename)


_HANDLERS = {
    "omni": _omni,
    "veo31": lambda prompt, path, image_path=None: _veo(
        prompt, path, model=_env("GEMINI_VEO31_MODEL", "veo-3.1-generate-preview"), image_path=image_path
    ),
    "veo31_fast": lambda prompt, path, image_path=None: _veo(
        prompt, path, model=_env("GEMINI_VEO31_FAST_MODEL", "veo-3.1-fast-generate-preview"), image_path=image_path
    ),
    "veo31_lite": lambda prompt, path, image_path=None: _veo(
        prompt, path, model=_env("GEMINI_VEO31_LITE_MODEL", "veo-3.1-lite-generate-preview"), image_path=image_path
    ),
}


def generate(
    prompt: str,
    filename: Path,
    *,
    image_path: Path | None = None,
    label: str = "AI video",
) -> VideoResponse:
    """Generate a video using the configured Google provider chain."""
    failures: list[str] = []

    for provider in _provider_order():
        if provider not in _HANDLERS:
            log.warning("%s: unknown video provider %s; skipping", label, provider)
            continue
        if not _configured(provider):
            log.info("%s: provider %s skipped; GEMINI_API_KEY missing", label, provider)
            continue
        if _cooled(provider):
            log.info("%s: provider %s skipped; cooldown active", label, provider)
            continue

        started = time.monotonic()
        try:
            result = _HANDLERS[provider](prompt, filename, image_path=image_path)
            if not result.path.exists() or result.path.stat().st_size < 50_000:
                raise VideoProviderError("provider reported success but output video is missing/invalid")
            log.info(
                "%s succeeded via %s/%s in %.1fs",
                label,
                result.provider,
                result.model,
                time.monotonic() - started,
            )
            return result
        except Exception as exc:
            failures.append(f"{provider}: {type(exc).__name__}: {exc}")
            log.warning("%s failed via %s: %s", label, provider, exc)
            cooldown = getattr(exc, "cooldown_seconds", None)
            if cooldown:
                _cooldown(provider, cooldown)

    if not failures:
        raise RuntimeError("No Google video provider is configured. Set GEMINI_API_KEY.")

    raise RuntimeError("All Google video providers failed: " + " | ".join(failures[-8:]))


def provider_status() -> list[dict]:
    now = time.monotonic()
    return [
        {
            "provider": provider,
            "configured": _configured(provider),
            "cooldown_remaining": round(max(0.0, _PROVIDER_COOLDOWN_UNTIL.get(provider, 0.0) - now), 1),
        }
        for provider in _provider_order()
    ]
