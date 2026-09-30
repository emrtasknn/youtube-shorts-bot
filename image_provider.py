"""Google Gemini native image generation for the Shorts pipeline.

Primary model: Nano Banana 2.
Fallback model: Nano Banana 2 Lite.

Both use the existing GEMINI_API_KEY. Provider-specific HTTP fallbacks were
removed so the production image path stays deterministic and easy to debug.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from google import genai
from PIL import Image, ImageFilter

log = logging.getLogger("shorts-bot.image")

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920

DEFAULT_MODEL_ORDER = [
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
]


class ImageProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class ImageResponse:
    provider: str
    model: str
    path: Path


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _model_order() -> list[str]:
    raw = _env("GEMINI_IMAGE_MODEL_ORDER", ",".join(DEFAULT_MODEL_ORDER))
    return [x.strip() for x in raw.split(",") if x.strip()] or list(DEFAULT_MODEL_ORDER)


def _normalize_image(filename: Path) -> None:
    with Image.open(filename) as source:
        img = source.convert("RGB")
        w, h = img.size
        target_ratio = VIDEO_WIDTH / VIDEO_HEIGHT

        if w / h > target_ratio:
            crop_w = int(h * target_ratio)
            left = max((w - crop_w) // 2, 0)
            img = img.crop((left, 0, left + crop_w, h))
        else:
            crop_h = int(w / target_ratio)
            top = max((h - crop_h) // 2, 0)
            img = img.crop((0, top, w, top + crop_h))

        img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=110, threshold=3))
        img.save(filename, format="JPEG", quality=95)


def _generate(model: str, prompt: str, filename: Path) -> ImageResponse:
    client = genai.Client(api_key=_env("GEMINI_API_KEY"))

    try:
        interaction = client.interactions.create(
            model=model,
            input=prompt,
            response_format={
                "type": "image",
                "mime_type": "image/jpeg",
                "aspect_ratio": "9:16",
                "image_size": "2K" if model == "gemini-3.1-flash-image" else "1K",
            },
        )
    except Exception as exc:
        lower = str(exc).lower()
        raise ImageProviderError(
            f"{model} request failed: {exc}",
            retryable=any(
                marker in lower
                for marker in ("429", "500", "502", "503", "504", "timeout", "timed out", "unavailable", "resource_exhausted")
            ),
        ) from exc

    output_image = getattr(interaction, "output_image", None)
    data = getattr(output_image, "data", None) if output_image is not None else None
    if not data:
        raise ImageProviderError(f"{model} returned no image data")

    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_bytes(base64.b64decode(data))
        with Image.open(filename) as image:
            image.verify()
        _normalize_image(filename)
    except Exception as exc:
        filename.unlink(missing_ok=True)
        raise ImageProviderError(f"{model} returned invalid image data: {exc}") from exc

    return ImageResponse(
        provider="nano_banana_2" if model == "gemini-3.1-flash-image" else "nano_banana_2_lite",
        model=model,
        path=filename,
    )


def generate(prompt: str, filename: Path, *, label: str = "AI image") -> ImageResponse:
    if not _env("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is missing")

    failures: list[str] = []
    for model in _model_order():
        try:
            result = _generate(model, prompt, filename)
            if not result.path.exists() or result.path.stat().st_size < 5000:
                raise ImageProviderError("image output file is missing or too small")
            log.info("%s succeeded via %s/%s", label, result.provider, result.model)
            return result
        except Exception as exc:
            failures.append(f"{model}: {type(exc).__name__}: {exc}")
            log.warning("%s failed: %s", label, failures[-1])

    raise RuntimeError(
        "All Google Nano Banana image providers failed: " + " | ".join(failures)
    )


def provider_status() -> list[dict[str, str]]:
    configured = bool(_env("GEMINI_API_KEY"))
    return [
        {"provider": "nano_banana_2", "model": model, "configured": str(configured).lower()}
        for model in _model_order()
    ]
