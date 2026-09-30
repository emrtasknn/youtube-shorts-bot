"""Multi-provider image generation for the Shorts pipeline.

Provider order:
1. Google Gemini native image generation
2. fal.ai FLUX.1 schnell
3. DeepAI text2img (if the account/API key is enabled)

The provider router is deliberately independent from the text AI router so a
Gemini text outage or image quota does not take the whole pipeline down.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from google import genai
from PIL import Image, ImageFilter

log = logging.getLogger("shorts-bot.image")

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920

DEFAULT_GEMINI_MODEL_ORDER = [
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
]
DEFAULT_PROVIDER_ORDER = ["gemini", "fal", "deepai"]
DEFAULT_FAL_MODEL = "fal-ai/flux/schnell"


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


def _provider_order() -> list[str]:
    raw = _env("IMAGE_PROVIDER_ORDER", ",".join(DEFAULT_PROVIDER_ORDER))
    result = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return result or list(DEFAULT_PROVIDER_ORDER)


def _model_order() -> list[str]:
    raw = _env(
        "GEMINI_IMAGE_MODEL_ORDER",
        ",".join(DEFAULT_GEMINI_MODEL_ORDER),
    )
    return [x.strip() for x in raw.split(",") if x.strip()] or list(DEFAULT_GEMINI_MODEL_ORDER)


def _configured(provider: str) -> bool:
    key_name = {
        "gemini": "GEMINI_API_KEY",
        "fal": "FAL_KEY",
        "deepai": "DEEPAI_API_KEY",
    }.get(provider)
    return bool(key_name and _env(key_name))


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


def _write_downloaded_image(url: str, filename: Path, *, timeout: float = 30.0) -> None:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "YouTubeShortsBot/2.1"},
    )
    if not response.ok:
        detail = response.text[:500].replace("\n", " ")
        error = ImageProviderError(
            f"image download HTTP {response.status_code}: {detail}",
            retryable=response.status_code == 429 or response.status_code >= 500,
        )
        raise error

    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_bytes(response.content)
    try:
        with Image.open(filename) as image:
            image.verify()
        _normalize_image(filename)
    except Exception as exc:
        filename.unlink(missing_ok=True)
        raise ImageProviderError(f"downloaded image is invalid: {exc}") from exc


def _generate_gemini(model: str, prompt: str, filename: Path) -> ImageResponse:
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
                for marker in (
                    "429", "500", "502", "503", "504",
                    "timeout", "timed out", "unavailable", "resource_exhausted",
                )
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


def _generate_fal(prompt: str, filename: Path) -> ImageResponse:
    model = _env("FAL_IMAGE_MODEL", DEFAULT_FAL_MODEL)
    key = _env("FAL_KEY")
    if not key:
        raise ImageProviderError("FAL_KEY is missing")

    timeout = max(30.0, float(_env("FAL_IMAGE_TIMEOUT_SECONDS", "120")))
    poll_seconds = max(1.0, float(_env("FAL_IMAGE_POLL_SECONDS", "2")))

    payload: dict[str, Any] = {
        "prompt": prompt,
        "image_size": {"width": 768, "height": 1365},
        "num_inference_steps": 4,
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": True,
    }

    headers = {
        "Authorization": f"Key {key}",
        "Content-Type": "application/json",
    }
    submit = requests.post(
        f"https://queue.fal.run/{model}",
        headers=headers,
        json=payload,
        timeout=30,
    )
    if not submit.ok:
        detail = submit.text[:800].replace("\n", " ")
        raise ImageProviderError(
            f"fal submit HTTP {submit.status_code}: {detail}",
            retryable=submit.status_code == 429 or submit.status_code >= 500,
        )

    body = submit.json()
    request_id = body.get("request_id")
    if not request_id:
        raise ImageProviderError(f"fal submit returned no request_id: {body}")

    status_url = f"https://queue.fal.run/{model}/requests/{request_id}/status"
    result_url = f"https://queue.fal.run/{model}/requests/{request_id}"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        status_response = requests.get(status_url, headers=headers, timeout=20)
        if not status_response.ok:
            detail = status_response.text[:500].replace("\n", " ")
            raise ImageProviderError(
                f"fal status HTTP {status_response.status_code}: {detail}",
                retryable=status_response.status_code == 429 or status_response.status_code >= 500,
            )

        status = status_response.json()
        state = str(status.get("status", "")).upper()
        if state == "COMPLETED":
            break
        if state == "FAILED":
            raise ImageProviderError(
                f"fal generation failed: {status.get('error') or status}"
            )
        time.sleep(poll_seconds)
    else:
        raise ImageProviderError("fal image generation timed out", retryable=True)

    result_response = requests.get(result_url, headers=headers, timeout=20)
    if not result_response.ok:
        detail = result_response.text[:500].replace("\n", " ")
        raise ImageProviderError(
            f"fal result HTTP {result_response.status_code}: {detail}",
            retryable=result_response.status_code == 429 or result_response.status_code >= 500,
        )

    result = result_response.json()
    images = result.get("images") or []
    image_url = images[0].get("url") if images and isinstance(images[0], dict) else None
    if not image_url:
        raise ImageProviderError(f"fal returned no image URL: {result}")

    _write_downloaded_image(image_url, filename)
    return ImageResponse(provider="fal", model=model, path=filename)


def _generate_deepai(prompt: str, filename: Path) -> ImageResponse:
    model = _env("DEEPAI_IMAGE_MODEL", "text2img")
    key = _env("DEEPAI_API_KEY")
    if not key:
        raise ImageProviderError("DEEPAI_API_KEY is missing")

    response = requests.post(
        f"https://api.deepai.org/api/{model}",
        headers={"api-key": key},
        data={
            "text": prompt,
            "width": "768",
            "height": "1365",
            "negative_prompt": "text, watermark, logo, caption, subtitles, UI, collage, distorted anatomy",
        },
        timeout=60,
    )
    if not response.ok:
        detail = response.text[:800].replace("\n", " ")
        raise ImageProviderError(
            f"DeepAI HTTP {response.status_code}: {detail}",
            retryable=response.status_code == 429 or response.status_code >= 500,
        )

    body = response.json()
    image_url = body.get("output_url")
    if not image_url:
        raise ImageProviderError(f"DeepAI returned no output_url: {body}")

    _write_downloaded_image(image_url, filename)
    return ImageResponse(provider="deepai", model=model, path=filename)


def generate(prompt: str, filename: Path, *, label: str = "AI image") -> ImageResponse:
    failures: list[str] = []

    for provider in _provider_order():
        if not _configured(provider):
            log.debug("%s: provider %s skipped; API key not configured", label, provider)
            continue

        if provider == "gemini":
            for model in _model_order():
                try:
                    result = _generate_gemini(model, prompt, filename)
                    if not result.path.exists() or result.path.stat().st_size < 5000:
                        raise ImageProviderError("image output file is missing or too small")
                    log.info("%s succeeded via %s/%s", label, result.provider, result.model)
                    return result
                except Exception as exc:
                    failures.append(f"{provider}/{model}: {type(exc).__name__}: {exc}")
                    log.warning("%s failed: %s", label, failures[-1])
            continue

        try:
            result = _generate_fal(prompt, filename) if provider == "fal" else _generate_deepai(prompt, filename)
            if not result.path.exists() or result.path.stat().st_size < 5000:
                raise ImageProviderError("image output file is missing or too small")
            log.info("%s succeeded via %s/%s", label, result.provider, result.model)
            return result
        except Exception as exc:
            failures.append(f"{provider}: {type(exc).__name__}: {exc}")
            log.warning("%s failed: %s", label, failures[-1])

    if not failures:
        raise RuntimeError(
            "No image provider is configured. Set GEMINI_API_KEY, FAL_KEY, or DEEPAI_API_KEY."
        )
    raise RuntimeError("All configured image providers failed: " + " | ".join(failures))


def provider_status() -> list[dict[str, str]]:
    return [
        {
            "provider": provider,
            "configured": str(_configured(provider)).lower(),
            "models": ",".join(_model_order()) if provider == "gemini" else (
                _env("FAL_IMAGE_MODEL", DEFAULT_FAL_MODEL) if provider == "fal"
                else _env("DEEPAI_IMAGE_MODEL", "text2img")
            ),
        }
        for provider in _provider_order()
    ]
