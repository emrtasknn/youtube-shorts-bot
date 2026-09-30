"""Centralized image-provider routing for the Shorts pipeline.

Providers are optional and configured through environment variables. The router
keeps provider-specific HTTP/API details out of pipeline.py, applies bounded
retries, cools down providers after hard failures, and only returns after a
real image file has been written and validated.
"""

from __future__ import annotations

import base64
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests
from PIL import Image, ImageFilter

log = logging.getLogger("shorts-bot.image")

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920

DEFAULT_PROVIDER_ORDER = [
    "imagen",
    "cloudflare",
    "fal",
    "together",
    "deepai",
    "pollinations_turbo",
    "pollinations_flux",
    "huggingface",
]

_PROVIDER_COOLDOWN_UNTIL: dict[str, float] = {}


class ImageProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, cooldown_seconds: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.cooldown_seconds = cooldown_seconds


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
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def _configured(provider: str) -> bool:
    if provider == "cloudflare":
        return bool(_env("CLOUDFLARE_ACCOUNT_ID") and _env("CLOUDFLARE_API_TOKEN"))
    required = {
        "imagen": "GEMINI_API_KEY",
        "fal": "FAL_KEY",
        "together": "TOGETHER_API_KEY",
        "deepai": "DEEPAI_API_KEY",
        "pollinations_turbo": "",
        "pollinations_flux": "",
        "huggingface": "HF_TOKEN",
    }
    key = required.get(provider)
    return True if key == "" else bool(_env(key))


def _cooldown(provider: str, seconds: float) -> None:
    _PROVIDER_COOLDOWN_UNTIL[provider] = time.monotonic() + seconds


def _cooled(provider: str) -> bool:
    return time.monotonic() < _PROVIDER_COOLDOWN_UNTIL.get(provider, 0.0)


def _retryable_status(status: int) -> bool:
    return status in {408, 409, 425, 429} or status >= 500


def _save_bytes(content: bytes, filename: Path) -> Path:
    if not content or len(content) < 5000:
        raise ImageProviderError("image response is empty or unexpectedly small")
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_bytes(content)
    try:
        with Image.open(filename) as img:
            img.verify()
    except Exception as exc:
        filename.unlink(missing_ok=True)
        raise ImageProviderError(f"provider returned invalid image data: {exc}") from exc
    return filename


def _download_url(url: str, filename: Path, timeout: float = 45) -> Path:
    if not url:
        raise ImageProviderError("provider returned an empty image URL")
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "YouTubeShortsBot/3.0"},
    )
    if not response.ok:
        err = ImageProviderError(
            f"image download HTTP {response.status_code}: {response.text[:300]}",
            retryable=_retryable_status(response.status_code),
        )
        raise err
    return _save_bytes(response.content, filename)


def _normalize_image(filename: Path) -> None:
    """Convert provider output to the exact 1080x1920 JPEG used by MoviePy."""
    with Image.open(filename) as img:
        img = img.convert("RGB")
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


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: dict | None = None,
    data: dict | None = None,
    timeout: float = 90,
) -> dict:
    try:
        response = requests.request(
            method,
            url,
            headers=headers or {},
            json=json_body,
            data=data,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise ImageProviderError(f"request failed: {exc}", retryable=True) from exc

    if not response.ok:
        status = response.status_code
        cooldown = 3600 if status in {401, 402, 403} else None
        raise ImageProviderError(
            f"HTTP {status}: {response.text[:500].replace(chr(10), ' ')}",
            retryable=_retryable_status(status),
            cooldown_seconds=cooldown,
        )
    try:
        return response.json()
    except ValueError as exc:
        raise ImageProviderError("provider returned non-JSON response") from exc


def _imagen(prompt: str, filename: Path) -> ImageResponse:
    from google import genai

    client = genai.Client(api_key=_env("GEMINI_API_KEY"))
    try:
        result = client.models.generate_images(
            model=_env("GEMINI_IMAGE_MODEL", "imagen-3.0-generate-002"),
            prompt=prompt,
            config={
                "number_of_images": 1,
                "output_mime_type": "image/jpeg",
                "aspect_ratio": "9:16",
            },
        )
    except Exception as exc:
        text = str(exc)
        retryable = any(x in text.lower() for x in ("429", "503", "unavailable", "resource_exhausted", "timeout"))
        raise ImageProviderError(f"Imagen request failed: {text}", retryable=retryable) from exc

    images = getattr(result, "generated_images", None) or []
    if not images:
        raise ImageProviderError("Imagen returned no generated images")
    image_bytes = images[0].image.image_bytes
    _save_bytes(image_bytes, filename)
    _normalize_image(filename)
    return ImageResponse("imagen", _env("GEMINI_IMAGE_MODEL", "imagen-3.0-generate-002"), filename)


def _cloudflare(prompt: str, filename: Path) -> ImageResponse:
    account_id = _env("CLOUDFLARE_ACCOUNT_ID")
    token = _env("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        raise ImageProviderError("Cloudflare credentials are not configured")

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    payload = {"prompt": prompt, "steps": int(_env("CLOUDFLARE_IMAGE_STEPS", "4"))}
    try:
        data = _request_json("POST", url, headers={"Authorization": f"Bearer {token}"}, json_body=payload)
        encoded = data.get("result", {}).get("image") or data.get("image")
        if not encoded:
            raise ImageProviderError("Cloudflare returned no image payload")
        _save_bytes(base64.b64decode(encoded), filename)
        _normalize_image(filename)
        return ImageResponse("cloudflare", "@cf/black-forest-labs/flux-1-schnell", filename)
    except ImageProviderError:
        raise
    except Exception as exc:
        raise ImageProviderError(f"Cloudflare image decode failed: {exc}") from exc


def _fal(prompt: str, filename: Path) -> ImageResponse:
    key = _env("FAL_KEY")
    if not key:
        raise ImageProviderError("FAL_KEY is not configured")
    payload = {
        "prompt": prompt,
        "image_size": "portrait_16_9",
        "num_inference_steps": int(_env("FAL_IMAGE_STEPS", "4")),
        "num_images": 1,
        "output_format": "jpeg",
        "sync_mode": True,
    }
    data = _request_json(
        "POST",
        "https://fal.run/fal-ai/flux/schnell",
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
        json_body=payload,
        timeout=float(_env("FAL_TIMEOUT_SECONDS", "120")),
    )
    images = data.get("images") or []
    url = images[0].get("url") if images else ""
    _download_url(url, filename, timeout=60)
    _normalize_image(filename)
    return ImageResponse("fal", "fal-ai/flux/schnell", filename)


def _together(prompt: str, filename: Path) -> ImageResponse:
    key = _env("TOGETHER_API_KEY")
    if not key:
        raise ImageProviderError("TOGETHER_API_KEY is not configured")
    model = _env("TOGETHER_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
    data = _request_json(
        "POST",
        "https://api.together.xyz/v1/images/generations",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json_body={
            "model": model,
            "prompt": prompt,
            "width": 1024,
            "height": 1536,
            "n": 1,
            "response_format": "url",
        },
    )
    items = data.get("data") or []
    url = items[0].get("url") if items else ""
    _download_url(url, filename)
    _normalize_image(filename)
    return ImageResponse("together", model, filename)


def _deepai(prompt: str, filename: Path) -> ImageResponse:
    key = _env("DEEPAI_API_KEY")
    if not key:
        raise ImageProviderError("DEEPAI_API_KEY is not configured")
    try:
        response = requests.post(
            "https://api.deepai.org/api/text2img",
            headers={"api-key": key},
            data={
                "text": prompt,
                "width": "832",
                "height": "1216",
                "image_generator_version": _env("DEEPAI_IMAGE_VERSION", "standard"),
                "negative_prompt": "text, watermark, logo, blurry, low quality, distorted anatomy",
            },
            timeout=90,
        )
    except requests.RequestException as exc:
        raise ImageProviderError(f"DeepAI request failed: {exc}", retryable=True) from exc
    if not response.ok:
        raise ImageProviderError(
            f"DeepAI HTTP {response.status_code}: {response.text[:500]}",
            retryable=_retryable_status(response.status_code),
            cooldown_seconds=3600 if response.status_code in {401, 402, 403} else None,
        )
    data = response.json()
    _download_url(data.get("output_url", ""), filename)
    _normalize_image(filename)
    return ImageResponse("deepai", "text2img", filename)


def _pollinations(prompt: str, filename: Path, model: str) -> ImageResponse:
    from urllib.parse import quote

    url = (
        f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        f"?width=1024&height=1024&model={quote(model)}&nologo=true"
    )
    response = requests.get(url, timeout=75)
    if not response.ok:
        raise ImageProviderError(
            f"Pollinations/{model} HTTP {response.status_code}: {response.text[:300]}",
            retryable=_retryable_status(response.status_code),
            cooldown_seconds=3600 if response.status_code in {401, 402, 403} else None,
        )
    _save_bytes(response.content, filename)
    _normalize_image(filename)
    return ImageResponse(f"pollinations_{model}", model, filename)


def _huggingface(prompt: str, filename: Path) -> ImageResponse:
    from huggingface_hub import InferenceClient

    token = _env("HF_TOKEN")
    if not token:
        raise ImageProviderError("HF_TOKEN is not configured")
    model = _env("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
    try:
        client = InferenceClient(api_key=token, provider="auto")
        image = client.text_to_image(prompt, model=model, width=1024, height=1024)
        if image is None:
            raise ImageProviderError("Hugging Face returned no image")
        image.save(filename, format="PNG")
        _normalize_image(filename)
        return ImageResponse("huggingface", model, filename)
    except ImageProviderError:
        raise
    except Exception as exc:
        text = str(exc)
        raise ImageProviderError(
            f"Hugging Face request failed: {text}",
            retryable=any(x in text.lower() for x in ("429", "503", "timeout", "unavailable")),
            cooldown_seconds=3600 if "402" in text or "quota" in text.lower() else None,
        ) from exc


_HANDLERS: dict[str, Callable[[str, Path], ImageResponse]] = {
    "imagen": _imagen,
    "cloudflare": _cloudflare,
    "fal": _fal,
    "together": _together,
    "deepai": _deepai,
    "pollinations_turbo": lambda p, f: _pollinations(p, f, "turbo"),
    "pollinations_flux": lambda p, f: _pollinations(p, f, "flux"),
    "huggingface": _huggingface,
}


def generate(prompt: str, filename: Path, *, label: str = "AI image") -> ImageResponse:
    max_attempts = max(1, int(_env("IMAGE_MAX_ATTEMPTS_PER_PROVIDER", "1")))
    cooldown_default = max(60.0, float(_env("IMAGE_PROVIDER_COOLDOWN_SECONDS", "1800")))
    failures: list[str] = []

    for provider in _provider_order():
        if provider not in _HANDLERS:
            log.warning("%s: unknown image provider %s; skipping", label, provider)
            continue
        if not _configured(provider):
            log.info("%s: provider %s skipped; credentials not configured", label, provider)
            continue
        if _cooled(provider):
            log.info("%s: provider %s skipped; cooldown active", label, provider)
            continue

        for attempt in range(1, max_attempts + 1):
            started = time.monotonic()
            try:
                result = _HANDLERS[provider](prompt, filename)
                if not result.path.exists() or result.path.stat().st_size < 5000:
                    raise ImageProviderError("provider reported success but output file is missing/invalid")
                log.info(
                    "%s succeeded via %s/%s in %.1fs",
                    label, result.provider, result.model, time.monotonic() - started,
                )
                return result
            except Exception as exc:
                retryable = bool(getattr(exc, "retryable", False))
                message = f"{provider} attempt {attempt}: {type(exc).__name__}: {exc}"
                failures.append(message)
                log.warning("%s failed: %s", label, message)
                if attempt < max_attempts and retryable:
                    time.sleep(min(15.0, (2 ** attempt) + random.uniform(0, 0.5)))
                    continue
                cooldown = getattr(exc, "cooldown_seconds", None) or cooldown_default
                _cooldown(provider, cooldown)
                break

    configured = [p for p in _provider_order() if _configured(p)]
    if not configured:
        raise RuntimeError(
            "No image provider is configured. Add CLOUDFLARE_API_TOKEN + "
            "CLOUDFLARE_ACCOUNT_ID, FAL_KEY, TOGETHER_API_KEY, DEEPAI_API_KEY, "
            "or keep an existing GEMINI_API_KEY/HF_TOKEN/Pollinations provider."
        )
    raise RuntimeError("All AI image providers failed: " + " | ".join(failures[-10:]))


def provider_status() -> list[dict]:
    now = time.monotonic()
    return [
        {
            "provider": p,
            "configured": _configured(p),
            "cooldown_remaining": round(max(0.0, _PROVIDER_COOLDOWN_UNTIL.get(p, 0.0) - now), 1),
        }
        for p in _provider_order()
    ]
