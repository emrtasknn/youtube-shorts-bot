"""Central AI provider routing for the Shorts pipeline.

The pipeline should never depend on one AI vendor. Providers are attempted in
configured order, transient failures are retried with bounded exponential
backoff, and a failed provider is temporarily cooled down for the rest of the
process. Provider-specific SDKs are isolated here so the rest of the pipeline
uses one stable interface.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import requests

log = logging.getLogger("shorts-bot.ai")

DEFAULT_PROVIDER_ORDER = ["gemini", "openrouter", "groq", "openai"]
_PROVIDER_COOLDOWN_UNTIL: dict[str, float] = {}


@dataclass
class AIResponse:
    text: str
    provider: str
    model: str
    raw: Any = None


def _provider_order() -> list[str]:
    raw = os.getenv("AI_PROVIDER_ORDER", ",".join(DEFAULT_PROVIDER_ORDER))
    result = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return result or list(DEFAULT_PROVIDER_ORDER)


def _key_for(provider: str) -> str:
    return {
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider, "")


def _configured(provider: str) -> bool:
    key = _key_for(provider)
    return bool(key and os.getenv(key, "").strip())


def _env_or_default(name: str, default: str) -> str:
    # GitHub Actions can inject an unset optional secret/env var as an empty
    # string. Treat blank values as unset so provider defaults still apply.
    value = os.getenv(name, "").strip()
    return value or default


def _model_for(provider: str) -> str:
    defaults = {
        "gemini": _env_or_default("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        "openrouter": _env_or_default("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
        "groq": _env_or_default("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "openai": _env_or_default("OPENAI_MODEL", "gpt-4.1-mini"),
    }
    return defaults[provider]


def _cooldown(provider: str, seconds: float) -> None:
    _PROVIDER_COOLDOWN_UNTIL[provider] = time.monotonic() + seconds


def _is_cooled_down(provider: str) -> bool:
    return time.monotonic() < _PROVIDER_COOLDOWN_UNTIL.get(provider, 0.0)


def _retryable_status(status: int) -> bool:
    return status == 408 or status == 409 or status == 425 or status == 429 or status >= 500


def _backoff(attempt: int) -> float:
    # 2, 4, 8... with small jitter. Never sleep indefinitely.
    return min(20.0, (2 ** attempt) + random.uniform(0, 0.5))


def _gemini(prompt: str, model: str, response_mime_type: str | None) -> AIResponse:
    from google import genai

    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    config = {"response_mime_type": response_mime_type} if response_mime_type else None
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    text = getattr(response, "text", "") or ""
    if not text.strip():
        raise RuntimeError("Gemini returned an empty response")
    return AIResponse(text=text, provider="gemini", model=model, raw=response)


def _openai_compatible(
    provider: str,
    prompt: str,
    model: str,
    response_mime_type: str | None,
) -> AIResponse:
    endpoints = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "groq": "https://api.groq.com/openai/v1/chat/completions",
    }
    endpoint = endpoints[provider]
    api_key = os.environ[_key_for(provider)]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "https://github.com/emrtasknn/youtube-shorts-bot")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "Historical Shorts Bot")

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a production content engine. Follow the user's output format exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    if response_mime_type == "application/json":
        body["response_format"] = {"type": "json_object"}

    timeout = float(os.getenv("AI_HTTP_TIMEOUT_SECONDS", "90"))
    response = requests.post(endpoint, headers=headers, json=body, timeout=timeout)

    if not response.ok:
        detail = response.text[:800].replace("\n", " ")
        error = RuntimeError(f"{provider} HTTP {response.status_code}: {detail}")
        error.retryable = _retryable_status(response.status_code)  # type: ignore[attr-defined]
        raise error

    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"{provider} returned no choices")

    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in text
        )
    if not str(text).strip():
        raise RuntimeError(f"{provider} returned empty content")

    return AIResponse(text=str(text), provider=provider, model=model, raw=payload)


def _call_provider(provider: str, prompt: str, response_mime_type: str | None) -> AIResponse:
    model = _model_for(provider)
    if provider == "gemini":
        return _gemini(prompt, model, response_mime_type)
    if provider in {"openai", "openrouter", "groq"}:
        return _openai_compatible(provider, prompt, model, response_mime_type)
    raise RuntimeError(f"Unknown AI provider: {provider}")


def generate(
    prompt: str,
    *,
    label: str = "AI generation",
    response_mime_type: str | None = None,
) -> AIResponse:
    """Generate text through the configured provider chain.

    Non-retryable provider errors immediately move to the next provider.
    Retryable errors receive bounded retries before failover. A provider that
    repeatedly fails is cooled down so a 10-video batch does not hammer it.
    """

    max_attempts = max(1, int(os.getenv("AI_MAX_ATTEMPTS_PER_PROVIDER", "2")))
    cooldown_seconds = max(30.0, float(os.getenv("AI_PROVIDER_COOLDOWN_SECONDS", "300")))
    failures: list[str] = []

    for provider in _provider_order():
        if not _configured(provider):
            log.debug("%s: provider %s skipped; API key not configured", label, provider)
            continue
        if _is_cooled_down(provider):
            log.warning("%s: provider %s is temporarily cooled down", label, provider)
            continue

        for attempt in range(1, max_attempts + 1):
            started = time.monotonic()
            try:
                result = _call_provider(provider, prompt, response_mime_type)
                elapsed = time.monotonic() - started
                log.info(
                    "%s succeeded via %s/%s in %.1fs",
                    label, result.provider, result.model, elapsed,
                )
                return result
            except Exception as exc:
                error_text = str(exc).lower()
                retryable = bool(getattr(exc, "retryable", False)) or any(
                    marker in error_text
                    for marker in (
                        "429", "too many requests", "rate limit",
                        "resource_exhausted", "quota exceeded",
                        "503", "service unavailable", "unavailable",
                        "deadline exceeded", "timed out", "timeout",
                    )
                )
                message = f"{provider}/{_model_for(provider)} attempt {attempt}: {type(exc).__name__}: {exc}"
                failures.append(message)
                log.warning("%s failed: %s", label, message)

                if attempt < max_attempts and (retryable or isinstance(exc, (TimeoutError, requests.RequestException))):
                    time.sleep(_backoff(attempt))
                    continue
                break

        _cooldown(provider, cooldown_seconds)
        log.warning("%s: failing over from provider %s", label, provider)

    configured = [p for p in _provider_order() if _configured(p)]
    if not configured:
        raise RuntimeError(
            "No AI provider is configured. Set GEMINI_API_KEY or configure "
            "OPENAI_API_KEY / OPENROUTER_API_KEY / GROQ_API_KEY."
        )

    raise RuntimeError(
        f"{label} failed across all configured AI providers. "
        + " | ".join(failures[-8:])
    )


def provider_status() -> list[dict[str, Any]]:
    """Return safe provider diagnostics without exposing API keys."""
    now = time.monotonic()
    return [
        {
            "provider": provider,
            "configured": _configured(provider),
            "model": _model_for(provider) if _configured(provider) else "",
            "cooldown_remaining": round(max(0.0, _PROVIDER_COOLDOWN_UNTIL.get(provider, 0.0) - now), 1),
        }
        for provider in _provider_order()
    ]
