"""Backward-compatible AI entrypoint.

Existing pipeline modules call call_gemini_with_retry(). The implementation is
now provider-agnostic and routes through ai_provider, while retaining the old
function name so existing callers do not need a risky one-shot rewrite.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import ai_provider

log = logging.getLogger("shorts-bot")

DEFAULT_CANDIDATE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
]


def get_candidate_models() -> list[str]:
    """Return the configured Gemini model plus legacy candidates."""
    user_model = os.getenv("GEMINI_MODEL")
    if user_model:
        return [user_model] + [m for m in DEFAULT_CANDIDATE_MODELS if m != user_model]
    return list(DEFAULT_CANDIDATE_MODELS)


def get_genai_client():
    """Compatibility helper for code that still needs the native Gemini client."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    return genai.Client(api_key=api_key)


def call_gemini_with_retry(
    prompt: str,
    client=None,
    label: str = "AI call",
    response_mime_type: Optional[str] = None,
):
    """Compatibility wrapper around the centralized multi-provider router.

    The optional client argument is retained for compatibility with older
    callers. Provider selection is now centralized in ai_provider.
    """
    return ai_provider.generate(
        prompt,
        label=label,
        response_mime_type=response_mime_type,
    )


def provider_status() -> list[dict]:
    return ai_provider.provider_status()
