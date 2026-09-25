import os
import time
import logging
from typing import Optional

try:
    from google import genai
except ImportError:
    genai = None

log = logging.getLogger("shorts-bot")

DEFAULT_CANDIDATE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
]

def get_candidate_models() -> list[str]:
    """Return candidate Gemini models (prefer fast/cheap), with user override."""
    user_model = os.getenv("GEMINI_MODEL")
    if user_model:
        return [user_model] + [m for m in DEFAULT_CANDIDATE_MODELS if m != user_model]
    return list(DEFAULT_CANDIDATE_MODELS)

def get_genai_client():
    """Lazily acquire the genai client."""
    if not genai:
        raise RuntimeError("google-genai SDK is not installed")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    return genai.Client(api_key=api_key)

def call_gemini_with_retry(
    prompt: str,
    client=None,
    label: str = "gemini call",
    response_mime_type: Optional[str] = None
):
    """
    Calls Gemini API with exponential backoff on 503/UNAVAILABLE/429 errors.
    If the main model fails after retries, falls back to the next candidate model.
    Returns the response object (which has .text) or None on total failure.
    """
    if not client:
        client = get_genai_client()
        
    models = get_candidate_models()
    max_retries_per_model = 3
    
    config = {}
    if response_mime_type:
        config["response_mime_type"] = response_mime_type

    for model in models:
        base_wait = 2
        for attempt in range(1, max_retries_per_model + 1):
            try:
                # API Call
                res = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config if config else None
                )
                if res and res.text:
                    return res
                log.warning("%s: '%s' returned empty response on attempt %d", label, model, attempt)
            except Exception as exc:
                err_str = str(exc)
                log.warning("%s: '%s' failed on attempt %d with: %s", label, model, attempt, exc)
                
                # Check if error is retryable (transient/rate limit)
                if any(err in err_str for err in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "Too Many Requests", "Internal Server Error"]):
                    if attempt < max_retries_per_model:
                        wait_time = base_wait ** attempt  # Exponential: 2, 4, 8...
                        log.info("Waiting %d seconds before retry...", wait_time)
                        time.sleep(wait_time)
                        continue
                else:
                    # Non-retryable error (e.g. 400 Bad Request, 404 Not Found), skip to next model
                    break
        log.warning("%s: All attempts failed for model '%s'. Falling back...", label, model)
        
    log.error("%s: All models and retries exhausted.", label)
    return None
