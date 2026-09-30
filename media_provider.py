"""Real-media search router for historical Shorts.

Searches licensed/attributed public media APIs before AI reconstruction.
Providers are intentionally isolated from image_provider.py so real media
selection and generative fallback remain separate concerns.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import requests

log = logging.getLogger("shorts-bot.media")

DEFAULT_ORDER = "pexels,pixabay,unsplash"
TIMEOUT = int(os.getenv("MEDIA_PROVIDER_TIMEOUT_SECONDS", "15"))


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _order() -> list[str]:
    raw = os.getenv("REAL_MEDIA_PROVIDER_ORDER", DEFAULT_ORDER)
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z0-9çğıöşüÇĞİÖŞÜ]+", str(value).lower())
        if len(token) >= 3
    }


def _score(query: str, title: str, tags: str = "", intent: dict | None = None) -> float:
    q = _tokens(query)
    hay = _tokens(" ".join([title, tags, str((intent or {}).get("primary_subject", ""))]))
    if not q or not hay:
        return 0.0
    overlap = len(q & hay) / max(1, len(q))
    return round(min(1.0, overlap), 3)


def _configured(provider: str) -> bool:
    return {
        "pexels": bool(_env("PEXELS_API_KEY")),
        "pixabay": bool(_env("PIXABAY_API_KEY")),
        "unsplash": bool(_env("UNSPLASH_ACCESS_KEY")),
    }.get(provider, False)


def _result(provider: str, *, title: str, image_url: str, page_url: str,
            query: str, relevance: float, author: str = "", width: int = 0,
            height: int = 0) -> dict[str, Any]:
    return {
        "source_type": provider,
        "title": title or f"{provider} visual",
        "image_url": image_url,
        "page_url": page_url,
        "author": author,
        "query": query,
        "relevance_score": relevance,
        "event_specificity": relevance,
        "information_density": relevance,
        "media_kind": "image",
        "width": width,
        "height": height,
        "provider": provider,
        "attribution_required": True,
    }


def _search_pexels(query: str, excluded: set[str], intent: dict) -> dict | None:
    response = requests.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization": _env("PEXELS_API_KEY")},
        params={"query": query, "per_page": 10, "orientation": "portrait", "locale": "en-US"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    for photo in response.json().get("photos", []):
        image_url = (photo.get("src") or {}).get("large2x") or (photo.get("src") or {}).get("large")
        if not image_url or image_url in excluded:
            continue
        title = str(photo.get("alt") or "")
        tags = " ".join(str(x) for x in photo.get("tags", []) if isinstance(x, str))
        score = _score(query, title, tags, intent)
        if score >= float(os.getenv("REAL_MEDIA_MIN_SEARCH_SCORE", "0.35")):
            return _result(
                "pexels", title=title, image_url=image_url,
                page_url=str((photo.get("url") or "")),
                query=query, relevance=score,
                author=str((photo.get("photographer") or "")),
                width=int(photo.get("width") or 0), height=int(photo.get("height") or 0),
            )
    return None


def _search_pixabay(query: str, excluded: set[str], intent: dict) -> dict | None:
    response = requests.get(
        "https://pixabay.com/api/",
        params={
            "key": _env("PIXABAY_API_KEY"), "q": query, "lang": "en",
            "image_type": "photo", "orientation": "vertical",
            "per_page": 10, "safesearch": "true",
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    for hit in response.json().get("hits", []):
        image_url = str(hit.get("largeImageURL") or hit.get("webformatURL") or "")
        if not image_url or image_url in excluded:
            continue
        title = str(hit.get("tags") or "")
        score = _score(query, title, title, intent)
        if score >= float(os.getenv("REAL_MEDIA_MIN_SEARCH_SCORE", "0.35")):
            return _result(
                "pixabay", title=title, image_url=image_url,
                page_url=str(hit.get("pageURL") or ""), query=query,
                relevance=score, author=str(hit.get("user") or ""),
                width=int(hit.get("imageWidth") or 0), height=int(hit.get("imageHeight") or 0),
            )
    return None


def _search_unsplash(query: str, excluded: set[str], intent: dict) -> dict | None:
    response = requests.get(
        "https://api.unsplash.com/search/photos",
        headers={"Authorization": f"Client-ID {_env('UNSPLASH_ACCESS_KEY')}"},
        params={"query": query, "per_page": 10, "orientation": "portrait", "content_filter": "high"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    for photo in response.json().get("results", []):
        urls = photo.get("urls") or {}
        image_url = str(urls.get("regular") or urls.get("full") or "")
        if not image_url or image_url in excluded:
            continue
        tags = " ".join(str(t.get("title") or "") for t in (photo.get("tags") or []) if isinstance(t, dict))
        title = str(photo.get("alt_description") or photo.get("description") or "")
        score = _score(query, title, tags, intent)
        if score >= float(os.getenv("REAL_MEDIA_MIN_SEARCH_SCORE", "0.35")):
            links = photo.get("links") or {}
            user = photo.get("user") or {}
            links = photo.get("links") or {}
        download_location = str(links.get("download_location") or "")
        if download_location:
            try:
                requests.get(
                    download_location,
                    params={"client_id": _env("UNSPLASH_ACCESS_KEY")},
                    timeout=TIMEOUT,
                ).raise_for_status()
            except requests.RequestException as exc:
                log.warning("MEDIA: Unsplash download tracking failed: %s", exc)
        return _result(
                "unsplash", title=title, image_url=image_url,
                page_url=str(links.get("html") or ""), query=query,
                relevance=score, author=str(user.get("name") or ""),
                width=int(photo.get("width") or 0), height=int(photo.get("height") or 0),
            )
    return None


_HANDLERS = {
    "pexels": _search_pexels,
    "pixabay": _search_pixabay,
    "unsplash": _search_unsplash,
}


def search(
    queries: list[str],
    *,
    excluded_urls: list[str] | None = None,
    visual_intent: dict | None = None,
) -> dict | None:
    """Return the first sufficiently relevant real-media result."""
    excluded = set(excluded_urls or [])
    intent = visual_intent or {}
    for provider in _order():
        if provider not in _HANDLERS:
            continue
        if not _configured(provider):
            log.info("MEDIA: %s skipped; credentials not configured", provider)
            continue
        for query in queries[:3]:
            query = str(query).strip()
            if not query:
                continue
            try:
                result = _HANDLERS[provider](query, excluded, intent)
                if result:
                    log.info(
                        "MEDIA: %s → %s relevance=%.2f query=%s",
                        provider, result["title"], result["relevance_score"], query,
                    )
                    return result
            except requests.RequestException as exc:
                log.warning("MEDIA: %s request failed for %r: %s", provider, query, exc)
            except Exception as exc:
                log.warning("MEDIA: %s failed for %r: %s", provider, query, exc)
    return None
