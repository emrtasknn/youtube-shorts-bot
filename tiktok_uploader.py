"""TikTok Content Posting API Direct Post uploader for Bir Garip Tarih."""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

API_BASE = "https://open.tiktokapis.com/v2"
DEFAULT_CHUNK_SIZE = 10 * 1024 * 1024


class TikTokPublishError(RuntimeError):
    pass


def _access_token() -> str:
    token = os.environ.get("TIKTOK_ACCESS_TOKEN", "").strip()
    if not token:
        raise TikTokPublishError("TIKTOK_ACCESS_TOKEN is missing.")
    return token


def _request(method: str, url: str, token: str, **kwargs) -> dict:
    headers = kwargs.pop("headers", {})
    headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=UTF-8",
    })
    response = requests.request(method, url, headers=headers, timeout=60, **kwargs)
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": {"code": "http_error", "message": response.text[:500]}}
    if not response.ok:
        raise TikTokPublishError(
            f"TikTok API HTTP {response.status_code}: {payload}"
        )
    error = payload.get("error") or {}
    if error.get("code") not in (None, "", "ok"):
        raise TikTokPublishError(
            f"TikTok API error {error.get('code')}: {error.get('message', '')}"
        )
    return payload


def build_tiktok_caption(topic: dict | None = None) -> str:
    topic = topic or {}
    title = str(topic.get("tiktok_caption") or topic.get("youtube_title") or topic.get("title") or "Tarihin Garip Hikâyeleri").strip()
    # Keep TikTok copy platform-specific and concise.
    hashtags = "#tarih #birgariptarih #tarihgizemleri"
    return f"{title}\n\n{hashtags}"[:2200]


def get_creator_info(token: str) -> dict:
    payload = _request(
        "POST",
        f"{API_BASE}/post/publish/creator_info/query/",
        token,
    )
    return payload.get("data") or {}


def _upload_file(upload_url: str, video_path: Path, chunk_size: int) -> None:
    total = video_path.stat().st_size
    with video_path.open("rb") as fh:
        offset = 0
        while offset < total:
            chunk = fh.read(min(chunk_size, total - offset))
            if not chunk:
                break
            end = offset + len(chunk) - 1
            headers = {
                "Content-Type": "video/mp4",
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {offset}-{end}/{total}",
            }
            response = requests.put(
                upload_url,
                headers=headers,
                data=chunk,
                timeout=180,
            )
            if response.status_code not in (200, 201, 206):
                raise TikTokPublishError(
                    f"TikTok media upload failed: HTTP {response.status_code} {response.text[:500]}"
                )
            offset = end + 1


def fetch_status(token: str, publish_id: str) -> dict:
    payload = _request(
        "POST",
        f"{API_BASE}/post/publish/status/fetch/",
        token,
        json={"publish_id": publish_id},
    )
    return payload.get("data") or {}


def publish_tiktok_video(
    video_path: Path,
    topic: dict | None = None,
    is_aigc: bool = True,
    poll_seconds: int = 3,
    poll_attempts: int = 20,
) -> dict:
    """Direct-post an MP4 to TikTok and return publish/status metadata."""
    if not video_path.exists():
        raise FileNotFoundError(f"TikTok video not found: {video_path}")

    token = _access_token()
    creator = get_creator_info(token)
    privacy_options = creator.get("privacy_level_options") or []
    if "PUBLIC_TO_EVERYONE" in privacy_options:
        privacy = "PUBLIC_TO_EVERYONE"
    elif privacy_options:
        privacy = privacy_options[0]
    else:
        raise TikTokPublishError("TikTok returned no privacy_level_options.")

    size = video_path.stat().st_size
    chunk_size = min(DEFAULT_CHUNK_SIZE, size) if size else DEFAULT_CHUNK_SIZE
    total_chunks = max(1, (size + chunk_size - 1) // chunk_size)

    body = {
        "post_info": {
            "title": build_tiktok_caption(topic),
            "privacy_level": privacy,
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
            "brand_organic_toggle": False,
            "is_aigc": bool(is_aigc),
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": size,
            "chunk_size": chunk_size,
            "total_chunk_count": total_chunks,
        },
    }

    payload = _request(
        "POST",
        f"{API_BASE}/post/publish/video/init/",
        token,
        json=body,
    )
    data = payload.get("data") or {}
    publish_id = data.get("publish_id")
    upload_url = data.get("upload_url")
    if not publish_id or not upload_url:
        raise TikTokPublishError(f"TikTok did not return publish_id/upload_url: {payload}")

    _upload_file(upload_url, video_path, chunk_size)

    status = {}
    for _ in range(max(1, poll_attempts)):
        status = fetch_status(token, publish_id)
        current = status.get("status")
        if current in {"PUBLISH_COMPLETE", "FAILED"}:
            break
        time.sleep(max(1, poll_seconds))

    post_ids = status.get("publicaly_available_post_id") or []
    post_id = post_ids[0] if post_ids else None

    return {
        "publish_id": publish_id,
        "status": status.get("status", "PROCESSING_UPLOAD"),
        "fail_reason": status.get("fail_reason", ""),
        "post_id": post_id,
        "url": f"https://www.tiktok.com/@birgariptarih/video/{post_id}" if post_id else "",
        "privacy_level": privacy,
        "caption": body["post_info"]["title"],
        "creator_username": creator.get("creator_username", ""),
    }
