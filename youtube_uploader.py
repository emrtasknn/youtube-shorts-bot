"""YouTube Data API v3 Shorts Uploader Module.

Handles OAuth2 authentication, Shorts metadata optimization (title, description, tags, category),
and resumable video uploading with error handling.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

log = logging.getLogger("shorts-bot.youtube")

YOUTUBE_UPLOAD_SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]
DEFAULT_CLIENT_SECRETS_FILE = Path("client_secrets.json")
DEFAULT_TOKEN_FILE = Path("token.json")


def build_shorts_metadata(
    topic: dict | None = None,
    full_text: str = "",
    privacy_status: str = "public",
) -> dict:
    """Build YouTube Shorts optimized metadata dictionary."""
    raw_title = (
        topic.get("youtube_title")
        or topic.get("title", "Tarihin Bilinmeyen Gizemi")
        or "Tarihin Bilinmeyen Gizemi"
    ) if topic else "Tarihin Bilinmeyen Gizemi"
    hook_question = topic.get("hook_question", "") if topic else ""

    # YouTube max title length is 100 chars
    suffix = " #Shorts"
    max_raw_len = 100 - len(suffix)
    clean_title = raw_title.strip()
    if len(clean_title) > max_raw_len:
        clean_title = clean_title[:max_raw_len - 3] + "..."
    title = f"{clean_title}{suffix}"

    # Build compelling description
    desc_lines = [
        f"🔥 {raw_title}\n",
    ]
    if hook_question:
        desc_lines.append(f"❓ {hook_question}\n")

    if full_text:
        desc_lines.append(f"{full_text[:400]}...\n")

    desc_lines.extend([
        "🎬 Her gün yeni bir gizemli tarih hikayesi!",
        "🔔 Abone olmayı ve beğenmeyi unutmayın!\n",
        "#Shorts #Tarih #Gizem #Belgesel #AntikTarih #TarihShorts #İlginçBilgiler #Keşfet",
    ])
    description = "\n".join(desc_lines)

    # Tags list
    tags = [
        "Shorts",
        "Tarih",
        "Gizem",
        "Antik Tarih",
        "Belgesel",
        "İlginç Bilgiler",
        "Tarihsel Olaylar",
        "Keşfet",
    ]
    if topic and "title" in topic:
        for word in topic["title"].split():
            clean_word = word.strip(",.?!")
            if len(clean_word) >= 4 and clean_word not in tags:
                tags.append(clean_word)

    valid_statuses = {"public", "unlisted", "private"}
    normalized_privacy = privacy_status.lower() if privacy_status.lower() in valid_statuses else "public"

    return {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags[:15],
            "categoryId": "27",  # 27 = Education
            "defaultLanguage": "tr",
            "defaultAudioLanguage": "tr",
        },
        "status": {
            "privacyStatus": normalized_privacy,
            "selfDeclaredMadeForKids": False,
        },
    }


def get_youtube_client(
    client_secrets_path: Path | None = None,
    token_path: Path | None = None,
):
    """Authenticate and create a YouTube Data API v3 client."""
    secrets_file = client_secrets_path if client_secrets_path is not None else DEFAULT_CLIENT_SECRETS_FILE
    tok_file = token_path if token_path is not None else DEFAULT_TOKEN_FILE

    creds = None

    # 1. Check local token.json
    if tok_file.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(tok_file), YOUTUBE_UPLOAD_SCOPE)
        except Exception as exc:
            log.warning("Could not read token file %s: %s", tok_file, exc)

    # 2. Check YOUTUBE_TOKEN_JSON environment variable (e.g. for GitHub Actions)
    if not creds and os.environ.get("YOUTUBE_TOKEN_JSON"):
        try:
            token_data = json.loads(os.environ["YOUTUBE_TOKEN_JSON"])
            creds = Credentials.from_authorized_user_info(token_data, YOUTUBE_UPLOAD_SCOPE)
        except Exception as exc:
            log.warning("Could not load YOUTUBE_TOKEN_JSON env: %s", exc)

    # 3. Direct refresh-token credentials (for Render / other headless hosts).
    # Prefer these environment variables over local token files because Render
    # does not have the developer's local OAuth files.
    if not creds:
        refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN", "").strip()
        client_id = os.environ.get("YOUTUBE_CLIENT_ID", "").strip()
        client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET", "").strip()

        if refresh_token and client_id and client_secret:
            log.info(
                "Using Render YouTube refresh-token credentials "
                "(refresh_token=%s, client_id=%s, client_secret=%s)",
                "set",
                "set",
                "set",
            )
            creds = Credentials(
                token=None,
                refresh_token=refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
                scopes=YOUTUBE_UPLOAD_SCOPE,
            )
        elif os.environ.get("YOUTUBE_REFRESH_TOKEN") or os.environ.get("YOUTUBE_CLIENT_ID") or os.environ.get("YOUTUBE_CLIENT_SECRET"):
            raise RuntimeError(
                "YouTube Render credentials are incomplete. "
                "YOUTUBE_REFRESH_TOKEN, YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET must all be set."
            )

    # 4. Refresh credentials when there is no access token yet or the token
    # has expired. This is essential for headless Render deployments: a
    # refresh-token-only Credentials object starts with token=None and is not
    # necessarily marked expired by google-auth.
    if creds and creds.refresh_token and (creds.token is None or creds.expired):
        try:
            creds.refresh(Request())
            log.info("YouTube OAuth access token refreshed successfully.")
            # Persist only when a writable local token file is actually in use.
            if tok_file.parent.exists() and tok_file.exists():
                tok_file.write_text(creds.to_json(), encoding="utf-8")
        except Exception as exc:
            log.exception("YouTube OAuth refresh failed")
            raise RuntimeError(
                f"YouTube OAuth refresh failed: {type(exc).__name__}: {exc}"
            ) from exc

    # 5. If still no valid creds, try client_secrets flow
    if not creds or not creds.valid:
        if secrets_file.exists():
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), YOUTUBE_UPLOAD_SCOPE)
            creds = flow.run_local_server(port=0)
            tok_file.write_text(creds.to_json(), encoding="utf-8")
        elif os.environ.get("YOUTUBE_CLIENT_SECRETS_JSON"):
            secrets_data = json.loads(os.environ["YOUTUBE_CLIENT_SECRETS_JSON"])
            flow = InstalledAppFlow.from_client_config(secrets_data, YOUTUBE_UPLOAD_SCOPE)
            creds = flow.run_local_server(port=0)
            tok_file.write_text(creds.to_json(), encoding="utf-8")
        else:
            raise FileNotFoundError(
                "YouTube credentials not found. Please provide 'client_secrets.json' or 'token.json'."
            )

    return build("youtube", "v3", credentials=creds)


def upload_shorts_video(
    video_path: Path,
    topic: dict | None = None,
    full_text: str = "",
    privacy_status: str | None = None,
    youtube_client=None,
    client_secrets_path: Path | None = None,
    token_path: Path | None = None,
) -> dict:
    """Upload a video to YouTube as a Short with optimized metadata and error handling."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    env_privacy = os.environ.get("YOUTUBE_PRIVACY_STATUS", "public")
    effective_privacy = privacy_status or env_privacy

    metadata = build_shorts_metadata(
        topic=topic,
        full_text=full_text,
        privacy_status=effective_privacy,
    )

    client = youtube_client
    if client is None:
        client = get_youtube_client(
            client_secrets_path=client_secrets_path,
            token_path=token_path,
        )

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        chunksize=1024 * 1024,
        resumable=True,
    )

    log.info("Uploading video to YouTube: %s", metadata["snippet"]["title"])
    request = client.videos().insert(
        part="snippet,status",
        body=metadata,
        media_body=media,
    )

    response = request.execute()
    video_id = response.get("id")
    if not video_id:
        raise RuntimeError("YouTube API did not return a valid video ID.")

    youtube_url = f"https://youtube.com/shorts/{video_id}"
    log.info("Successfully uploaded video to YouTube Shorts: %s", youtube_url)

    return {
        "video_id": video_id,
        "url": youtube_url,
        "title": metadata["snippet"]["title"],
        "privacy_status": metadata["status"]["privacyStatus"],
    }
