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
YOUTUBE_PLAYLIST_SCOPE = "https://www.googleapis.com/auth/youtube"
YOUTUBE_SCOPES = [*YOUTUBE_UPLOAD_SCOPE, YOUTUBE_PLAYLIST_SCOPE]
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
    media_credits = topic.get("real_media_credits", []) if topic else []

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
        # Keep the complete narration in the YouTube description. The previous
        # 400-character slice was cutting every description mid-sentence.
        # YouTube allows up to 5,000 characters; reserve a small margin so
        # future metadata additions cannot accidentally exceed the limit.
        complete_text = str(full_text).strip()
        if len(complete_text) > 4800:
            complete_text = complete_text[:4797].rstrip() + "..."
        desc_lines.append(f"{complete_text}\n")

    if media_credits:
        desc_lines.append("📚 Görsel kaynaklar:")
        for credit in media_credits[:10]:
            provider = str(credit.get("provider", "")).capitalize()
            author = str(credit.get("author", "")).strip()
            page_url = str(credit.get("page_url", "")).strip()
            if page_url:
                label = provider + (f" — {author}" if author else "")
                desc_lines.append(f"{label}: {page_url}")
        desc_lines.append("")

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
            creds = Credentials.from_authorized_user_file(str(tok_file))
        except Exception as exc:
            log.warning("Could not read token file %s: %s", tok_file, exc)

    # 2. Check YOUTUBE_TOKEN_JSON environment variable (e.g. for GitHub Actions)
    if not creds and os.environ.get("YOUTUBE_TOKEN_JSON"):
        try:
            token_data = json.loads(os.environ["YOUTUBE_TOKEN_JSON"])
            creds = Credentials.from_authorized_user_info(token_data)
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
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), YOUTUBE_SCOPES)
            creds = flow.run_local_server(port=0)
            tok_file.write_text(creds.to_json(), encoding="utf-8")
        elif os.environ.get("YOUTUBE_CLIENT_SECRETS_JSON"):
            secrets_data = json.loads(os.environ["YOUTUBE_CLIENT_SECRETS_JSON"])
            flow = InstalledAppFlow.from_client_config(secrets_data, YOUTUBE_SCOPES)
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

    # videos.insert already returns the uploaded video resource, including
    # status because part="snippet,status" was requested above. Do not make a
    # second videos.list call here: that read-back requires broader OAuth
    # permissions than the upload itself and can make a successful upload look
    # like a failed publish when the token is scoped only for upload.
    actual_privacy = (
        (response.get("status") or {}).get("privacyStatus")
        or metadata["status"]["privacyStatus"]
    )
    if actual_privacy != metadata["status"]["privacyStatus"]:
        raise RuntimeError(
            "YouTube accepted the upload but returned a different privacy status: "
            f"requested={metadata['status']['privacyStatus']}, actual={actual_privacy}"
        )

    youtube_url = f"https://youtube.com/shorts/{video_id}"
    log.info(
        "Successfully uploaded video to YouTube Shorts: %s (privacy=%s)",
        youtube_url,
        actual_privacy,
    )

    playlist_id = ""
    content_type = str((topic or {}).get("content_type", "TREND_HISTORY"))
    playlist_env = {
        "TREND_HISTORY": "YOUTUBE_PLAYLIST_TREND_HISTORY",
        "TODAY_IN_HISTORY": "YOUTUBE_PLAYLIST_TODAY_IN_HISTORY",
        "AYT_HISTORY": "YOUTUBE_PLAYLIST_AYT_HISTORY",
        "HISTORY_FACT": "YOUTUBE_PLAYLIST_HISTORY_FACT",
        "CUSTOM": "YOUTUBE_PLAYLIST_CUSTOM",
    }.get(content_type, "")
    if playlist_env:
        playlist_id = os.environ.get(playlist_env, "").strip()

    playlist_result = {"status": "skipped", "playlist_id": playlist_id}
    if playlist_id:
        try:
            # playlistItems.insert requires a playlist-management OAuth scope
            # such as https://www.googleapis.com/auth/youtube.
            if not hasattr(client, "playlistItems"):
                raise RuntimeError("YouTube client does not expose playlistItems")

            item = client.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id,
                        },
                    }
                },
            ).execute()

            returned_playlist_id = (
                (item.get("snippet") or {}).get("playlistId") or ""
            )
            returned_video_id = (
                (item.get("snippet") or {}).get("resourceId", {}).get("videoId") or ""
            )
            if returned_playlist_id != playlist_id or returned_video_id != video_id:
                raise RuntimeError(
                    "YouTube returned an unexpected playlist item: "
                    f"playlist={returned_playlist_id!r}, video={returned_video_id!r}"
                )

            playlist_result = {
                "status": "added",
                "playlist_id": playlist_id,
                "playlist_item_id": item.get("id", ""),
            }
            log.info(
                "Added video %s to playlist %s (playlist_item=%s)",
                video_id,
                playlist_id,
                item.get("id", ""),
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            # YouTube reports a duplicate with a 400 duplicate/videoAlreadyInPlaylist
            # error. Treat that as success because the desired final state is already
            # achieved.
            if "videoAlreadyInPlaylist" in error_text:
                playlist_result = {
                    "status": "already_added",
                    "playlist_id": playlist_id,
                    "error": error_text,
                }
                log.info(
                    "Video %s is already in playlist %s",
                    video_id,
                    playlist_id,
                )
            else:
                # Never turn a successful YouTube upload into a failed publish just
                # because playlist insertion failed. The Telegram result will expose
                # the playlist-specific failure separately.
                playlist_result = {
                    "status": "failed",
                    "playlist_id": playlist_id,
                    "error": error_text,
                }
                log.warning(
                    "Playlist insertion failed for video %s / playlist %s: %s",
                    video_id,
                    playlist_id,
                    exc,
                )

    return {
        "video_id": video_id,
        "url": youtube_url,
        "title": metadata["snippet"]["title"],
        "privacy_status": actual_privacy,
        "content_type": content_type,
        "playlist": playlist_result,
    }
