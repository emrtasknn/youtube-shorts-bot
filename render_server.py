from __future__ import annotations

import io
import json
import logging
import os
import secrets
import tempfile
import threading
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

import youtube_uploader

log = logging.getLogger("render-server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = str(os.environ["TELEGRAM_CHAT_ID"])
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "emrtasknn/youtube-shorts-bot").strip()

PORT = int(os.environ.get("PORT", "10000"))
TELEGRAM_WEBHOOK_SECRET = os.environ.get(
    "TELEGRAM_WEBHOOK_SECRET",
    secrets.token_urlsafe(32),
)

PROCESSED_CALLBACKS: set[str] = set()
CALLBACK_LOCK = threading.Lock()


def telegram(method: str, payload: dict) -> dict:
    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {data}")
    return data


def github_headers() -> dict:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is missing in Render environment")
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "youtube-shorts-render-control",
    }


def github_dispatch(workflow_file: str, inputs: dict) -> None:
    response = requests.post(
        f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{workflow_file}/dispatches",
        headers={**github_headers(), "Content-Type": "application/json"},
        json={"ref": "main", "inputs": inputs},
        timeout=30,
    )
    if response.status_code not in {200, 201, 204}:
        raise RuntimeError(
            f"GitHub workflow dispatch failed: {response.status_code} {response.text[:1000]}"
        )


def download_run_artifact(source_run_id: str, source_run_number: str) -> tuple[Path, Path]:
    artifact_name = f"shorts-run-{source_run_number}"
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{source_run_id}/artifacts",
        headers=github_headers(),
        params={"per_page": 100},
        timeout=30,
    )
    response.raise_for_status()

    artifacts = response.json().get("artifacts", [])
    artifact = next(
        (
            item
            for item in artifacts
            if item.get("name") == artifact_name and not item.get("expired")
        ),
        None,
    )
    if not artifact:
        raise FileNotFoundError(
            f"GitHub artifact '{artifact_name}' not found for run {source_run_id}"
        )

    archive_response = requests.get(
        artifact["archive_download_url"],
        headers=github_headers(),
        timeout=120,
    )
    archive_response.raise_for_status()

    target_dir = Path(tempfile.mkdtemp(prefix="youtube_short_publish_"))
    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
        archive.extractall(target_dir)

    video_candidates = list(target_dir.rglob("final_short.mp4"))
    metadata_candidates = list(target_dir.rglob("metadata.json"))

    if not video_candidates:
        raise FileNotFoundError("final_short.mp4 was not found in GitHub artifact")
    if not metadata_candidates:
        raise FileNotFoundError("metadata.json was not found in GitHub artifact")

    metadata_path = metadata_candidates[0]
    video_path = video_candidates[0]

    log.info(
        "Downloaded publication artifact: video=%s metadata=%s",
        video_path,
        metadata_path,
    )
    return video_path, metadata_path


def notify(chat_id: str, text: str) -> None:
    telegram(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": False,
        },
    )


def answer_callback(callback_id: str, text: str) -> None:
    telegram(
        "answerCallbackQuery",
        {
            "callback_query_id": callback_id,
            "text": text,
            "show_alert": False,
        },
    )


def remove_buttons(chat_id: str, message_id: int) -> None:
    telegram(
        "editMessageReplyMarkup",
        {
            "chat_id": chat_id,
            "message_id": message_id,
            "reply_markup": {"inline_keyboard": []},
        },
    )


def process_publish(source_run_id: str, source_run_number: str, chat_id: str) -> None:
    try:
        video_path, metadata_path = download_run_artifact(
            source_run_id, source_run_number
        )

        # Reuse the topic/script metadata produced by pipeline.py so the
        # published YouTube Short keeps the actual generated title and
        # description instead of falling back to a generic default.
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        event_record = metadata.get("event_record") or {}
        scenes = metadata.get("scenes") or []
        full_text = " ".join(
            str(scene.get("narration", "")).strip()
            for scene in scenes
            if scene.get("narration")
        ).strip()

        default_title = os.getenv(
            "YOUTUBE_DEFAULT_TITLE", "Tarihin Bilinmeyen Gizemi"
        )
        topic = {
            "title": event_record.get("canonical_title") or default_title,
            "hook_question": "",
        }

        # youtube_uploader supports direct OAuth refresh-token credentials
        # through Render environment variables.
        result = youtube_uploader.upload_shorts_video(
            video_path=video_path,
            topic=topic,
            full_text=full_text,
            privacy_status=os.getenv("YOUTUBE_PRIVACY_STATUS"),
        )

        title = result.get("title", "Video")
        youtube_url = result.get("url", "")
        video_id = result.get("video_id", "")

        notify(
            chat_id,
            "🎉 YouTube Shorts'a yükleme tamamlandı!\n\n"
            f"🎬 {title}\n"
            f"🔗 {youtube_url}\n"
            f"🆔 Video ID: {video_id}",
        )

    except Exception as exc:
        log.exception("YouTube publication failed")
        notify(
            chat_id,
            "❌ YouTube yüklemesi başarısız oldu.\n\n"
            f"{type(exc).__name__}: {exc}",
        )


def process_callback(update: dict) -> None:
    callback = update.get("callback_query")
    if not callback:
        return

    callback_id = str(callback.get("id", ""))
    with CALLBACK_LOCK:
        if callback_id in PROCESSED_CALLBACKS:
            return
        PROCESSED_CALLBACKS.add(callback_id)

    message = callback.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = str(chat.get("id", ""))

    if chat_id != TELEGRAM_CHAT_ID:
        answer_callback(callback_id, "Yetkisiz sohbet.")
        return

    data = str(callback.get("data", ""))
    parts = data.split(":")
    action = parts[0] if parts else ""

    if action == "publish" and len(parts) >= 3:
        source_run_id = parts[1]
        source_run_number = parts[2]
        answer_callback(callback_id, "Yayınlama başlatılıyor...")
        try:
            remove_buttons(chat_id, int(message.get("message_id")))
        except Exception:
            log.exception("Could not remove Telegram buttons")

        notify(
            chat_id,
            "🚀 Video onaylandı. GitHub artifact indiriliyor ve YouTube'a yükleniyor...",
        )
        threading.Thread(
            target=process_publish,
            args=(source_run_id, source_run_number, chat_id),
            daemon=True,
        ).start()
        return

    if action == "regen" and len(parts) >= 2:
        source_run_id = parts[1]
        answer_callback(callback_id, "Yeni üretim başlatılıyor...")
        try:
            remove_buttons(chat_id, int(message.get("message_id")))
        except Exception:
            log.exception("Could not remove Telegram buttons")

        try:
            github_dispatch(
                "daily_short.yml",
                {
                    "mode": "regenerate",
                    "source_run_id": source_run_id,
                    "source_event_id": "",
                },
            )
            notify(chat_id, "🔄 Yeni video üretimi başlatıldı.")
        except Exception as exc:
            log.exception("Regeneration dispatch failed")
            notify(
                chat_id,
                f"❌ Yeni üretim başlatılamadı.\n\n{type(exc).__name__}: {exc}",
            )
        return

    if action == "cancel":
        answer_callback(callback_id, "Video iptal edildi.")
        try:
            remove_buttons(chat_id, int(message.get("message_id")))
        except Exception:
            log.exception("Could not remove Telegram buttons")
        notify(chat_id, "❌ Video yayını iptal edildi.")
        return

    if action == "script":
        answer_callback(
            callback_id,
            "Senaryo görüntüleme bu sürümde yayın akışından ayrıldı.",
        )
        return

    answer_callback(callback_id, "Geçersiz veya eski buton.")


def register_webhook(public_url: str) -> None:
    webhook_url = public_url.rstrip("/") + "/telegram/webhook"
    telegram(
        "setWebhook",
        {
            "url": webhook_url,
            "secret_token": TELEGRAM_WEBHOOK_SECRET,
            "allowed_updates": ["callback_query"],
            "drop_pending_updates": False,
        },
    )
    log.info("Telegram webhook configured: %s", webhook_url)


class Handler(BaseHTTPRequestHandler):
    def _json(self, data: dict, status: int = 200) -> None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/", "/health"}:
            self._json({"ok": True, "service": "youtube-shorts-bot"})
            return
        self._json({"error": "not_found"}, 404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        if parsed.path == "/telegram/webhook":
            supplied = self.headers.get(
                "X-Telegram-Bot-Api-Secret-Token", ""
            )
            if not secrets.compare_digest(
                supplied, TELEGRAM_WEBHOOK_SECRET
            ):
                self._json({"error": "unauthorized"}, 401)
                return

            try:
                update = json.loads(body.decode("utf-8"))
                process_callback(update)
                self._json({"ok": True})
            except Exception as exc:
                log.exception("Webhook processing failed")
                self._json({"error": str(exc)}, 500)
            return

        self._json({"error": "not_found"}, 404)

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    public_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()

    if public_url:
        try:
            register_webhook(public_url)
        except Exception:
            log.exception("Initial Telegram webhook registration failed")

    log.info("HTTP server listening on 0.0.0.0:%s", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
