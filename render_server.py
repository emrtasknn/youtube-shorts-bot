from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

import pipeline
import youtube_uploader

log = logging.getLogger("render-server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = str(os.environ["TELEGRAM_CHAT_ID"])
CONTROL_API_SECRET = os.environ.get("CONTROL_API_SECRET", "")
TELEGRAM_WEBHOOK_SECRET = os.environ.get(
    "TELEGRAM_WEBHOOK_SECRET",
    secrets.token_urlsafe(24),
)

PORT = int(os.environ.get("PORT", "10000"))
MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024

bot = pipeline.bot


def telegram(method: str, payload: dict) -> dict:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {data}")
    return data


def register_webhook(public_url: str) -> None:
    webhook_url = public_url.rstrip("/") + "/telegram/webhook"
    telegram(
        "setWebhook",
        {
            "url": webhook_url,
            "secret_token": TELEGRAM_WEBHOOK_SECRET,
            "allowed_updates": ["callback_query", "message"],
            "drop_pending_updates": False,
        },
    )
    log.info("Telegram webhook configured: %s", webhook_url)


def send_publish_status(chat_id: str, text: str) -> None:
    telegram(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": False,
        },
    )


def process_callback(update: dict) -> None:
    callback = update.get("callback_query")
    if not callback:
        return

    message = callback.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    if chat_id != TELEGRAM_CHAT_ID:
        telegram(
            "answerCallbackQuery",
            {
                "callback_query_id": callback.get("id", ""),
                "text": "Yetkisiz sohbet.",
                "show_alert": True,
            },
        )
        return

    data = str(callback.get("data") or "")
    action, _, run_id = data.partition(":")
    if not action or not run_id:
        telegram(
            "answerCallbackQuery",
            {
                "callback_query_id": callback.get("id", ""),
                "text": "Geçersiz işlem.",
                "show_alert": True,
            },
        )
        return

    state = pipeline.get_approval_state(run_id)
    if not state:
        telegram(
            "answerCallbackQuery",
            {
                "callback_query_id": callback.get("id", ""),
                "text": "Video kaydı bulunamadı.",
                "show_alert": True,
            },
        )
        return

    if state.get("status") not in {"pending"} and action in {"publish", "cancel", "regen"}:
        telegram(
            "answerCallbackQuery",
            {
                "callback_query_id": callback.get("id", ""),
                "text": f"Bu video zaten {state.get('status', 'işleniyor')} durumunda.",
                "show_alert": False,
            },
        )
        return

    telegram(
        "answerCallbackQuery",
        {
            "callback_query_id": callback.get("id", ""),
            "text": {
                "publish": "Yayınlama başlatılıyor...",
                "cancel": "Video iptal edildi.",
                "regen": "Yeni üretim başlatılıyor...",
                "script": "Senaryo getiriliyor...",
            }.get(action, "İşlem başlatılıyor..."),
            "show_alert": False,
        },
    )

    if action == "script":
        scenes = state.get("scenes", [])
        title = state.get("topic", {}).get("title", "Short")
        lines = [f"📜 {title}", ""]
        for index, scene in enumerate(scenes, 1):
            lines.append(f"{index}. Sahne")
            lines.append(str(scene.get("narration", "")))
            lines.append("")
        telegram(
            "sendMessage",
            {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": "\n".join(lines)[:4000],
            },
        )
        return

    try:
        telegram(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": message.get("message_id"),
                "reply_markup": {"inline_keyboard": []},
            },
        )
    except Exception:
        log.exception("Could not remove Telegram buttons")

    if action == "cancel":
        pipeline.update_approval_status(run_id, "cancelled")
        send_publish_status(chat_id, "❌ Video yayını iptal edildi.")
        return

    if action == "regen":
        pipeline.update_approval_status(run_id, "regenerating")
        send_publish_status(chat_id, "🔄 Yeni video üretimi başlatılıyor...")

        def regenerate() -> None:
            try:
                pipeline.run()
            except Exception as exc:
                log.exception("Regeneration failed")
                send_publish_status(chat_id, f"❌ Yeniden üretim başarısız: {type(exc).__name__}: {exc}")

        threading.Thread(target=regenerate, daemon=True).start()
        return

    if action == "publish":
        pipeline.update_approval_status(run_id, "uploading")

        def publish() -> None:
            try:
                video_path = Path(str(state.get("video_path", "")))
                if not video_path.exists():
                    raise FileNotFoundError(f"Video file not found: {video_path}")

                result = youtube_uploader.upload_shorts_video(
                    video_path=video_path,
                    topic=state.get("topic"),
                    full_text=state.get("full_text", ""),
                )

                youtube_url = result.get("url", "")
                video_id = result.get("video_id", "")
                pipeline.update_approval_status(
                    run_id,
                    "published",
                    extra={
                        "youtube_video_id": video_id,
                        "youtube_url": youtube_url,
                    },
                )
                send_publish_status(
                    chat_id,
                    "🎉 YouTube Shorts'a yükleme tamamlandı!\n\n"
                    f"🔗 {youtube_url}\n"
                    f"🆔 Video ID: {video_id}",
                )
            except Exception as exc:
                log.exception("YouTube upload failed")
                pipeline.update_approval_status(
                    run_id,
                    "upload_failed",
                    extra={"upload_error": str(exc)},
                )
                send_publish_status(
                    chat_id,
                    f"❌ YouTube yüklemesi başarısız oldu.\n\n{type(exc).__name__}: {exc}",
                )

        threading.Thread(target=publish, daemon=True).start()
        return


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
            supplied = self.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if not secrets.compare_digest(supplied, TELEGRAM_WEBHOOK_SECRET):
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

        if parsed.path == "/control/publish":
            supplied = self.headers.get("X-Control-Secret", "")
            if not CONTROL_API_SECRET or not secrets.compare_digest(
                supplied, CONTROL_API_SECRET
            ):
                self._json({"error": "unauthorized"}, 401)
                return

            try:
                payload = json.loads(body.decode("utf-8"))
                self._json({"ok": True, "payload": payload})
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
            return

        self._json({"error": "not_found"}, 404)

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    public_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()
    if public_url:
        register_webhook(public_url)
    else:
        log.warning(
            "RENDER_EXTERNAL_URL is not available yet; webhook can be registered after deployment."
        )

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    log.info("HTTP server listening on 0.0.0.0:%s", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
