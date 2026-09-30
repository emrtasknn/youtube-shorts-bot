# Bir Garip Tarih — YouTube Shorts Bot

Üretim odaklı, insan onaylı tarih Shorts pipeline'ı. İçerik araştırması, 7 sahneli Türkçe script, Google üretken medya, seslendirme, MoviePy render, Telegram onayı ve YouTube/TikTok yayınını tek akışta yönetir.

## Akış

```text
Event discovery + research
        ↓
7-scene Turkish script
        ↓
Visual / video prompt
        ↓
Nano Banana 2 / Omni / Veo
        ↓
1080x1920 final render
        ↓
Telegram approval
        ↓
YouTube + optional TikTok
```

## Media modes

`MEDIA_MODE=image`

Nano Banana 2 → Nano Banana 2 Lite → MoviePy Ken Burns.

`MEDIA_MODE=video`

Gemini Omni Flash → Veo 3.1 Fast → Veo 3.1 Lite → Veo 3.1 → scene-level image fallback.

Google Gemini API model IDs used by the project:

- Nano Banana 2: `gemini-3.1-flash-image`
- Nano Banana 2 Lite: `gemini-3.1-flash-lite-image`
- Gemini Omni Flash: `gemini-omni-1.1-flash`
- Veo 3.1: `veo-3.1-generate-preview`
- Veo 3.1 Fast: `veo-3.1-fast-generate-preview`
- Veo 3.1 Lite: `veo-3.1-lite-generate-preview`

All four Google media families use the same `GEMINI_API_KEY`.

## Local setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python pipeline.py
```

For tests:

```bash
pytest -q
```

## Environment

Required:

- `GEMINI_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Publishing/control:

- `YOUTUBE_TOKEN_JSON`
- `YOUTUBE_CLIENT_SECRETS_JSON`
- `CONTROL_API_URL`
- `CONTROL_API_SECRET`
- `AUTO_PUBLISH=false`

Media:

- `MEDIA_MODE=image` or `video`
- `GEMINI_IMAGE_MODEL_ORDER`
- `VIDEO_PROVIDER_ORDER`
- `GEMINI_OMNI_MODEL`
- `GEMINI_VEO31_MODEL`
- `GEMINI_VEO31_FAST_MODEL`
- `GEMINI_VEO31_LITE_MODEL`

## Daily content plan

`4 TREND_HISTORY + 2 TODAY_IN_HISTORY + 2 AYT_HISTORY + 1 HISTORY_FACT + 1 CUSTOM`.

For a single manual video, set `batch_count=1` and select the desired `content_type`.

`TODAY_IN_HISTORY` is designed for 45–75 second mini-stories.

## Production workflows

- `daily_short.yml` — scheduled generation at 05:00 UTC / 08:00 Türkiye, plus manual runs.
- `publish_short.yml` — publishes the exact approved artifact to YouTube and optionally TikTok.
- `ci.yml` — compiles production modules, imports core modules and runs the test suite.

## Project structure

See `ARCHITECTURE.md` for the current production design. The repository deliberately keeps provider-specific logic out of `pipeline.py`.