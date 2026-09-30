# Architecture

The bot has one production path with two media modes.

## Flow

1. Content discovery — content_engine.py selects content type/candidate; event_memory.py verifies the event and prevents reuse.
2. Script — pipeline.py creates the 7-scene Turkish script; prompt_engine.py creates deterministic visual prompts.
3. Media — MEDIA_MODE=image uses Nano Banana 2 -> Nano Banana 2 Lite -> MoviePy. MEDIA_MODE=video uses Gemini Omni Flash -> Veo 3.1 Fast -> Veo 3.1 Lite -> Veo 3.1; failed video scenes fall back to image rendering.
4. Audio/render — Edge TTS narration, local music/SFX, MoviePy final 1080x1920 MP4 and deterministic QA.
5. Approval — Telegram preview plus Cloudflare Worker state and buttons.
6. Publication — publish_short.yml downloads the exact artifact and publishes to YouTube; TikTok is optional in the same workflow.

## Production files

- pipeline.py — orchestration, rendering, Telegram handoff
- content_engine.py — content types and candidate selection
- event_memory.py — event identity, research, visual-source resolution
- content_memory.py — content/audio/visual reuse memory
- ai_provider.py — text-model failover
- image_provider.py — Nano Banana image generation
- video_provider.py — Gemini Omni/Veo video generation
- prompt_engine.py — deterministic visual prompts
- visual_qc.py / visual_telemetry.py — media QA and diagnostics
- scene_motion.py / scene_transitions.py — image-mode motion and transitions
- media_provider.py — real-media search
- youtube_uploader.py — YouTube publishing
- tiktok_uploader.py — TikTok publishing
- telegram_control/ — Cloudflare Worker for Telegram approval state and OAuth

## Workflows

Only three workflows are intended to remain:

- .github/workflows/daily_short.yml — scheduled/manual generation
- .github/workflows/publish_short.yml — approved publication
- .github/workflows/ci.yml — syntax, imports, and tests

## Required Google credential

`GEMINI_API_KEY` is shared by text generation, Nano Banana, Gemini Omni Flash and Veo.

Current model IDs:

- `gemini-3.1-flash-image` — Nano Banana 2
- `gemini-3.1-flash-lite-image` — Nano Banana 2 Lite
- `gemini-omni-1.1-flash` — Gemini Omni Flash
- `veo-3.1-generate-preview`
- `veo-3.1-fast-generate-preview`
- `veo-3.1-lite-generate-preview`

The repository intentionally does not use retired Imagen or Veo 3.0 models.