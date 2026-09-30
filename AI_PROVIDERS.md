# AI provider configuration

The pipeline uses a provider chain instead of binding content generation to one vendor.

## Recommended order

AI_PROVIDER_ORDER=gemini,openrouter,groq,openai

Providers without an API key are skipped automatically.

## Environment variables

    AI_PROVIDER_ORDER=gemini,openrouter,groq,openai

    GEMINI_API_KEY=
    GEMINI_MODEL=

    OPENROUTER_API_KEY=
    OPENROUTER_MODEL=openai/gpt-4.1-mini
    OPENROUTER_SITE_URL=https://github.com/emrtasknn/youtube-shorts-bot
    OPENROUTER_APP_NAME=Historical Shorts Bot

    GROQ_API_KEY=
    GROQ_MODEL=llama-3.3-70b-versatile

    OPENAI_API_KEY=
    OPENAI_MODEL=gpt-4.1-mini

    AI_MAX_ATTEMPTS_PER_PROVIDER=2
    AI_PROVIDER_COOLDOWN_SECONDS=300
    AI_HTTP_TIMEOUT_SECONDS=90

## Failure behavior

1. Try providers in AI_PROVIDER_ORDER.
2. Retry transient errors such as 429/503/timeouts with bounded exponential backoff.
3. Temporarily cool down a repeatedly failing provider.
4. Fail over to the next configured provider.
5. Never expose API keys in logs.
6. If every configured provider fails, abort the current job rather than producing partial/corrupt content.

## First production test

Keep AUTO_PUBLISH=false and run one Short manually with batch_count=1.
Only after Telegram preview, video playback, metadata, and the YouTube OAuth upload are verified should the 10-video daily batch be enabled.
