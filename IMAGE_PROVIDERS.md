# Image Provider Architecture

The Shorts pipeline uses a centralized image-provider router in `image_provider.py`.
Provider-specific API code is isolated from `pipeline.py`.

## Default order

1. Google Imagen 3
2. Cloudflare Workers AI — FLUX.1 [schnell]
3. fal.ai — FLUX.1 [schnell]
4. Together AI — FLUX.1 [schnell]
5. DeepAI Text-to-Image
6. Pollinations turbo
7. Pollinations flux
8. Hugging Face

The order can be overridden with `IMAGE_PROVIDER_ORDER`.

## Credentials

Optional GitHub Actions repository secrets:

- `GEMINI_API_KEY`
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`
- `FAL_KEY`
- `TOGETHER_API_KEY`
- `DEEPAI_API_KEY`
- `HF_TOKEN`

Pollinations does not require a secret.

## Failure policy

- 1 attempt per provider by default.
- Retryable HTTP failures (408/409/425/429/5xx) can be retried by setting `IMAGE_MAX_ATTEMPTS_PER_PROVIDER`.
- Authentication/payment/quota-style failures (401/402/403) are cooled down rather than repeatedly retried.
- Default provider cooldown is 30 minutes.
- The router writes and validates an image before reporting success.
- If every provider fails, the pipeline raises a controlled image-generation error.
- The pipeline never enters enhancement/compositing with a missing image.

## Quality pipeline

```
Storyboard
   ↓
Real visual source resolution
   ├─ Wikimedia/Openverse → download → visual QC
   └─ AI reconstruction
          ↓
   Image Provider Router
          ↓
   Provider-specific generation
          ↓
   Image validation
          ↓
   Normalize to 1080×1920 JPEG
          ↓
   Visual QC
          ↓
   Source-aware enhancement
          ↓
   Motion / transition
          ↓
   Render
```

## Recommended initial production setup

Start with:

```
IMAGE_PROVIDER_ORDER=imagen,cloudflare,fal,together,deepai,pollinations_turbo,pollinations_flux,huggingface
IMAGE_MAX_ATTEMPTS_PER_PROVIDER=1
IMAGE_PROVIDER_COOLDOWN_SECONDS=1800
```

Configure at least Cloudflare and fal.ai before testing a full 7-scene Short. Together and DeepAI remain additional fallbacks.

Cloudflare's FLUX.1 [schnell] endpoint accepts a prompt and returns base64 image data. fal.ai's FLUX Schnell endpoint supports portrait output and synchronous responses. Together exposes FLUX Schnell through its image-generation API. DeepAI exposes a REST text-to-image endpoint with configurable dimensions.
