# Real Media Provider Router

The visual engine now separates real-media search from AI image generation.

## Flow

1. Wikimedia Commons
2. Openverse
3. Pexels
4. Pixabay
5. Unsplash
6. AI image provider router

Pexels, Pixabay, and Unsplash are searched only when earlier historical/public-domain sources do not return a sufficiently relevant result.

## Environment

- PEXELS_API_KEY
- PIXABAY_API_KEY
- UNSPLASH_ACCESS_KEY
- REAL_MEDIA_PROVIDER_ORDER (default: pexels,pixabay,unsplash)
- REAL_MEDIA_MIN_SEARCH_SCORE (default: 0.35)
- MEDIA_PROVIDER_TIMEOUT_SECONDS (default: 15)

## Attribution

The pipeline preserves provider, author, title, and source page URL in scene metadata. These credits are also added to the YouTube description when a video is published.

Unsplash download tracking is registered through the API download endpoint before the returned hotlinked image URL is used.

## Scope

The first implementation uses still images. Pexels and Pixabay expose video APIs as well, but video retrieval is intentionally a separate phase so it does not complicate the current scene renderer and QC path.
