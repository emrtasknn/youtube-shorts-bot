# ShortBot — Overnight Development Report

Date: 2026-09-28
Repository: emrtasknn/youtube-shorts-bot

## 1. Run #58 incident

Run #58 was canceled after investigation.

- Workflow: Daily Shorts Generator
- Run: #58
- The job was stuck in **Generate Short**.
- It never reached artifact upload or YouTube publishing.
- The Telegram error was:
  `GitHub artifact 'shorts-run-58' not found`
- Root cause: the publish workflow attempted to download the source artifact while the generation run was still running.
- Fix committed: publish workflow now waits for the source run to finish successfully and verifies that the expected artifact exists before downloading it.

## 2. Daily content portfolio implemented

The new daily plan is:

1. 4 × TREND_HISTORY
2. 2 × TODAY_IN_HISTORY
   - primary candidate
   - alternative candidate
3. 2 × AYT_HISTORY
4. 1 × HISTORY_FACT
5. 1 × CUSTOM

Total: **10 videos**.

The routing is implemented in `content_engine.py` and integrated into `pipeline.py`.

## 3. Series duration rules

Content types now have independent targets:

- TREND_HISTORY: 28–40s, 56–72 words
- TODAY_IN_HISTORY: 45–75s, 90–145 words
- AYT_HISTORY: 25–45s, 55–90 words
- HISTORY_FACT: 30–55s, 65–105 words
- CUSTOM: 30–65s, 65–115 words

TARİHTE BUGÜN is intentionally designed as the longer mini-documentary format.

YouTube currently classifies vertical videos up to 3 minutes as Shorts for standard channels, so a 45–75 second Tarihte Bugün format is compatible with the current platform rules.

## 4. Telegram

Telegram cards now carry the content-series label.

Examples:
- 🔥 Trend History
- 📅 Tarihte Bugün
- 🎓 AYT Tarih
- 🧠 History Fact
- 🧪 Custom

New command routing was added to the Telegram control worker:

- `/custom <fikir>`
- `/ayt`
- `/today`
- `/generate`

The custom command starts a single CUSTOM generation using the supplied idea.

## 5. YouTube public publishing test

The public test has **not yet been completed**.

The publish workflow is configured to request:
`privacyStatus = public`

After upload, the code reads the actual YouTube privacy status back from the API. If YouTube stores a different status, the publish is treated as failed instead of falsely reporting success.

The previous test did not reach YouTube because the source artifact was missing.

Important current platform constraint:
YouTube states that uploads through `videos.insert` from unverified API projects created after 28 July 2020 are restricted to private viewing until the API project passes the required audit. Therefore the next real upload is the decisive test.

## 6. Playlist automation

Content-type → playlist routing is implemented.

Environment variables expected by the publish workflow:

- `YOUTUBE_PLAYLIST_TREND_HISTORY`
- `YOUTUBE_PLAYLIST_TODAY_IN_HISTORY`
- `YOUTUBE_PLAYLIST_AYT_HISTORY`
- `YOUTUBE_PLAYLIST_HISTORY_FACT`
- `YOUTUBE_PLAYLIST_CUSTOM`

Playlist insertion is non-fatal: a successful YouTube upload will not be marked as failed solely because playlist insertion fails.

### Remaining manual step

YouTube's `playlistItems.insert` requires a broader OAuth scope than the current upload-only scope. The OAuth authenticator has therefore been updated to request the `youtube` scope as well.

A fresh OAuth token/refresh token with the broader scope still needs to be generated and placed into the GitHub/Render credential configuration before playlist insertion can be fully activated.

## 7. Quality / safety improvements

- Added lightweight Python syntax CI workflow.
- Added per-content-type duration QA.
- Made YouTube titles series-aware.
- Added content type to metadata and Telegram approval state.
- Publish result now reports playlist status back to Telegram.
- Existing YouTube title Turkish-language guard remains active.

## 8. What is intentionally NOT claimed as completed

The following have not been falsely marked as successful:

- Public YouTube upload permission — **pending real upload test**
- Playlist insertion — **code ready; OAuth scope + playlist IDs still required**
- Full 10-video production batch — **not run after the new routing changes**
- Telegram Cloudflare worker deployment — **code committed; deployment status not independently verified from this session**

## 9. Next validation sequence

1. Run a single content generation after the new routing is live.
2. Confirm Telegram receives the correct content-series label.
3. Approve one video.
4. Confirm artifact waiting logic.
5. Confirm YouTube upload.
6. Read back actual privacy status.
7. If actual status is public: keep direct-public publishing.
8. If actual status is private: treat the API-project audit restriction as the blocker and keep the existing safe workflow.
9. Complete OAuth reauthorization for playlist management.
10. Add the five playlist IDs.
11. Run the full 10-video daily portfolio.

## Current status

**Code architecture: implemented.**

**Public upload: test pending.**

**Playlists: implementation ready, credentials/config pending.**

**Full daily 10-video run: next validation step.**
