# V2 Implementation Plan — Historical Event Discovery, Novelty & Visual Sourcing

## 0. Mission

The goal of this version is to change the bot from a system that repeatedly generates popular historical topics into a **continuous historical-event discovery engine**.

The desired behavior is:

> Discover genuinely interesting, mysterious, unusual or curiosity-inducing historical events → verify/research them → reject events already used → generate a unique script → find real historical visuals where possible → use web visuals when appropriate → use AI reconstruction only when suitable real visuals cannot be found → render with consistent atmospheric audio.

The most important rule is:

# ONE HISTORICAL EVENT = ONE VIDEO

A different angle does NOT make an already-used historical event eligible again.

Example:

```text
Mary Celeste
├── crew disappearance        ❌
├── lifeboat disappearance   ❌
├── possible explanations    ❌
├── discovery of the ship    ❌
└── another Mary Celeste video ❌
```

Once the Mary Celeste event has been used, the system should move on to a different historical event.

---

# 1. Current System Diagnosis

The current repository already contains useful foundations:

- `topics_history.json`
- `content_memory.json`
- topic/angle extraction
- Gemini novelty evaluation
- ACCEPT / REVISE / REJECT
- mood/energy/tension extraction
- content-aware audio selection
- visual prompt reuse checking

However, the current architecture still has several fundamental limitations.

## 1.1 Topic discovery is too narrow

The current discovery process asks Gemini for only a small number of historical topics and relies heavily on Gemini's general knowledge.

This causes repeated selection of famous stories.

## 1.2 Fallback stories can cause repetition

The current pipeline contains a hardcoded fallback pool:

```python
FALLBACK_STORIES = [
    "Kayıp Koloni Roanoke Gizemi",
    "Hayalet Gemi Mary Celeste",
    "Göbeklitepe'nin Taş Çağı Sırrı",
]
```

This must no longer be a normal recovery path for topic generation.

A failed discovery request must NOT silently produce another video about one of these already-used events.

## 1.3 Current memory is content-oriented, not event-oriented

`content_memory.py` stores:

- topic
- angle
- summary
- main claim
- key facts
- entities
- script

This is useful, but it does not establish a canonical identity for the underlying historical event.

## 1.4 Current novelty logic allows the same event with a different angle

The current novelty rules explicitly allow:

> same broad topic + different informational angle = ACCEPT

That behavior is NOT desired for the new system.

For this project:

```text
same historical event = reject
```

regardless of angle.

## 1.5 Novelty checking currently considers only recent memory

The current implementation formats only the most recent subset of memory for comparison.

The event history must be persistent and searchable across the entire lifetime of the project.

## 1.6 Novelty failure currently defaults to ACCEPT

The current novelty checker accepts a candidate when all Gemini analysis models fail.

This is unsafe for duplicate prevention.

The new behavior must be:

```text
novelty check unavailable
        ↓
UNCERTAIN
        ↓
retry / reject candidate
```

Never:

```text
novelty check unavailable → ACCEPT
```

## 1.7 Visual sourcing does not follow the desired priority

The current pipeline primarily generates AI images.

The desired pipeline is:

```text
Wikimedia/Wikipedia historical visual
        ↓ if unavailable
Web visual from an appropriate source
        ↓ if unavailable / unsuitable
AI-generated reconstruction
```

---

# 2. Target Architecture

The new pipeline should conceptually become:

```text
                HISTORICAL DISCOVERY
                        ↓
               15–30 candidate events
                        ↓
                EVENT DEDUPLICATION
                        ↓
          REMOVE PREVIOUSLY USED EVENTS
                        ↓
              RESEARCH / VERIFICATION
                        ↓
             INTEREST + CURIOSITY SCORE
                        ↓
                 SELECT EVENT
                        ↓
              SCRIPT GENERATION
                        ↓
          SCRIPT FACTUAL / NOVELTY QA
                        ↓
                 SCENE PLANNING
                        ↓
             VISUAL SOURCE RESOLUTION
                        ↓
       ┌────────────────┼─────────────────┐
       ↓                ↓                 ↓
 Wikimedia/       Appropriate web       AI
 Wikipedia          visual            reconstruction
       └────────────────┼─────────────────┘
                        ↓
                 AUDIO PLAN
                        ↓
             AUDIO MIX / DUCKING
                        ↓
                     RENDER
                        ↓
               SAVE EVENT MEMORY
                        ↓
                  PUBLISH / APPROVAL
```

The key architectural change is:

> **Discovery and event identity happen BEFORE expensive script/image/audio generation.**

Do not generate a complete video and only then discover that the underlying event is a duplicate.

---

# 3. Phase 0 — Repository Audit Before Modification

Before implementing anything:

1. Inspect:
   - `pipeline.py`
   - `content_memory.py`
   - `topics_history.json`
   - `approvals.json`
   - `output/*/metadata.json`
   - existing image utilities
   - existing web search utilities
   - existing Wikipedia/Wikimedia utilities, if any
   - existing audio utilities
2. Identify all existing topic discovery and fallback paths.
3. Identify all places where a topic is selected.
4. Identify all places where a script is generated.
5. Identify all places where images are selected/generated.
6. Identify the exact persistence mechanism already used.
7. Do NOT rewrite working infrastructure unnecessarily.
8. Reuse existing Gemini/API utilities where appropriate.

After the audit, produce a short implementation report before making major architectural changes.

---

# 4. Phase 1 — Historical Event Memory

Create or extend a persistent event-level memory.

Do not replace working `content_memory.json` blindly. Extend it or create a compatible `event_memory.json` if that is cleaner for the current architecture.

Recommended structure:

```json
{
  "event_id": "evt_...",
  "canonical_title": "Mary Celeste",
  "aliases": [
    "Mary Celeste mystery",
    "Mary Celeste disappearance",
    "Mary Celeste abandonment"
  ],
  "date": "1872",
  "date_normalized": "1872",
  "location": "Atlantic Ocean",
  "entities": [
    "Mary Celeste",
    "Benjamin Briggs"
  ],
  "event_summary": "...",
  "core_facts": [],
  "claims": [],
  "sources": [],
  "first_video_id": "...",
  "created_at": "...",
  "status": "used"
}
```

## Important

The event memory must represent the **underlying historical event**, not just the title.

For example:

```text
"Mary Celeste'nin Gizemi"
"Mary Celeste Mürettebatı Nereye Kayboldu?"
"Hayalet Gemi Mary Celeste"

```

must resolve to the same `event_id`.

---

# 5. Phase 2 — Historical Event Discovery Engine

Replace the current narrow topic discovery behavior with a broader candidate discovery process.

## Target

Generate approximately:

```text
15–30 candidate historical events
```

per discovery cycle.

The candidates should be:

- historically documented
- unusual, mysterious, surprising or highly curiosity-inducing
- capable of supporting a short-form story
- sufficiently distinct from previous events
- researchable
- visually representable

## Discovery categories

The system should explore a broad range of historical phenomena, for example:

```text
unexplained disappearances
strange deaths
historical mysteries
lost expeditions
bizarre accidents
unusual disasters
forgotten events
archaeological mysteries
historical coincidences
unusual inventions
strange traditions
mass phenomena
lost places
mysterious artifacts
unusual battles/incidents
unexpected survival stories
historical frauds
unusual scientific events
```

Do not hardcode only these categories. They are discovery directions, not a fixed list.

## Avoid over-focusing on famous mysteries

The discovery engine should deliberately seek less commonly used events.

Do not repeatedly prioritize:

- Mary Celeste
- Roanoke
- Göbeklitepe
- Titanic
- Bermuda Triangle

or other highly saturated topics simply because they are famous.

Fame may contribute to curiosity, but it must NOT override event novelty.

---

# 6. Phase 3 — Discovery Must Produce Structured Candidates

Every candidate should be represented approximately as:

```json
{
  "canonical_title": "...",
  "aliases": [],
  "date": "...",
  "location": "...",
  "entities": [],
  "event_summary": "...",
  "why_interesting": "...",
  "known_facts": [],
  "source_urls": [],
  "visual_search_terms": []
}
```

The candidate must contain enough information for the system to identify whether it is a previously used event.

---

# 7. Phase 4 — Event Identity Matching

Before a candidate can be selected, compare it against the COMPLETE historical event memory.

Do not only compare title strings.

Use:

```text
canonical event name
aliases
date
location
entities
event description
core facts
semantic similarity
```

The system should determine:

```text
NEW EVENT
or
KNOWN EVENT
or
UNCERTAIN
```

## Example

Previous:

```text
Mary Celeste
1872
Atlantic Ocean
Benjamin Briggs
crew disappearance
```

Candidate:

```text
The abandoned Mary Celeste
1872
Atlantic Ocean
missing crew
```

Expected:

```text
KNOWN EVENT
```

Reject.

---

# 8. Phase 5 — Event-Level Hard Rejection

If a candidate matches an existing historical event with high confidence:

```text
REJECT
```

Do NOT allow:

```text
same event + different angle
```

to bypass the event-level duplicate filter.

The distinction is:

```text
Topic:
Ancient Rome

Event:
Great Fire of Rome
```

Ancient Rome can support many independent events.

But:

```text
Event:
Great Fire of Rome
```

should not produce multiple videos.

---

# 9. Phase 6 — Batch-Level Deduplication

Candidates generated in the SAME discovery batch must also be compared with each other.

Example:

```text
Candidate A:
Mary Celeste

Candidate B:
The abandoned ship Mary Celeste

Candidate C:
Mary Celeste crew disappearance
```

Only one underlying event should survive.

After every accepted candidate is added to temporary memory:

```text
batch_memory += accepted_event
```

The next candidate must be checked against:

```text
historical_event_memory
+
current_batch_memory
```

---

# 10. Phase 7 — Historical Research / Verification

Once an event survives novelty filtering, perform a dedicated research step before script generation.

Research should establish:

1. What happened?
2. When did it happen?
3. Where did it happen?
4. Who was involved?
5. Which facts are well documented?
6. Which details are disputed?
7. Which details are folklore/speculation?
8. What makes the event genuinely interesting?
9. What primary/secondary sources are available?
10. What real visuals may exist?

Return a structured research dossier.

Example:

```json
{
  "event_id": "...",
  "verified_facts": [],
  "disputed_claims": [],
  "possible_myths": [],
  "sources": [],
  "visual_sources": []
}
```

---

# 11. Phase 8 — Factuality Rules

The script generator must distinguish:

```text
verified fact
historical interpretation
disputed claim
legend / speculation
```

Do not present an uncertain theory as an established fact merely because it sounds more interesting.

If the mystery itself is unresolved, the script should preserve that uncertainty.

---

# 12. Phase 9 — Script Generation

Generate the script from the research dossier.

The script should:

- focus on one historical event
- have a strong curiosity hook
- introduce concrete facts
- avoid generic filler
- avoid repeating facts unnecessarily
- preserve uncertainty where appropriate
- avoid inventing details
- provide a satisfying progression
- end with an open question or memorable conclusion when appropriate

The script generator should receive the research dossier rather than relying only on its general knowledge.

---

# 13. Phase 10 — Script-Level QA

After script generation:

Run a QA step checking:

```text
event identity
fact consistency
research consistency
duplicate information
unsupported claims
```

If the script contains unsupported claims:

```text
REVISE
```

If it is fundamentally invalid:

```text
REJECT
```

Do not publish a script simply because generation succeeded.

---

# 14. Phase 11 — Visual Source Resolution

Every scene should have a visual source strategy.

Priority:

```text
1. Wikimedia / Wikipedia-compatible historical visual
2. Appropriate web visual
3. AI-generated reconstruction
```

The system must NOT use AI generation simply because it is convenient.

## 14.1 Wikimedia / Wikipedia

Search for:

- historical photographs
- portraits
- maps
- artifacts
- archaeological sites
- documents
- paintings
- diagrams
- public-domain historical imagery

Store source metadata where available.

Example:

```json
{
  "source_type": "wikimedia",
  "source_url": "...",
  "image_url": "...",
  "title": "...",
  "license": "...",
  "attribution": "..."
}
```

## 14.2 Web image

If a suitable Wikimedia/Wikipedia image does not exist, search the web.

Prefer reputable and contextually appropriate sources.

Do not blindly scrape arbitrary image results.

Record:

```text
source URL
image URL
source name
```

when available.

## 14.3 AI reconstruction

Use AI only when:

- no suitable real image exists
- a historical scene needs visual reconstruction
- available images are inappropriate for the specific scene
- a conceptual scene is required

AI images must be treated as reconstructions, not authentic historical photographs.

---

# 15. Phase 12 — Scene-Level Visual Diversity

Existing visual reuse checking is useful, but prompt word-overlap alone is insufficient.

Current implementation uses Jaccard similarity on normalized prompts.

Keep this as a lightweight guard, but improve the system if practical.

At minimum:

- prevent exact duplicate prompts
- prevent exact duplicate image assets
- avoid consecutive visually identical scenes
- preserve intentional visual continuity where appropriate

Do NOT simply append random phrases like:

```text
different camera angle
different lighting
```

to an otherwise identical prompt and consider the problem solved.

The scene should have a genuinely different visual purpose.

---

# 16. Phase 13 — Content Memory Migration

The project already has:

```text
topics_history.json
content_memory.json
approvals.json
output/*/metadata.json
```

Use existing data to populate historical event memory where possible.

The migration process should:

1. Inspect previous generated videos.
2. Extract script/title/topic metadata.
3. Ask Gemini to identify the underlying historical event.
4. Generate canonical event identity.
5. Deduplicate event aliases.
6. Store the historical events as already used.

This is essential.

The existing 10-ish generated videos must NOT be ignored.

If 7 videos represent only 3 underlying events, event memory should record those 3 events as used.

---

# 17. Phase 14 — Remove the Repetition-Causing Fallback

Do not allow:

```python
random.choice(FALLBACK_STORIES)
```

to create a normal production video.

Instead:

```text
Discovery failure
      ↓
retry discovery
      ↓
alternate discovery strategy
      ↓
if still unavailable
      ↓
abort generation safely
```

An aborted generation is preferable to producing another duplicate historical story.

A fallback pool may remain only as an explicit emergency test fixture, never as an automatic production content source.

---

# 18. Phase 15 — Novelty Failure Behavior

Current behavior:

```text
Gemini novelty API failure
        ↓
ACCEPT
```

must change.

New behavior:

```text
Novelty check failure
        ↓
retry
        ↓
if still unavailable
        ↓
UNCERTAIN
        ↓
do not generate expensive assets
```

Never treat an inability to check novelty as proof of novelty.

---

# 19. Phase 16 — Full Content Selection Scoring

After event deduplication and research, score remaining candidates using multiple dimensions:

```text
historical interest
curiosity
story potential
research quality
visual availability
novelty confidence
```

Do NOT allow popularity/viral score to override:

```text
event novelty
factual reliability
research quality
```

A highly viral duplicate must still be rejected.

---

# 20. Phase 17 — Audio System

Keep the existing content-aware audio selection architecture.

However, ensure:

```text
every generated video
=
narration
+
background atmosphere/music
```

unless silence is explicitly intentional.

The existing mood/energy/tension metadata can continue to drive selection.

---

# 21. Phase 18 — Fix Audio Ducking

The current project already has a `compute_ducking_volume()` helper.

Ensure it is actually used during final audio composition.

The desired behavior:

```text
Narration active
    ↓
background volume lower

Narration pause
    ↓
background volume rises slightly
```

Use smooth transitions.

Do not rely solely on a fixed:

```text
with_volume_scaled(0.18)
```

value.

A configurable base volume is acceptable, but final mixing should be narration-aware.

---

# 22. Phase 19 — Audio Validation

Before final render completion, validate:

```text
audio stream exists
narration exists
background layer exists
background is not effectively silent
narration remains intelligible
```

If background audio cannot be loaded, log an explicit warning/error.

Do not silently produce inconsistent narration-only videos.

---

# 23. Phase 20 — Logging

Add clear logs for the complete decision chain.

Example:

```text
DISCOVERY
Found 24 candidate historical events

EVENT FILTER
24 discovered
8 previously used
3 duplicate candidates within current batch
13 remaining

RESEARCH
Selected: London Beer Flood
Research confidence: high

VISUALS
Scene 1 → Wikimedia
Scene 2 → Wikimedia
Scene 3 → Web
Scene 4 → AI reconstruction

AUDIO
Mood: mysterious
Track: mystery_investigation.mp3
Base volume: 0.18
Ducking: enabled

FINAL
Event ID: evt_...
Status: completed
```

For rejected events:

```text
REJECTED EVENT
Candidate: Mary Celeste disappearance

Matched event:
Mary Celeste

Reason:
Historical event already used.

Confidence:
0.98
```

This will make future debugging much easier.

---

# 24. Acceptance Tests

## Test A — Existing event

Input:

```text
Mary Celeste
```

Expected:

```text
REJECT
```

if Mary Celeste already exists in historical event memory.

---

## Test B — Same event with different wording

Input:

```text
The abandoned ship found in the Atlantic in 1872
```

Expected:

```text
REJECT
```

if it maps to Mary Celeste.

---

## Test C — Same event, different angle

Input:

```text
What happened to the Mary Celeste lifeboat?
```

Expected:

```text
REJECT
```

Same underlying event.

---

## Test D — Completely different historical event

Input:

```text
London Beer Flood
```

Expected:

```text
ACCEPT
```

assuming it is not already in event memory.

---

## Test E — Duplicate candidates within one batch

Candidates:

```text
Mary Celeste
Mary Celeste disappearance
The abandoned Mary Celeste
```

Expected:

```text
1 event maximum
```

and if the event was previously used:

```text
0 accepted
```

---

## Test F — Discovery failure

Simulate Gemini discovery failure.

Expected:

```text
retry
→ alternate strategy
→ safe abort if still unavailable
```

NOT:

```text
Mary Celeste fallback
```

---

## Test G — Novelty API failure

Simulate novelty API failure.

Expected:

```text
UNCERTAIN / retry
```

NOT:

```text
ACCEPT
```

---

## Test H — Historical visual available

For an event with a suitable Wikimedia image:

Expected:

```text
Wikimedia image selected
AI generation skipped
```

---

## Test I — No historical visual available

Expected:

```text
Web visual search
```

and if no suitable web visual exists:

```text
AI reconstruction
```

---

## Test J — 20 candidate discovery cycle

Run discovery.

Expected:

```text
multiple distinct historical events
```

not repeated variations of:

```text
Mary Celeste
Roanoke
Göbeklitepe
```

---

# 25. Definition of Done

The implementation is complete only when:

### Discovery

- The system can discover a broad range of historical events.
- Discovery is not dependent on a fixed list of famous stories.
- Candidate generation returns many candidates before selection.

### Novelty

- Historical events have persistent identities.
- Previously used events are permanently rejected.
- Same event with different wording is rejected.
- Same event with a different angle is rejected.
- Current-batch duplicates are rejected.
- Novelty failure never defaults to ACCEPT.

### Research

- Selected events are researched before scripting.
- Facts and uncertain claims are separated.
- Scripts are generated from the research dossier.

### Visuals

- Real historical visuals are preferred.
- Wikimedia/Wikipedia is searched first where appropriate.
- Web visuals are the second option.
- AI generation is the final fallback.
- Visual sources are recorded.
- AI reconstructions are not represented as authentic historical images.

### Audio

- Background audio is consistently present.
- Track selection is content-aware.
- Narration-aware ducking actually affects the final mix.
- Background audio is audible without overpowering narration.

### Memory

- Existing generated videos are migrated into event memory.
- Event memory persists across executions.
- Event identity is independent of title wording.

---

# 26. Engineering Constraints

- Preserve existing working functionality.
- Avoid unnecessary rewrites.
- Reuse existing APIs/utilities where appropriate.
- Do not introduce unnecessary dependencies.
- Do not generate expensive images/audio for an event before it passes event-level novelty.
- Keep API calls observable through logs.
- Keep credentials out of source control.
- Keep `client_secrets.json`, `token.json`, `.env` and secrets protected by `.gitignore`.
- Do not expose credential values in logs.
- Do not silently fall back to duplicate content.
- Make new thresholds configurable through environment/config values.
- Keep the pipeline runnable after each phase.

---

# 27. Recommended Implementation Order

Implement incrementally.

```text
PHASE 0
Repository audit
        ↓
PHASE 1
Historical event memory
        ↓
PHASE 2
Historical content backfill
        ↓
PHASE 3
Broad event discovery
        ↓
PHASE 4
Event identity matching
        ↓
PHASE 5
Batch-level deduplication
        ↓
PHASE 6
Research / verification
        ↓
PHASE 7
Research-based script generation
        ↓
PHASE 8
Visual source resolver
        ↓
PHASE 9
Audio consistency + ducking
        ↓
PHASE 10
Logging + regression tests
```

Do not implement all phases blindly in one large rewrite.

After each phase:

1. run tests
2. inspect logs
3. verify existing behavior
4. commit the changes
5. proceed to the next phase

---

# 28. Final Product Behavior

The final bot should behave like this:

```text
"Find me an interesting historical mystery."

             ↓

Search / discover historical events

             ↓

Generate 15–30 candidates

             ↓

Identify underlying historical event

             ↓

Compare against ALL previously used events

             ↓

Remove duplicates

             ↓

Research remaining candidates

             ↓

Select one genuinely new event

             ↓

Build factual research dossier

             ↓

Generate unique script

             ↓

Validate script

             ↓

For every scene:
    Wikimedia/Wikipedia?
        ↓ yes → use real visual
        ↓ no
    Web visual?
        ↓ yes → use web visual
        ↓ no
    AI reconstruction

             ↓

Generate narration

             ↓

Select appropriate atmosphere

             ↓

Mix narration + background
with real ducking

             ↓

Render

             ↓

Save event identity to permanent memory

             ↓

Next run:
NEVER USE THAT EVENT AGAIN
```

## Core principle

The system should not ask:

> "Have I used this title before?"

It should ask:

> **"Have I already told this historical story before?"**

That is the central requirement of this version.
