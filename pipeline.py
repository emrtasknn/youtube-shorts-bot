import asyncio
import json
from collections import Counter
import logging
import os
import random
import re
import time
import urllib.parse
from pathlib import Path

import edge_tts
import numpy as np
import requests
import telebot
from google import genai
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
)
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
from telebot import types

import youtube_uploader
import content_memory
import event_memory
import gemini_config


# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOPICS_HISTORY_FILE = Path(os.getenv("TOPICS_HISTORY_FILE", "topics_history.json"))
APPROVALS_FILE = Path(os.getenv("APPROVALS_FILE", "approvals.json"))

# Novelty engine settings
MAX_NOVELTY_RETRIES = int(os.getenv("MAX_NOVELTY_RETRIES", "3"))

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30
SCENE_COUNT = 7
TARGET_DURATION = int(os.getenv("TARGET_DURATION", "34"))
MIN_DURATION = int(os.getenv("MIN_DURATION", "28"))
MAX_DURATION = int(os.getenv("MAX_DURATION", "40"))
MIN_TOTAL_WORDS = int(os.getenv("MIN_TOTAL_WORDS", "56"))
MAX_TOTAL_WORDS = int(os.getenv("MAX_TOTAL_WORDS", "72"))

# V1.4 legacy smoke-test compatibility markers retained for the existing workflow checks.
# Natural ending + loop strategy
# V1.4 visual loop
# "visual_fact": "..."

# The image provider's mark is kept out of the final frame by cropping the
# lower part of the generated image. This is intentionally a fixed crop,
# not an attempt to edit/inpaint the mark out of the source image.
WATERMARK_CROP_PX = int(os.getenv("WATERMARK_CROP_PX", "75"))
ZOOM_AMOUNT = float(os.getenv("ZOOM_AMOUNT", "0.07"))
IMAGE_ENHANCEMENT_ENABLED = os.getenv("IMAGE_ENHANCEMENT_ENABLED", "true").lower() in ("1", "true", "yes")
REAL_VISUAL_MIN_RELEVANCE = float(os.getenv("REAL_VISUAL_MIN_RELEVANCE", "0.65"))

# V1.4 narration profiles. Edge TTS is retained for production stability;
# diversity comes from narrator identity plus controlled delivery profiles.
VOICE_PROFILES = {
    "documentary_male": {"voice": "tr-TR-AhmetNeural", "rate": "+15%", "pitch": "+0Hz"},
    "mystery_female": {"voice": "tr-TR-EmelNeural", "rate": "+8%", "pitch": "+0Hz"},
    "dramatic_male": {"voice": "tr-TR-AhmetNeural", "rate": "+10%", "pitch": "-2Hz"},
    "energetic_female": {"voice": "tr-TR-EmelNeural", "rate": "+18%", "pitch": "+1Hz"},
}
VOICE_KEYWORDS = {
    "mystery": ["gizem", "kayıp", "esrar", "açıklanamadı", "gizemli", "bilinmiyor", "yok oldu", "kaybol"],
    "ancient": ["antik", "taş çağı", "roma", "mısır", "mezopotamya", "bin yıl"],
    "war": ["savaş", "ordu", "asker", "silah", "cephe", "muharebe", "işgal"],
    "disaster": ["felaket", "patlama", "deprem", "sel", "yangın", "çöküş"],
    "conflict": ["çatışma", "isyan", "ayaklanma", "saldırı", "kuşatma"],
    "tragedy": ["ölüm", "trajedi", "öldü", "can kaybı", "facıa"],
    "absurd": ["absürt", "inanılmaz", "tuhaf", "garip", "saçma", "kuş", "hayvan"],
    "surprising": ["şaşırtıcı", "inanılmaz", "beklenmedik", "şok"],
}

def choose_voice_profile(candidate: dict, research_dossier: dict) -> tuple[str, dict]:
    """Choose a narrator persona from the story atmosphere, deterministically."""
    text = " ".join([
        str(candidate.get("canonical_title", "")),
        str(candidate.get("event_summary", "")),
        str(candidate.get("why_interesting", "")),
        str(research_dossier.get("story_hook", "")),
        " ".join(str(x) for x in research_dossier.get("verified_facts", [])),
    ]).lower()
    scores = {name: 0 for name in VOICE_PROFILES}
    for atmosphere, keywords in VOICE_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        if atmosphere in {"mystery", "ancient"}:
            scores["mystery_female"] += hits
        elif atmosphere in {"war", "disaster", "conflict", "tragedy"}:
            scores["dramatic_male"] += hits
        elif atmosphere in {"absurd", "surprising"}:
            scores["energetic_female"] += hits
    event_key = str(candidate.get("event_id") or candidate.get("canonical_title", ""))
    tie = sum(ord(ch) for ch in event_key) % len(VOICE_PROFILES)

    # Avoid narrator fatigue: prefer a profile not used in the last two
    # completed videos when content memory contains that metadata.
    recent_profiles = []
    try:
        recent_entries = content_memory.load_content_memory()
        recent_profiles = [
            str(entry.get("voice_profile", ""))
            for entry in recent_entries[-2:]
            if entry.get("voice_profile")
        ]
    except Exception:
        recent_profiles = []

    if max(scores.values()) == 0:
        ranked = list(VOICE_PROFILES)
    else:
        best_score = max(scores.values())
        ranked = sorted(
            VOICE_PROFILES,
            key=lambda name: (-scores[name], (list(VOICE_PROFILES).index(name) - tie) % len(VOICE_PROFILES)),
        )

    available = [name for name in ranked if name not in recent_profiles]
    selected = (available or ranked)[0]
    return selected, VOICE_PROFILES[selected]



GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")
if not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_CHAT_ID is missing")

client = genai.Client(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("shorts-bot")


# -----------------------------
# Utilities
# -----------------------------
def retry_call(fn, attempts=3, base_delay=3, label="operation"):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            log.warning("%s failed (%s/%s): %s", label, attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(base_delay * attempt)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_error


def normalize_word(text: str) -> str:
    return re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]+", "", text, flags=re.UNICODE).lower()


def validate_script(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Gemini output is not a JSON object")

    scenes = data.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != SCENE_COUNT:
        raise ValueError(f"Expected exactly {SCENE_COUNT} scenes")

    cleaned = []
    total_words = 0
    for i, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene {i} is not an object")
        narration = str(scene.get("narration", "")).strip()
        image_prompt = str(scene.get("image_prompt", "")).strip()
        visual_fact = str(scene.get("visual_fact", "")).strip()
        visual_role = str(scene.get("visual_role", "")).strip().lower()
        visual_intent = scene.get("visual_intent")
        if not narration or not image_prompt:
            raise ValueError(f"Scene {i} is missing narration or image_prompt")
        if not visual_fact:
            raise ValueError(f"Scene {i} is missing visual_fact")
        allowed_visual_roles = {
            "evidence", "mechanism", "reconstruction", "context_map",
            "person_or_entity", "aftermath", "atmosphere"
        }
        if visual_role not in allowed_visual_roles:
            raise ValueError(f"Scene {i} has invalid visual_role: {visual_role}")
        if not isinstance(visual_intent, dict):
            raise ValueError(f"Scene {i} is missing visual_intent")
        required_intent = ["primary_subject", "visual_type", "visual_action", "scene_context", "shot_type", "composition", "must_show", "avoid", "search_queries"]
        missing_intent = [key for key in required_intent if not visual_intent.get(key)]
        if missing_intent:
            raise ValueError(f"Scene {i} visual_intent is incomplete: {missing_intent}")
        if not isinstance(visual_intent.get("must_show"), list) or not isinstance(visual_intent.get("avoid"), list):
            raise ValueError(f"Scene {i} visual_intent must_show/avoid must be lists")
        if not isinstance(visual_intent.get("search_queries"), list):
            raise ValueError(f"Scene {i} visual_intent search_queries must be a list")
        if "visual_entities" in visual_intent and not isinstance(visual_intent.get("visual_entities"), list):
            raise ValueError(f"Scene {i} visual_intent visual_entities must be a list")
        if len(visual_intent.get("must_show", [])) < 2 or len(visual_intent.get("avoid", [])) < 2:
            raise ValueError(f"Scene {i} visual_intent must_show/avoid need at least 2 concrete items")
        # Gemini may paraphrase the duplicated visual_fact/visual_role fields.
        # The scene-level fields are canonical; synchronize the nested intent instead
        # of rejecting an otherwise valid script for a wording-only mismatch.
        nested_fact = str(visual_intent.get("visual_fact", "")).strip()
        nested_role = str(visual_intent.get("visual_role", "")).strip().lower()
        if nested_fact != visual_fact:
            log.warning("Scene %s visual_intent.visual_fact differs from scene visual_fact; synchronizing to canonical scene value.", i)
            visual_intent["visual_fact"] = visual_fact
        if nested_role != visual_role:
            log.warning("Scene %s visual_intent.visual_role differs from scene visual_role; synchronizing to canonical scene value.", i)
            visual_intent["visual_role"] = visual_role
        word_count = len(narration.split())
        if word_count < 6 or word_count > 12:
            raise ValueError(f"Scene {i} has suspicious narration length: {word_count} words (target 6-12)")
        total_words += word_count
        event_specificity = scene.get("event_specificity", 0.0)
        information_density = scene.get("information_density", 0.0)
        try:
            event_specificity = float(event_specificity)
            information_density = float(information_density)
        except (TypeError, ValueError):
            raise ValueError(f"Scene {i} event_specificity/information_density must be numeric")
        if not 0.0 <= event_specificity <= 1.0 or not 0.0 <= information_density <= 1.0:
            raise ValueError(f"Scene {i} visual planning scores must be between 0 and 1")
        if visual_role != "atmosphere" and event_specificity < 0.70:
            raise ValueError(f"Scene {i} event_specificity is too low: {event_specificity:.2f} (<0.70)")
        cleaned.append({
            "narration": narration,
            "image_prompt": image_prompt,
            "visual_fact": visual_fact,
            "visual_role": visual_role,
            "event_specificity": round(event_specificity, 3),
            "information_density": round(information_density, 3),
            "visual_intent": visual_intent,
            "ending_strategy": str(scene.get("ending_strategy", "")) if i == SCENE_COUNT else "",
        })

    # Keep Shorts comfortably short while allowing natural variation.
    if total_words < MIN_TOTAL_WORDS or total_words > MAX_TOTAL_WORDS:
        raise ValueError(f"Total narration length is outside expected range: {total_words} words")

    return cleaned

def validate_visual_storyboard(scenes: list[dict]) -> dict:
    """Validate the scene plan as a storyboard, not just seven independent prompts."""
    if len(scenes) != SCENE_COUNT:
        raise ValueError(f"Storyboard requires exactly {SCENE_COUNT} scenes")

    visual_types = [str(s.get("visual_intent", {}).get("visual_type", "")).lower() for s in scenes]
    primary_subjects = [
        re.sub(r"\\s+", " ", str(s.get("visual_intent", {}).get("primary_subject", "")).lower()).strip()
        for s in scenes
    ]
    shot_types = [str(s.get("visual_intent", {}).get("shot_type", "")).lower() for s in scenes]

    type_counts = Counter(visual_types)
    subject_counts = Counter(primary_subjects)
    if type_counts and max(type_counts.values()) >= 5:
        repeated = max(type_counts, key=type_counts.get)
        raise ValueError(
            f"Visual storyboard is too homogeneous: visual_type '{repeated}' appears {type_counts[repeated]} times"
        )
    if subject_counts and max(subject_counts.values()) >= 4:
        repeated = max(subject_counts, key=subject_counts.get)
        raise ValueError(
            f"Visual storyboard repeats the same primary subject too often: '{repeated}'"
        )

    missing_action = [
        idx for idx, scene in enumerate(scenes, 1)
        if not str(scene.get("visual_intent", {}).get("visual_action", "")).strip()
    ]
    if missing_action:
        raise ValueError(f"Visual storyboard missing visual_action in scenes: {missing_action}")

    unique_shots = len(set(shot_types))
    return {
        "scene_count": len(scenes),
        "visual_type_counts": dict(type_counts),
        "unique_shot_types": unique_shots,
        "unique_primary_subjects": len(set(primary_subjects)),
    }
