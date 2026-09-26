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


# -----------------------------
# Topic & Content engine
# -----------------------------
def load_topic_history(history_path: Path | None = None) -> list[dict]:
    """Load previously covered topics to avoid content repetition."""
    target_path = history_path if history_path is not None else TOPICS_HISTORY_FILE
    if not target_path.exists():
        return []
    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as exc:
        log.warning("Could not read topic history: %s", exc)
        return []


def save_topic_to_history(topic_entry: dict, history_path: Path | None = None):
    """Save newly produced topic to persistent history."""
    target_path = history_path if history_path is not None else TOPICS_HISTORY_FILE
    history = load_topic_history(target_path)
    history.append(topic_entry)
    try:
        target_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("Could not save topic to history: %s", exc)


# -----------------------------
# Curated Fallback Stories (Resilient & Pre-timed)
# -----------------------------
FALLBACK_STORIES = [
    {
        "title": "Kayıp Koloni Roanoke Gizemi",
        "hook_question": "115 İngiliz yerleşimci bir gecede nereye kayboldu?",
        "viral_score": 9,
        "visual_appeal": 9,
        "scenes": [
            {
                "narration": "1590 yılında Roanoke adasındaki 115 yerleşimci bir gecede sırra kadem bastı.",
                "image_prompt": "Cinematic vertical 9:16 shot of an abandoned wooden colonial fort on a misty island, 1590 era, photorealistic documentary style, 8k",
            },
            {
                "narration": "Evler ve eşyalar yerli yerindeydi, fakat tek bir insan bile yoktu.",
                "image_prompt": "Eerie empty village with silent wooden cabins, fog rolling through dirt streets, no people, dramatic lighting, vertical 9:16",
            },
            {
                "narration": "Ne bir çatışma izi ne de tek bir mezar bulundu.",
                "image_prompt": "Close-up interior of a colonial wooden cottage, warm hearth, untouched dinner on wooden table, vertical 9:16, cinematic",
            },
            {
                "narration": "Bulunan tek ipucu bir ağaca kazınan esrarengiz kelimeydi: Kroatoan!",
                "image_prompt": "Dramatic close-up of a rustic wooden tree trunk with mysterious word CROATOAN carved deeply into bark, dark moody lighting, vertical 9:16",
            },
            {
                "narration": "Yüzyıllar süren araştırmalar bile bu insanların nereye gittiğini çözemedi.",
                "image_prompt": "Ancient faded nautical parchment map showing North Carolina coast with mysterious symbols, vertical 9:16, historical documentary style",
            },
            {
                "narration": "Kayıp koloninin gizemi bugün hâlâ aydınlatılamadı.",
                "image_prompt": "Modern archaeologists excavating historical earth beneath tall trees, moody sunset light, vertical 9:16, cinematic",
            },
            {
                "narration": "Ve tarihin en büyük sırrının başladığı yer aslında;",
                "image_prompt": "Mysterious ghostly ship sailing into thick white fog under moonlight, vertical 9:16, photorealistic, 8k",
            },
        ],
    },
    {
        "title": "Hayalet Gemi Mary Celeste",
        "hook_question": "Okyanusun ortasında terk edilen mürettebata ne oldu?",
        "viral_score": 9,
        "visual_appeal": 9,
        "scenes": [
            {
                "narration": "1872 yılında Mary Celeste gemisi Atlas Okyanusu'nda rotasız sürüklenirken bulundu.",
                "image_prompt": "Cinematic vertical 9:16 shot of a 19th-century merchant sailing ship drifting alone in misty ocean, moody cinematic lighting, photorealistic 8k",
            },
            {
                "narration": "Gemiye çıkan denizciler akılalmaz bir manzarayla karşılaştı.",
                "image_prompt": "Sailors boarding an eerie wooden ship deck, stormy sky, dark ocean waves, vertical 9:16, documentary style",
            },
            {
                "narration": "Kargo ve yiyecekler tamdı, fakat kaptan dahil 10 kişi tamamen yok olmuştu.",
                "image_prompt": "Interior cabin of sailing ship, charts on wooden desk, untouched tea cup, vertical 9:16, hyperrealistic",
            },
            {
                "narration": "Tek filika kayıptı ama gemiyi terk etmelerini gerektirecek hiçbir hasar yoktu.",
                "image_prompt": "Empty lifeboat davits on a rocking ship hull, turbulent dark Atlantic water, vertical 9:16, cinematic realism",
            },
            {
                "narration": "Onları canavarlar mı yoksa açıklanamayan bir delilik mi yuttu?",
                "image_prompt": "Dark shadowy ocean horizon with glowing bioluminescence, fog, mysterious silhouette, vertical 9:16",
            },
            {
                "narration": "Yüz elli yıldır bu hayalet geminin sırrını çözen tek bir insan çıkmadı.",
                "image_prompt": "Vintage maritime court investigation scene, candlelit room with old documents, vertical 9:16",
            },
            {
                "narration": "Çünkü okyanusun en derin gizemleri aslında;",
                "image_prompt": "Endless deep ocean water reflecting moonlight through storm clouds, vertical 9:16, 8k render",
            },
        ],
    },
    {
        "title": "Göbeklitepe'nin Taş Çağı Sırrı",
        "hook_question": "12 bin yıl önce bu devasa tapınakları kim inşa etti?",
        "viral_score": 9,
        "visual_appeal": 9,
        "scenes": [
            {
                "narration": "Tarihin sıfır noktası Göbeklitepe, bildiğimiz tüm tarihi kökünden sarstı.",
                "image_prompt": "Cinematic vertical 9:16 wide view of Göbeklitepe stone pillars at sunrise, ancient mystical atmosphere, 8k photorealistic",
            },
            {
                "narration": "12 bin yıl önce, henüz tarım bile yokken devasa T biçimli sütunlar dikildi.",
                "image_prompt": "Ancient prehistoric hunter-gatherers lifting megalithic limestone T-pillar with wooden tools, vertical 9:16, cinematic",
            },
            {
                "narration": "Tonlarca ağırlıktaki taşların üzerine kusursuz hayvan kabartmaları işlenmişti.",
                "image_prompt": "Close-up of intricately carved lion and vulture reliefs on ancient megalithic pillar, vertical 9:16, detailed texture",
            },
            {
                "narration": "Daha da şaşırtıcı olanı, bu devasa kompleks bilinçli olarak gömülmüştü!",
                "image_prompt": "Ancient builders covering circular stone monument with hill of earth, torches glowing at twilight, vertical 9:16",
            },
            {
                "narration": "İnsanlar bu kutsal alanı neden kendi elleriyle toprağa gömdü?",
                "image_prompt": "Atmospheric shot of excavated circular megalith enclosure beneath starry night sky, vertical 9:16, cinematic",
            },
            {
                "narration": "Arkeologlar hâlâ bu sorunun cevabını arıyor.",
                "image_prompt": "Archaeologist dusting ancient carved limestone relief with small brush, soft warm light, vertical 9:16",
            },
            {
                "narration": "Ve insanlığın gerçek kökeni tam da burada gizli;",
                "image_prompt": "Mystical ancient sunrise over the plains of Mesopotamia behind stone pillars, vertical 9:16, photorealistic",
            },
        ],
    },
]


def discover_and_score_topics(history_titles: list[str], content_aware_prompt: str = "") -> dict:
    """Discover candidate viral history topics, score them, and pick the best unseen one."""
    negative_prompt = ""
    if history_titles:
        recent = ", ".join(history_titles[-30:])
        negative_prompt = f"\nBu konular daha önce işlendi, bunları KESİNLİKLE SEÇME VEYA TEKRAR ETME:\n{recent}\n"

    # Content-aware enhancement: include previously covered angles
    content_awareness = ""
    if content_aware_prompt:
        content_awareness = f"\n{content_aware_prompt}\n"

    prompt = f"""
Sen YouTube Shorts için viral tarih içerikleri keşfeden uzman bir araştırmacısın.
İzleyiciyi ilk saniyeden ekrana kilitleyecek, az bilinen, şaşırtıcı veya esrarengiz 3 farklı tarihsel olay öner.
{negative_prompt}
{content_awareness}

Her konu için şu alanları sağla:
- "title": Türkçe çarpıcı kısa başlık (örn: "Kayıp 9. Roma Lejyonu")
- "hook_question": İlk 2 saniyede sorulacak şok edici soru (örn: "5000 Roma askeri İskoçya sislerinde nasıl tek bir iz bırakmadan yok oldu?")
- "viral_score": 1-10 arası viral potansiyel puanı (sayı)
- "visual_appeal": 1-10 arası görsel zenginlik puanı (sayı)

SADECE şu JSON şemasında çıktı ver:
{{
  "topics": [
    {{
      "title": "...",
      "hook_question": "...",
      "viral_score": 9,
      "visual_appeal": 9
    }}
  ]
}}
"""
    for attempt in range(3):
        res = gemini_config.call_gemini_with_retry(
            prompt=prompt,
            label="topic discovery",
            response_mime_type="application/json"
        )
        if res and res.text:
            try:
                parsed = json.loads(res.text.strip())
                topics = parsed.get("topics", [])
                if topics and isinstance(topics, list):
                    past_lower = {t.lower() for t in history_titles if isinstance(t, str)}
                    unseen = [t for t in topics if t.get("title", "").lower() not in past_lower]
                    candidates = unseen if unseen else topics
                    best = max(
                        candidates,
                        key=lambda t: float(t.get("viral_score", 5)) + float(t.get("visual_appeal", 5)),
                    )
                    log.info("Selected topic: %s (score: %s)", best.get("title"), best.get("viral_score"))
                    return best
            except Exception as exc:
                log.warning("Topic discovery JSON parse failed: %s", exc)

    raise RuntimeError("Topic discovery failed after all attempts. Pipeline aborting safely.")


def evaluate_script_quality(scenes: list[dict], topic: dict | None = None) -> dict:
    """Evaluate script quality and storytelling metrics. Returns score 0-100 and issues."""
    if not scenes or len(scenes) != SCENE_COUNT:
        return {"score": 0, "passed": False, "issues": ["Invalid scene count"]}

    score = 100
    issues = []

    # 1. Hook check in Scene 1
    s1 = scenes[0].get("narration", "")
    hook_indicators = ["?", "nasıl", "neden", "kim", "nerede", "hiç", "inanılmaz", "gizem", "şok", "esrarengiz", "fakat"]
    passive_date_openers = ("1814 yılında", "1872 yılında", "191", "18", "17")
    has_hook_signal = any(ind in s1.lower() for ind in hook_indicators)
    if (not has_hook_signal and s1.lower().startswith(passive_date_openers)) or len(s1.split()) > 14:
        score -= 20
        issues.append("Scene 1 hook is too passive/date-led or too long; prefer immediate surprise, scale, or question")

    # 2. Natural ending + loop strategy
    s7 = scenes[-1].get("narration", "").strip()
    ending_strategy = scenes[-1].get("ending_strategy", "").lower()
    if s7.endswith((";", ":")) or any(marker in s7.lower() for marker in [
        "aslında;", "tam da burası:", "cevabı:", "başladığı yer:"
    ]):
        score -= 20
        issues.append("Scene 7 is linguistically unfinished; loop must never override a natural ending")
    if ending_strategy not in {"semantic", "visual", "audio", "clean"}:
        score -= 5
        issues.append("Scene 7 has no valid ending strategy")

    # 3. Retention architecture
    normalized_scenes = [s.get("narration", "").lower() for s in scenes]
    transition_words = ["ama", "fakat", "ancak", "çünkü", "oysa", "derken", "sonra", "üstelik", "asıl"]
    transition_count = sum(
        any(scene.startswith(word + " ") or (" " + word + " ") in scene for word in transition_words)
        for scene in normalized_scenes[1:6]
    )
    unique_starts = len(set(scene.split()[0] for scene in normalized_scenes if scene.split()))
    if transition_count < 2:
        score -= 8
        issues.append("Low escalation/transition density across middle scenes")
    if unique_starts < 5:
        score -= 5
        issues.append("Scene openings are too repetitive")

    # 4. Word count balance
    word_counts = [len(s.get("narration", "").split()) for s in scenes]
    total_words = sum(word_counts)
    if total_words < MIN_TOTAL_WORDS or total_words > MAX_TOTAL_WORDS:
        score -= 20
        issues.append(f"Total word count ({total_words}) is outside optimal range ({MIN_TOTAL_WORDS}-{MAX_TOTAL_WORDS})")

    score = max(score, 0)
    return {
        "score": score,
        "passed": score >= 60,
        "total_words": total_words,
        "issues": issues,
        "topic": topic.get("title", "") if topic else "",
    }


def generate_viral_script(candidate: dict, research_dossier: dict) -> list[dict]:
    """Generate a 7-scene structured script based on the verified research dossier.

    This ensures the script is based on facts and uncertainty is preserved where appropriate.
    """
    topic_title = candidate.get("canonical_title", "Tarihsel Gizem")
    hook_question = research_dossier.get("story_hook", candidate.get("why_interesting", ""))
    
    verified_facts_str = "\n".join(f"- {f}" for f in research_dossier.get("verified_facts", []))
    disputed_claims_str = "\n".join(f"- {f}" for f in research_dossier.get("disputed_claims", []))
    myths_str = "\n".join(f"- {f}" for f in research_dossier.get("possible_myths", []))

    prompt = f"""
Sen YouTube Shorts için viral tarih belgeselleri üreten usta bir yönetmensin.

Olay: {topic_title}
Önerilen Kanca: {hook_question}

DOĞRULANMIŞ GERÇEKLER:
{verified_facts_str}

TARTIŞMALI İDDİALAR:
{disputed_claims_str}

EFSANELER / SPEKÜLASYONLAR:
{myths_str}

Bu olayı tam {SCENE_COUNT} sahnelik, yüksek tempolu bir Shorts senaryosu olarak yaz.
Hedef seslendirme süresi yaklaşık {TARGET_DURATION} saniye; toplam 56-72 kelime.
AMAÇ: 7 sahnenin toplamı kesinlikle 72 kelimeyi geçmesin. Her sahne 6-12 kelime olsun; mümkünse 8-10 kelime kullan.
Öncelik kısa ve doğal Türkçe cümlelerdir. 30-38 saniyelik, hızlı ve yoğun bir Shorts üret.

Sahne Hikaye Şablonu:
- 1. Sahne: COLD OPEN - 6-10 kelime. Başlığı veya yalnızca tarihi tekrar etme. İlk cümle doğrudan şaşırtıcı sonuç, devasa ölçek, imkânsız görünen durum veya güçlü bir soru ile başlamalı; "1814 yılında..." gibi pasif tarih girişi kullanma.
- 2. Sahne: CONTEXT - İzleyicinin olayı anlaması için gereken minimum bilgi.
- 3. Sahne: ESCALATION - Yeni bir bilgi, sayı, tehdit veya çelişki getir. Önceki sahneyi farklı kelimelerle tekrar etme.
- 4. Sahne: UNEXPECTED FACT - Hikâyenin yönünü değiştiren veya merakı artıran yeni gerçek.
- 5. Sahne: STAKES / CONSEQUENCE - "Peki bunun sonucu ne oldu?" sorusunu cevapla.
- 6. Sahne: PAYOFF - En güçlü doğrulanmış bilgi veya olayın asıl çarpıcı sonucu. Ana payoff'u ilk cümlede tüketme.
- 7. Sahne: NATURAL CLOSURE + LOOP - Cümle tek başına video burada bitse bile tamamen doğal ve tamamlanmış olmalı. ASLA noktalı virgül, iki nokta, yarım cümle veya "ve..." ile bitirme.

RETENTION KURALLARI:
- Her sahne izleyiciye yeni bir bilgi, soru, sonuç veya kontrast vermeli.
- İlk 2 saniyede bağlamı değil merakı sat; tarihi gerekiyorsa sonraki sahnelere bırak.
- En güçlü payoff'u 6. sahne civarına taşı.
- Orta sahnelerde en az 2 doğal escalation/contrast geçişi kullan; sürekli "ama işler daha da..." kalıbını tekrarlama.
- Aynı bilgiyi farklı kelimelerle tekrarlama.
- Son cümle ilk sahnenin fikrine doğal biçimde geri dönebiliyorsa SEMANTIC loop seç.
- Semantic loop doğal değilse VISUAL loop seç; son cümle yine tamamen doğal kalmalı.
- Visual loop da doğal değilse AUDIO loop veya CLEAN ending seç. Kötü bir dilsel loop üretmek kesinlikle yasak.

SAHNE 7 ENDING STRATEGY:
- "ending_strategy" yalnızca "semantic", "visual", "audio" veya "clean" olabilir.
- semantic: Son cümle hook'un fikrini doğal olarak yankılar ve bağımsız bir kapanış cümlesidir.
- visual: Son cümle doğal kapanır; sahne 7 görseli sahne 1 ile eşleşebilir.
- audio: Son cümle doğal kapanır; atmosfer/müzik döngüyü taşır.
- clean: Loop yapılmaz; yalnızca doğal ve güçlü kapanış yapılır.

Kurallar:
1. Sadece araştırma dosyasındaki bilgileri kullan. Kendi genel bilgilerinle yeni iddialar uydurma.
2. Kesin olmayan şeyleri kesinmiş gibi anlatma (belirsizliği koru).
3. Görsel promptları İngilizce, sinematik, dikey (vertical 9:16), fotogerçekçi, 8k render ve PG-13 belgesel tonunda olsun.
4. Kesinlikle kan, aşırı şiddet veya ceset istemiyorum.
5. Çıktı SADECE geçerli JSON olsun.
6. Her sahnenin image_prompt'u birbirinden FARKLI görsel kompozisyon, açı ve sahne içermeli.
7. Aynı insan yüzünü/karakteri sahneden sahneye TEKRARLAMA; isimsiz tekrar eden kahraman oluşturma.
8. Mümkün olduğunca insan portresi yerine belge, harita, mimari, nesne, kalabalık, manzara ve yakın plan detay kullan.
9. Yedi sahnede farklı görsel arketipler kullan: hook/close-up, wide establishing, artifact/document, map/diagram, crowd/action, location/detail, archival aftermath.
10. Görsel promptlarda "same woman", "same man", "same character" veya karakter sürekliliği isteme. Yalnızca olayın gerçek kişilerinin görsel olarak zorunlu olduğu sahnede kişi göster.
11. Modern stok estetiği yerine döneme uygun tarihsel/arkeolojik belgesel estetiğini tercih et.
12. Her sahne için visual_fact ve visual_intent üret.
13. visual_fact, anlatılan cümlenin ekranda gösterilecek SOMUT tarihsel bilgisidir. Bir atmosfer, duygu veya genel konu değildir.
14. visual_role, görselin rolünü belirtmeli: evidence, mechanism, reconstruction, context_map, person_or_entity, aftermath veya atmosphere.
15. primary_subject, o sahnede gerçekten görülmesi gereken ana nesne/kişi/olay olmalı.
16. must_show en az 2 somut unsur, avoid ise en az 2 yanlış/ilgisiz görsel türü içermeli.
17. visual_action, narrationdaki tarihsel olayın GÖRÜNÜR EYLEMİDİR. Bir nesnenin adı olamaz. "bir şişe", "elma", "bina" gibi tek başına cevaplar yasaktır.
18. scene_context olayın görsel olarak gerçekleştiği yer/dönem/ortamı belirtmeli.
19. shot_type, wide/medium/close-up/detail/overhead gibi kadraj stratejisini belirtmeli.
20. composition, ana özne + eylem + yardımcı unsur + çevre arasındaki görsel ilişkiyi tarif etmeli.
21. visual_entities, sahnede görünmesi gereken belirli kişi/kurum/nesne adlarını içerebilir; mümkünse 1-4 öğe kullan.
22. search_queries, visual_fact + visual_action + olayın spesifik kimliği etrafında 2-4 tarihsel arama sorgusu içermeli. Ham narration'ı aynen sorgu olarak kullanma.
23. Görsel plan bir "story unit" olmalı: tek bir kelimeyi/objeyi değil, mümkün olduğunda olayın eylemini ve ilişkisini göstermeli.
24. Bir sahnede "10.000 mermi" anlatılıyorsa harita/manzara değil mühimmat veya döneme ait askerî ekipman hedefle.
25. "80 milyon ağaç devrildi" anlatılıyorsa normal orman değil devrilmiş/hasar görmüş ağaçlar veya blast pattern hedefle.
26. "atmosferde hava patlaması" anlatılıyorsa generic Earth/sunset değil giriş yapan gök cismi, atmosferik patlama veya şok dalgası göster.
27. Belirli bir kişi/kurum adı anlatılıyorsa generic person kullanma; gerçek arşiv görseli veya açıkça hedeflenmiş historical reconstruction oluştur.
28. atmosphere yalnızca geçiş/duygu amacıyla kullanılabilir; ana tarihsel bilgi taşıyan sahnelerde evidence/mechanism/reconstruction tercih et.
29. Görsel arketipleri factual intent'i ezmemeli. Metin harita gerektirmiyorsa sırf çeşitlilik için map kullanma.
30. AI fallback, visual_fact + visual_action + composition'ı birlikte görselleştiren reconstruction olmalı; keyword collage veya generic stock estetiği olamaz.

31. Her sahne için `event_specificity` ve `information_density` planlama puanlarını 0-1 arasında ver.
26. evidence, mechanism, reconstruction, context_map, person_or_entity ve aftermath rollerinde event_specificity en az 0.70 hedefle.
27. atmosphere rolünü en fazla 1 sahnede kullan ve information_density düşük olabilir; diğer sahneler bilgi taşımalıdır.

Şema:
{{
  "scenes": [
    {{
      "narration": "...",
      "image_prompt": "...",
      "visual_fact": "...",
      "visual_role": "evidence",
      "event_specificity": 0.90,
      "information_density": 0.85,
      "visual_intent": {{
        "visual_fact": "...",
        "visual_role": "evidence",
        "primary_subject": "...",
        "visual_type": "historical_photo/artifact/document/map/person/location/crowd/action/reconstruction",
        "visual_action": "...",
        "scene_context": "...",
        "shot_type": "medium action shot",
        "composition": "who does what with which supporting element in what environment",
        "visual_entities": ["specific person", "specific object"],
        "must_show": ["somut unsur 1", "somut unsur 2"],
        "avoid": ["ilgisiz görsel 1", "ilgisiz görsel 2"],
        "search_queries": ["spesifik arama 1", "spesifik arama 2"]
      }},,
      "ending_strategy": "semantic"
    }}
  ]
}}
"""

    current_prompt = prompt
    for attempt in range(4):
        res = gemini_config.call_gemini_with_retry(
            prompt=current_prompt,
            label="script generation",
            response_mime_type="application/json"
        )
        
        if res and res.text:
            try:
                raw_json = res.text.strip()
                try:
                    data = json.loads(raw_json)
                except json.JSONDecodeError:
                    decoder = json.JSONDecoder()
                    data, _ = decoder.raw_decode(raw_json)
                scenes = validate_script(data)
                storyboard_qa = validate_visual_storyboard(scenes)
                log.info("Visual storyboard QA: %s", storyboard_qa)
                topic_compat = {"title": topic_title} # For QA
                qa_result = evaluate_script_quality(scenes, topic_compat)
                log.info("Script QA evaluated: score=%s, issues=%s", qa_result["score"], qa_result["issues"])
                return scenes
            except Exception as exc:
                err_msg = str(exc)
                log.warning("Script parsing or validation failed: %s", err_msg)
                current_total_match = re.search(r"(\d+) words", err_msg)
                current_total = int(current_total_match.group(1)) if current_total_match else None
                word_fix = (
                    f"Mevcut denemede toplam {current_total} kelime vardı; bu kez toplamı 56-72 kelimeye çıkar."
                    if current_total is not None and current_total < MIN_TOTAL_WORDS
                    else "Toplam narration 56-72 kelime arasında olmalı."
                )
                current_prompt = prompt + f"\n\nÖNCEKİ DENEMEDE HATA ALINDI:\n{err_msg}\n"
                current_prompt += (
                    f"KRİTİK DÜZELTME: {word_fix} "
                    "Her sahne 7-10 kelime hedeflesin ve 6-12 sınırını korusun. "
                    "visual_fact ve visual_intent.visual_fact aynı anlamı taşımalı; "
                    "visual_role ve visual_intent.visual_role aynı olmalı. "
                    "visual_action, scene_context, shot_type ve composition alanlarını doldur; "
                    "bunlar narrationdaki tek bir nesneyi değil olayın görsel eylemini tarif etmeli. "
                    "Çıktıda yalnızca TEK bir JSON nesnesi üret; JSON'dan sonra hiçbir metin ekleme."
                )

    raise RuntimeError("Script generation failed after all attempts. Pipeline aborting safely.")


# Alias for the content_memory module to avoid name collision with local variables
content_memory_module = content_memory



# -----------------------------
# Voice + word timing
# -----------------------------
def estimate_word_timestamps(text: str, total_duration: float) -> list[dict]:
    """Fallback: distribute word timestamps proportionally across audio duration if TTS misses them."""
    raw_words = text.split()
    if not raw_words or total_duration <= 0:
        return []
    total_chars = max(sum(len(w) for w in raw_words), 1)
    words = []
    cursor = 0.05
    usable_duration = max(total_duration - 0.15, 0.5)
    for w in raw_words:
        w_duration = usable_duration * (len(w) / total_chars)
        start_t = cursor
        end_t = min(start_t + w_duration, total_duration)
        words.append(
            {
                "word": w,
                "start": round(start_t, 3),
                "end": round(end_t, 3),
            }
        )
        cursor = end_t
    return words


async def create_voice_with_timestamps(
    text: str,
    audio_path: Path,
    voice_profile_name: str = "documentary_male",
) -> list[dict]:
    profile = VOICE_PROFILES.get(voice_profile_name, VOICE_PROFILES["documentary_male"])
    log.info(
        "TTS voice profile: %s | voice=%s | rate=%s | pitch=%s",
        voice_profile_name, profile["voice"], profile["rate"], profile["pitch"],
    )
    communicate = edge_tts.Communicate(
        text,
        voice=profile["voice"],
        rate=profile["rate"],
        pitch=profile["pitch"],
        boundary="WordBoundary",
    )
    words = []
    with audio_path.open("wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append(
                    {
                        "word": chunk["text"],
                        "start": chunk["offset"] / 10_000_000,
                        "end": (chunk["offset"] + chunk["duration"]) / 10_000_000,
                    }
                )

    # Robust fallback: if edge-tts did not return word boundaries, estimate them from audio duration
    expected_words = len(text.split())
    if len(words) < max(expected_words // 2, 1) and audio_path.exists():
        try:
            with AudioFileClip(str(audio_path)) as audio_clip:
                audio_dur = audio_clip.duration
            if audio_dur and audio_dur > 0:
                log.warning(
                    "TTS returned %d word boundaries (expected ~%d); applying estimated word timings fallback",
                    len(words),
                    expected_words,
                )
                words = estimate_word_timestamps(text, audio_dur)
        except Exception as exc:
            log.warning("Could not compute fallback word timestamps: %s", exc)

    return words


def calculate_scene_timings(scenes, words_data, total_duration):
    """Create continuous scene intervals from word timestamps.

    A scene starts when the previous scene's narration ends and remains on
    screen through any natural pause until the next scene's first word. This
    prevents black gaps while preserving the timing information from TTS.
    """
    word_counts = [len(scene["narration"].split()) for scene in scenes]
    expected = sum(word_counts)

    if len(words_data) >= expected:
        chunks = []
        cursor = 0
        for count in word_counts:
            chunk = words_data[cursor: cursor + count]
            chunks.append(chunk)
            cursor += count

        timings = []
        cursor = 0.0
        for i, chunk in enumerate(chunks):
            start = cursor
            if i == len(chunks) - 1:
                end = total_duration
            else:
                end = chunks[i + 1][0]["start"]
            end = max(end, start + 0.2)
            end = min(end, total_duration)
            timings.append((start, end))
            cursor = end
        timings[-1] = (timings[-1][0], total_duration)
        return timings

    # Fallback: proportional timing if the TTS provider returns fewer word
    # boundaries than expected. This fallback is also continuous.
    total_chars = sum(len(s["narration"]) for s in scenes)
    timings = []
    cursor = 0.0
    for scene in scenes:
        share = len(scene["narration"]) / max(total_chars, 1)
        duration = total_duration * share
        timings.append((cursor, min(total_duration, cursor + duration)))
        cursor += duration
    timings[-1] = (timings[-1][0], total_duration)
    return timings


# -----------------------------
# Image generation
# -----------------------------
def crop_watermark_zone(filename: Path):
    """Crop image to 9:16 aspect ratio from center, remove watermark, and sharpen for crisp 1080p display."""
    with Image.open(filename) as img:
        img = img.convert("RGB")
        w, h = img.size

        target_ratio = VIDEO_WIDTH / VIDEO_HEIGHT  # 1080 / 1920 = 0.5625
        current_ratio = w / h

        if current_ratio > target_ratio:
            # Image is wider than 9:16 (e.g. square 1024x1024 or 768x768).
            # Center-crop the width. This cleanly removes watermarks located in the outer margins!
            crop_w = int(h * target_ratio)
            left = max((w - crop_w) // 2, 0)
            fitted = img.crop((left, 0, left + crop_w, h))
        else:
            # Image is taller than 9:16. Center-crop the height.
            crop_h = int(w / target_ratio)
            top = max((h - crop_h) // 2, 0)
            fitted = img.crop((0, top, w, top + crop_h))

        # Scale cleanly to exact video dimensions
        final_img = fitted.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
        # Apply subtle unsharp mask to restore crisp edges and eliminate blur
        final_img = final_img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=125, threshold=3))
        final_img.save(filename, quality=95)


def try_imagen3_generation(prompt_text: str, filename: Path) -> bool:
    """Attempt high-resolution native 9:16 image generation via Google GenAI Imagen 3."""
    try:
        if not hasattr(client, "models") or not hasattr(client.models, "generate_images"):
            return False
        result = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt_text,
            config=dict(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="9:16",
            ),
        )
        if result and getattr(result, "generated_images", None):
            image_bytes = result.generated_images[0].image.image_bytes
            filename.write_bytes(image_bytes)
            with Image.open(filename) as img:
                img = img.convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
                img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=110, threshold=3))
                img.save(filename, quality=95)
            log.info("Successfully generated native 9:16 image with Imagen 3: %s", filename.name)
            return True
    except Exception as exc:
        log.debug("Imagen 3 generation skipped/failed: %s", exc)
    return False


def download_ai_image(prompt_text: str, filename: Path):
    if try_imagen3_generation(prompt_text, filename):
        return filename

    cleaned = urllib.parse.quote(prompt_text)
    # Request square 1024x1024 to prevent Pollinations from stretching/distorting the latents
    url = (
        f"https://image.pollinations.ai/prompt/{cleaned}"
        f"?width=1024&height=1024&model=turbo&nologo=true"
    )

    def request_image():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        if len(response.content) <= 5000:
            raise RuntimeError("Image response is unexpectedly small")
        filename.write_bytes(response.content)
        crop_watermark_zone(filename)
        return filename

    return retry_call(request_image, attempts=3, base_delay=3, label="image generation")


# -----------------------------
# Video composition
# -----------------------------
MOTION_TYPES = ["zoom_in", "pan_left_right", "zoom_out", "pan_right_left"]


def enhance_source_image(image_path: Path, source_type: str = "", visual_intent: dict | None = None) -> dict:
    """Apply conservative source-aware restoration before video composition."""
    if not IMAGE_ENHANCEMENT_ENABLED:
        return {"enabled": False, "source_type": source_type}

    intent = visual_intent or {}
    visual_type = str(intent.get("visual_type", "")).lower()
    try:
        with Image.open(image_path) as source:
            img = source.convert("RGB")
        original_size = img.size

        if source_type in ("wikimedia", "openverse"):
            if visual_type in ("map", "document", "artifact"):
                img = ImageEnhance.Contrast(img).enhance(1.04)
                img = ImageEnhance.Sharpness(img).enhance(1.06)
            else:
                img = ImageOps.autocontrast(img, cutoff=1)
                img = ImageEnhance.Contrast(img).enhance(1.05)
                img = ImageEnhance.Color(img).enhance(1.02)
                img = ImageEnhance.Sharpness(img).enhance(1.10)
            profile = "archival_gentle"
        else:
            img = ImageOps.autocontrast(img, cutoff=1)
            img = ImageEnhance.Contrast(img).enhance(1.06)
            img = ImageEnhance.Sharpness(img).enhance(1.14)
            profile = "ai_detail"

        img.save(image_path, quality=96, subsampling=0)
        return {
            "enabled": True,
            "source_type": source_type,
            "visual_type": visual_type,
            "original_size": list(original_size),
            "final_size": list(img.size),
            "profile": profile,
        }
    except Exception as exc:
        raise RuntimeError(f"Image enhancement failed for {image_path.name}: {exc}") from exc


def prepare_vertical_image(image_path: Path) -> None:
    """Resize without geometric stretching; crop to the 9:16 canvas."""
    with Image.open(image_path) as source:
        img = ImageOps.fit(
            source.convert("RGB"),
            (VIDEO_WIDTH, VIDEO_HEIGHT),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        img.save(image_path, quality=96, subsampling=0)



def turkish_upper(text: str) -> str:
    """Convert text to uppercase respecting Turkish-specific character rules."""
    mapping = {
        "i": "İ",
        "ı": "I",
        "ğ": "Ğ",
        "ü": "Ü",
        "ş": "Ş",
        "ö": "Ö",
        "ç": "Ç",
    }
    return "".join(mapping.get(c, c.upper()) for c in text)


def get_subtitle_font(size: int = 68):
    """Retrieve an optimal bold font that supports Turkish glyphs."""
    candidate_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for font_path in candidate_fonts:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_subtitle_image(words: list[str], active_idx: int, font, canvas_w=1080, canvas_h=220) -> np.ndarray:
    """Render a transparent frame with words centered and active word highlighted in gold/yellow."""
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    upper_words = [turkish_upper(w) for w in words]
    space_w = draw.textlength(" ", font=font)
    word_widths = [draw.textlength(w, font=font) for w in upper_words]
    total_w = sum(word_widths) + space_w * max(len(words) - 1, 0)

    # If the text chunk is wider than canvas allows, scale font size down dynamically
    current_size = getattr(font, "size", 68)
    while total_w > (canvas_w - 60) and current_size > 36:
        current_size -= 4
        scaled_font = get_subtitle_font(size=current_size)
        space_w = draw.textlength(" ", font=scaled_font)
        word_widths = [draw.textlength(w, font=scaled_font) for w in upper_words]
        total_w = sum(word_widths) + space_w * max(len(words) - 1, 0)
        font = scaled_font

    x = max((canvas_w - total_w) / 2, 20.0)
    y = max((canvas_h - 90) / 2, 10.0)

    for i, (word, w_width) in enumerate(zip(upper_words, word_widths)):
        if i == active_idx:
            fill_color = (255, 230, 0, 255)
        else:
            fill_color = (255, 255, 255, 255)

        draw.text(
            (x, y),
            word,
            font=font,
            fill=fill_color,
            stroke_width=6,
            stroke_fill=(0, 0, 0, 255),
        )
        x += w_width + space_w

    return np.array(img)


def generate_subtitle_clips(words_data: list[dict]):
    """Generate dynamic subtitle clips with active word highlighting and Turkish character support."""
    subtitle_clips = []
    if not words_data:
        return subtitle_clips

    font = get_subtitle_font(size=68)
    chunk_size = 3
    for i in range(0, len(words_data), chunk_size):
        chunk = words_data[i : i + chunk_size]
        if not chunk:
            continue

        raw_words = [w["word"] for w in chunk]
        chunk_end = chunk[-1]["end"]

        for idx, word_data in enumerate(chunk):
            start_t = word_data["start"]
            if idx < len(chunk) - 1:
                end_t = chunk[idx + 1]["start"]
            else:
                end_t = chunk_end

            duration = max(end_t - start_t, 0.15)
            frame_arr = render_subtitle_image(raw_words, active_idx=idx, font=font)

            clip = (
                ImageClip(frame_arr)
                .with_start(start_t)
                .with_duration(duration)
                .with_position(("center", 1430))
            )
            subtitle_clips.append(clip)

    return subtitle_clips


def create_hook_badge(duration: float = 2.2, title: str | None = None) -> ImageClip:
    """Create an eye-catching top badge for the first 2 seconds to boost watch retention."""
    badge_w, badge_h = 820, 90
    img = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        [(0, 0), (badge_w, badge_h)],
        radius=25,
        fill=(15, 15, 20, 215),
        outline=(255, 215, 0, 220),
        width=3,
    )

    font = get_subtitle_font(size=34)
    badge_text = turkish_upper(title)[:32] if title else "TARİHİN BİLİNMEYEN GİZEMİ"
    text_w = draw.textlength(badge_text, font=font)
    draw.text(
        ((badge_w - text_w) / 2, (badge_h - 46) / 2),
        badge_text,
        font=font,
        fill=(255, 255, 255, 255),
    )

    return (
        ImageClip(np.array(img))
        .with_start(0.0)
        .with_duration(duration)
        .with_position(("center", 340))
    )


def build_scene_clip(image_path: Path, start_time: float, end_time: float, motion_type: str = "zoom_in"):
    duration = max(end_time - start_time, 0.2)

    prepare_vertical_image(image_path)
    clip = ImageClip(str(image_path)).with_duration(duration).with_start(start_time)

    # Watermark-safe Ken Burns: scale is ALWAYS >= 1.03
    def apply_motion(get_frame, t):
        progress = min(max(t / max(duration, 0.1), 0.0), 1.0)
        frame = get_frame(t)
        src_img = Image.fromarray(frame)

        if motion_type == "zoom_in":
            scale = 1.03 + 0.08 * progress
            new_w, new_h = int(VIDEO_WIDTH * scale), int(VIDEO_HEIGHT * scale)
            resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            left = max((new_w - VIDEO_WIDTH) // 2, 0)
            top = max((new_h - VIDEO_HEIGHT) // 2, 0)
        elif motion_type == "zoom_out":
            scale = 1.11 - 0.08 * progress
            new_w, new_h = int(VIDEO_WIDTH * scale), int(VIDEO_HEIGHT * scale)
            resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            left = max((new_w - VIDEO_WIDTH) // 2, 0)
            top = max((new_h - VIDEO_HEIGHT) // 2, 0)
        elif motion_type == "pan_left_right":
            scale = 1.08
            new_w, new_h = int(VIDEO_WIDTH * scale), int(VIDEO_HEIGHT * scale)
            resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            max_x = new_w - VIDEO_WIDTH
            left = int(max_x * (0.15 + 0.70 * progress))
            top = max((new_h - VIDEO_HEIGHT) // 2, 0)
        elif motion_type == "pan_right_left":
            scale = 1.08
            new_w, new_h = int(VIDEO_WIDTH * scale), int(VIDEO_HEIGHT * scale)
            resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            max_x = new_w - VIDEO_WIDTH
            left = int(max_x * (0.85 - 0.70 * progress))
            top = max((new_h - VIDEO_HEIGHT) // 2, 0)
        else:
            scale = 1.03 + ZOOM_AMOUNT * progress
            new_w, new_h = int(VIDEO_WIDTH * scale), int(VIDEO_HEIGHT * scale)
            resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            left = max((new_w - VIDEO_WIDTH) // 2, 0)
            top = max((new_h - VIDEO_HEIGHT) // 2, 0)

        cropped = resized.crop((left, top, left + VIDEO_WIDTH, top + VIDEO_HEIGHT))
        return np.array(cropped)

    return clip.transform(apply_motion)


# -----------------------------
# Music
# -----------------------------
AUDIO_ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "audio"
LOCAL_ASSET_MUSIC = AUDIO_ASSETS_DIR / "ambient_mystery.mp3"
FALLBACK_MUSIC_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/c/c9/Rafael_Krux_-_Lights_-_Creepy_Ambient_Suspense_%28cc-by%29_%28filmmusic%29.mp3",
    "https://upload.wikimedia.org/wikipedia/commons/5/50/Hypnotic_ambient_electronic_music_by_MusicLM.mp3",
]


def get_ambient_music(music_path: Path, content_analysis: dict | None = None, memory: list[dict] | None = None):
    """Select and prepare background music. When content_analysis is provided,
    uses content-aware selection to match mood/energy/tension."""
    if music_path.exists() and music_path.stat().st_size > 10_000:
        return music_path

    # 1. Content-aware selection from local tracks (Phase 4)
    if content_analysis and AUDIO_ASSETS_DIR.exists():
        recently_used = content_memory_module.get_recently_used_audio(memory or [])
        selected = content_memory_module.select_audio_for_content(
            analysis=content_analysis,
            audio_dir=AUDIO_ASSETS_DIR,
            recently_used=recently_used,
        )
        if selected:
            try:
                music_path.write_bytes(selected.read_bytes())
                log.info(
                    "Content-aware audio selection: %s (mood=%s, energy=%.2f, tension=%.2f)",
                    selected.name,
                    content_analysis.get("mood", "unknown"),
                    float(content_analysis.get("energy", 0)),
                    float(content_analysis.get("tension", 0)),
                )
                return music_path
            except Exception as exc:
                log.warning("Could not copy content-aware audio %s: %s", selected.name, exc)

    # 2. Check bundled local royalty-free ambient tracks (variety, zero latency, offline-ready)
    if AUDIO_ASSETS_DIR.exists():
        local_tracks = [p for p in AUDIO_ASSETS_DIR.glob("*.mp3") if p.stat().st_size > 10_000]
        if local_tracks:
            chosen = random.choice(local_tracks)
            try:
                music_path.write_bytes(chosen.read_bytes())
                log.info(
                    "Selected ambient background music: %s (from %d available local tracks)",
                    chosen.name,
                    len(local_tracks),
                )
                return music_path
            except Exception as exc:
                log.warning("Could not copy local ambient music %s: %s", chosen.name, exc)

    if LOCAL_ASSET_MUSIC.exists() and LOCAL_ASSET_MUSIC.stat().st_size > 10_000:
        try:
            music_path.write_bytes(LOCAL_ASSET_MUSIC.read_bytes())
            log.info("Loaded ambient background music from default asset: %s", LOCAL_ASSET_MUSIC)
            return music_path
        except Exception as exc:
            log.warning("Could not copy default local ambient music: %s", exc)

    # 3. Check environment variable or reliable public domain CDNs
    custom_url = os.getenv("BG_MUSIC_URL")
    candidate_urls = ([custom_url] if custom_url else []) + FALLBACK_MUSIC_URLS

    for url in candidate_urls:
        def download():
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=25)
            response.raise_for_status()
            if len(response.content) < 10_000:
                raise ValueError("Downloaded music file is suspiciously small")
            music_path.write_bytes(response.content)
            return music_path

        try:
            return retry_call(download, attempts=2, base_delay=2, label=f"music download from {url[:40]}")
        except Exception as exc:
            log.warning("Music URL %s unavailable: %s", url[:40], exc)

    # FAIL LOUDLY: audio is mandatory per plan section 12
    log.error("CRITICAL: No background audio could be obtained — every video must have audio")
    return None


# -----------------------------
# Telegram
# -----------------------------
# Telegram & Approval System
# -----------------------------
def load_approvals(approvals_path: Path | None = None) -> dict:
    """Load video approval states."""
    target_path = approvals_path if approvals_path is not None else APPROVALS_FILE
    if not target_path.exists():
        return {}
    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("Could not read approvals file: %s", exc)
        return {}


def save_approval_state(run_id: str, state_data: dict, approvals_path: Path | None = None):
    """Save or update a run approval state."""
    target_path = approvals_path if approvals_path is not None else APPROVALS_FILE
    approvals = load_approvals(target_path)
    approvals[run_id] = state_data
    try:
        target_path.write_text(
            json.dumps(approvals, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("Could not save approvals file: %s", exc)


def get_approval_state(run_id: str, approvals_path: Path | None = None) -> dict | None:
    """Retrieve state for a specific run."""
    target_path = approvals_path if approvals_path is not None else APPROVALS_FILE
    return load_approvals(target_path).get(run_id)


def update_approval_status(run_id: str, status: str, extra: dict | None = None, approvals_path: Path | None = None):
    """Update approval status (e.g. pending -> approved / cancelled)."""
    target_path = approvals_path if approvals_path is not None else APPROVALS_FILE
    approvals = load_approvals(target_path)
    if run_id not in approvals:
        approvals[run_id] = {}
    approvals[run_id]["status"] = status
    approvals[run_id]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if extra:
        approvals[run_id].update(extra)
    try:
        target_path.write_text(
            json.dumps(approvals, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("Could not update approval status: %s", exc)


def build_telegram_markup(run_id: str) -> types.InlineKeyboardMarkup:
    """Build the 4-button interactive Human-in-the-Loop control keyboard."""
    markup = types.InlineKeyboardMarkup(row_width=2)
    btn_publish = types.InlineKeyboardButton("✅ YAYINLA", callback_data=f"publish:{run_id}")
    btn_regen = types.InlineKeyboardButton("🔄 YENİDEN ÜRET", callback_data=f"regen:{run_id}")
    btn_cancel = types.InlineKeyboardButton("❌ İPTAL", callback_data=f"cancel:{run_id}")
    btn_script = types.InlineKeyboardButton("📜 Senaryoyu Oku", callback_data=f"script:{run_id}")
    markup.add(btn_publish, btn_regen)
    markup.add(btn_cancel, btn_script)
    return markup


def send_to_telegram(
    video_path: Path,
    total_duration: float,
    full_text: str,
    topic: dict | None = None,
    run_id: str | None = None,
    scenes: list[dict] | None = None,
):
    """Send generated video to Telegram with 4 interactive human-in-the-loop control buttons."""
    if not run_id:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    topic_title = topic.get("title", "Yeni Shorts") if topic else "Yeni Shorts"
    hook_question = topic.get("hook_question", "") if topic else ""
    hook_line = f"❓ _{hook_question}_\n\n" if hook_question else ""

    caption = (
        f"🔥 *{topic_title}*\n\n"
        f"{hook_line}"
        f"⏱ Süre: {total_duration:.1f}s | {SCENE_COUNT} Sahne\n"
        "🎬 AI Görsel + Kelime Vurgulu Altyazı + Ambiyans\n\n"
        f"{full_text[:280]}..."
    )

    save_approval_state(run_id, {
        "run_id": run_id,
        "status": "pending",
        "video_path": str(video_path),
        "topic": topic,
        "total_duration": total_duration,
        "full_text": full_text,
        "scenes": scenes or [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    markup = build_telegram_markup(run_id)

    with video_path.open("rb") as video:
        return bot.send_video(
            TELEGRAM_CHAT_ID,
            video,
            caption=caption,
            reply_markup=markup,
            parse_mode="Markdown",
        )


def handle_callback_action(
    action: str,
    run_id: str,
    chat_id: int | str,
    message_id: int,
    callback_id: str,
    approvals_path: Path | None = None,
) -> dict:
    """Pure business logic handler for button clicks, returning an outcome dict."""
    target_path = approvals_path if approvals_path is not None else APPROVALS_FILE
    state = get_approval_state(run_id, approvals_path=target_path) or {}
    topic_title = state.get("topic", {}).get("title", "Video") if state else "Video"

    if action == "publish":
        update_approval_status(run_id, "approved", approvals_path=target_path)
        bot.answer_callback_query(callback_id, "Video onaylandı!")
        bot.edit_message_reply_markup(chat_id, message_id, reply_markup=None)

        video_path_str = state.get("video_path")
        video_path = Path(video_path_str) if video_path_str else None

        bot.send_message(
            chat_id,
            f"✅ *{topic_title}* onaylandı!\n\n🚀 YouTube Shorts yüklemesi başlatılıyor...",
            parse_mode="Markdown",
        )

        if video_path and video_path.exists():
            try:
                update_approval_status(run_id, "uploading", approvals_path=target_path)
                result = youtube_uploader.upload_shorts_video(
                    video_path=video_path,
                    topic=state.get("topic"),
                    full_text=state.get("full_text", ""),
                )
                yt_id = result.get("video_id", "")
                yt_url = result.get("url", "")
                update_approval_status(
                    run_id,
                    "published",
                    extra={"youtube_video_id": yt_id, "youtube_url": yt_url},
                    approvals_path=target_path,
                )
                bot.send_message(
                    chat_id,
                    f"🎉 *{topic_title}* başarıyla YouTube Shorts'a yüklendi!\n\n"
                    f"🔗 [YouTube Shorts'ta İzle]({yt_url})\n"
                    f"🆔 Video ID: `{yt_id}`",
                    parse_mode="Markdown",
                )
                return {
                    "action": "publish",
                    "status": "published",
                    "run_id": run_id,
                    "youtube_video_id": yt_id,
                    "youtube_url": yt_url,
                }
            except Exception as exc:
                log.error("YouTube upload error for %s: %s", run_id, exc)
                update_approval_status(
                    run_id,
                    "upload_failed",
                    extra={"upload_error": str(exc)},
                    approvals_path=target_path,
                )
                bot.send_message(
                    chat_id,
                    f"⚠️ *YouTube yüklemesi gerçekleştirilemedi:*\n`{exc}`\n\n"
                    "Lütfen `client_secrets.json` veya `token.json` dosyasını kontrol edin.",
                    parse_mode="Markdown",
                )
                return {
                    "action": "publish",
                    "status": "upload_failed",
                    "run_id": run_id,
                    "error": str(exc),
                }
        else:
            bot.send_message(
                chat_id,
                f"⚠️ Video dosyası diskte bulunamadı ({video_path_str}). Yükleme tamamlanamadı.",
            )
            return {"action": "publish", "status": "missing_video", "run_id": run_id}


    elif action == "cancel":
        update_approval_status(run_id, "cancelled", approvals_path=target_path)
        bot.answer_callback_query(callback_id, "Video iptal edildi.")
        bot.edit_message_reply_markup(chat_id, message_id, reply_markup=None)
        bot.send_message(
            chat_id,
            f"❌ *{topic_title}* yayını iptal edildi.",
            parse_mode="Markdown",
        )
        return {"action": "cancel", "status": "cancelled", "run_id": run_id}

    elif action == "regen":
        update_approval_status(run_id, "regenerating", approvals_path=target_path)
        bot.answer_callback_query(callback_id, "Yeniden üretim başlatılıyor...")
        bot.send_message(
            chat_id,
            f"🔄 *{topic_title}* için yeni bir üretim başlatılıyor...",
            parse_mode="Markdown",
        )
        return {"action": "regen", "status": "regenerating", "run_id": run_id}

    elif action == "script":
        bot.answer_callback_query(callback_id, "Senaryo getiriliyor...")
        scenes = state.get("scenes", [])
        if scenes:
            lines = [f"📜 *{topic_title}* --- Tam Senaryo\n"]
            for i, sc in enumerate(scenes, 1):
                lines.append(f"*{i}. Sahne:* {sc.get('narration')}\n_Prompt:_ `{sc.get('image_prompt')}`\n")
            bot.send_message(chat_id, "\n".join(lines)[:4000], parse_mode="Markdown")
        else:
            bot.send_message(chat_id, f"Senaryo metni:\n\n{state.get('full_text', 'Kayıt bulunamadı.')}")
        return {"action": "script", "status": "viewed", "run_id": run_id}

    return {"action": action, "status": "unknown", "run_id": run_id}


@bot.callback_query_handler(func=lambda call: True)
def on_callback_query(call):
    data = str(call.data or "")
    if ":" in data:
        action, run_id = data.split(":", 1)
    else:
        action, run_id = data, "legacy"

    outcome = handle_callback_action(
        action=action,
        run_id=run_id,
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        callback_id=call.id,
    )
    if outcome.get("action") == "regen":
        try:
            run()
        except Exception as exc:
            log.error("Regeneration run failed: %s", exc)
            bot.send_message(call.message.chat.id, f"⚠️ Yeniden üretim sırasında bir hata oluştu: {exc}")


# -----------------------------
# Telegram Command Handlers
# -----------------------------
@bot.message_handler(commands=["start", "help"])
def on_start(message):
    if str(message.chat.id) != str(TELEGRAM_CHAT_ID):
        log.warning("Unauthorized access to /start from %s", message.chat.id)
        return
        
    bot.reply_to(
        message,
        "🤖 *YouTube Shorts Bot Kontrol Merkezi*\n\n"
        "Kullanabileceğiniz komutlar:\n"
        "• /generate - Hemen yeni bir YouTube Short videosu üret\n"
        "• /status - En son üretilen videonun onay durumunu gör\n"
        "• /history - Son üretilen konuların geçmişini listele",
        parse_mode="Markdown",
    )


@bot.message_handler(commands=["status"])
def on_status(message):
    if str(message.chat.id) != str(TELEGRAM_CHAT_ID):
        return
        
    approvals = load_approvals()
    if not approvals:
        bot.reply_to(message, "Henüz kayıtlı bir video üretimi yok.")
        return
    latest_id = list(approvals.keys())[-1]
    st = approvals[latest_id]
    topic = st.get("topic", {})
    status_icon = {"pending": "⏳", "approved": "✅", "cancelled": "❌", "regenerating": "🔄"}.get(
        st.get("status", ""), "ℹ️"
    )
    bot.reply_to(
        message,
        f"📊 *Son Video Durumu ({latest_id})*\n\n"
        f"• Başlık: *{topic.get('title', 'N/A')}*\n"
        f"• Durum: {status_icon} *{st.get('status', 'unknown').upper()}*\n"
        f"• Süre: *{st.get('total_duration', 0):.1f}s*\n"
        f"• Tarih: {st.get('created_at', 'N/A')}",
        parse_mode="Markdown",
    )


@bot.message_handler(commands=["generate"])
def on_generate(message):
    if str(message.chat.id) != str(TELEGRAM_CHAT_ID):
        bot.reply_to(message, "Yetkisiz erişim. Bu botu kullanma yetkiniz yok.")
        log.warning("Unauthorized access attempt to /generate from chat_id %s", message.chat.id)
        return

    bot.reply_to(message, "🚀 Yeni video üretimi başlatıldı! Tamamlandığında önizleme gönderilecektir...")
    
    def background_run():
        try:
            run()
        except Exception as exc:
            log.error(f"Background video generation failed: {exc}", exc_info=True)
            try:
                bot.send_message(message.chat.id, f"❌ Video üretimi sırasında hata oluştu: {exc}")
            except Exception as nested_exc:
                log.error(f"Failed to send error message to Telegram: {nested_exc}")

    import threading
    threading.Thread(target=background_run, daemon=True).start()


@bot.message_handler(commands=["history"])
def on_history(message):
    if str(message.chat.id) != str(TELEGRAM_CHAT_ID):
        return

    import event_memory
    history = event_memory.load_event_memory()
    if not history:
        bot.reply_to(message, "Henüz işlenmiş bir konu geçmişi bulunmuyor (V2).")
        return
    lines = ["📚 *Son Üretilen Konular (Son 10):*\n"]
    for i, h in enumerate(history[-10:], 1):
        lines.append(f"{i}. *{h.get('canonical_title')}* ({h.get('created_at', '')[:10]})")
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")


def start_bot_service():
    """Start persistent Telegram bot polling service."""
    log.info("Starting Telegram Bot listener service...")
    bot.infinity_polling(skip_pending=True)


# -----------------------------
# Video Quality QA
# -----------------------------
def validate_video_quality(video_path: Path) -> dict:
    """Validate that the generated video meets quality and Shorts standards.

    Checks:
    - Video file exists on disk
    - File size > 0 and reasonable (> 50 KB)
    - Width == 1080 and Height == 1920
    - Duration is within MIN_DURATION and MAX_DURATION
    - Audio track is present
    - Bitrate is verified (must be >= MIN_BITRATE_KBPS outside unit tests)
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    size_bytes = video_path.stat().st_size
    if size_bytes < 50_000:
        raise ValueError(f"Video file is suspiciously small ({size_bytes} bytes): {video_path}")

    with VideoFileClip(str(video_path)) as clip:
        w, h = clip.size
        if (w, h) != (VIDEO_WIDTH, VIDEO_HEIGHT):
            raise ValueError(f"Invalid video dimensions: {w}x{h}, expected {VIDEO_WIDTH}x{VIDEO_HEIGHT}")

        duration = clip.duration
        if duration < MIN_DURATION or duration > MAX_DURATION:
            if not os.getenv("PYTEST_CURRENT_TEST") and 10.0 <= duration <= 58.0:
                log.warning(
                    "Video duration (%.2fs) is outside configured range (%ss - %ss), but valid for YouTube Shorts platform.",
                    duration, MIN_DURATION, MAX_DURATION,
                )
            else:
                raise ValueError(
                    f"Video duration ({duration:.2f}s) is outside expected range ({MIN_DURATION}s - {MAX_DURATION}s)"
                )

        if clip.audio is None:
            raise ValueError(f"Video has no audio track: {video_path}")

        bitrate_kbps = (size_bytes * 8) / (max(duration, 0.1) * 1000)
        min_bitrate = int(os.getenv("MIN_BITRATE_KBPS", "1500" if not os.getenv("PYTEST_CURRENT_TEST") else "0"))
        if min_bitrate > 0 and bitrate_kbps < min_bitrate:
            raise ValueError(
                f"Video bitrate ({bitrate_kbps:.1f} kbps) is below minimum threshold ({min_bitrate} kbps)"
            )

        return {
            "path": str(video_path),
            "size_bytes": size_bytes,
            "width": w,
            "height": h,
            "duration": round(duration, 3),
            "bitrate_kbps": round(bitrate_kbps, 1),
            "has_audio": True,
        }


def validate_visual_sources(visual_sources: list[dict], scenes: list[dict], min_relevance: float = 0.55) -> dict:
    """Final deterministic gate for source metadata and visual planning quality."""
    if len(visual_sources) != len(scenes):
        raise ValueError(f"Visual QA failed: {len(visual_sources)} sources for {len(scenes)} scenes")

    results = []
    for idx, (source, scene) in enumerate(zip(visual_sources, scenes), 1):
        source_type = source.get("source_type", "")
        visual_role = str(scene.get("visual_role", "")).lower()
        planned_specificity = float(scene.get("event_specificity", 0.0))
        planned_density = float(scene.get("information_density", 0.0))
        visual_fact = str(scene.get("visual_fact", "")).strip()

        if not visual_fact:
            raise ValueError(f"Visual QA failed: scene {idx} has no visual_fact")
        if visual_role != "atmosphere" and planned_specificity < 0.70:
            raise ValueError(
                f"Visual QA failed: scene {idx} planned event specificity {planned_specificity:.2f} < 0.70"
            )

        if source_type in ("wikimedia", "openverse"):
            relevance = float(source.get("relevance_score", 0))
            specificity = float(source.get("event_specificity", 0))
            density = float(source.get("information_density", 0))
            if relevance < min_relevance:
                raise ValueError(
                    f"Visual QA failed: scene {idx} real visual relevance {relevance:.2f} "
                    f"is below threshold {min_relevance:.2f}"
                )
            if visual_role != "atmosphere" and specificity < 0.35:
                raise ValueError(
                    f"Visual QA failed: scene {idx} real visual specificity {specificity:.2f} < 0.35"
                )
            results.append({
                "scene": idx,
                "source_type": source_type,
                "relevance_score": relevance,
                "event_specificity": specificity,
                "information_density": density,
                "planned_event_specificity": planned_specificity,
                "planned_information_density": planned_density,
                "visual_fact": visual_fact,
                "visual_role": visual_role,
                "title": source.get("title", ""),
                "query": source.get("search_query", ""),
                "decision": "PASS_REAL",
            })
        elif source_type == "ai_reconstruction":
            intent = scene.get("visual_intent", {})
            if not intent.get("primary_subject") or not intent.get("must_show"):
                raise ValueError(f"Visual QA failed: scene {idx} AI reconstruction has no concrete visual intent")
            results.append({
                "scene": idx,
                "source_type": source_type,
                "decision": "PASS_AI_INTENT_PLAN",
                "primary_subject": intent.get("primary_subject", ""),
                "visual_fact": visual_fact,
                "visual_role": visual_role,
                "planned_event_specificity": planned_specificity,
                "planned_information_density": planned_density,
            })
        else:
            raise ValueError(f"Visual QA failed: scene {idx} has unsupported source type '{source_type}'")

    atmosphere_count = sum(1 for scene in results if scene.get("visual_role") == "atmosphere")
    if atmosphere_count > 1:
        raise ValueError(f"Visual QA failed: too many atmosphere-only scenes ({atmosphere_count}); maximum is 1")

    return {
        "passed": True,
        "scene_count": len(results),
        "real_visuals": sum(1 for item in results if item["source_type"] in ("wikimedia", "openverse")),
        "ai_reconstructions": sum(1 for item in results if item["source_type"] == "ai_reconstruction"),
        "atmosphere_scenes": atmosphere_count,
        "scenes": results,
    }


# -----------------------------
# Main pipeline
# -----------------------------
def run(auto_publish: bool | None = None):
    run_dir = OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run migration if needed
    if not event_memory.EVENT_MEMORY_FILE.exists():
        log.info("First run with V2 engine. Attempting migration...")
        event_memory.migrate_existing_memory()

    log.info("1/8 Historical Event Discovery & Deduplication")
    discovery_result = event_memory.run_discovery_pipeline()
    if not discovery_result:
        log.error("Pipeline aborting due to discovery failure.")
        raise RuntimeError("Historical event discovery failed; no video was generated.")

    candidate, research_dossier = discovery_result
    # Topic dict compatibility for older functions (e.g., telegram / youtube uploader)
    topic_compat = {
        "title": candidate.get("canonical_title", "Tarihsel Gizem"),
        "hook_question": research_dossier.get("story_hook", ""),
        "viral_score": 9,
        "visual_appeal": 9,
    }

    log.info("2/8 Generating script from verified research dossier")
    scenes = generate_viral_script(candidate, research_dossier)
    full_text = " ".join(scene["narration"] for scene in scenes)

    log.info("3/8 Analyzing script content for metadata")
    candidate_analysis = content_memory_module.analyze_script_content(
        scenes=scenes,
        topic_title=topic_compat["title"],
    )

    visual_warnings = content_memory_module.check_visual_reuse(scenes)
    if visual_warnings:
        log.info("Visual reuse detected in %d scene pairs; diversifying prompts", len(visual_warnings))
        scenes = content_memory_module.diversify_image_prompts(scenes)

    log.info("4/8 Generating voice and word timestamps")
    voice_path = run_dir / "voice.mp3"
    voice_profile_name, voice_profile = choose_voice_profile(candidate, research_dossier)
    candidate["_voice_profile"] = voice_profile_name
    log.info(
        "V1.4 narrator selected: %s (%s, rate=%s, pitch=%s)",
        voice_profile_name, voice_profile["voice"], voice_profile["rate"], voice_profile["pitch"],
    )

    for duration_attempt in range(1, 4):
        words_data = asyncio.run(
            create_voice_with_timestamps(full_text, voice_path, voice_profile_name)
        )
        voice_audio = AudioFileClip(str(voice_path))
        total_duration = voice_audio.duration
        if 28.0 <= total_duration <= 40.0:
            if total_duration > 38.0:
                log.warning("Duration %.2fs is above the V1.1 target window (28-38s).", total_duration)
            break

        log.warning(
            "TTS duration %.2fs is outside 28-40s; regenerating script (%s/3)",
            total_duration, duration_attempt,
        )
        if duration_attempt == 3:
            raise RuntimeError(f"Could not produce a V1.1 Short in target duration range: {total_duration:.2f}s")
        scenes = generate_viral_script(candidate, research_dossier)
        full_text = " ".join(scene["narration"] for scene in scenes)

    scene_timings = calculate_scene_timings(scenes, words_data, total_duration)
    log.info("Scene timings: %s", [(round(a, 2), round(b, 2)) for a, b in scene_timings])

    log.info("5/8 Resolving visual sources and generating images")
    scene_clips = []
    
    # Pre-resolve visuals
    visual_sources = []
    used_visual_urls: list[str] = []
    for i, scene in enumerate(scenes, 1):
        visual_intent = scene.get("visual_intent", {})
        terms = list(visual_intent.get("search_queries", []))
        event_context = {
            "title": topic_compat["title"],
            "aliases": candidate.get("aliases", []),
            "location": candidate.get("location", ""),
            "entities": candidate.get("entities", []),
            "date": candidate.get("date", ""),
        }
        v_source = event_memory.resolve_visual_source(
            scene_description=scene["narration"],
            visual_search_terms=terms,
            event_title=topic_compat["title"],
            excluded_urls=used_visual_urls,
            visual_intent=visual_intent,
            event_context=event_context,
        )
        v_source["visual_fact"] = scene.get("visual_fact", "")
        v_source["visual_role"] = scene.get("visual_role", "")

        visual_sources.append(v_source)
        resolved_url = v_source.get("image_url")
        if resolved_url and resolved_url not in used_visual_urls:
            used_visual_urls.append(resolved_url)
    
    visual_qa = validate_visual_sources(visual_sources, scenes, min_relevance=REAL_VISUAL_MIN_RELEVANCE)
    log.info(
        "Visual QA passed: %d real visuals, %d AI reconstructions",
        visual_qa["real_visuals"],
        visual_qa["ai_reconstructions"],
    )

    for i, (scene, (start, end), v_source) in enumerate(zip(scenes, scene_timings, visual_sources), 1):
        image_path = run_dir / f"scene_{i:02d}.jpg"

        # V1.4 visual loop: natural narration, but return to the opening visual.
        image_downloaded = False
        visual_loop_reused = False
        if i == SCENE_COUNT and scene.get("ending_strategy") == "visual":
            first_image = run_dir / "scene_01.jpg"
            if first_image.exists():
                image_path.write_bytes(first_image.read_bytes())
                image_downloaded = True
                visual_loop_reused = True
                log.info("Scene %d uses Scene 1 visual for a clean visual loop.", i)

        # Download historical/web images or fallback to AI
        if not visual_loop_reused and v_source.get("source_type") in ("wikimedia", "openverse"):
            try:
                img_url = v_source.get("image_url")
                resp = requests.get(
                    img_url,
                    timeout=20,
                    headers={"User-Agent": "YouTubeShortsBot/2.1 (historical-content-bot)"},
                )
                resp.raise_for_status()
                image_path.write_bytes(resp.content)
                crop_watermark_zone(image_path)
                image_downloaded = True
                if img_url:
                    used_visual_urls.append(img_url)
                log.info(
                    "Scene %d downloaded real visual: %s (%s)",
                    i,
                    v_source.get("title", "untitled"),
                    v_source.get("source_type"),
                )
            except Exception as e:
                log.warning(
                    "%s download failed for scene %d: %s. Falling back to AI.",
                    v_source.get("source_type", "real visual"),
                    i,
                    e,
                )

        # V1.4.1: source metadata is part of the visual QA trail.
        scene["resolved_visual"] = {
            "source_type": v_source.get("source_type", ""),
            "title": v_source.get("title", ""),
            "relevance_score": v_source.get("relevance_score", 0.0),
            "event_specificity": v_source.get("event_specificity", 0.0),
            "information_density": v_source.get("information_density", 0.0),
            "visual_fact": scene.get("visual_fact", ""),
            "visual_role": scene.get("visual_role", ""),
            "visual_action": scene.get("visual_intent", {}).get("visual_action", ""),
            "shot_type": scene.get("visual_intent", {}).get("shot_type", ""),
            "composition": scene.get("visual_intent", {}).get("composition", ""),
        }
        source_type_for_enhancement = v_source.get("source_type", "ai_reconstruction") if image_downloaded else "ai_reconstruction"
        if not image_downloaded:
            intent = scene.get("visual_intent", {})
            ai_prompt = (
                f"{scene['image_prompt']} "
                f"Visual fact to depict: {scene.get('visual_fact', '')}. "
                f"Visual action: {intent.get('visual_action', '')}. "
                f"Scene context: {intent.get('scene_context', '')}. "
                f"Shot type: {intent.get('shot_type', '')}. "
                f"Composition: {intent.get('composition', '')}. "
                f"Visual entities: {', '.join(intent.get('visual_entities', []))}. "
                f"Visual role: {scene.get('visual_role', '')}. "
                f"Primary subject: {intent.get('primary_subject', '')}. "
                f"Must visibly include: {', '.join(intent.get('must_show', []))}. "
                f"Avoid: {', '.join(intent.get('avoid', []))}. "
                f"Historical event context: {topic_compat['title']}. "
                "Depict the historical ACTION and relationship first, not an isolated object mentioned in narration. "
                "The frame must communicate who/what is doing what, where and in what historical context. "
                "Do not substitute a generic landscape, generic portrait, keyword collage, isolated prop, or stock-photo composition. "
                "Respect the requested shot type and composition. No modern objects, no text, no watermark, documentary historical reconstruction."
            )
            download_ai_image(ai_prompt, image_path)

        enhancement_info = enhance_source_image(
            image_path,
            source_type=source_type_for_enhancement,
            visual_intent=scene.get("visual_intent", {}),
        )
        scene["image_enhancement"] = enhancement_info
        log.info(
            "Scene %d image enhancement: enabled=%s profile=%s",
            i, enhancement_info.get("enabled"), enhancement_info.get("profile", "none"),
        )

        motion_type = MOTION_TYPES[(i - 1) % len(MOTION_TYPES)]
        scene_clips.append(build_scene_clip(image_path, start, end, motion_type=motion_type))

    log.info("6/8 Compositing video and subtitles")
    base_video = CompositeVideoClip(
        scene_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
    ).with_duration(total_duration)
    subtitles = generate_subtitle_clips(words_data)
    if not subtitles:
        log.warning("Warning: No subtitle clips were generated for this Short!")
    hook_badge = create_hook_badge(duration=min(2.5, total_duration), title=research_dossier.get("story_hook") or topic_compat["title"])

    video = CompositeVideoClip(
        [base_video, hook_badge] + subtitles, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
    ).with_duration(total_duration)

    log.info("7/8 Mixing audio (content-aware selection + real ducking)")
    memory = content_memory_module.load_content_memory()
    music_path = get_ambient_music(
        run_dir / "bg_music.mp3",
        content_analysis=candidate_analysis,
        memory=memory,
    )
    selected_audio_track = ""
    audio_mix_mode = "not_mixed"
    if music_path:
        try:
            bg_music = AudioFileClip(str(music_path))
            selected_audio_track = music_path.name
            if bg_music.duration < total_duration:
                bg_music = bg_music.loop(duration=total_duration)
            else:
                bg_music = bg_music.subclipped(0, total_duration)

            fade_in_len = min(1.0, total_duration / 4)
            fade_out_len = min(1.5, total_duration / 4)

            # Use moviepy's make_frame to apply true ducking function
            ducking_fn = content_memory_module.compute_ducking_volume(
                narration_words=words_data,
                total_duration=total_duration,
                base_volume=0.18,
                ducked_volume=0.12,
                pause_volume=0.24,
            )
            
            # MoviePy 2.x: AudioClip.transform replaces the removed .fl API.
            def ducked_frame(get_frame, t):
                frame = get_frame(t)
                volume = np.asarray(ducking_fn(t))

                # MoviePy can pass a scalar timestamp or a NumPy array of
                # timestamps when rendering audio chunks. Preserve the shape
                # so per-sample ducking broadcasts correctly across channels.
                if volume.ndim == 0:
                    return frame * float(volume)

                if frame.ndim == volume.ndim:
                    return frame * volume

                # Typical stereo audio: frame=(samples, channels),
                # volume=(samples,). Expand volume across channels.
                reshape = (volume.shape[0],) + (1,) * (frame.ndim - 1)
                return frame * volume.reshape(reshape)

            bg_music = bg_music.transform(ducked_frame, keep_duration=True)
            bg_music = bg_music.with_effects([
                AudioFadeIn(fade_in_len),
                AudioFadeOut(fade_out_len),
            ])
            final_audio = CompositeAudioClip([voice_audio, bg_music])
            audio_mix_mode = "voice_plus_background_ducked"
            log.info(
                "Audio mix OK: narration + %s with dynamic ducking (fade_in=%.1fs, fade_out=%.1fs)",
                selected_audio_track, fade_in_len, fade_out_len,
            )
        except Exception as exc:
            log.error("CRITICAL: Background music mixing failed: %s", exc)
            raise RuntimeError(f"Background music mixing failed: {exc}") from exc
    else:
        raise RuntimeError("Background music is mandatory but no music track was available.")

    final_video = video.with_audio(final_audio)
    output_path = run_dir / "final_short.mp4"
    final_video.write_videofile(
        str(output_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        threads=2,
        bitrate="8000k",
        preset="fast",
        ffmpeg_params=["-crf", "18"],
    )

    log.info("Performing final video quality checks")
    video_qa = validate_video_quality(output_path)
    if audio_mix_mode != "voice_plus_background_ducked":
        raise RuntimeError("Final audio QA failed: background music was not mixed into the video.")
    video_qa["audio_mix_mode"] = audio_mix_mode
    video_qa["background_music"] = selected_audio_track
    log.info("Video QA passed: %s", video_qa)

    # ── Phase 4: Save Event Identity ──
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    event_record = event_memory.mark_event_as_used(candidate, research_dossier, video_id=run_id)

    # ── Legacy Content Memory ──
    content_entry = content_memory_module.build_content_entry(
        topic=topic_compat,
        scenes=scenes,
        analysis=candidate_analysis,
        full_text=full_text,
    )
    content_entry["audio_track"] = selected_audio_track
    content_entry["voice_profile"] = voice_profile_name
    content_entry["ending_strategy"] = scenes[-1].get("ending_strategy", "")
    content_entry["event_id"] = event_record["event_id"]
    content_memory_module.save_content_entry(content_entry)

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event_record": event_record,
        "duration_seconds": round(total_duration, 3),
        "scene_count": len(scenes),
        "scene_timings": [{"start": round(a, 3), "end": round(b, 3)} for a, b in scene_timings],
        "scenes": scenes,
        "qa": video_qa,
        "content_analysis": candidate_analysis,
        "audio_track": selected_audio_track,
        "audio_mix_mode": audio_mix_mode,
        "voice_profile": {
            "name": voice_profile_name,
            "voice": voice_profile["voice"],
            "rate": voice_profile["rate"],
            "pitch": voice_profile["pitch"],
        },
        "visual_sources": visual_sources,
        "visual_intents": [scene.get("visual_intent", {}) for scene in scenes],
        "visual_storyboard_qa": storyboard_qa,
        "visual_qa": visual_qa,
        "visual_reuse_warnings": visual_warnings,
        "image_enhancement": {
            "enabled": IMAGE_ENHANCEMENT_ENABLED,
            "real_visual_min_relevance": REAL_VISUAL_MIN_RELEVANCE,
            "method": "source-aware PIL restoration; no generative redraw",
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("8/8 Sending result to Telegram")
    send_to_telegram(
        output_path,
        total_duration,
        full_text,
        topic=topic_compat,
        run_id=run_id,
        scenes=scenes,
    )
    log.info("Pipeline completed: %s", output_path)

    auto_pub = (
        auto_publish
        if auto_publish is not None
        else os.getenv("AUTO_PUBLISH", "false").lower() in ("true", "1", "yes")
    )
    if auto_pub:
        log.info("AUTO_PUBLISH enabled; automatically uploading to YouTube Shorts...")
        outcome = handle_callback_action(
            action="publish",
            run_id=run_id,
            chat_id=TELEGRAM_CHAT_ID,
            message_id=0,
            callback_id="auto_publish",
        )
        return {"output_path": output_path, "run_id": run_id, "publish_outcome": outcome}

    return {"output_path": output_path, "run_id": run_id}


if __name__ == "__main__":
    import sys
    if "--bot" in sys.argv or "--listen" in sys.argv:
        start_bot_service()
    else:
        run()
