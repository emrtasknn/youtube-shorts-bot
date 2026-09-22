import asyncio
import json
import logging
import os
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
from PIL import Image, ImageDraw, ImageFont
from telebot import types

import youtube_uploader



# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOPICS_HISTORY_FILE = Path(os.getenv("TOPICS_HISTORY_FILE", "topics_history.json"))
APPROVALS_FILE = Path(os.getenv("APPROVALS_FILE", "approvals.json"))

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30
SCENE_COUNT = 7
TARGET_DURATION = 25
MIN_DURATION = 18
MAX_DURATION = 32
MIN_TOTAL_WORDS = 35
MAX_TOTAL_WORDS = 78

# The image provider's mark is kept out of the final frame by cropping the
# lower part of the generated image. This is intentionally a fixed crop,
# not an attempt to edit/inpaint the mark out of the source image.
WATERMARK_CROP_PX = int(os.getenv("WATERMARK_CROP_PX", "75"))
ZOOM_AMOUNT = float(os.getenv("ZOOM_AMOUNT", "0.07"))

DEFAULT_CANDIDATE_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest",
    "gemini-1.5-pro",
    "gemini-pro-latest",
]


def get_candidate_models() -> list[str]:
    user_model = os.getenv("GEMINI_MODEL")
    if user_model:
        return [user_model] + [m for m in DEFAULT_CANDIDATE_MODELS if m != user_model]
    return list(DEFAULT_CANDIDATE_MODELS)


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
        if not narration or not image_prompt:
            raise ValueError(f"Scene {i} is missing narration or image_prompt")
        word_count = len(narration.split())
        if word_count < 3 or word_count > 35:
            raise ValueError(f"Scene {i} has suspicious narration length: {word_count} words")
        total_words += word_count
        cleaned.append({"narration": narration, "image_prompt": image_prompt})

    # Keep Shorts comfortably short while allowing natural variation.
    if total_words < MIN_TOTAL_WORDS or total_words > MAX_TOTAL_WORDS:
        raise ValueError(f"Total narration length is outside expected range: {total_words} words")

    return cleaned


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


def discover_and_score_topics(history_titles: list[str]) -> dict:
    """Discover candidate viral history topics, score them, and pick the best unseen one."""
    negative_prompt = ""
    if history_titles:
        recent = ", ".join(history_titles[-30:])
        negative_prompt = f"\nBu konular daha önce işlendi, bunları KESİNLİKLE SEÇME VEYA TEKRAR ETME:\n{recent}\n"

    prompt = f"""
Sen YouTube Shorts için viral tarih içerikleri keşfeden uzman bir araştırmacısın.
İzleyiciyi ilk saniyeden ekrana kilitleyecek, az bilinen, şaşırtıcı veya esrarengiz 3 farklı tarihsel olay öner.
{negative_prompt}

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
    candidate_models = get_candidate_models()

    for attempt in range(3):
        for model in candidate_models:
            try:
                log.info("Discovering topics: %s (attempt %s)", model, attempt + 1)
                res = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                if res and res.text:
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
                log.warning("Topic discovery failed with %s: %s", model, exc)
            time.sleep(1)
        time.sleep((attempt + 1) * 3)

    # Reliable fallback if API is unavailable
    return {
        "title": "Kayıp Koloni Roanoke Gizemi",
        "hook_question": "115 İngiliz yerleşimci bir gecede nereye kayboldu?",
        "viral_score": 8,
        "visual_appeal": 8,
    }


def evaluate_script_quality(scenes: list[dict], topic: dict | None = None) -> dict:
    """Evaluate script quality and storytelling metrics. Returns score 0-100 and issues."""
    if not scenes or len(scenes) != SCENE_COUNT:
        return {"score": 0, "passed": False, "issues": ["Invalid scene count"]}

    score = 100
    issues = []

    # 1. Hook check in Scene 1
    s1 = scenes[0].get("narration", "")
    hook_indicators = ["?", "nasıl", "neden", "kim", "nerede", "hiç", "inanılmaz", "gizem", "şok", "esrarengiz", "fakat"]
    if not any(ind in s1.lower() for ind in hook_indicators):
        score -= 15
        issues.append("Scene 1 lacks a distinct hook question or trigger word")

    # 2. Seamless loop check in Scene 7
    s7 = scenes[-1].get("narration", "").strip()
    loop_indicators = [";", ":", "aslında", "çünkü", "işte bu yüzden", "başlıyor", "bunun cevabı", "ve sır", "sebebi"]
    if not any(ind in s7.lower() for ind in loop_indicators) and not s7.endswith((";", ":", "...")):
        score -= 15
        issues.append("Scene 7 lacks seamless loop connector")

    # 3. Word count balance
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


def generate_viral_script(topic: dict | None = None) -> tuple[list[dict], dict]:
    """Generate a 7-scene structured script based on topic research with storytelling arc and seamless loop."""
    if topic is None:
        history = load_topic_history()
        history_titles = [item.get("title", "") for item in history if isinstance(item, dict)]
        topic = discover_and_score_topics(history_titles)

    topic_title = topic.get("title", "Tarihsel Gizem")
    hook_question = topic.get("hook_question", "")

    prompt = f"""
Sen YouTube Shorts için viral tarih belgeselleri üreten usta bir yönetmensin.
Konu: {topic_title}
Önerilen Açılış Kancası: {hook_question}

Bu olayı tam {SCENE_COUNT} sahnelik, yüksek tempolu bir Shorts senaryosu olarak yaz.
Hedef seslendirme süresi yaklaşık {TARGET_DURATION} saniye (toplam 40-65 kelime).

Sahne Hikaye Şablonu:
- 1. Sahne: Güçlü Kanca (Hook) - İlk 1-2 saniyede izleyiciyi ekrana bağlayacak şok edici soru veya gerçek.
- 2-3. Sahne: Merak ve Tırmanış (Escalation) - Olayın karanlık ve gizemli ayrıntıları.
- 4-5. Sahne: Çarpıcı Kırılma (The Twist) - Tarihçileri şaşkına çeviren beklenmedik boyut.
- 6. Sahne: Yankı - Bu olayın tarihte bıraktığı silinmez iz.
- 7. Sahne: Kusursuz Döngü Köprüsü (Seamless Loop) - Asla veda veya bitiş bildirmemeli! Cümle öyle bir bağlaçla bitmeli ki, video başa sarınca 1. sahneye pürüzsüzce bağlansın (örn: "...ve bu dehşet verici sırrın cevabı aslında;", "...çünkü tarihin en büyük gizemi tam da burada başlıyor:").

Kurallar:
1. Görsel promptları İngilizce, sinematik, dikey (vertical 9:16), fotogerçekçi, 8k render ve PG-13 belgesel tonunda olsun.
2. Kesinlikle kan, aşırı şiddet veya ceset istemiyorum.
3. Tarihsel iddiaları uydurma.
4. Çıktı SADECE geçerli JSON olsun.

Şema:
{{
  "scenes": [
    {{"narration": "...", "image_prompt": "..."}}
  ]
}}
"""

    candidate_models = get_candidate_models()

    for attempt in range(4):
        for model in candidate_models:
            try:
                log.info("Requesting structured script: %s (attempt %s)", model, attempt + 1)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                if response and response.text:
                    data = json.loads(response.text.strip())
                    scenes = validate_script(data)
                    qa_result = evaluate_script_quality(scenes, topic)
                    log.info("Script QA evaluated: score=%s, issues=%s", qa_result["score"], qa_result["issues"])
                    return scenes, topic
            except Exception as exc:
                log.warning("Script generation failed with %s: %s", model, exc)
            time.sleep(1)
        time.sleep((attempt + 1) * 3)

    log.warning("All Gemini model attempts failed (503/network); using curated historical fallback story.")
    fallback_scenes = [
        {
            "narration": "1590 yılında Kuzey Karolina'daki Roanoke adasında akılalmaz bir gizem yaşandı.",
            "image_prompt": "Cinematic vertical 9:16 shot of an abandoned wooden colonial fort on a misty island, 1590 era, photorealistic documentary style, 8k",
        },
        {
            "narration": "115 İngiliz yerleşimci arkalarında hiçbir savaş veya saldırı izi bırakmadan kayboldu.",
            "image_prompt": "Eerie empty village with silent wooden cabins, fog rolling through dirt streets, no people, dramatic lighting, vertical 9:16",
        },
        {
            "narration": "Evler, eşyalar ve yiyecekler yerli yerinde öylece terk edilmişti.",
            "image_prompt": "Close-up interior of a colonial wooden cottage, warm hearth, untouched dinner on wooden table, vertical 9:16, cinematic",
        },
        {
            "narration": "Bulunan tek ipucu, yaşlı bir ağaca kazınmış gizemli bir kelimeydi:",
            "image_prompt": "Dramatic close-up of a rustic wooden tree trunk with mysterious word CROATOAN carved deeply into bark, dark moody lighting, vertical 9:16",
        },
        {
            "narration": "Kroatoan! Bu kelimenin anlamını ve yerleşimcilerin akıbetini kimse çözemedi.",
            "image_prompt": "Ancient faded nautical parchment map showing North Carolina coast with mysterious symbols, vertical 9:16, historical documentary style",
        },
        {
            "narration": "Yüzyıllar boyunca yapılan kazılar bile bu kayıp koloniden tek bir iz bulamadı.",
            "image_prompt": "Modern archaeologists excavating historical earth beneath tall trees, moody sunset light, vertical 9:16, cinematic",
        },
        {
            "narration": "Ve tarihin en karanlık sırrının cevabı aslında;",
            "image_prompt": "Mysterious ghostly ship sailing into thick white fog under moonlight, vertical 9:16, photorealistic, 8k",
        },
    ]
    return fallback_scenes, topic



# -----------------------------
# Voice + word timing
# -----------------------------
async def create_voice_with_timestamps(text: str, audio_path: Path):
    communicate = edge_tts.Communicate(
        text,
        voice="tr-TR-AhmetNeural",
        rate="+15%",
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
    """Remove the bottom watermark area by cropping, then fit to 9:16.

    We do not edit/inpaint the source image. The removed area is simply kept
    outside the final composition, as requested.
    """
    with Image.open(filename) as img:
        img = img.convert("RGB")
        w, h = img.size
        crop_px = min(max(WATERMARK_CROP_PX, 0), max(h - 2, 1))
        cropped = img.crop((0, 0, w, h - crop_px))

        target_ratio = VIDEO_WIDTH / VIDEO_HEIGHT
        crop_w = min(cropped.width, int(cropped.height * target_ratio))
        left = max((cropped.width - crop_w) // 2, 0)
        fitted = cropped.crop((left, 0, left + crop_w, cropped.height))
        final_img = fitted.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
        final_img.save(filename, quality=95)


def download_ai_image(prompt_text: str, filename: Path):
    cleaned = urllib.parse.quote(prompt_text)
    url = (
        f"https://image.pollinations.ai/prompt/{cleaned}"
        f"?width={VIDEO_WIDTH}&height={VIDEO_HEIGHT}&model=turbo&nologo=true"
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

    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
        img.save(image_path, quality=95)

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
def get_ambient_music(music_path: Path):
    if music_path.exists():
        return music_path

    url = "https://assets.mixkit.co/music/preview/mixkit-cinematic-mystery-suspense-hum-2852.mp3"

    def download():
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        music_path.write_bytes(response.content)
        return music_path

    try:
        return retry_call(download, attempts=2, base_delay=2, label="music download")
    except Exception as exc:
        log.warning("Music unavailable: %s", exc)
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
    bot.reply_to(message, "🚀 Yeni video üretimi başlatıldı! Tamamlandığında önizleme gönderilecektir...")
    try:
        run()
    except Exception as exc:
        bot.reply_to(message, f"❌ Video üretimi sırasında hata oluştu: {exc}")


@bot.message_handler(commands=["history"])
def on_history(message):
    history = load_topic_history()
    if not history:
        bot.reply_to(message, "Henüz işlenmiş bir konu geçmişi bulunmuyor.")
        return
    lines = ["📚 *Son Üretilen Konular (Son 10):*\n"]
    for i, h in enumerate(history[-10:], 1):
        lines.append(f"{i}. *{h.get('title')}* ({h.get('created_at', '')[:10]})")
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
            raise ValueError(
                f"Video duration ({duration:.2f}s) is outside expected range ({MIN_DURATION}s - {MAX_DURATION}s)"
            )

        if clip.audio is None:
            raise ValueError(f"Video has no audio track: {video_path}")

        return {
            "path": str(video_path),
            "size_bytes": size_bytes,
            "width": w,
            "height": h,
            "duration": round(duration, 3),
            "has_audio": True,
        }


# -----------------------------
# Main pipeline
# -----------------------------
def run(auto_publish: bool | None = None):
    run_dir = OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("1/6 Discovering topic and generating structured script")
    scenes, topic = generate_viral_script()
    full_text = " ".join(scene["narration"] for scene in scenes)

    log.info("2/6 Generating voice and word timestamps")
    voice_path = run_dir / "voice.mp3"

    # Regenerate the script if the actual TTS duration is far from the target.
    # This is more reliable than accepting a 40+ second Short just because the
    # prompt asked for ~25 seconds.
    for duration_attempt in range(1, 4):
        words_data = asyncio.run(create_voice_with_timestamps(full_text, voice_path))
        voice_audio = AudioFileClip(str(voice_path))
        total_duration = voice_audio.duration
        if MIN_DURATION <= total_duration <= MAX_DURATION:
            break

        log.warning(
            "TTS duration %.2fs is outside %s-%ss; regenerating script (%s/3)",
            total_duration, MIN_DURATION, MAX_DURATION, duration_attempt,
        )
        if duration_attempt == 3:
            raise RuntimeError(
                f"Could not produce a Short in the target duration range after 3 attempts: {total_duration:.2f}s"
            )
        scenes, topic = generate_viral_script(topic=topic)
        full_text = " ".join(scene["narration"] for scene in scenes)

    scene_timings = calculate_scene_timings(scenes, words_data, total_duration)
    log.info("Scene timings: %s", [(round(a, 2), round(b, 2)) for a, b in scene_timings])

    log.info("3/6 Generating %s AI images", SCENE_COUNT)
    scene_clips = []
    for i, (scene, (start, end)) in enumerate(zip(scenes, scene_timings), 1):
        image_path = run_dir / f"scene_{i:02d}.jpg"
        download_ai_image(scene["image_prompt"], image_path)
        motion_type = MOTION_TYPES[(i - 1) % len(MOTION_TYPES)]
        scene_clips.append(build_scene_clip(image_path, start, end, motion_type=motion_type))

    log.info("4/6 Compositing video and subtitles")
    base_video = CompositeVideoClip(
        scene_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
    ).with_duration(total_duration)
    subtitles = generate_subtitle_clips(words_data)
    hook_badge = create_hook_badge(duration=min(2.5, total_duration), title=topic.get("title"))

    video = CompositeVideoClip(
        [base_video, hook_badge] + subtitles, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
    ).with_duration(total_duration)

    log.info("5/6 Mixing audio")
    music_path = get_ambient_music(run_dir / "bg_music.mp3")
    if music_path:
        try:
            bg_music = AudioFileClip(str(music_path))
            if bg_music.duration < total_duration:
                bg_music = bg_music.loop(duration=total_duration)
            else:
                bg_music = bg_music.subclipped(0, total_duration)

            fade_in_len = min(1.0, total_duration / 4)
            fade_out_len = min(1.5, total_duration / 4)
            bg_music = (
                bg_music
                .with_volume_scaled(0.10)
                .with_effects([
                    AudioFadeIn(fade_in_len),
                    AudioFadeOut(fade_out_len),
                ])
            )
            final_audio = CompositeAudioClip([voice_audio, bg_music])
        except Exception as exc:
            log.warning("Music mixing failed; using voice only: %s", exc)
            final_audio = voice_audio
    else:
        final_audio = voice_audio

    final_video = video.with_audio(final_audio)
    output_path = run_dir / "final_short.mp4"
    final_video.write_videofile(
        str(output_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        threads=2,
    )

    log.info("Performing final video quality checks")
    video_qa = validate_video_quality(output_path)
    log.info("Video QA passed: %s", video_qa)

    # Save newly completed topic to persistent history to avoid repeating it
    save_topic_to_history({
        "title": topic.get("title"),
        "hook_question": topic.get("hook_question"),
        "viral_score": topic.get("viral_score"),
        "visual_appeal": topic.get("visual_appeal"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": round(total_duration, 2),
    })

    # Save machine-readable metadata for debugging and future analytics.
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "topic": topic,
        "duration_seconds": round(total_duration, 3),
        "scene_count": len(scenes),
        "scene_timings": [
            {"start": round(a, 3), "end": round(b, 3)} for a, b in scene_timings
        ],
        "scenes": scenes,
        "qa": video_qa,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log.info("6/6 Sending result to Telegram")
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    send_to_telegram(
        output_path,
        total_duration,
        full_text,
        topic=topic,
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

