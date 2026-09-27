"""Content routing and topic generation for the Shorts bot.

Daily portfolio:
4 TREND_HISTORY + 2 TODAY_IN_HISTORY + 2 AYT_HISTORY +
1 HISTORY_FACT + 1 CUSTOM.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import gemini_config

log = logging.getLogger("shorts-bot.content")

CONTENT_PLAN = [
    {"content_type": "TREND_HISTORY", "label": "🔥 Trend History", "count": 4},
    {"content_type": "TODAY_IN_HISTORY", "label": "📅 Tarihte Bugün", "count": 2},
    {"content_type": "AYT_HISTORY", "label": "🎓 AYT Tarih", "count": 2},
    {"content_type": "HISTORY_FACT", "label": "🧠 History Fact", "count": 1},
    {"content_type": "CUSTOM", "label": "🧪 Custom", "count": 1},
]

CONTENT_CONFIG = {
    "TREND_HISTORY": {"target_duration": 34, "min_duration": 28, "max_duration": 40, "min_words": 56, "max_words": 72},
    "TODAY_IN_HISTORY": {"target_duration": 60, "min_duration": 45, "max_duration": 75, "min_words": 90, "max_words": 145},
    "AYT_HISTORY": {"target_duration": 34, "min_duration": 25, "max_duration": 45, "min_words": 55, "max_words": 90},
    "HISTORY_FACT": {"target_duration": 40, "min_duration": 30, "max_duration": 55, "min_words": 65, "max_words": 105},
    "CUSTOM": {"target_duration": 45, "min_duration": 30, "max_duration": 65, "min_words": 65, "max_words": 115},
}

TYPE_PLAYLIST_ENV = {
    "TREND_HISTORY": "YOUTUBE_PLAYLIST_TREND_HISTORY",
    "TODAY_IN_HISTORY": "YOUTUBE_PLAYLIST_TODAY_IN_HISTORY",
    "AYT_HISTORY": "YOUTUBE_PLAYLIST_AYT_HISTORY",
    "HISTORY_FACT": "YOUTUBE_PLAYLIST_HISTORY_FACT",
    "CUSTOM": "YOUTUBE_PLAYLIST_CUSTOM",
}


def today_tr() -> str:
    return datetime.now(ZoneInfo("Europe/Istanbul")).strftime("%Y-%m-%d")


def config_for(content_type: str) -> dict:
    return dict(CONTENT_CONFIG.get(content_type, CONTENT_CONFIG["TREND_HISTORY"]))


def playlist_id_for(content_type: str) -> str:
    env_name = TYPE_PLAYLIST_ENV.get(content_type, "")
    return os.getenv(env_name, "").strip() if env_name else ""


def daily_plan() -> list[str]:
    plan = []
    for item in CONTENT_PLAN:
        plan.extend([item["content_type"]] * int(item["count"]))
    return plan


def _recent_titles(limit: int = 60) -> list[str]:
    try:
        from event_memory import load_event_memory
        events = load_event_memory()
        return [str(x.get("canonical_title", "")).strip() for x in events[-limit:] if x.get("canonical_title")]
    except Exception:
        return []


def _candidate_prompt(content_type: str, custom_prompt: str = "", date_str: str = "", exclude_titles: list[str] | None = None) -> str:
    recent = "\n".join(f"- {x}" for x in (_recent_titles() + list(exclude_titles or []))[-80:])
    date_str = date_str or today_tr()

    if content_type == "TODAY_IN_HISTORY":
        task = f"""
BUGÜNÜN TARİHİ: {date_str}
Bu tarihte gerçekleşmiş 5-8 gerçek olay arasından Shorts için en güçlü bir olayı seç.
Hikâye potansiyeli, şaşırtıcılık, görsel potansiyel ve güvenilir biçimde anlatılabilirlik güçlü olsun.
Sadece {date_str} gün/ayına gerçekten denk gelen bir olay seç. Yıl farklı olabilir.
Aşağıdaki olayları seçme.
"""
    elif content_type == "AYT_HISTORY":
        task = """
AYT Tarih / TYT-AYT tarih müfredatından tek bir yüksek değerli ezber veya karşılaştırma bilgisi seç.
Osmanlı ilkleri, antlaşma-sonuç, padişah-dönem, kronoloji, kurumlar, Milli Mücadele,
Atatürk dönemi inkılapları gibi sınavda kullanılabilecek net bir konu seç.
Tartışmalı akademik iddia kullanma.
"""
    elif content_type == "HISTORY_FACT":
        task = """
Evergreen bir tarih gerçeği seç. İzleyicinin "Bunu bilmiyordum" diyeceği,
tek videoda açıklanabilecek ve güvenilir kaynaklarla doğrulanabilir bir konu olsun.
Sadece bir isim/yer vermek yerine şaşırtıcı mekanizmayı veya sonucu seç.
"""
    elif content_type == "CUSTOM":
        task = f"""
KULLANICI CUSTOM FİKRİ:
{custom_prompt or "Tarih kanalının mevcut formatlarından farklı ama tarih odaklı, yaratıcı ve yüksek merak potansiyelli tek bir deneysel konu seç."}
Bu fikir doğrulanabilir tarihsel gerçeklere dönüştürülebilmeli.
"""
    else:
        task = "Genel tarih kanalında yüksek merak ve görsel potansiyel taşıyan tek bir olay seç. Daha önce kullanılanları tekrarlama."

    return f"""
Sen tarih içerik motorunun konu seçme ajanısın.
İçerik tipi: {content_type}
{task}

DAHA ÖNCE KULLANILAN / KAÇINILACAK BAŞLIKLAR:
{recent or "- yok"}

SADECE GEÇERLİ JSON DÖNDÜR:
{{
  "canonical_title": "kısa ve net konu adı",
  "aliases": ["alternatif ad"],
  "date": "olayın tarihi veya ilgili tarih",
  "date_normalized": "YYYY-MM-DD veya boş",
  "location": "yer",
  "entities": ["kişi/kurum/nesne"],
  "event_summary": "olayı tek paragrafta özetle",
  "known_facts": ["en az 4 doğrulanabilir başlangıç gerçeği"],
  "why_interesting": "neden Shorts için ilginç",
  "source_urls": [],
  "content_type": "{content_type}",
  "today_date": "{date_str}"
}}

Kurallar:
- Türkçe yaz.
- Bilmediğin kesin ayrıntıları uydurma.
- Aynı olayın farklı yazımını yeni olay gibi gösterme.
- Tarih formatında gün/ay doğruluğuna dikkat et.
- AYT içeriğinde sınav odaklı bilgi ver, spekülasyon verme.
"""


def build_candidate(content_type: str, custom_prompt: str = "", date_str: str = "", exclude_titles: list[str] | None = None) -> dict:
    prompt = _candidate_prompt(content_type, custom_prompt, date_str, exclude_titles)
    result = gemini_config.call_gemini_with_retry(
        prompt=prompt,
        label=f"content candidate: {content_type}",
        response_mime_type="application/json",
    )
    raw = getattr(result, "text", "") if result else ""
    if not raw:
        raise RuntimeError(f"No candidate returned for {content_type}")
    data = json.loads(raw)
    if not isinstance(data, dict) or not data.get("canonical_title"):
        raise ValueError(f"Invalid candidate for {content_type}")
    data["content_type"] = content_type
    data["content_label"] = next(
        (x["label"] for x in CONTENT_PLAN if x["content_type"] == content_type),
        content_type,
    )
    return data
