# AI-driven SFX library

Bu klasör, Gemini'nin sahne bazında seçtiği semantik SFX cue'larını gerçek ses dosyalarına bağlar.

## Dosya adlandırma

Dosya adının ilk bölümü kontrollü SFX tiplerinden biri olmalı. Örnek:

- `whoosh.mp3`
- `impact_heavy.mp3`
- `sword_clash.mp3`
- `door_creak.mp3`
- `crowd_ancient.mp3`
- `fire_crackle.mp3`
- `thunder.mp3`
- `wind.mp3`
- `water.mp3`
- `horse.mp3`
- `footsteps.mp3`
- `paper.mp3`
- `clock.mp3`
- `bell.mp3`
- `ship.mp3`
- `explosion.mp3`
- `metal.mp3`
- `stone.mp3`
- `coin.mp3`
- `whisper.mp3`

Sistem önce tam eşleşmeyi, sonra `<type>_` ile başlayan dosyaları arar. Aynı tipte birden fazla dosya varsa rastgele çeşitlilik sağlar.

## YouTube Audio Library kullanımı

YouTube Studio > Ses Kitaplığı > Ses efektleri bölümünden MP3 olarak indirilen efektleri buraya koyabiliriz. Kullanacağımız dosyanın lisans türünü ayrıca kontrol et; Creative Commons ise gerekli atfı video açıklamasına eklemek gerekir.

## AI davranışı

Gemini dosya adını seçmez. Örneğin:

```json
{
  "type": "sword",
  "cue": "kılıçların kısa ve sert şekilde çarpışması",
  "offset_ratio": 0.55,
  "volume": 0.35,
  "max_duration": 1.5
}
```

Renderer daha sonra `sword*.mp3` dosyalarından birini seçip doğru sahne zamanına yerleştirir.

SFX dosyası bulunamazsa video üretimi durmaz; cue atlanır ve QA metadata'sında görünür.
