# 🎬 YouTube Shorts Bot --- Yapay Zeka ile Otomatik Tarih Kanalı

Bu proje, yapay zeka teknolojilerini (Google Gemini, Edge-TTS, MoviePy 2.x, PIL, Telegram Bot API ve YouTube Data API v3) bir araya getirerek **tam otomatik, insan onaylı (Human-in-the-Loop) veya tam otonom** dikey YouTube Shorts videoları üreten ve yayınlayan profesyonel bir içerik üretim boru hattıdır (pipeline).

---

## 🌟 Temel Özellikler

### 1. Akıllı İçerik & Hikaye Motoru
- **Viral Konu Keşfi**: Gemini yapay zekası ile her gün gizemli, ilgi çekici 3 farklı tarihsel konu alternatifi keşfedilir, viral potansiyel ve görsel zenginlik puanına göre en iyisi seçilir.
- **Tekrarları Önleme (`topics_history.json`)**: Daha önce üretilmiş tüm konular kalıcı hafızada saklanır ve sonraki üretimlerde prompta negatif filtre olarak verilir; aynı konu asla tekrarlanmaz.
- **7 Sahnelik Hikaye Şablonu**:
  - *1. Sahne (Kanca / Hook)*: İzleyiciyi ilk 2 saniyede durduran merak uyandırıcı soru.
  - *2-5. Sahne (Gerilim & Kırılma)*: Olayın bilinmeyenleri ve beklenmedik tarihi detaylar.
  - *6. Sahne (Sonuç)*: Olayın tarihteki yankısı.
  - *7. Sahne (Kusursuz Döngü / Seamless Loop)*: Başa döndüğünde 1. sahneyle kusursuz birleşen döngü sonu.

### 2. Üstün Video & Görsel Kalitesi
- **1080x1920 Dikey Format**: Tam ekran YouTube Shorts standartlarına tam uyumlu.
- **Watermark-Safe Ken Burns Efektleri**: AI görsel sağlayıcısının alt filigranını gizlemek için 75px güvenli kırpma uygulanır. Zoom ve Pan hareketleri minimum `1.03x` ölçekte kilitlenerek hiçbir karede filigran veya siyah kenar gösterilmez.
- **Kelime Vurgulu Dinamik Altyazı Motoru**:
  - Konuşulan aktif kelime parlak sarı (`#FFE500`), diğer kelimeler beyaz ve 6px siyah dış hat ile çizilir.
  - Türkçe büyük harf dönüşümü (`i -> İ`, `ı -> I`) ve UTF-8 karakterler kusursuz desteklenir.
- **Kanca Rozeti (Hook Badge)**: İlk 2.2 saniyede ekranın üst kısmında konunun çarpıcı başlığı dikkat çekici altın çerçeveli rozet olarak yer alır.
- **Ambiyans Müziği**: Arka plan müziği seslendirme altına dengelenerek `%10` ses seviyesi ve yumuşak fade-in / fade-out efektleriyle mikslenir.

### 3. Telegram Kontrol Merkezi (Human-in-the-Loop)
Her üretilen video Telegram üzerinden önizleme ve 4 interaktif inline buton ile gönderilir:
- `✅ YAYINLA`: YouTube Data API v3 ile videoyu YouTube Shorts'a yükler ve izleme bağlantısını iletir.
- `🔄 YENİDEN ÜRET`: Senaryoyu veya konuyu beğenmezseniz tek dokunuşla yeni bir üretim başlatır.
- `❌ İPTAL`: Videonun yayınlanmasını iptal eder.
- `📜 Senaryoyu Oku`: 7 sahnenin tam metnini ve görsel promptlarını Telegram üzerinden döker.

**Telegram Komutları (7/24 Bot Modu):**
- `/start` veya `/help`: Bot kullanım kılavuzu.
- `/status`: Son üretilen videonun onay durumunu (`PENDING`, `APPROVED`, `PUBLISHED`, `CANCELLED`) gösterir.
- `/generate`: Telegram üzerinden tek bir mesajla anında yeni video üretimini tetikler.
- `/history`: Son üretilen 10 konunun listesini ve tarihlerini döker.

### 4. YouTube Data API v3 Otomasyonu
- **Metadata Optimizasyonu**: Başlık sonuna `#Shorts` eklenir, maksimum 100 karakter sınırına göre dinamik optimize edilir.
- **Açıklama & Etiketler**: Kanca sorusu, hikaye özeti, kategorik etiketler (`#Shorts #Tarih #Gizem #Belgesel`) ve kategori ID `27` (Eğitim) ayarlanır.
- **OAuth2 Token Yönetimi**: `client_secrets.json` veya `token.json` ile yetkilendirme; süresi dolan token'lar için otomatik yenileme (`refresh_token`).

### 5. GitHub Actions CI/CD & State Caching
- Her gün saat 07:00 UTC'de (`daily_short.yml`) otomatik çalışır.
- `actions/cache@v4` sayesinde `topics_history.json` ve `approvals.json` dosyaları runner'lar arasında korunur; geçmiş kaybolmaz.
- Üretilen tüm çıktılar GitHub Artifacts olarak 7 gün saklanır.

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum
Python 3.10 veya üzeri bir sürüm gereklidir.

```bash
# Depoyu klonlayın
git clone <repo_url>
cd youtube_short_botu_calismam

# Sanal ortam oluşturun ve aktif edin
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate     # Linux / macOS

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Ortam Değişkenleri (`.env`)
Proje kök dizininde bir `.env` dosyası oluşturun:

```env
GEMINI_API_KEY=your_gemini_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# İsteğe Bağlı Ayarlar
AUTO_PUBLISH=false              # true yapılırsa Telegram onayını beklemeden YouTube'a yükler
YOUTUBE_PRIVACY_STATUS=public   # public | unlisted | private
WATERMARK_CROP_PX=75            # AI watermark güvenli kırpma payı
ZOOM_AMOUNT=0.07                # Ken Burns hareket payı
```

### 3. YouTube API Kurulumu (İsteğe Bağlı)
1. Google Cloud Console'da bir proje açın ve **YouTube Data API v3** servisini etkinleştirin.
2. OAuth 2.0 İstemci Kimliği (Masaüstü Uygulaması) oluşturun ve `client_secrets.json` olarak proje kök dizinine indirin.
3. İlk yüklemede tarayıcı üzerinden onay verildiğinde `token.json` otomatik üretilecek ve sonraki tüm çağrılarda yenilenecektir.
4. *GitHub Actions için:* `token.json` içeriğini GitHub Repository Secrets içine `YOUTUBE_TOKEN_JSON` olarak ekleyebilirsiniz.

---

## 🛠️ Çalıştırma Modları

### A. Tek Seferlik Video Üretimi (Lokal / Manuel)
```bash
python pipeline.py
```
Video üretilir, kalite kontrollerinden geçer ve Telegram'a butonlarla gönderilir.

### B. Telegram Bot Servisi (7/24 Kesintisiz Mod)
```bash
python pipeline.py --bot
```
Bot arka planda sürekli çalışarak Telegram'dan gelen `/generate` komutlarını ve buton tıklamalarını (`Yayınla`, `Yeniden Üret`, vb.) anında işler.

### C. Otomatik Testleri Çalıştırma
```bash
pytest -q
```
Tüm 33 birim testi (senaryo doğrulama, seslendirme, Ken Burns, altyazı, onay döngüsü, YouTube uploader) test edilir.

---

## ⚙️ Ortam Değişkenleri Tablosu

| Değişken | Açıklama | Zorunlu? | Varsayılan |
| :--- | :--- | :---: | :---: |
| `GEMINI_API_KEY` | Google Gemini API anahtarı | Evet | - |
| `TELEGRAM_BOT_TOKEN` | Telegram BotFather API token | Evet | - |
| `TELEGRAM_CHAT_ID` | Bildirimlerin gideceği Telegram Chat ID | Evet | - |
| `AUTO_PUBLISH` | Doğrudan YouTube'a yükleme modu (`true`/`false`) | Hayır | `false` |
| `YOUTUBE_PRIVACY_STATUS` | YouTube video gizliliği (`public`/`unlisted`/`private`) | Hayır | `public` |
| `YOUTUBE_TOKEN_JSON` | Headless/CI için JSON formatında OAuth2 token | Hayır | - |
| `YOUTUBE_CLIENT_SECRETS_JSON` | Headless/CI için JSON formatında client secrets | Hayır | - |
| `WATERMARK_CROP_PX` | Alt filigran kırpma pikseli | Hayır | `75` |
| `OUTPUT_DIR` | Video çıktılarının kaydedileceği dizin | Hayır | `output` |

---

## 📂 Dosya Yapısı

```text
├── .github/workflows/
│   └── daily_short.yml       # Günlük otomatik GitHub Actions iş akışı
├── output/                   # Üretilen videolar, sesler ve görsel çıktıları
├── tests/
│   └── test_pipeline.py      # 33 kapsamlı birim testi
├── pipeline.py               # Ana boru hattı, bot servisi ve onay döngüsü
├── youtube_uploader.py       # YouTube Data API v3 yükleme ve metadata modülü
├── requirements.txt          # Python bağımlılıkları
├── approvals.json            # Kalıcı video onay durumu kaydı (gitignore)
├── topics_history.json       # İşlenmiş konu geçmişi (gitignore)
└── README.md                 # Proje dokümantasyonu
```
