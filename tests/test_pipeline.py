import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path for pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST")
os.environ.setdefault("TELEGRAM_CHAT_ID", "1")

import pipeline
import youtube_uploader




def test_validate_script_accepts_expected_shape():
    scenes = [
        {"narration": "Bu çok ilginç bir tarih olayı", "image_prompt": "cinematic ancient scene"}
        for _ in range(7)
    ]
    result = pipeline.validate_script({"scenes": scenes})
    assert len(result) == 7


def test_validate_script_rejects_wrong_scene_count():
    try:
        pipeline.validate_script({"scenes": []})
    except ValueError as exc:
        assert "exactly 7 scenes" in str(exc)
    else:
        raise AssertionError("Expected validation failure")


def test_scene_timings_are_continuous_and_cover_full_audio():
    scenes = [{"narration": "bir iki üç"}, {"narration": "dört beş"}, {"narration": "altı yedi sekiz"}]
    words = [
        {"word": "bir", "start": 0.2, "end": 0.5},
        {"word": "iki", "start": 0.6, "end": 0.9},
        {"word": "üç", "start": 1.0, "end": 1.3},
        {"word": "dört", "start": 2.0, "end": 2.3},
        {"word": "beş", "start": 2.4, "end": 2.7},
        {"word": "altı", "start": 3.4, "end": 3.7},
        {"word": "yedi", "start": 3.8, "end": 4.1},
        {"word": "sekiz", "start": 4.2, "end": 4.5},
    ]
    timings = pipeline.calculate_scene_timings(scenes, words, 5.0)
    assert timings[0][0] == 0.0
    assert timings[-1][1] == 5.0
    for previous, current in zip(timings, timings[1:]):
        assert abs(previous[1] - current[0]) < 1e-9


def test_scene_timings_fallback_is_continuous():
    scenes = [{"narration": "one two three"}, {"narration": "four five six"}]
    timings = pipeline.calculate_scene_timings(scenes, [], 20.0)
    assert timings[0][0] == 0.0
    assert timings[-1][1] == 20.0
    assert abs(timings[0][1] - timings[1][0]) < 1e-9


def test_validate_script_rejects_missing_fields():
    # Missing narration
    bad_scenes_1 = [{"image_prompt": "cinematic scene"} for _ in range(7)]
    try:
        pipeline.validate_script({"scenes": bad_scenes_1})
    except ValueError as exc:
        assert "missing narration" in str(exc).lower()
    else:
        raise AssertionError("Expected failure for missing narration")

    # Missing image_prompt
    bad_scenes_2 = [{"narration": "bir iki üç dört beş"} for _ in range(7)]
    try:
        pipeline.validate_script({"scenes": bad_scenes_2})
    except ValueError as exc:
        assert "missing narration or image_prompt" in str(exc).lower()
    else:
        raise AssertionError("Expected failure for missing image_prompt")


def test_validate_video_quality_rejects_missing_file():
    try:
        pipeline.validate_video_quality(Path("non_existent_output_short.mp4"))
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_validate_video_quality_rejects_small_file(tmp_path):
    small_file = tmp_path / "tiny.mp4"
    small_file.write_bytes(b"dummy data" * 10)
    try:
        pipeline.validate_video_quality(small_file)
    except ValueError as exc:
        assert "suspiciously small" in str(exc)
    else:
        raise AssertionError("Expected ValueError for suspiciously small file")


class DummyAudio:
    duration = 25.0


class DummyVideoClip:
    def __init__(self, size=(1080, 1920), duration=25.0, audio=DummyAudio()):
        self.size = size
        self.duration = duration
        self.audio = audio

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_validate_video_quality_valid_clip(monkeypatch, tmp_path):
    fake_video = tmp_path / "good.mp4"
    fake_video.write_bytes(b"\x00" * 60_000)

    monkeypatch.setattr(pipeline, "VideoFileClip", lambda path: DummyVideoClip())
    qa = pipeline.validate_video_quality(fake_video)
    assert qa["width"] == 1080
    assert qa["height"] == 1920
    assert qa["duration"] == 25.0
    assert qa["has_audio"] is True


def test_validate_video_quality_rejects_bad_dimensions(monkeypatch, tmp_path):
    fake_video = tmp_path / "bad_dim.mp4"
    fake_video.write_bytes(b"\x00" * 60_000)

    monkeypatch.setattr(pipeline, "VideoFileClip", lambda path: DummyVideoClip(size=(720, 1280)))
    try:
        pipeline.validate_video_quality(fake_video)
    except ValueError as exc:
        assert "Invalid video dimensions" in str(exc)
    else:
        raise AssertionError("Expected ValueError for bad dimensions")


def test_validate_video_quality_rejects_missing_audio(monkeypatch, tmp_path):
    fake_video = tmp_path / "no_audio.mp4"
    fake_video.write_bytes(b"\x00" * 60_000)

    monkeypatch.setattr(pipeline, "VideoFileClip", lambda path: DummyVideoClip(audio=None))
    try:
        pipeline.validate_video_quality(fake_video)
    except ValueError as exc:
        assert "no audio track" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing audio track")


def test_turkish_upper():
    assert pipeline.turkish_upper("istanbul") == "İSTANBUL"
    assert pipeline.turkish_upper("ışık") == "IŞIK"
    assert pipeline.turkish_upper("türkçe çeşme şemsiye öğüt") == "TÜRKÇE ÇEŞME ŞEMSİYE ÖĞÜT"
    assert pipeline.turkish_upper("HELLO world") == "HELLO WORLD"


def test_render_subtitle_image():
    font = pipeline.get_subtitle_font(size=40)
    arr = pipeline.render_subtitle_image(["BİR", "İKİ", "ÜÇ"], active_idx=1, font=font)
    assert arr.shape == (220, 1080, 4)  # RGBA


def test_generate_subtitle_clips_creates_timed_clips():
    words = [
        {"word": "bir", "start": 0.0, "end": 0.5},
        {"word": "iki", "start": 0.6, "end": 1.0},
        {"word": "üç", "start": 1.1, "end": 1.5},
    ]
    clips = pipeline.generate_subtitle_clips(words)
    assert len(clips) == 3
    assert clips[0].start == 0.0
    assert clips[1].start == 0.6
    assert clips[2].start == 1.1


def test_create_hook_badge():
    badge = pipeline.create_hook_badge(duration=2.0)
    assert badge.duration == 2.0
    assert badge.start == 0.0


def test_build_scene_clip_motions(tmp_path):
    from PIL import Image

    test_img_path = tmp_path / "scene_test.jpg"
    img = Image.new("RGB", (1080, 1920), color=(100, 150, 200))
    img.save(test_img_path)

    for motion in pipeline.MOTION_TYPES:
        clip = pipeline.build_scene_clip(test_img_path, start_time=0.0, end_time=2.0, motion_type=motion)
        assert clip.duration == 2.0
        frame_start = clip.get_frame(0.0)
        frame_mid = clip.get_frame(1.0)
        frame_end = clip.get_frame(1.9)
        assert frame_start.shape == (1920, 1080, 3)
        assert frame_mid.shape == (1920, 1080, 3)
        assert frame_end.shape == (1920, 1080, 3)


def test_topic_history_persistence(tmp_path):
    history_file = tmp_path / "test_history.json"
    assert pipeline.load_topic_history(history_file) == []

    pipeline.save_topic_to_history({"title": "Antik Roma Gizemi"}, history_path=history_file)
    loaded = pipeline.load_topic_history(history_file)
    assert len(loaded) == 1
    assert loaded[0]["title"] == "Antik Roma Gizemi"


def test_evaluate_script_quality_scoring():
    # 1. High-quality script with hook and loop
    good_scenes = [
        {"narration": "Bu gizemli olay tarihte nasıl gerçekleşti?", "image_prompt": "prompt 1"},
        {"narration": "Arkeologlar yıllarca bu sırrı çözmeye çalıştı.", "image_prompt": "prompt 2"},
        {"narration": "Toprak altından çıkan bulgular herkesi şaşırttı.", "image_prompt": "prompt 3"},
        {"narration": "Kimse bu kadar büyük bir yapıyı beklemiyordu.", "image_prompt": "prompt 4"},
        {"narration": "Ancak en karanlık ayrıntı yeni keşfedildi.", "image_prompt": "prompt 5"},
        {"narration": "O günden sonra bu krallık haritadan silindi.", "image_prompt": "prompt 6"},
        {"narration": "Ve bu sırrın cevabı aslında;", "image_prompt": "prompt 7"},
    ]
    qa = pipeline.evaluate_script_quality(good_scenes, topic={"title": "Test"})
    assert qa["passed"] is True
    assert qa["score"] >= 80

    # 2. Script lacking hook in scene 1 and loop in scene 7
    bad_scenes = [
        {"narration": "Ali ata baktı ve gitti.", "image_prompt": "prompt 1"},
        {"narration": "Hava bugün oldukça güneşliydi ve güzeldi.", "image_prompt": "prompt 2"},
        {"narration": "Yolda yürürken küçük bir kedi gördüler.", "image_prompt": "prompt 3"},
        {"narration": "Kedi ağacın dalına doğru tırmanmaya başladı.", "image_prompt": "prompt 4"},
        {"narration": "Sonra hep birlikte eve geri döndüler.", "image_prompt": "prompt 5"},
        {"narration": "Akşam yemeğinde lezzetli bir çorba içildi.", "image_prompt": "prompt 6"},
        {"narration": "Ve böylece güzel bir gün sona erdi.", "image_prompt": "prompt 7"},
    ]
    bad_qa = pipeline.evaluate_script_quality(bad_scenes)
    assert any("hook" in issue.lower() for issue in bad_qa["issues"])
    assert any("loop" in issue.lower() for issue in bad_qa["issues"])
    assert bad_qa["score"] < 80


def test_discover_and_score_topics_selection(monkeypatch):
    class FakeContent:
        text = json.dumps({
            "topics": [
                {"title": "Eski Konu", "hook_question": "Soru 1", "viral_score": 10, "visual_appeal": 10},
                {"title": "Yeni Harika Konu", "hook_question": "Soru 2", "viral_score": 9, "visual_appeal": 9},
                {"title": "Düşük Puanlı Konu", "hook_question": "Soru 3", "viral_score": 6, "visual_appeal": 6},
            ]
        })

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeContent()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(pipeline, "client", FakeClient())

    # "Eski Konu" is in history; function must ignore it and pick "Yeni Harika Konu"
    best = pipeline.discover_and_score_topics(history_titles=["Eski Konu"])
    assert best["title"] == "Yeni Harika Konu"
    assert best["viral_score"] == 9


# -----------------------------
# Sprint 4 Tests: Telegram Control System
# -----------------------------
def test_approval_state_lifecycle(tmp_path):
    approvals_file = tmp_path / "approvals.json"
    assert pipeline.load_approvals(approvals_file) == {}

    # Save state
    run_id = "run_20260922_001"
    initial_data = {
        "run_id": run_id,
        "status": "pending",
        "topic": {"title": "Babil'in Asma Bahçeleri"},
        "total_duration": 24.5,
    }
    pipeline.save_approval_state(run_id, initial_data, approvals_path=approvals_file)

    # Get state
    retrieved = pipeline.get_approval_state(run_id, approvals_path=approvals_file)
    assert retrieved is not None
    assert retrieved["status"] == "pending"
    assert retrieved["topic"]["title"] == "Babil'in Asma Bahçeleri"

    # Update status
    pipeline.update_approval_status(
        run_id,
        status="approved",
        extra={"youtube_video_id": "yt_xyz123"},
        approvals_path=approvals_file,
    )
    updated = pipeline.get_approval_state(run_id, approvals_path=approvals_file)
    assert updated["status"] == "approved"
    assert updated["youtube_video_id"] == "yt_xyz123"
    assert "updated_at" in updated


def test_build_telegram_markup():
    markup = pipeline.build_telegram_markup("run_test_456")
    assert markup is not None
    # 2 rows, 2 buttons each
    keyboard = markup.keyboard
    assert len(keyboard) == 2
    row1 = keyboard[0]
    row2 = keyboard[1]
    assert len(row1) == 2
    assert len(row2) == 2

    btn_publish, btn_regen = row1[0], row1[1]
    btn_cancel, btn_script = row2[0], row2[1]

    assert "YAYINLA" in btn_publish.text
    assert btn_publish.callback_data == "publish:run_test_456"

    assert "YENİDEN ÜRET" in btn_regen.text
    assert btn_regen.callback_data == "regen:run_test_456"

    assert "İPTAL" in btn_cancel.text
    assert btn_cancel.callback_data == "cancel:run_test_456"

    assert "Senaryo" in btn_script.text
    assert btn_script.callback_data == "script:run_test_456"


def test_handle_callback_action_publish_success(monkeypatch, tmp_path):
    approvals_file = tmp_path / "test_approvals.json"
    monkeypatch.setattr(pipeline, "APPROVALS_FILE", approvals_file)

    test_video = tmp_path / "shorts_sample.mp4"
    test_video.write_bytes(b"dummy mp4 video content")

    run_id = "run_pub_01"
    pipeline.save_approval_state(run_id, {
        "run_id": run_id,
        "status": "pending",
        "video_path": str(test_video),
        "topic": {"title": "Göbeklitepe Sırrı"},
        "full_text": "Tarihin en eski tapınak kompleksi.",
    }, approvals_path=approvals_file)

    bot_calls = {"answered": [], "edited": [], "sent": []}
    monkeypatch.setattr(pipeline.bot, "answer_callback_query", lambda cb_id, text: bot_calls["answered"].append((cb_id, text)))
    monkeypatch.setattr(pipeline.bot, "edit_message_reply_markup", lambda chat_id, msg_id, reply_markup: bot_calls["edited"].append((chat_id, msg_id, reply_markup)))
    monkeypatch.setattr(pipeline.bot, "send_message", lambda chat_id, text, **kw: bot_calls["sent"].append((chat_id, text)))

    # Mock upload_shorts_video
    monkeypatch.setattr(
        pipeline.youtube_uploader,
        "upload_shorts_video",
        lambda video_path, topic, full_text: {
            "video_id": "yt_sample_123",
            "url": "https://youtube.com/shorts/yt_sample_123",
            "title": "Göbeklitepe Sırrı #Shorts",
            "privacy_status": "public",
        },
    )

    res = pipeline.handle_callback_action(
        action="publish",
        run_id=run_id,
        chat_id=12345,
        message_id=99,
        callback_id="cb_publish",
    )

    assert res["status"] == "published"
    assert res["youtube_video_id"] == "yt_sample_123"
    assert "https://youtube.com/shorts/yt_sample_123" in res["youtube_url"]
    assert bot_calls["answered"] == [("cb_publish", "Video onaylandı!")]
    assert bot_calls["edited"] == [(12345, 99, None)]

    # Sent start message + success message with YouTube URL
    assert any("https://youtube.com/shorts/yt_sample_123" in msg[1] for msg in bot_calls["sent"])

    # Verify state updated on disk to published with youtube ID
    state = pipeline.get_approval_state(run_id, approvals_path=approvals_file)
    assert state["status"] == "published"
    assert state["youtube_video_id"] == "yt_sample_123"


def test_handle_callback_action_publish_upload_error(monkeypatch, tmp_path):
    approvals_file = tmp_path / "test_approvals.json"
    monkeypatch.setattr(pipeline, "APPROVALS_FILE", approvals_file)

    test_video = tmp_path / "shorts_err.mp4"
    test_video.write_bytes(b"dummy mp4 video content")

    run_id = "run_pub_err_01"
    pipeline.save_approval_state(run_id, {
        "run_id": run_id,
        "status": "pending",
        "video_path": str(test_video),
        "topic": {"title": "Kayıp Kıta Mu"},
    }, approvals_path=approvals_file)

    bot_calls = {"answered": [], "edited": [], "sent": []}
    monkeypatch.setattr(pipeline.bot, "answer_callback_query", lambda cb_id, text: bot_calls["answered"].append((cb_id, text)))
    monkeypatch.setattr(pipeline.bot, "edit_message_reply_markup", lambda chat_id, msg_id, reply_markup: bot_calls["edited"].append((chat_id, msg_id, reply_markup)))
    monkeypatch.setattr(pipeline.bot, "send_message", lambda chat_id, text, **kw: bot_calls["sent"].append((chat_id, text)))

    def raise_quota_error(**kwargs):
        raise RuntimeError("YouTube API Quota Exceeded")

    monkeypatch.setattr(pipeline.youtube_uploader, "upload_shorts_video", raise_quota_error)

    res = pipeline.handle_callback_action(
        action="publish",
        run_id=run_id,
        chat_id=12345,
        message_id=99,
        callback_id="cb_publish_err",
    )

    assert res["status"] == "upload_failed"
    assert "Quota Exceeded" in res["error"]

    state = pipeline.get_approval_state(run_id, approvals_path=approvals_file)
    assert state["status"] == "upload_failed"
    assert any("YouTube yüklemesi gerçekleştirilemedi" in msg[1] for msg in bot_calls["sent"])


def test_handle_callback_action_publish_missing_video(monkeypatch, tmp_path):
    approvals_file = tmp_path / "test_approvals.json"
    monkeypatch.setattr(pipeline, "APPROVALS_FILE", approvals_file)

    run_id = "run_pub_missing_01"
    pipeline.save_approval_state(run_id, {
        "run_id": run_id,
        "status": "pending",
        "video_path": "non_existent_folder/does_not_exist.mp4",
        "topic": {"title": "Kayıp Şehir"},
    }, approvals_path=approvals_file)

    bot_calls = {"answered": [], "edited": [], "sent": []}
    monkeypatch.setattr(pipeline.bot, "answer_callback_query", lambda cb_id, text: bot_calls["answered"].append((cb_id, text)))
    monkeypatch.setattr(pipeline.bot, "edit_message_reply_markup", lambda chat_id, msg_id, reply_markup: bot_calls["edited"].append((chat_id, msg_id, reply_markup)))
    monkeypatch.setattr(pipeline.bot, "send_message", lambda chat_id, text, **kw: bot_calls["sent"].append((chat_id, text)))

    res = pipeline.handle_callback_action(
        action="publish",
        run_id=run_id,
        chat_id=12345,
        message_id=99,
        callback_id="cb_pub_missing",
    )

    assert res["status"] == "missing_video"
    assert any("bulunamadı" in msg[1] for msg in bot_calls["sent"])




def test_handle_callback_action_cancel(monkeypatch, tmp_path):
    approvals_file = tmp_path / "test_approvals.json"
    monkeypatch.setattr(pipeline, "APPROVALS_FILE", approvals_file)

    run_id = "run_cancel_01"
    pipeline.save_approval_state(run_id, {
        "run_id": run_id,
        "status": "pending",
        "topic": {"title": "Atlantis Efsanesi"},
    }, approvals_path=approvals_file)

    bot_calls = {"answered": [], "edited": [], "sent": []}
    monkeypatch.setattr(pipeline.bot, "answer_callback_query", lambda cb_id, text: bot_calls["answered"].append((cb_id, text)))
    monkeypatch.setattr(pipeline.bot, "edit_message_reply_markup", lambda chat_id, msg_id, reply_markup: bot_calls["edited"].append((chat_id, msg_id, reply_markup)))
    monkeypatch.setattr(pipeline.bot, "send_message", lambda chat_id, text, **kw: bot_calls["sent"].append((chat_id, text)))

    res = pipeline.handle_callback_action(
        action="cancel",
        run_id=run_id,
        chat_id=12345,
        message_id=99,
        callback_id="cb_cancel",
    )

    assert res["status"] == "cancelled"
    assert bot_calls["answered"] == [("cb_cancel", "Video iptal edildi.")]
    assert bot_calls["edited"] == [(12345, 99, None)]
    assert "iptal edildi" in bot_calls["sent"][0][1]

    state = pipeline.get_approval_state(run_id, approvals_path=approvals_file)
    assert state["status"] == "cancelled"


def test_handle_callback_action_script(monkeypatch, tmp_path):
    approvals_file = tmp_path / "test_approvals.json"
    monkeypatch.setattr(pipeline, "APPROVALS_FILE", approvals_file)

    run_id = "run_script_01"
    pipeline.save_approval_state(run_id, {
        "run_id": run_id,
        "status": "pending",
        "topic": {"title": "Truva Atı"},
        "scenes": [
            {"narration": "Truva surları aşılamazdı.", "image_prompt": "prompt 1"},
            {"narration": "Bir gece tahta bir at bırakıldı.", "image_prompt": "prompt 2"},
        ],
    }, approvals_path=approvals_file)

    bot_calls = {"answered": [], "sent": []}
    monkeypatch.setattr(pipeline.bot, "answer_callback_query", lambda cb_id, text: bot_calls["answered"].append((cb_id, text)))
    monkeypatch.setattr(pipeline.bot, "send_message", lambda chat_id, text, **kw: bot_calls["sent"].append((chat_id, text)))

    res = pipeline.handle_callback_action(
        action="script",
        run_id=run_id,
        chat_id=12345,
        message_id=99,
        callback_id="cb_script",
    )

    assert res["status"] == "viewed"
    assert len(bot_calls["sent"]) == 1
    sent_text = bot_calls["sent"][0][1]
    assert "Truva Atı" in sent_text
    assert "Truva surları aşılamazdı" in sent_text
    assert "prompt 1" in sent_text


def test_on_status_command(monkeypatch, tmp_path):
    approvals_file = tmp_path / "test_approvals.json"
    monkeypatch.setattr(pipeline, "APPROVALS_FILE", approvals_file)

    replies = []
    class FakeMessage:
        pass
    msg = FakeMessage()
    monkeypatch.setattr(pipeline.bot, "reply_to", lambda message, text, **kw: replies.append(text))

    # Empty case
    pipeline.on_status(msg)
    assert "Henüz kayıtlı bir video üretimi yok" in replies[0]

    # Populated case
    pipeline.save_approval_state("run_stat_01", {
        "status": "pending",
        "total_duration": 22.3,
        "topic": {"title": "Antik Mısır Mumyaları"},
        "created_at": "2026-09-22T01:00:00",
    }, approvals_path=approvals_file)

    pipeline.on_status(msg)
    assert "Antik Mısır Mumyaları" in replies[1]
    assert "PENDING" in replies[1]
    assert "22.3s" in replies[1]


def test_on_callback_query_dispatch(monkeypatch):
    dispatched = []

    def mock_handle_action(action, run_id, chat_id, message_id, callback_id):
        dispatched.append({
            "action": action,
            "run_id": run_id,
            "chat_id": chat_id,
            "message_id": message_id,
            "callback_id": callback_id,
        })
        return {"action": action, "status": "ok"}

    monkeypatch.setattr(pipeline, "handle_callback_action", mock_handle_action)

    class FakeChat:
        id = 998877

    class FakeMessage:
        chat = FakeChat()
        message_id = 456

    class FakeCall:
        id = "call_abc"
        data = "publish:run_2026_test"
        message = FakeMessage()

    pipeline.on_callback_query(FakeCall())

    assert len(dispatched) == 1
    assert dispatched[0]["action"] == "publish"
    assert dispatched[0]["run_id"] == "run_2026_test"
    assert dispatched[0]["chat_id"] == 998877
    assert dispatched[0]["message_id"] == 456
    assert dispatched[0]["callback_id"] == "call_abc"


# -----------------------------
# Sprint 5 Tests: YouTube Automation & Uploader
# -----------------------------
def test_build_shorts_metadata():
    topic = {
        "title": "Babil'in Gizemli Asma Bahçeleri",
        "hook_question": "Bu devasa yapay cennet gerçekten var mıydı?",
    }
    meta = youtube_uploader.build_shorts_metadata(
        topic=topic,
        full_text="Dünyanın yedi harikasından biri olan Babil...",
        privacy_status="public",
    )

    snippet = meta["snippet"]
    status = meta["status"]

    assert snippet["title"].endswith(" #Shorts")
    assert len(snippet["title"]) <= 100
    assert "Babil'in Gizemli Asma Bahçeleri" in snippet["title"]

    assert "❓ Bu devasa yapay cennet gerçekten var mıydı?" in snippet["description"]
    assert "#Shorts" in snippet["description"]
    assert snippet["categoryId"] == "27"
    assert "tr" in snippet["defaultLanguage"]

    assert status["privacyStatus"] == "public"
    assert status["selfDeclaredMadeForKids"] is False


def test_build_shorts_metadata_long_title_truncation():
    super_long_title = "A" * 150
    meta = youtube_uploader.build_shorts_metadata(topic={"title": super_long_title})
    title = meta["snippet"]["title"]
    assert len(title) <= 100
    assert title.endswith(" #Shorts")
    assert "..." in title


def test_get_youtube_client_raises_when_no_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("YOUTUBE_TOKEN_JSON", raising=False)
    monkeypatch.delenv("YOUTUBE_CLIENT_SECRETS_JSON", raising=False)

    fake_secrets = tmp_path / "non_existent_secrets.json"
    fake_token = tmp_path / "non_existent_token.json"

    try:
        youtube_uploader.get_youtube_client(
            client_secrets_path=fake_secrets,
            token_path=fake_token,
        )
    except FileNotFoundError as exc:
        assert "credentials not found" in str(exc).lower()
    else:
        raise AssertionError("Expected FileNotFoundError when credentials missing")


def test_upload_shorts_video_missing_file():
    non_existent = Path("does_not_exist_xyz.mp4")
    try:
        youtube_uploader.upload_shorts_video(non_existent)
    except FileNotFoundError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_upload_shorts_video_mocked_success(tmp_path):
    test_video = tmp_path / "final_short.mp4"
    test_video.write_bytes(b"dummy video data")

    # Mock YouTube client structure: client.videos().insert().execute()
    class FakeInsertRequest:
        def execute(self):
            return {"id": "yt_video_98765"}

    class FakeVideosResource:
        def insert(self, part, body, media_body):
            assert "snippet,status" in part
            assert body["snippet"]["title"].endswith(" #Shorts")
            return FakeInsertRequest()

    class FakeYouTubeClient:
        def videos(self):
            return FakeVideosResource()

    fake_client = FakeYouTubeClient()

    result = youtube_uploader.upload_shorts_video(
        video_path=test_video,
        topic={"title": "Antik Roma Sırrı"},
        full_text="Roma İmparatorluğu'nun gizli yer altı tünelleri.",
        privacy_status="unlisted",
        youtube_client=fake_client,
    )

    assert result["video_id"] == "yt_video_98765"
    assert result["url"] == "https://youtube.com/shorts/yt_video_98765"
    assert result["privacy_status"] == "unlisted"
    assert "Antik Roma Sırrı" in result["title"]


# -----------------------------
# Sprint 6 Tests: Production & Auto-Publish
# -----------------------------
def test_auto_publish_flag_behavior(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "generate_viral_script",
        lambda topic=None, content_memory=None: (
            [{"narration": f"Sahne {i} anlatımı burada yer alıyor.", "image_prompt": f"prompt {i}"} for i in range(7)],
            {"title": "Test Konu", "hook_question": "Soru?"},
        ),
    )

    dummy_audio = tmp_path / "voice.mp3"
    dummy_audio.write_bytes(b"audio")

    async def fake_voice(text, path):
        path.write_bytes(b"voice")
        return [{"word": "Sahne", "start": 0.0, "end": 20.0}]

    monkeypatch.setattr(pipeline, "create_voice_with_timestamps", fake_voice)

    class FakeAudio:
        duration = 22.0

    monkeypatch.setattr(pipeline, "AudioFileClip", lambda p: FakeAudio())

    class FakeVideo:
        def with_duration(self, d):
            return self

        def with_audio(self, a):
            return self

        def write_videofile(self, path, **kw):
            Path(path).write_bytes(b"final")

    monkeypatch.setattr(pipeline, "CompositeVideoClip", lambda *a, **kw: FakeVideo())
    monkeypatch.setattr(pipeline, "download_ai_image", lambda prompt, path: Path(path).write_bytes(b"img"))
    monkeypatch.setattr(pipeline, "build_scene_clip", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "generate_subtitle_clips", lambda *a: [])
    monkeypatch.setattr(pipeline, "create_hook_badge", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "get_ambient_music", lambda p, content_analysis=None, memory=None: None)
    monkeypatch.setattr(pipeline, "validate_video_quality", lambda p: {"passed": True})
    monkeypatch.setattr(pipeline, "save_topic_to_history", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "send_to_telegram", lambda *a, **kw: None)

    # Mock the content_memory module functions used by run()
    cm = pipeline.content_memory_module
    monkeypatch.setattr(cm, "load_content_memory", lambda *a, **kw: [])
    monkeypatch.setattr(cm, "analyze_script_content", lambda *a, **kw: {
        "topic": "Test", "angle": "test angle", "summary": "s",
        "main_claim": "c", "key_facts": [], "entities": [],
        "mood": "mysterious", "energy": 0.5, "tension": 0.3,
    })
    monkeypatch.setattr(cm, "check_content_novelty", lambda *a, **kw: {
        "decision": "ACCEPT", "reason": "test", "similarity_score": 0.0,
        "fact_overlap_pct": 0, "nearest_content": None, "revision_hint": None,
    })
    monkeypatch.setattr(cm, "check_visual_reuse", lambda *a, **kw: [])
    monkeypatch.setattr(cm, "build_content_entry", lambda *a, **kw: {"content_id": "test"})
    monkeypatch.setattr(cm, "save_content_entry", lambda *a, **kw: None)
    monkeypatch.setattr(cm, "compute_ducking_volume", lambda *a, **kw: lambda t: 0.18)

    published_calls = []
    monkeypatch.setattr(
        pipeline,
        "handle_callback_action",
        lambda action, run_id, **kw: published_calls.append((action, run_id)) or {"status": "published"},
    )

    # 1. auto_publish=False -> handle_callback_action NOT called
    res_no = pipeline.run(auto_publish=False)
    assert len(published_calls) == 0
    assert "publish_outcome" not in res_no

    # 2. auto_publish=True -> handle_callback_action called
    res_yes = pipeline.run(auto_publish=True)
    assert len(published_calls) == 1
    assert published_calls[0][0] == "publish"
    assert "publish_outcome" in res_yes


def test_estimate_word_timestamps_distribution():
    text = "Tarihin bilinmeyen en büyük sırrı ortaya çıktı"
    words = pipeline.estimate_word_timestamps(text, total_duration=10.0)
    assert len(words) == len(text.split())
    assert words[0]["start"] >= 0.0
    assert words[-1]["end"] <= 10.0
    for i in range(len(words) - 1):
        assert words[i]["end"] <= words[i + 1]["start"] + 0.05


def test_get_ambient_music_local_asset(tmp_path):
    dest = tmp_path / "test_music.mp3"
    result = pipeline.get_ambient_music(dest)
    assert result is not None
    assert result.exists()
    assert result.stat().st_size > 10_000
    # Also verify that multiple royalty-free audio tracks exist in the assets directory
    if pipeline.AUDIO_ASSETS_DIR.exists():
        tracks = list(pipeline.AUDIO_ASSETS_DIR.glob("*.mp3"))
        assert len(tracks) >= 3


def test_crop_watermark_zone_maintains_dimensions(tmp_path):
    from PIL import Image

    # Square image test (e.g. 1024x1024)
    img_path = tmp_path / "square_test.jpg"
    img = Image.new("RGB", (1024, 1024), color=(120, 80, 40))
    img.save(img_path)

    pipeline.crop_watermark_zone(img_path)
    with Image.open(img_path) as processed:
        assert processed.size == (pipeline.VIDEO_WIDTH, pipeline.VIDEO_HEIGHT)


def test_render_subtitle_image_long_text_scaling():
    font = pipeline.get_subtitle_font(size=68)
    long_words = ["GERÇEKLEŞTİRİLEMEDİ", "KAVRAMSALLAŞTIRILAMAZ", "MUVAFFAKİYETSİZLEŞTİRİCİ"]
    arr = pipeline.render_subtitle_image(long_words, active_idx=0, font=font)
    assert arr.shape == (220, 1080, 4)


def test_validate_video_quality_bitrate_check(monkeypatch, tmp_path):
    fake_video = tmp_path / "low_bitrate.mp4"
    # 55KB file for 25s duration gives only ~17 kbps
    fake_video.write_bytes(b"\x00" * 55_000)

    class DummyClip:
        size = (1080, 1920)
        duration = 25.0
        audio = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(pipeline, "VideoFileClip", lambda p: DummyClip())
    monkeypatch.setenv("MIN_BITRATE_KBPS", "1500")
    try:
        pipeline.validate_video_quality(fake_video)
    except ValueError as exc:
        assert "below minimum threshold" in str(exc)
    else:
        raise AssertionError("Expected ValueError for low bitrate")


def test_fallback_stories_structure_and_duration_bounds():
    assert len(pipeline.FALLBACK_STORIES) >= 3
    for story in pipeline.FALLBACK_STORIES:
        assert "title" in story
        assert "hook_question" in story
        assert len(story["scenes"]) == pipeline.SCENE_COUNT
        total_words = sum(len(s["narration"].split()) for s in story["scenes"])
        assert pipeline.MIN_TOTAL_WORDS <= total_words <= pipeline.MAX_TOTAL_WORDS


def test_discover_topics_fallback_when_api_fails(monkeypatch):
    class FailingModels:
        def generate_content(self, *a, **kw):
            raise RuntimeError("API quota exhausted 429")

    class FailingClient:
        models = FailingModels()

    monkeypatch.setattr(pipeline, "client", FailingClient())
    monkeypatch.setattr(pipeline, "DEFAULT_CANDIDATE_MODELS", ["gemini-test"])
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    topic = pipeline.discover_and_score_topics(history_titles=["Kayıp Koloni Roanoke Gizemi"])
    assert topic is not None
    assert "title" in topic
    assert "scenes" in topic
    assert len(topic["scenes"]) == pipeline.SCENE_COUNT


def test_generate_viral_script_fallback_when_api_fails(monkeypatch):
    class FailingModels:
        def generate_content(self, *a, **kw):
            raise RuntimeError("API quota exhausted 429")

    class FailingClient:
        models = FailingModels()

    monkeypatch.setattr(pipeline, "client", FailingClient())
    monkeypatch.setattr(pipeline, "DEFAULT_CANDIDATE_MODELS", ["gemini-test"])
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    scenes, topic = pipeline.generate_viral_script()
    assert len(scenes) == pipeline.SCENE_COUNT
    assert topic is not None

