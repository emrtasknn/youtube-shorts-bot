import video_provider


def test_default_video_provider_order(monkeypatch):
    monkeypatch.delenv("VIDEO_PROVIDER_ORDER", raising=False)
    assert video_provider._provider_order()[:2] == ["omni", "veo31"]


def test_google_video_providers_share_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    assert video_provider._configured("omni")
    assert video_provider._configured("veo31_lite")


def test_video_router_fails_over(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_PROVIDER_ORDER", "omni,veo31")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    video_provider._PROVIDER_COOLDOWN_UNTIL.clear()

    calls = []

    def fail(prompt, path, image_path=None):
        calls.append("omni")
        raise video_provider.VideoProviderError("quota", cooldown_seconds=3600)

    def succeed(prompt, path, image_path=None):
        calls.append("veo31")
        path.write_bytes(b"x" * 60000)
        return video_provider.VideoResponse("veo31", "test-model", path)

    monkeypatch.setitem(video_provider._HANDLERS, "omni", fail)
    monkeypatch.setitem(video_provider._HANDLERS, "veo31", succeed)

    result = video_provider.generate("historical scene", tmp_path / "scene.mp4")

    assert result.provider == "veo31"
    assert calls == ["omni", "veo31"]
