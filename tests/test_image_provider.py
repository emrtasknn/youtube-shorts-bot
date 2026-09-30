import image_provider


def test_image_router_fails_over_after_provider_error(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_PROVIDER_ORDER", "cloudflare,fal")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-test")
    monkeypatch.setenv("FAL_KEY", "fal-test")
    image_provider._PROVIDER_COOLDOWN_UNTIL.clear()

    calls = []

    def fail_cf(prompt, path):
        calls.append("cloudflare")
        raise image_provider.ImageProviderError("HTTP 402", cooldown_seconds=3600)

    def succeed_fal(prompt, path):
        calls.append("fal")
        path.write_bytes(b"x" * 6000)
        return image_provider.ImageResponse("fal", "test-model", path)

    monkeypatch.setitem(image_provider._HANDLERS, "cloudflare", fail_cf)
    monkeypatch.setitem(image_provider._HANDLERS, "fal", succeed_fal)

    result = image_provider.generate("historical scene", tmp_path / "scene.jpg")

    assert result.provider == "fal"
    assert calls == ["cloudflare", "fal"]


def test_image_router_skips_unconfigured_providers(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_PROVIDER_ORDER", "cloudflare,together")
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setenv("TOGETHER_API_KEY", "together-test")
    image_provider._PROVIDER_COOLDOWN_UNTIL.clear()

    def succeed(prompt, path):
        path.write_bytes(b"x" * 6000)
        return image_provider.ImageResponse("together", "test-model", path)

    monkeypatch.setitem(image_provider._HANDLERS, "together", succeed)

    result = image_provider.generate("historical scene", tmp_path / "scene.jpg")
    assert result.provider == "together"


def test_image_router_has_clear_error_when_no_provider_is_configured(monkeypatch, tmp_path):
    for key in (
        "GEMINI_API_KEY",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "FAL_KEY",
        "TOGETHER_API_KEY",
        "DEEPAI_API_KEY",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("IMAGE_PROVIDER_ORDER", "cloudflare,fal,together,deepai,huggingface")
    image_provider._PROVIDER_COOLDOWN_UNTIL.clear()

    try:
        image_provider.generate("historical scene", tmp_path / "scene.jpg")
    except RuntimeError as exc:
        assert "No image provider is configured" in str(exc)
    else:
        raise AssertionError("Expected image-provider configuration error")


def test_default_image_provider_order_uses_current_nano_banana_models(monkeypatch):
    monkeypatch.delenv("IMAGE_PROVIDER_ORDER", raising=False)
    order = image_provider._provider_order()
    assert order[:2] == ["nano_banana_2", "nano_banana_2_lite"]
    assert "imagen" not in order


def test_nano_banana_providers_share_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    assert image_provider._configured("nano_banana_2")
    assert image_provider._configured("nano_banana_2_lite")
