import image_provider


def test_default_image_models_are_current_nano_banana(monkeypatch):
    monkeypatch.delenv("GEMINI_IMAGE_MODEL_ORDER", raising=False)
    assert image_provider._model_order() == [
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
    ]


def test_missing_all_image_keys(monkeypatch, tmp_path):
    for name in ("GEMINI_API_KEY", "FAL_KEY", "DEEPAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    try:
        image_provider.generate("historical scene", tmp_path / "scene.jpg")
    except RuntimeError as exc:
        assert "No image provider is configured" in str(exc)
    else:
        raise AssertionError("Expected missing image provider error")


def test_image_router_fails_over_from_gemini_to_fal(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    monkeypatch.setenv("FAL_KEY", "fal-test")
    monkeypatch.setenv("IMAGE_PROVIDER_ORDER", "gemini,fal")

    calls = []

    def fake_gemini(model, prompt, path):
        calls.append(("gemini", model))
        raise image_provider.ImageProviderError("quota", retryable=False)

    def fake_fal(prompt, path):
        calls.append(("fal", "fal-ai/flux/schnell"))
        path.write_bytes(b"x" * 6000)
        return image_provider.ImageResponse("fal", "fal-ai/flux/schnell", path)

    monkeypatch.setattr(image_provider, "_generate_gemini", fake_gemini)
    monkeypatch.setattr(image_provider, "_generate_fal", fake_fal)

    result = image_provider.generate("historical scene", tmp_path / "scene.jpg")

    assert result.provider == "fal"
    assert calls == [
        ("gemini", "gemini-3.1-flash-image"),
        ("gemini", "gemini-3.1-flash-lite-image"),
        ("fal", "fal-ai/flux/schnell"),
    ]


def test_deepai_is_available_as_last_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPAI_API_KEY", "deepai-test")
    monkeypatch.setenv("IMAGE_PROVIDER_ORDER", "deepai")

    def fake_deepai(prompt, path):
        path.write_bytes(b"x" * 6000)
        return image_provider.ImageResponse("deepai", "text2img", path)

    monkeypatch.setattr(image_provider, "_generate_deepai", fake_deepai)

    result = image_provider.generate("historical scene", tmp_path / "scene.jpg")
    assert result.provider == "deepai"
