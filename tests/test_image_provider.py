import image_provider


def test_default_image_models_are_current_nano_banana(monkeypatch):
    monkeypatch.delenv("GEMINI_IMAGE_MODEL_ORDER", raising=False)
    assert image_provider._model_order() == [
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
    ]


def test_image_generation_requires_gemini_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        image_provider.generate("historical scene", tmp_path / "scene.jpg")
    except RuntimeError as exc:
        assert "GEMINI_API_KEY is missing" in str(exc)
    else:
        raise AssertionError("Expected missing Gemini API key error")


def test_image_router_fails_over_from_nano_banana_2_to_lite(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    monkeypatch.setenv(
        "GEMINI_IMAGE_MODEL_ORDER",
        "gemini-3.1-flash-image,gemini-3.1-flash-lite-image",
    )

    calls = []

    def fake_generate(model, prompt, path):
        calls.append(model)
        if model == "gemini-3.1-flash-image":
            raise image_provider.ImageProviderError("quota", retryable=False)
        path.write_bytes(b"x" * 6000)
        return image_provider.ImageResponse("nano_banana_2_lite", model, path)

    monkeypatch.setattr(image_provider, "_generate", fake_generate)
    result = image_provider.generate("historical scene", tmp_path / "scene.jpg")

    assert result.provider == "nano_banana_2_lite"
    assert calls == ["gemini-3.1-flash-image", "gemini-3.1-flash-lite-image"]
