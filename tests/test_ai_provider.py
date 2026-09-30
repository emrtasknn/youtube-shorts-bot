import os

import ai_provider


def test_provider_router_fails_over_after_retryable_error(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_ORDER", "openrouter,groq")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test")
    monkeypatch.setenv("AI_MAX_ATTEMPTS_PER_PROVIDER", "1")

    calls = []

    def fake_call(provider, prompt, response_mime_type):
        calls.append(provider)
        if provider == "openrouter":
            error = RuntimeError("429 rate limit")
            error.retryable = True
            raise error
        return ai_provider.AIResponse(
            text='{"ok":true}',
            provider=provider,
            model="test-model",
        )

    monkeypatch.setattr(ai_provider, "_call_provider", fake_call)
    ai_provider._PROVIDER_COOLDOWN_UNTIL.clear()

    result = ai_provider.generate(
        "return json",
        label="test",
        response_mime_type="application/json",
    )

    assert result.provider == "groq"
    assert result.text == '{"ok":true}'
    assert calls == ["openrouter", "groq"]


def test_provider_router_skips_unconfigured_provider(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_ORDER", "openai,groq")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-test")

    monkeypatch.setattr(
        ai_provider,
        "_call_provider",
        lambda provider, prompt, response_mime_type: ai_provider.AIResponse(
            text="ok",
            provider=provider,
            model="test",
        ),
    )
    ai_provider._PROVIDER_COOLDOWN_UNTIL.clear()

    result = ai_provider.generate("hello", label="test")
    assert result.provider == "groq"


def test_no_provider_has_clear_configuration_error(monkeypatch):
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AI_PROVIDER_ORDER", "gemini,openrouter,groq,openai")
    ai_provider._PROVIDER_COOLDOWN_UNTIL.clear()

    try:
        ai_provider.generate("hello", label="test")
    except RuntimeError as exc:
        assert "No AI provider is configured" in str(exc)
    else:
        raise AssertionError("Expected a configuration error")
