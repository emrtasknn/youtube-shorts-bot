import media_provider


def test_media_router_uses_pexels_first(monkeypatch):
    monkeypatch.setenv("REAL_MEDIA_PROVIDER_ORDER", "pexels,pixabay,unsplash")
    monkeypatch.setenv("PEXELS_API_KEY", "test")
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)

    def fake_pexels(query, excluded, intent):
        return {
            "source_type": "pexels",
            "title": "Historic ship",
            "image_url": "https://example.com/ship.jpg",
            "page_url": "https://pexels.com/photo/1",
            "relevance_score": 0.8,
            "provider": "pexels",
        }

    monkeypatch.setitem(media_provider._HANDLERS, "pexels", fake_pexels)
    result = media_provider.search(["historic ship"], visual_intent={"primary_subject": "ship"})
    assert result["source_type"] == "pexels"


def test_media_router_skips_unconfigured_provider(monkeypatch):
    monkeypatch.setenv("REAL_MEDIA_PROVIDER_ORDER", "pexels,pixabay")
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setenv("PIXABAY_API_KEY", "test")

    def fake_pixabay(query, excluded, intent):
        return {
            "source_type": "pixabay",
            "title": "Old ship",
            "image_url": "https://example.com/ship.jpg",
            "page_url": "https://pixabay.com/images/1",
            "relevance_score": 0.7,
            "provider": "pixabay",
        }

    monkeypatch.setitem(media_provider._HANDLERS, "pixabay", fake_pixabay)
    result = media_provider.search(["old ship"], visual_intent={"primary_subject": "ship"})
    assert result["source_type"] == "pixabay"
