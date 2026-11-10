"""Model selection tests are offline. Never discover models on import."""
from app.extraction.anthropic_batch import MODEL, message_params
from app.graph.ontology import load_ontology


def test_active_model_is_fixed_anthropic_fable():
    assert MODEL == "claude-fable-5-2"


def test_model_selection_does_not_use_environment_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-not-used")
    monkeypatch.setenv("OPENAI_MODEL", "not-authorized")
    ontology = load_ontology("configs/ontology.json")
    params = message_params({"text":"Cover page."}, ontology, 256, "1h")
    assert params["model"] == MODEL
    assert params["max_tokens"] == 256
    assert "fallback" not in params
