"""Model selection tests are offline. Never discover models on import.

Migrated from the retired Anthropic batch module (fixed MODEL constant) to
the current providers: the model is always an explicit constructor/CLI
choice, recorded in the recipe, and never read from the environment.
"""
import json

import pytest

from app.extraction.providers import JSONProvider
from app.graph.ontology import load_ontology


def test_model_is_always_explicit_and_recorded():
    provider = JSONProvider('gemini', 'gemini-fixture-model', 256)
    assert provider.recipe['model'] == 'gemini-fixture-model'
    assert provider.recipe['provider'] == 'gemini'
    with pytest.raises(ValueError):
        JSONProvider('gemini', '', 256)
    with pytest.raises(ValueError):
        JSONProvider('gemini', 'not a model with spaces', 256)
    with pytest.raises(ValueError):
        JSONProvider('anthropic', 'claude', 256)


def test_model_selection_does_not_use_environment_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-not-used")
    monkeypatch.setenv("OPENAI_MODEL", "not-authorized")
    monkeypatch.setenv("GEMINI_MODEL", "also-not-authorized")
    ontology = load_ontology("configs/ontology.json")
    provider = JSONProvider('gemini', 'explicit-model-name', 256)
    body = provider.body("prompt text", {'type': 'object'})
    url, headers = provider.endpoint()
    assert 'explicit-model-name' in url
    assert 'not-authorized' not in json.dumps(body)
    assert 'also-not-authorized' not in url
    assert body['generationConfig']['maxOutputTokens'] == 256


def test_importing_providers_makes_no_model_discovery_calls():
    # The session network guard fails this test if import or construction
    # attempts any external DNS or socket work.
    import importlib
    import app.extraction.providers as providers
    importlib.reload(providers)
    JSONProvider('openai', 'explicit-openai-model', 128)
