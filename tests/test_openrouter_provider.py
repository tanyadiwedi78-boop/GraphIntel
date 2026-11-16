"""OpenRouter adapter: offline shape and validation tests. No network calls.

These prove the request/response envelopes and job rules, never live model
quality. Model choice and free-tier status are verified separately before any
real run.
"""
import json
from pathlib import Path

import pytest

from app.extraction.pipeline import prepare, retry_policy
from app.extraction.providers import JSONProvider
from app.graph.ontology import load_ontology

ROOT = Path(__file__).resolve().parent.parent
MODEL = 'nvidia/nemotron-3.5-super-vl:free'


@pytest.fixture
def ontology():
    return load_ontology(ROOT / 'configs' / 'ontology.json')


@pytest.fixture
def chunk():
    return {'chunk_id': 'c1', 'text': 'Acme develops Widget.', 'document_hash': 'a' * 64}


def test_openrouter_body_is_chat_completions_with_strict_schema_and_routing():
    provider = JSONProvider('openrouter', MODEL, 100)
    body = provider.body('same prompt', {'type': 'object'})
    assert body['model'] == MODEL
    assert body['messages'] == [{'role': 'user', 'content': 'same prompt'}]
    assert body['max_tokens'] == 100
    assert body['response_format']['type'] == 'json_schema'
    assert body['response_format']['json_schema']['strict'] is True
    assert body['response_format']['json_schema']['schema'] == {'type': 'object'}
    # Route only to endpoints that can enforce the schema.
    assert body['provider'] == {'require_parameters': True}


def test_openrouter_endpoint_and_key_variable(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-not-a-key')
    provider = JSONProvider('openrouter', MODEL, 100)
    assert provider.key_variable() == 'OPENROUTER_API_KEY'
    url, headers = provider.endpoint()
    assert url == 'https://openrouter.ai/api/v1/chat/completions'
    assert headers['Authorization'] == 'Bearer offline-not-a-key'


def test_openrouter_ready_raises_without_key(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    provider = JSONProvider('openrouter', MODEL, 100)
    with pytest.raises(ValueError, match='provider key missing'):
        provider.ready()


def test_openrouter_output_text_parses_message_and_refusal():
    provider = JSONProvider('openrouter', MODEL, 100)
    envelope = {'choices': [{'message': {'content': '{"entities": []}', 'refusal': None}}]}
    assert provider.output_text(envelope) == '{"entities": []}'
    with pytest.raises(ValueError, match='refused'):
        provider.output_text({'choices': [{'message': {'content': None, 'refusal': 'no'}}]})
    with pytest.raises(ValueError):
        provider.output_text({'choices': []})
    with pytest.raises(ValueError):
        provider.output_text({'choices': [{'message': {'content': None, 'refusal': None}}]})


def test_openrouter_model_id_allows_vendor_slash_and_free_suffix():
    provider = JSONProvider('openrouter', 'google/gemma-4-31b-it:free', 100)
    assert provider.model == 'google/gemma-4-31b-it:free'
    with pytest.raises(ValueError):
        JSONProvider('gemini', 'google/gemma:free', 100)
    with pytest.raises(ValueError):
        JSONProvider('openai', 'google/gemma:free', 100)


def test_openrouter_batch_control_raises():
    provider = JSONProvider('openrouter', MODEL, 100)
    with pytest.raises(ValueError, match='not wired for openrouter'):
        provider.api('batches')


def test_prepare_allows_openrouter_primary_alone(tmp_path, chunk, ontology):
    recipe = JSONProvider('openrouter', MODEL, 100).recipe
    plan = prepare(tmp_path / 'job', [chunk], ontology, recipe, policy=retry_policy(0))
    assert plan['primary']['provider'] == 'openrouter'
    assert plan['fallback'] is None


def test_prepare_rejects_openrouter_with_fallback(tmp_path, chunk, ontology):
    recipe = JSONProvider('openrouter', MODEL, 100).recipe
    fallback = JSONProvider('openai', 'explicit-model', 100).recipe
    with pytest.raises(ValueError, match='fallback'):
        prepare(tmp_path / 'job', [chunk], ontology, recipe, fallback, policy=retry_policy(0))


def test_prepare_rejects_openrouter_native_batch(tmp_path, chunk, ontology):
    recipe = JSONProvider('openrouter', MODEL, 100).recipe
    with pytest.raises(ValueError, match='sync'):
        prepare(tmp_path / 'job', [chunk], ontology, recipe, policy=retry_policy(0),
                execution_mode='native_batch')


def test_openrouter_body_round_trips_through_output_text():
    provider = JSONProvider('openrouter', MODEL, 100)
    body = provider.body('prompt', {'type': 'object'})
    raw = json.dumps({'choices': [{'message': {'content': '{"ok": true}', 'refusal': None}}]})
    assert provider.output_text(json.loads(raw)) == '{"ok": true}'
    assert json.loads(json.dumps(body))['provider']['require_parameters'] is True


def test_common_schema_keeps_strict_mode_shape(ontology):
    # Strict json_schema endpoints require additionalProperties: false and
    # every property listed in required on every object. Pin that invariant
    # so a future schema change fails here, not during a live request.
    from app.extraction.pipeline import common_schema
    schema = common_schema(ontology)
    objects = []

    def walk(node):
        if isinstance(node, dict):
            if node.get('type') == 'object' or 'properties' in node:
                objects.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    assert len(objects) >= 2
    for node in objects:
        assert node.get('additionalProperties') is False
        assert set(node.get('properties') or {}) == set(node.get('required') or [])
