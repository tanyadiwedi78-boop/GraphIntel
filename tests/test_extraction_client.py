"""JSONProvider transport and pipeline pacing limits, fully offline.

Migrated from the retired app.extraction.client (Anthropic adapter). Every
network effect here is an injected opener or fixture provider; nothing leaves
the machine. Native batch lives in tests/test_anthropic_batch.py.
"""
import io
import json
from urllib.error import HTTPError

import pytest

from app.extraction.contracts import ontology_contract
from app.extraction.pipeline import Pipeline, build_prompt, prepare
from app.extraction.providers import JSONProvider, CallFailure
from tests.conftest import FakeProvider, demo_chunks

PAYLOAD = '{"entities": [], "relationships": []}'


def gemini_envelope(text=PAYLOAD, finish='STOP', usage=None):
    data = {"candidates": [{"finishReason": finish,
                            "content": {"parts": [{"text": text}]}}]}
    if usage is not None:
        data["usageMetadata"] = usage
    return json.dumps(data).encode()


def openai_envelope(text=PAYLOAD, status='completed', usage=None, refusal=False):
    part = {"type": "refusal", "refusal": "no"} if refusal else {"type": "output_text", "text": text}
    data = {"status": status,
            "output": [{"type": "message", "content": [part]}]}
    if usage is not None:
        data["usage"] = usage
    return json.dumps(data).encode()


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status = status

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingOpener:
    """Stands in for urllib's opener: records requests, replays scripted answers."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(answer)


def http_error(status, retry_after=None):
    headers = {}
    if retry_after is not None:
        headers['Retry-After'] = retry_after
    return HTTPError('https://example.invalid', status, 'error', headers, io.BytesIO(b'x'))


def make_client(provider='gemini', model='m1', limit=128, opener=None, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv('GEMINI_API_KEY' if provider == 'gemini' else 'OPENAI_API_KEY', 'k-secret')
    return JSONProvider(provider, model, limit, opener=opener or RecordingOpener())


# ---------- contract enums come from the ontology ----------

def test_schema_required_enums_and_documented_fields_only(ontology):
    schema = ontology_contract(ontology).model_json_schema()
    blob = json.dumps(schema)
    for entity_type in ontology.entity_types:
        assert entity_type in blob
    for relation in ontology.relations:
        assert relation in blob
    entity_props = schema['$defs']['EntityExtraction']['properties']
    relation_props = schema['$defs']['RelationshipExtraction']['properties']
    assert sorted(entity_props) == ['aliases', 'canonical_name', 'confidence',
                                    'description', 'entity_type', 'evidence', 'source_chunk_id']
    assert sorted(relation_props) == ['confidence', 'evidence', 'relationship_type',
                                      'source_chunk_id', 'source_entity', 'target_entity']


# ---------- construction and request shape ----------

def test_constructor_makes_no_call_and_request_carries_schema(monkeypatch):
    opener = RecordingOpener(gemini_envelope())
    client = make_client(opener=opener, monkeypatch=monkeypatch)
    assert opener.requests == []  # building the client never calls out
    body = client.body('PROMPT TEXT', {'type': 'object'})
    assert body['contents'][0]['parts'][0]['text'] == 'PROMPT TEXT'
    assert body['generationConfig']['maxOutputTokens'] == 128
    assert body['generationConfig']['responseFormat']['text']['schema'] == {'type': 'object'}
    assert opener.requests == []
    openai = make_client(provider='openai', opener=opener, monkeypatch=monkeypatch)
    body = openai.body('PROMPT TEXT', {'type': 'object'})
    assert body['input'] == 'PROMPT TEXT'
    assert body['text']['format']['schema'] == {'type': 'object'}
    assert body['store'] is False
    assert opener.requests == []


def test_constructor_rejects_bad_arguments():
    with pytest.raises(ValueError):
        JSONProvider('anthropic', 'm1', 128)
    with pytest.raises(ValueError):
        JSONProvider('gemini', 'bad model!', 128)
    with pytest.raises(ValueError):
        JSONProvider('gemini', 'm1', 0)


def test_assume_free_reports_unknown_cost_not_zero(monkeypatch):
    client = make_client(opener=RecordingOpener(gemini_envelope()), monkeypatch=monkeypatch)
    result = client.call('p', {'type': 'object'})
    assert result['outcome'] == 'returned'
    assert result['cost_usd'] is None  # unknown is never reported as zero


def test_unconfigured_client_blocks_before_any_request(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    opener = RecordingOpener(gemini_envelope())
    client = make_client(opener=opener)
    with pytest.raises(ValueError):
        client.ready()
    with pytest.raises(ValueError):
        client.call('p', {'type': 'object'})
    assert opener.requests == []  # refused before any request left


def test_assume_free_without_limits_is_refused(tmp_path, ontology):
    """A live-transport recipe refuses to run without an explicit TPM limit."""
    monkey = pytest.MonkeyPatch()
    monkey.setenv('GEMINI_API_KEY', 'k-secret')
    client = JSONProvider('gemini', 'm1', 128, opener=RecordingOpener())
    prepare(tmp_path / 'job', demo_chunks(1), ontology, client.recipe)
    pipe = Pipeline(tmp_path / 'job', ontology, client)
    with pytest.raises(ValueError, match='token-per-minute'):
        pipe.run(max_calls=1, requests_per_minute=60, live_authorized=True)
    monkey.undo()


def test_single_prompt_over_token_limit_fails_fast(tmp_path, ontology):
    chunk = demo_chunks(1)[0]
    primary = FakeProvider([PAYLOAD])
    job = tmp_path / 'job'
    prepare(job, [chunk], ontology, primary.recipe)
    pipe = Pipeline(job, ontology, primary, sleep=lambda s: None)
    schema_bytes = len(json.dumps(pipe.plan['schema']).encode())
    reserved = len(build_prompt(chunk, ontology).encode()) + schema_bytes + 4096 + primary.recipe['max_output_tokens']
    # One request would exceed the conservative window: refused before any call.
    with pytest.raises(ValueError, match='conservative local token window'):
        pipe.run(max_calls=5, requests_per_minute=100000, live_authorized=True,
                 provider_limits={'gemini': {'rpm': 100000, 'tpm': reserved - 1}})
    assert primary.calls == 0
    # With headroom for the conservative reservation the same request runs.
    pipe.run(max_calls=5, requests_per_minute=100000, live_authorized=True,
             provider_limits={'gemini': {'rpm': 100000, 'tpm': reserved}})
    assert primary.calls == 1


def test_pacing_waits_before_sending(tmp_path, ontology):
    primary = FakeProvider([PAYLOAD, PAYLOAD])
    job = tmp_path / 'job'
    prepare(job, demo_chunks(2), ontology, primary.recipe)
    waits = []
    pipe = Pipeline(job, ontology, primary, clock=lambda: 0.0, sleep=waits.append)
    pipe.run(max_calls=5, requests_per_minute=2, live_authorized=True)
    assert primary.calls == 2
    assert waits == [30.0, 30.0]  # frozen clock: every request waits one full 60/rpm interval


def test_token_pacing_uses_conservative_estimate(tmp_path, ontology):
    """The reservation counts prompt bytes plus schema plus a fixed margin, not a guess."""
    chunk = demo_chunks(1)[0]
    primary = FakeProvider([PAYLOAD])
    job = tmp_path / 'job'
    prepare(job, [chunk], ontology, primary.recipe)
    pipe = Pipeline(job, ontology, primary, sleep=lambda s: None)
    schema_bytes = len(json.dumps(pipe.plan['schema']).encode())
    exact = len(build_prompt(chunk, ontology).encode()) + schema_bytes + 4096 + primary.recipe['max_output_tokens']
    # Exactly the reservation fits; one byte less is refused (see fail-fast test).
    pipe.run(max_calls=5, requests_per_minute=100000, live_authorized=True,
             provider_limits={'gemini': {'rpm': 100000, 'tpm': exact}})
    assert primary.calls == 1


# ---------- response handling ----------

def test_missing_usage_is_null_not_zero(monkeypatch):
    client = make_client(opener=RecordingOpener(gemini_envelope()), monkeypatch=monkeypatch)
    assert client.call('p', {'type': 'object'})['usage'] is None
    client = make_client(provider='openai', opener=RecordingOpener(openai_envelope()), monkeypatch=monkeypatch)
    assert client.call('p', {'type': 'object'})['usage'] is None


def test_usage_is_captured_when_present(monkeypatch):
    usage = {"promptTokenCount": 10, "candidatesTokenCount": 5}
    client = make_client(opener=RecordingOpener(gemini_envelope(usage=usage)), monkeypatch=monkeypatch)
    assert client.call('p', {'type': 'object'})['usage'] == usage


@pytest.mark.parametrize('status', [429, 500, 503])
def test_http_errors_have_one_attempt_and_safe_message(monkeypatch, status):
    opener = RecordingOpener(http_error(status))
    client = make_client(opener=opener, monkeypatch=monkeypatch)
    with pytest.raises(CallFailure) as caught:
        client.call('p', {'type': 'object'})
    assert caught.value.kind == 'http_failure'
    assert caught.value.status == status
    assert len(opener.requests) == 1  # the transport never retries on its own


def test_retry_after_header_is_parsed(monkeypatch):
    client = make_client(opener=RecordingOpener(http_error(429, '17')), monkeypatch=monkeypatch)
    with pytest.raises(CallFailure) as caught:
        client.call('p', {'type': 'object'})
    assert caught.value.retry_after == 17


def test_bad_envelope_is_not_a_valid_empty_answer(monkeypatch):
    client = make_client(opener=RecordingOpener(b'not json at all'), monkeypatch=monkeypatch)
    result = client.call('p', {'type': 'object'})
    assert result['outcome'] == 'unusable_response'
    assert result['raw_text'] is None


def test_cut_off_answer_fails_the_chunk(monkeypatch):
    client = make_client(opener=RecordingOpener(gemini_envelope(finish='MAX_TOKENS')), monkeypatch=monkeypatch)
    result = client.call('p', {'type': 'object'})
    assert result['outcome'] == 'unusable_response'
    assert result['raw_text'] is None


def test_continuation_request_fails_the_chunk(monkeypatch):
    two = json.dumps({"candidates": [
        {"finishReason": "STOP", "content": {"parts": [{"text": PAYLOAD}]}},
        {"finishReason": "STOP", "content": {"parts": [{"text": PAYLOAD}]}}]}).encode()
    client = make_client(opener=RecordingOpener(two), monkeypatch=monkeypatch)
    assert client.call('p', {'type': 'object'})['outcome'] == 'unusable_response'


def test_dropped_connection_is_a_retryable_transport_error(monkeypatch):
    client = make_client(opener=RecordingOpener(OSError('dropped')), monkeypatch=monkeypatch)
    with pytest.raises(CallFailure) as caught:
        client.call('p', {'type': 'object'})
    assert caught.value.kind == 'transport_unknown'
    assert caught.value.status is None


def test_cut_off_response_retains_nonretryable_adapter_error(tmp_path, ontology, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k-secret')
    client = make_client(opener=RecordingOpener(gemini_envelope(finish='MAX_TOKENS')))
    chunk = demo_chunks(1)[0]
    result = client.call('p', {'type': 'object'})
    job = tmp_path / 'job'
    primary = FakeProvider([])
    prepare(job, [chunk], ontology, primary.recipe)
    pipe = Pipeline(job, ontology, primary)
    assert result['outcome'] == 'unusable_response'
    assert pipe.retryable(result, chunk) is False  # never blindly resent


def test_missing_key_is_refused_without_leaking(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'super-secret-value')
    opener = RecordingOpener(http_error(401))
    client = make_client(opener=opener, monkeypatch=monkeypatch)
    with pytest.raises(CallFailure) as caught:
        client.call('p', {'type': 'object'})
    assert 'super-secret-value' not in str(caught.value)
    monkeypatch.delenv('GEMINI_API_KEY')
    with pytest.raises(ValueError) as caught:
        client.call('p', {'type': 'object'})
    assert 'super-secret-value' not in str(caught.value)
    assert len(opener.requests) == 1  # no second request without a key


def test_openai_refusal_is_unusable_not_empty_answer(monkeypatch):
    client = make_client(provider='openai', opener=RecordingOpener(openai_envelope(refusal=True)),
                         monkeypatch=monkeypatch)
    assert client.call('p', {'type': 'object'})['outcome'] == 'unusable_response'


def test_recipe_changes_with_model_and_generation_limit(monkeypatch):
    a = make_client(model='m1', limit=128, monkeypatch=monkeypatch).recipe
    b = make_client(model='m2', limit=256, monkeypatch=monkeypatch).recipe
    assert a != b
    assert a['model'] == 'm1' and b['model'] == 'm2'
    assert a['max_output_tokens'] == 128 and b['max_output_tokens'] == 256
    c = make_client(provider='openai', model='m1', limit=128, monkeypatch=monkeypatch).recipe
    assert c != a  # provider is part of the recipe identity


def test_legacy_interactions_envelope_is_read_not_sent(monkeypatch):
    """Old saved Interactions envelopes parse; nothing is ever sent to that API."""
    envelope = json.dumps({"status": "completed", "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": PAYLOAD}]}]}).encode()
    client = make_client(opener=RecordingOpener(envelope), monkeypatch=monkeypatch)
    result = client.call('p', {'type': 'object'})
    assert result['outcome'] == 'returned'
    assert result['raw_text'] == PAYLOAD
