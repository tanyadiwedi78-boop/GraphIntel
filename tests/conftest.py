"""Offline suite guard installed before collection, including Python children.

No test collection exclusions. Loopback fake HTTP servers remain available.
"""
import os
from pathlib import Path
import tempfile
from tests import network_guard

_child_guard = None
_old_pythonpath = None
_old_proxy_env = {}


def pytest_sessionstart(session):
    global _child_guard, _old_pythonpath
    network_guard.install()
    _child_guard = tempfile.TemporaryDirectory(prefix="graphintel-offline-")
    source = Path(network_guard.__file__).read_text(encoding="utf-8")
    (Path(_child_guard.name) / "sitecustomize.py").write_text(source, encoding="utf-8")
    _old_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = _child_guard.name + (os.pathsep + _old_pythonpath if _old_pythonpath else "")
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            _old_proxy_env[name] = os.environ.pop(name)


def pytest_sessionfinish(session, exitstatus):
    network_guard.restore()
    if _old_pythonpath is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = _old_pythonpath
    os.environ.update(_old_proxy_env)
    if _child_guard is not None:
        _child_guard.cleanup()


# ---------------------------------------------------------------------------
# Shared fixtures for the current two-provider architecture (Gemini primary,
# OpenAI fallback). All provider traffic is fake and offline.
# ---------------------------------------------------------------------------
import json

import pytest

from app.extraction.pipeline import prepare, retry_policy, Pipeline
from app.extraction.providers import CallFailure
from app.graph.ontology import load_ontology

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def ontology():
    return load_ontology(ROOT / "configs" / "ontology.json")


def demo_chunks(count=2):
    """Small source chunks carrying the required document hash."""
    chunks = []
    for number in range(1, count + 1):
        chunks.append({'chunk_id': 'c' + str(number),
                       'text': 'Acme develops Widget Engine number ' + str(number) + '.',
                       'document_hash': ('%064x' % number),
                       'accession': 'fixture-only', 'form': '10-K', 'period': '2026'})
    return chunks


def payload_json(chunk, quote=None):
    """A valid expanded-contract payload for the demo chunk."""
    quote = quote or chunk['text']
    return json.dumps({
        'entities': [
            {'canonical_name': 'Acme', 'entity_type': 'Company', 'aliases': [],
             'description': 'A company developing Widget Engine.', 'source_chunk_id': chunk['chunk_id'],
             'confidence': 0.9, 'evidence': quote},
            {'canonical_name': 'Widget Engine', 'entity_type': 'Product', 'aliases': [],
             'description': 'A product Acme develops.', 'source_chunk_id': chunk['chunk_id'],
             'confidence': 0.9, 'evidence': quote}],
        'relationships': [
            {'source_entity': 'Acme', 'target_entity': 'Widget Engine',
             'relationship_type': 'MENTIONS', 'source_chunk_id': chunk['chunk_id'],
             'confidence': 0.9, 'evidence': quote}]})


class FakeProvider:
    """Offline sync provider: scripted return values or CallFailure errors."""

    def __init__(self, values, provider='gemini'):
        self.values = list(values)
        self.recipe = {'provider': provider, 'model': 'offline-fixture',
                       'max_output_tokens': 1024, 'execution_mode': 'offline_fixture'}
        self.calls = 0
        self.prompts = []

    def call(self, prompt, schema):
        self.calls += 1
        self.prompts.append(prompt)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return {'raw_text': value, 'raw_envelope': '{"offline_fixture": true}',
                'http_status': 200, 'request_seconds': 0.25,
                'usage': {'fixture': True}, 'outcome': 'returned', 'cost_usd': None}

    def body(self, prompt, schema):
        return {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                'generationConfig': {'maxOutputTokens': self.recipe['max_output_tokens'],
                    'responseFormat': {'text': {'mimeType': 'application/json', 'schema': schema}}}}

    def output_text(self, data):
        candidates = data['candidates']
        if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
            raise ValueError('incomplete or refused')
        return candidates[0]['content']['parts'][0]['text']


class FakeBatchProvider(FakeProvider):
    """Offline native-batch provider with scripted submit/collect replies.

    submit_replies: values returned by batch_submit (or exceptions to raise).
    collect_replies: values returned by batch_collect (or exceptions).
    """

    def __init__(self, submit_replies, collect_replies, provider='gemini'):
        super().__init__([], provider)
        self.submit_replies = list(submit_replies)
        self.collect_replies = list(collect_replies)
        self.submit_calls = []
        self.collect_calls = []

    def batch_submit(self, items, display_name):
        self.submit_calls.append({'items': items, 'display_name': display_name})
        reply = self.submit_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def batch_collect(self, receipt, expected_ids):
        self.collect_calls.append({'receipt': receipt, 'expected_ids': expected_ids})
        reply = self.collect_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def done_collect(claim_items, rows_by_id=None, errors_by_id=None):
    """A completed collect reply: one returned row per claimed item by default."""
    results = []
    for item in claim_items:
        custom_id = item['custom_id']
        if errors_by_id and custom_id in errors_by_id:
            results.append({'custom_id': custom_id, 'body': None,
                            'error': errors_by_id[custom_id], 'http_status': None})
        else:
            text = None
            if rows_by_id and custom_id in rows_by_id:
                text = rows_by_id[custom_id]
            body = {'candidates': [{'finishReason': 'STOP',
                                    'content': {'parts': [{'text': text}]}}],
                    'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 5}}
            results.append({'custom_id': custom_id, 'body': body, 'error': None,
                            'http_status': None})
    return {'done': True, 'snapshot': {'done': True}, 'results': results}


def make_job(tmp_path, ontology, chunks, execution_mode='sync', name='job',
             fallback=False, retries=0, cache=None):
    """Prepare an immutable offline job and return (Pipeline, primary, fallback)."""
    primary = FakeBatchProvider([], []) if execution_mode == 'native_batch' else FakeProvider([])
    fallback_provider = FakeBatchProvider([], [], 'openai') if fallback and execution_mode == 'native_batch' else (
        FakeProvider([], 'openai') if fallback else None)
    job = tmp_path / name
    prepare(job, chunks, ontology, primary.recipe,
            fallback_provider.recipe if fallback_provider else None,
            policy=retry_policy(retries), execution_mode=execution_mode)
    pipe = Pipeline(job, ontology, primary, fallback_provider,
                    sleep=lambda seconds: None, cache_dir=cache)
    return pipe, primary, fallback_provider
