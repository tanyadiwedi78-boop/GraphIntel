"""Single-flight execution gates on the current pipeline.

The retired pilot's provider registry, admission flags and Anthropic-claims
tests are quarantined in legacy_anthropic_tests/. What survives here is the
safety intent: one flight at a time, no repair admission, no silent reuse.
"""
import json

import pytest

from app.extraction.pipeline import Pipeline
from app.extraction.providers import CallFailure
from tests.conftest import FakeProvider, demo_chunks, make_job, payload_json

RUN_ARGS = dict(max_calls=10, requests_per_minute=100000, live_authorized=True)


def test_wrong_or_missing_authorization_stops_before_network(tmp_path, ontology):
    pipe, primary, _ = make_job(tmp_path, ontology, demo_chunks(1))
    with pytest.raises(ValueError, match='not authorized'):
        pipe.run(max_calls=10, requests_per_minute=100000)
    assert primary.calls == 0


def test_mutated_plan_stops_before_any_call(tmp_path, ontology):
    pipe, primary, _ = make_job(tmp_path, ontology, demo_chunks(1))
    plan_path = pipe.job / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan['chunks'] = plan['chunks'] * 5
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='plan content changed'):
        Pipeline(pipe.job, ontology, primary)
    assert primary.calls == 0


def test_timeout_keeps_claim_and_blocks_blind_retry(tmp_path, ontology):
    chunks = demo_chunks(1)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks)
    primary.values = [CallFailure('transport_unknown')]
    pipe.run(**RUN_ARGS)
    # Unknown outcome: one attempt only, checkpointed as NOT accepted, and a
    # later run never blindly resubmits it (transport_unknown is not retryable).
    assert primary.calls == 1
    records = pipe.records()
    assert len(records) == 1 and records[0]['accepted'] is False
    attempts, unknown = pipe.attempts()
    assert unknown == 0  # the result claim is durable...
    assert attempts[0]['outcome'] == 'transport_unknown'  # ...and honestly unknown
    result = pipe.run(**RUN_ARGS)
    assert isinstance(result, dict)
    assert primary.calls == 1  # still no blind retry


def test_single_flight_lock_blocks_a_second_runner(tmp_path, ontology):
    pipe, primary, _ = make_job(tmp_path, ontology, demo_chunks(1))
    primary.values = [payload_json(demo_chunks(1)[0])]
    lock = pipe.job / 'run-lock.json'
    lock.write_text('{"note": "another runner"}')
    with pytest.raises((FileExistsError, ValueError)):
        pipe.run(**RUN_ARGS)
    assert primary.calls == 0


def test_cached_result_is_reused_without_a_new_call(tmp_path, ontology):
    cache = tmp_path / 'cache'
    cache.mkdir()
    chunks = demo_chunks(1)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, cache=cache)
    primary.values = [payload_json(chunks[0])]
    pipe.run(**RUN_ARGS)
    assert primary.calls == 1
    # A fresh job with the same inputs and cache needs no new call.
    pipe2, primary2, _ = make_job(tmp_path, ontology, chunks, name='job2', cache=cache)
    pipe2.run(**RUN_ARGS)
    assert primary2.calls == 0
    assert pipe2.records()[0]['cache_hit'] is True


def test_tampered_cache_is_rejected_not_used(tmp_path, ontology):
    cache = tmp_path / 'cache'
    cache.mkdir()
    chunks = demo_chunks(1)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, cache=cache)
    primary.values = [payload_json(chunks[0])]
    pipe.run(**RUN_ARGS)
    for path in cache.glob('*.json'):
        data = json.loads(path.read_text())
        data['record']['raw_text'] = '{"entities": [], "relationships": []}'
        path.write_text(json.dumps(data))
    pipe2, primary2, _ = make_job(tmp_path, ontology, chunks, name='job2', cache=cache)
    with pytest.raises(ValueError, match='cache identity or validation failed'):
        pipe2.run(**RUN_ARGS)
    assert primary2.calls == 0


def test_repairs_are_not_a_stage(tmp_path, ontology):
    """No repair-loop admission exists: bad output is refused, never auto-fixed."""
    from app.extraction.pipeline import validated_record
    chunk = demo_chunks(1)[0]
    record = validated_record('{"entities": [{"canonical_name": "X"}], "relationships": []}',
                              chunk, ontology)
    assert record['accepted'] is False  # refused as-is, no repair path
