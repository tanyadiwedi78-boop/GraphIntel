"""Timing honesty on the current pipeline.

Native batch has no per-request latency; the current contract refuses to
invent one. Sync calls keep real measured seconds only.
"""
import json

import pytest

from tests.conftest import FakeProvider, demo_chunks, done_collect, make_job, payload_json

SUBMIT_OK = {'name': 'batches/timing-wave'}
BATCH_ARGS = dict(max_calls=100, live_authorized=True, max_requests=10,
                  max_input_units=10 ** 9)


def test_provider_wall_time_has_explicit_boundaries(tmp_path, ontology):
    """Sync runs keep per-request seconds; batch collection stamps a finish
    time and never per-item latency."""
    chunks = demo_chunks(1)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks)
    primary.values = [payload_json(chunks[0])]
    pipe.run(max_calls=5, requests_per_minute=100000, live_authorized=True)
    attempts, _ = pipe.attempts()
    assert attempts[0]['request_seconds'] == 0.25  # measured by the provider call


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), 'fast', True])
def test_invalid_native_times_stay_rejected(tmp_path, ontology, bad):
    chunks = demo_chunks(1)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks)
    primary.values = [payload_json(chunks[0])]
    pipe.run(max_calls=5, requests_per_minute=100000, live_authorized=True)
    path = next(pipe.job.glob('attempt-*-result.json'))
    result = json.loads(path.read_text())
    result['request_seconds'] = bad
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='latency'):
        pipe.attempts()


def test_native_batch_never_fabricates_per_request_latency(tmp_path, ontology):
    chunks = demo_chunks(1)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, execution_mode='native_batch')
    primary.submit_replies = [SUBMIT_OK]
    pipe.batch_submit(**BATCH_ARGS)
    items = primary.submit_calls[0]['items']
    primary.collect_replies = [done_collect(items, {items[0]['custom_id']: payload_json(chunks[0])})]
    pipe.batch_collect(1)
    attempts, _ = pipe.attempts()
    assert attempts[0]['request_seconds'] is None  # honest unknown, not invented


def test_invented_native_latency_is_rejected(tmp_path, ontology):
    chunks = demo_chunks(1)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, execution_mode='native_batch')
    primary.submit_replies = [SUBMIT_OK]
    pipe.batch_submit(**BATCH_ARGS)
    items = primary.submit_calls[0]['items']
    primary.collect_replies = [done_collect(items, {items[0]['custom_id']: payload_json(chunks[0])})]
    pipe.batch_collect(1)
    path = next(pipe.job.glob('attempt-*-result.json'))
    result = json.loads(path.read_text())
    result['request_seconds'] = 12.0
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='cannot invent per-request latency'):
        pipe.attempts()


def test_collection_marks_finish_once_and_resume_does_not_remeasure(tmp_path, ontology):
    chunks = demo_chunks(1)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, execution_mode='native_batch')
    primary.submit_replies = [SUBMIT_OK]
    pipe.batch_submit(**BATCH_ARGS)
    items = primary.submit_calls[0]['items']
    primary.collect_replies = [done_collect(items, {items[0]['custom_id']: payload_json(chunks[0])})]
    pipe.batch_collect(1)
    collected = json.loads((pipe.job / 'batch-1-collected.json').read_text())
    first_stamp = collected['finished_at']
    assert first_stamp
    pipe.batch_collect(1)
    again = json.loads((pipe.job / 'batch-1-collected.json').read_text())
    assert again['finished_at'] == first_stamp  # immutable once written
    assert len(primary.collect_calls) == 1
