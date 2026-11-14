"""Scale and bound behavior on the current native-batch pipeline.

The old bounded-reader/download tests exercised the retired batch_storage
module; they are quarantined in legacy_anthropic_tests/ (see its README).
What remains here exercises the current implementation at scale.
"""
import json

import pytest

from app.extraction.pipeline import Pipeline, build_prompt
from tests.conftest import demo_chunks, done_collect, make_job, payload_json

SUBMIT_OK = {'name': 'batches/scale-wave'}
BATCH_ARGS = dict(max_calls=2000, live_authorized=True, max_requests=1000,
                  max_input_units=10 ** 12)


def test_many_requests_keep_separate_source_ids(tmp_path, ontology):
    """Approved shape: many separate requests per batch, one inference each,
    never packed into one prompt."""
    chunks = demo_chunks(250)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, execution_mode='native_batch')
    primary.submit_replies = [SUBMIT_OK]
    result = pipe.batch_submit(**BATCH_ARGS)
    assert result['submitted'] is True and result['items'] == 250
    items = primary.submit_calls[0]['items']
    assert len({item['custom_id'] for item in items}) == 250
    assert [item['chunk_id'] for item in items] == [c['chunk_id'] for c in chunks]
    # Each item carries exactly its own chunk's prompt, nobody else's.
    for item, chunk in zip(items, chunks):
        assert item['body']['contents'][0]['parts'][0]['text'] == build_prompt(chunk, ontology)


def test_max_requests_bounds_one_wave_without_truncating_ids(tmp_path, ontology):
    chunks = demo_chunks(5)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, execution_mode='native_batch')
    primary.submit_replies = [{'name': 'batches/w1'}, {'name': 'batches/w2'}]
    args = dict(BATCH_ARGS, max_requests=3)
    first = pipe.batch_submit(**args)
    assert first['items'] == 3
    items = primary.submit_calls[0]['items']
    primary.collect_replies = [done_collect(items, {i['custom_id']: payload_json(chunks[n]) for n, i in enumerate(items)})]
    pipe.batch_collect(1)
    second = pipe.batch_submit(**args)
    assert second['submitted'] is True and second['items'] == 2  # the remainder, complete
    remaining = primary.submit_calls[1]['items']
    assert [i['chunk_id'] for i in remaining] == [c['chunk_id'] for c in chunks[3:]]


def test_oversized_wave_never_leaves_partial_claim(tmp_path, ontology):
    chunks = demo_chunks(3)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, execution_mode='native_batch')
    with pytest.raises(ValueError, match='enqueue-token allowance'):
        pipe.batch_submit(max_calls=100, live_authorized=True, max_requests=10,
                          max_input_units=10)
    assert primary.submit_calls == []
    assert list(pipe.job.glob('batch-*-claim.json')) == []
    assert list(pipe.job.glob('attempt-*-claim.json')) == []


def test_fifteen_mib_items_bound_guarded_before_claims(tmp_path, ontology, monkeypatch):
    pipe, primary, _ = make_job(tmp_path, ontology, demo_chunks(2), execution_mode='native_batch')
    real_dumps = json.dumps
    def huge_dumps(value, *args, **kwargs):
        text = real_dumps(value, *args, **kwargs)
        return text + ' ' * (16 * 1024 * 1024)
    monkeypatch.setattr('app.extraction.pipeline.json.dumps', huge_dumps)
    with pytest.raises(ValueError, match='15MiB'):
        pipe.batch_submit(**BATCH_ARGS)
    assert primary.submit_calls == []
    assert list(pipe.job.glob('batch-*-claim.json')) == []


def test_bad_result_coverage_has_no_partial_extractions(tmp_path, ontology):
    chunks = demo_chunks(3)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, execution_mode='native_batch')
    primary.submit_replies = [SUBMIT_OK]
    pipe.batch_submit(**BATCH_ARGS)
    items = primary.submit_calls[0]['items']
    # Only the first item returns; the rest stay missing, never invented.
    reply = done_collect(items[:1], {items[0]['custom_id']: payload_json(chunks[0])})
    primary.collect_replies = [reply]
    pipe.batch_collect(1)
    assert len(pipe.records()) == 1
    attempts, unknown = pipe.attempts()
    assert unknown == 2  # held honestly, no partial summary, no auto-resubmit
    with pytest.raises(ValueError, match='unreconciled batch item'):
        pipe.batch_submit(**BATCH_ARGS)
