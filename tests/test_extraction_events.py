"""Durable event binding on the current pipeline: claims, receipts, attempts.

Migrated from the retired anthropic_batch event tests; the retired modules'
download/reader events are quarantined in legacy_anthropic_tests/.
"""
import json

import pytest

from app.extraction.pipeline import Pipeline
from tests.conftest import demo_chunks, done_collect, make_job, payload_json

SUBMIT_OK = {'name': 'batches/event-wave'}
BATCH_ARGS = dict(max_calls=100, live_authorized=True, max_requests=10,
                  max_input_units=10 ** 9)


def submitted(tmp_path, ontology, count=2):
    chunks = demo_chunks(count)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, execution_mode='native_batch')
    primary.submit_replies = [SUBMIT_OK]
    pipe.batch_submit(**BATCH_ARGS)
    return pipe, primary, chunks


def test_claim_and_receipt_bind_the_exact_plan_before_collection(tmp_path, ontology):
    pipe, primary, chunks = submitted(tmp_path, ontology)
    claim_path = pipe.job / 'batch-1-claim.json'
    claim = json.loads(claim_path.read_text())
    claim['plan_hash'] = 'tampered'
    claim_path.write_text(json.dumps(claim))
    with pytest.raises(ValueError, match='different plan'):
        pipe.batch_collect(1)
    assert len(primary.collect_calls) == 0


def test_pending_wave_is_durable_and_blocks_resubmission(tmp_path, ontology):
    pipe, primary, chunks = submitted(tmp_path, ontology)
    with pytest.raises(ValueError, match='pending or ambiguous batch'):
        pipe.batch_submit(**BATCH_ARGS)
    assert len(primary.submit_calls) == 1
    # Deleting the receipt does not free the claim either: reconcile, never resubmit.
    (pipe.job / 'batch-1-receipt.json').unlink()
    with pytest.raises(ValueError, match='pending or ambiguous batch'):
        pipe.batch_submit(**BATCH_ARGS)
    assert len(primary.submit_calls) == 1


def test_zero_cap_stops_before_claim_or_network(tmp_path, ontology):
    chunks = demo_chunks(1)
    pipe, primary, _ = make_job(tmp_path, ontology, chunks, execution_mode='native_batch')
    result = pipe.batch_submit(max_calls=0, live_authorized=True, max_requests=10,
                               max_input_units=10 ** 9)
    assert result['submitted'] is False
    assert primary.submit_calls == []
    assert list(pipe.job.glob('*-claim.json')) == []


def test_completion_retains_one_final_record_per_source(tmp_path, ontology):
    pipe, primary, chunks = submitted(tmp_path, ontology)
    items = primary.submit_calls[0]['items']
    primary.collect_replies = [done_collect(
        items, {item['custom_id']: payload_json(chunks[n]) for n, item in enumerate(items)})]
    pipe.batch_collect(1)
    records = pipe.records()
    assert len(records) == 2
    assert {r['chunk_id'] for r in records} == {c['chunk_id'] for c in chunks}
    assert all(r['accepted'] for r in records)
    again = pipe.batch_collect(1)
    assert again['already_collected'] is True
    assert len(pipe.records()) == 2  # resume adds nothing


def test_attempt_result_tampering_is_rejected(tmp_path, ontology):
    pipe, primary, chunks = submitted(tmp_path, ontology)
    items = primary.submit_calls[0]['items']
    primary.collect_replies = [done_collect(
        items, {item['custom_id']: payload_json(chunks[n]) for n, item in enumerate(items)})]
    pipe.batch_collect(1)
    result_path = sorted(pipe.job.glob('attempt-*-result.json'))[0]
    result = json.loads(result_path.read_text())
    result['chunk_id'] = 'tampered'
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='differs from durable claim'):
        pipe.attempts()


def test_resume_does_not_create_new_records_or_calls(tmp_path, ontology):
    pipe, primary, chunks = submitted(tmp_path, ontology)
    items = primary.submit_calls[0]['items']
    primary.collect_replies = [done_collect(
        items, {item['custom_id']: payload_json(chunks[n]) for n, item in enumerate(items)})]
    pipe.batch_collect(1)
    before = sorted(p.name for p in pipe.job.iterdir())
    result = pipe.batch_submit(**BATCH_ARGS)
    assert result['submitted'] is False  # nothing left to do
    assert sorted(p.name for p in pipe.job.iterdir()) == before
    assert len(primary.submit_calls) == 1 and len(primary.collect_calls) == 1
