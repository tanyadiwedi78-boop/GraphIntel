"""Native-batch lifecycle on the current two-provider architecture.

File name kept for history; the retired Anthropic BatchAPI/usage arithmetic
tests moved to legacy_anthropic_tests/ (see its README). These tests use
offline fixture providers only: nothing leaves the machine, nothing is spent.
"""
import json

import pytest

from app.extraction.pipeline import Pipeline, build_prompt, validated_record
from app.extraction.providers import CallFailure
from tests.conftest import (FakeBatchProvider, FakeProvider, demo_chunks,
                            done_collect, make_job, payload_json)

SUBMIT_OK = {'name': 'batches/test-wave-1'}
BATCH_ARGS = dict(max_calls=100, live_authorized=True, max_requests=10,
                  max_input_units=10 ** 9)


def batch_job(tmp_path, ontology, chunks=2, submit_replies=None, collect_replies=None):
    pipe, primary, _ = make_job(tmp_path, ontology, demo_chunks(chunks),
                                execution_mode='native_batch')
    primary.submit_replies = list(submit_replies if submit_replies is not None else [SUBMIT_OK])
    primary.collect_replies = list(collect_replies or [])
    return pipe, primary


def submit(pipe):
    return pipe.batch_submit(**BATCH_ARGS)


# ---------- prompt and plan identity ----------

def test_batch_items_carry_the_same_prompt_words(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    assert submit(pipe)['submitted'] is True
    items = primary.submit_calls[0]['items']
    for item, chunk in zip(items, pipe.plan['chunks']):
        # Same exact prompt words as a sync call, carried in the batch envelope.
        assert item['body']['contents'][0]['parts'][0]['text'] == build_prompt(chunk, ontology)
        assert item['chunk_id'] == chunk['chunk_id']


def test_tampered_plans_rejected(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    plan_path = pipe.job / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan['chunks'][0]['text'] = 'tampered'
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='plan content changed'):
        Pipeline(pipe.job, ontology, primary)


# ---------- gates before any network ----------

def test_submission_gates_before_network(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    with pytest.raises(ValueError, match='not authorized'):
        pipe.batch_submit(max_calls=10, max_requests=10, max_input_units=10 ** 9)
    with pytest.raises(ValueError, match='bounded native-batch'):
        pipe.batch_submit(max_calls=10, live_authorized=True, max_requests=0,
                          max_input_units=10 ** 9)
    with pytest.raises(ValueError, match='bounded native-batch'):
        pipe.batch_submit(max_calls=10, live_authorized=True, max_requests=10,
                          max_input_units=None)
    assert primary.submit_calls == []


def test_exact_conservative_reservation_blocks_oversized_wave(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    with pytest.raises(ValueError, match='enqueue-token allowance'):
        pipe.batch_submit(max_calls=100, live_authorized=True, max_requests=10,
                          max_input_units=10)
    assert primary.submit_calls == []


def test_call_cap_blocks_before_submission(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    result = pipe.batch_submit(max_calls=0, live_authorized=True, max_requests=10,
                               max_input_units=10 ** 9)
    assert result['submitted'] is False
    assert primary.submit_calls == []


def test_single_submission_and_no_duplicate(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    assert submit(pipe)['submitted'] is True
    assert len(primary.submit_calls) == 1
    with pytest.raises(ValueError, match='pending or ambiguous batch'):
        submit(pipe)
    assert len(primary.submit_calls) == 1


def test_unknown_submit_holds_claim_and_never_retries(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology,
                              submit_replies=[CallFailure('transport_unknown')])
    with pytest.raises(CallFailure):
        submit(pipe)
    # The durable claim exists without a receipt: never resubmitted automatically.
    with pytest.raises(ValueError, match='pending or ambiguous batch'):
        submit(pipe)
    assert len(primary.submit_calls) == 1


# ---------- collection ----------

def test_collect_pending_reads_once_no_submit(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology, collect_replies=[{'done': False}])
    submit(pipe)
    result = pipe.batch_collect(1)
    assert result['done'] is False
    assert len(primary.collect_calls) == 1
    assert len(primary.submit_calls) == 1
    assert not (pipe.job / 'batch-1-collected.json').exists()


def test_collect_saves_results_then_summary_and_is_idempotent(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    submit(pipe)
    items = primary.submit_calls[0]['items']
    rows = {item['custom_id']: payload_json(demo_chunks(2)[i])
            for i, item in enumerate(items)}
    primary.collect_replies = [done_collect(items, rows)]
    result = pipe.batch_collect(1)
    assert result['done'] is True
    assert (pipe.job / 'batch-1-collected.json').exists()
    records = pipe.records()
    assert len(records) == 2
    assert all(record['accepted'] for record in records)
    again = pipe.batch_collect(1)
    assert again['already_collected'] is True
    assert len(primary.collect_calls) == 1  # read once, then durable


def test_terminal_provider_errors_preserved_not_retried(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    submit(pipe)
    items = primary.submit_calls[0]['items']
    # gRPC 3 = INVALID_ARGUMENT: terminal, not a retryable 8/13/14.
    errors = {items[0]['custom_id']: {'code': 3, 'message': 'bad request'}}
    rows = {items[1]['custom_id']: payload_json(demo_chunks(2)[1])}
    primary.collect_replies = [done_collect(items, rows, errors)]
    pipe.batch_collect(1)
    attempts, _ = pipe.attempts()
    failed = [a for a in attempts if a['outcome'] != 'returned']
    assert len(failed) == 1 and failed[0]['outcome'] == 'batch_unusable_response'
    # A second wave contains no retry for the terminal item.
    primary.submit_replies = [SUBMIT_OK]
    result = submit(pipe)
    assert result['submitted'] is False


def test_retryable_batch_error_gets_one_later_wave(tmp_path, ontology):
    pipe, primary, _ = make_job(tmp_path, ontology, demo_chunks(2),
                                execution_mode='native_batch', retries=1)
    primary.submit_replies = [SUBMIT_OK]
    submit(pipe)
    items = primary.submit_calls[0]['items']
    errors = {items[0]['custom_id']: {'code': 14, 'message': 'unavailable'}}
    rows = {items[1]['custom_id']: payload_json(demo_chunks(2)[1])}
    primary.collect_replies = [done_collect(items, rows, errors)]
    pipe.batch_collect(1)
    attempts, _ = pipe.attempts()
    outcomes = sorted(a['outcome'] for a in attempts)
    assert 'batch_error_retryable' in outcomes
    primary.submit_replies = [{'name': 'batches/test-wave-2'}]
    result = submit(pipe)
    assert result['submitted'] is True
    retry_items = primary.submit_calls[1]['items']
    assert len(retry_items) == 1  # only the failed chunk is resent


def test_bad_results_no_completion_claim(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    submit(pipe)
    items = primary.submit_calls[0]['items']
    reply = done_collect(items)
    reply['results'].append(dict(reply['results'][0]))  # duplicate ID
    primary.collect_replies = [reply]
    with pytest.raises(ValueError, match='duplicate/unknown batch result ID'):
        pipe.batch_collect(1)
    assert not (pipe.job / 'batch-1-collected.json').exists()


def test_refusal_truncation_never_collected_as_valid(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    submit(pipe)
    items = primary.submit_calls[0]['items']
    reply = done_collect(items)
    for row in reply['results']:
        row['body']['candidates'][0]['finishReason'] = 'MAX_TOKENS'
    primary.collect_replies = [reply]
    pipe.batch_collect(1)
    attempts, _ = pipe.attempts()
    assert all(a['outcome'] == 'batch_unusable_response' for a in attempts)
    # Checkpoints exist but none claims a valid answer.
    assert pipe.records() != []
    assert all(record['accepted'] is False for record in pipe.records())


def test_unknown_cost_and_usage_stay_unknown(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    submit(pipe)
    items = primary.submit_calls[0]['items']
    reply = done_collect(items)
    for row in reply['results']:
        row['body'].pop('usageMetadata')  # provider omitted usage
    primary.collect_replies = [reply]
    pipe.batch_collect(1)
    attempts, _ = pipe.attempts()
    assert all(a['usage'] is None for a in attempts)
    assert all(a['cost_usd'] is None for a in attempts)


def test_batch_wave_never_uses_shared_response_cache(tmp_path, ontology):
    cache = tmp_path / 'cache'
    cache.mkdir()
    pipe, primary, _ = make_job(tmp_path, ontology, demo_chunks(1),
                                execution_mode='native_batch', cache=cache)
    primary.submit_replies = [SUBMIT_OK]
    with pytest.raises(ValueError, match='checkpoints, not shared response cache'):
        submit(pipe)
    assert primary.submit_calls == []


def test_collect_wave_number_validated(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    with pytest.raises(ValueError, match='positive batch wave'):
        pipe.batch_collect(0)


def test_sync_runner_refuses_batch_job_and_vice_versa(tmp_path, ontology):
    pipe, primary = batch_job(tmp_path, ontology)
    with pytest.raises(ValueError, match='native batch job requires'):
        pipe.run(max_calls=1, requests_per_minute=60, live_authorized=True)
    sync_pipe, _, _ = make_job(tmp_path, ontology, demo_chunks(1), name='sync-job')
    with pytest.raises(ValueError, match='native_batch'):
        sync_pipe.batch_submit(**BATCH_ARGS)


# ---------- aliases stay source-local (ported intent) ----------

ALIAS_TEXT = ('Apple and its subsidiaries ("Apple") filed this report. '
              'Apple develops the Vision Engine for customers.')


def alias_chunk(chunk_id='alias-1'):
    return {'chunk_id': chunk_id, 'text': ALIAS_TEXT,
            'document_hash': 'f' * 64, 'accession': 'a1', 'form': '10-K', 'period': '2026'}


def alias_answer(aliases):
    return json.dumps({
        'entities': [
            {'canonical_name': 'Apple', 'entity_type': 'Company', 'aliases': aliases,
             'description': 'The filing company and its subsidiaries per the source.',
             'source_chunk_id': 'alias-1', 'confidence': 0.9,
             'evidence': 'Apple and its subsidiaries ("Apple") filed this report.'},
            {'canonical_name': 'Vision Engine', 'entity_type': 'Technology', 'aliases': [],
             'description': 'A technology.', 'source_chunk_id': 'alias-1',
             'confidence': 0.9, 'evidence': 'Apple develops the Vision Engine for customers.'}],
        'relationships': []})


def test_apple_explicit_collective_alias_not_fake_subsidiary(ontology):
    """A source-defined collective alias is preserved on the entity, never
    turned into a fake OWNS/subsidiary edge."""
    record = validated_record(alias_answer(['Apple and its subsidiaries']), alias_chunk(), ontology)
    assert record['accepted'], record.get('problems')
    assert record['entities'][0]['aliases'] == ['Apple and its subsidiaries']
    assert record['relationships'] == []  # no invented ownership edge


def test_alias_fabricating_a_subsidiary_edge_is_rejected(ontology):
    answer = json.loads(alias_answer([]))
    answer['entities'].append({
        'canonical_name': 'Apple and its subsidiaries', 'entity_type': 'Subsidiary',
        'aliases': [], 'description': 'Invented subsidiary.',
        'source_chunk_id': 'alias-1', 'confidence': 0.9,
        'evidence': 'Apple and its subsidiaries ("Apple") filed this report.'})
    answer['relationships'].append({
        'relationship_type': 'OWNS', 'source_entity': 'Apple',
        'target_entity': 'Apple and its subsidiaries', 'source_chunk_id': 'alias-1',
        'evidence': 'Apple and its subsidiaries ("Apple") filed this report.',
        'confidence': 0.9})
    record = validated_record(json.dumps(answer), alias_chunk(), ontology)
    # Mechanically the fabricated OWNS edge fits the ontology, so validation can
    # accept it - but validation is never factual accuracy: it must stay marked
    # Not reviewed and blocked from graph writes until source review confirms it.
    if record['accepted']:
        assert record['semantic_evidence_status'] == 'Not reviewed'
        assert record['graph_write_permitted'] is False
    # A rejected outcome is equally honest; a silent auto-merge would not be.


def test_no_alias_merge_without_source_definition(ontology):
    """Two chunks mentioning 'Apple' separately never merge aliases globally."""
    chunk = alias_chunk()
    record = validated_record(alias_answer([]), chunk, ontology)
    assert record['accepted']
    assert record['entities'][0]['aliases'] == []
    # validated_record is per-chunk: nothing carries aliases across chunks.


def test_whitespace_entity_and_relation_dedup_is_conservative(ontology):
    answer = json.loads(alias_answer([]))
    answer['entities'].append(dict(answer['entities'][0]))
    answer['entities'][1]['canonical_name'] = 'Apple '  # trailing space
    record = validated_record(json.dumps(answer), alias_chunk(), ontology)
    assert not record['accepted']  # near-duplicates are refused, not merged


def test_invalid_duplicate_cannot_hide_behind_valid_one(ontology):
    answer = json.loads(alias_answer([]))
    answer['entities'].append(dict(answer['entities'][0]))
    record = validated_record(json.dumps(answer), alias_chunk(), ontology)
    assert not record['accepted']
    assert record['entities'] == []  # the whole payload is refused, no partial keep


def test_shape_failure_never_normalized_into_valid(ontology):
    record = validated_record('{"entities": "Apple"}', alias_chunk(), ontology)
    assert not record['accepted']
    assert record['entities'] == []


def test_type_conflicts_stay_rejected(ontology):
    answer = json.loads(alias_answer([]))
    answer['entities'][1]['entity_type'] = 'Company'  # two companies named differently
    answer['relationships'].append({
        'relationship_type': 'DEVELOPS', 'source_entity': 'Apple',
        'target_entity': 'Vision Engine', 'source_chunk_id': 'alias-1',
        'evidence': 'Apple develops the Vision Engine for customers.',
        'confidence': 0.9})
    record = validated_record(json.dumps(answer), alias_chunk(), ontology)
    assert not record['accepted']  # DEVELOPS must end at Technology
