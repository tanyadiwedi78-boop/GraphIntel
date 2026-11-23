"""Offline fault injection for the new async path. No network or graph writes."""

import asyncio
import json
from pathlib import Path

import pytest

from app.evaluation.extraction_benchmark import identity
from app.extraction.async_runner import AsyncRunner
from app.extraction.failures import classify_failure
from app.extraction.pipeline import Pipeline, prepare, retry_policy
from app.extraction.reporting import read_events, token_counts
from tests.test_phase1_pipeline import FakeProvider, chunk, ontology, payload_data


@pytest.fixture(autouse=True)
def immediate_async_wait(monkeypatch):
    async def immediate(seconds):
        return None
    monkeypatch.setattr('app.extraction.async_runner.asyncio.sleep', immediate)


def runner(tmp_path, ontology, chunk, primary_values, fallback_values=None, retries=0):
    primary = FakeProvider(primary_values)
    fallback = FakeProvider(fallback_values, 'openrouter') if fallback_values is not None else None
    job = tmp_path / 'job'
    prepare(job, [chunk], ontology, primary.recipe, fallback.recipe if fallback else None,
            policy=retry_policy(retries))
    pipeline = Pipeline(job, ontology, primary, fallback)
    return AsyncRunner(pipeline, tmp_path / 'reports')


def execute(instance, maximum=10, authorized=True):
    limits = {'gemini': {'rpm': 1e12, 'tpm': None}, 'openrouter': {'rpm': 1e12, 'tpm': None}}
    return asyncio.run(instance.run(maximum, limits, authorized, True))


def test_async_success_local_reports_and_resume(tmp_path, ontology, chunk, payload_data):
    instance = runner(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    result = execute(instance)
    folder = Path(result['reports'])
    assert result['status'] == 'OK'
    assert (folder / 'dashboard.html').exists()
    assert (folder / 'spans.jsonl').exists()
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['attempts_finished'] == 1
    assert summary['tokens']['total'] is None
    assert summary['mean_request_seconds'] == .25
    assert json.loads((folder / 'quality.json').read_text())['status'] == 'Not measured'
    assert chunk['text'] not in (folder / 'events.jsonl').read_text()
    second = AsyncRunner(instance.pipeline, tmp_path / 'reports')
    execute(second)
    assert instance.pipeline.primary.calls == 1
    assert instance.telemetry.run_id != second.telemetry.run_id


@pytest.mark.parametrize('status,category', [(429, 'rate_limit'), (401, 'authentication_error'), (403, 'authentication_error')])
def test_systemic_failure_stops_without_fallback(tmp_path, ontology, chunk, status, category):
    from app.extraction.pipeline import CallFailure
    instance = runner(tmp_path, ontology, chunk, [CallFailure('http_failure', status)], ['unused'], retries=2)
    result = execute(instance)
    assert result['status'] == 'STOPPED'
    assert instance.pipeline.primary.calls == 1
    assert instance.pipeline.fallback.calls == 0
    dead = read_events(instance.telemetry.folder / 'dead-letter.jsonl')
    assert dead[0]['category'] == category and dead[0]['replayable'] is False


def test_schema_failure_falls_back_after_bounded_primary(tmp_path, ontology, chunk, payload_data):
    instance = runner(tmp_path, ontology, chunk, ['{}'], [json.dumps(payload_data)])
    result = execute(instance)
    assert result['status'] == 'OK'
    assert instance.pipeline.primary.calls == 1 and instance.pipeline.fallback.calls == 1
    events = read_events(instance.telemetry.folder / 'events.jsonl')
    assert any(event['event'] == 'fallback_switched' for event in events)
    assert sum(event['event'] == 'attempt_started' for event in events) == 2


def test_valid_empty_is_not_automatically_a_failure(tmp_path, ontology, chunk):
    instance = runner(tmp_path, ontology, chunk, [json.dumps({'entities': [], 'relationships': []})], ['unused'])
    assert execute(instance)['status'] == 'OK'
    assert instance.pipeline.fallback.calls == 0


def test_timeout_unknown_never_resubmitted(tmp_path, ontology, chunk):
    from app.extraction.pipeline import CallFailure
    instance = runner(tmp_path, ontology, chunk, [CallFailure('transport_timeout')], ['unused'])
    assert execute(instance)['status'] == 'STOPPED'
    assert instance.pipeline.primary.calls == 1
    second = AsyncRunner(instance.pipeline, tmp_path / 'reports')
    assert execute(second)['status'] == 'STOPPED'
    assert instance.pipeline.primary.calls == 1


def test_preflight_and_unexpected_errors_are_recorded(tmp_path, ontology, chunk):
    instance = runner(tmp_path, ontology, chunk, [RuntimeError('secret custom message')])
    with pytest.raises(ValueError, match='authorized'):
        execute(instance, authorized=False)
    assert 'run_failed' in (instance.telemetry.folder / 'events.jsonl').read_text()
    second = AsyncRunner(instance.pipeline, tmp_path / 'reports')
    with pytest.raises(RuntimeError):
        execute(second)
    assert not (instance.pipeline.job / 'run-lock.json').exists()
    assert 'secret custom message' not in (second.telemetry.folder / 'events.jsonl').read_text()
    third = AsyncRunner(instance.pipeline, tmp_path / 'reports')
    with pytest.raises(ValueError, match='Unreconciled'):
        execute(third)
    assert instance.pipeline.primary.calls == 1


def test_total_budget_preserves_unfinished_chunk(tmp_path, ontology, chunk, payload_data):
    instance = runner(tmp_path, ontology, chunk, ['{}', json.dumps(payload_data)], retries=1)
    assert execute(instance, maximum=1)['status'] == 'STOPPED'
    assert not list(instance.pipeline.job.glob('chunk-*.json'))
    second = AsyncRunner(instance.pipeline, tmp_path / 'reports')
    assert execute(second, maximum=2)['status'] == 'OK'
    assert instance.pipeline.primary.calls == 2


@pytest.mark.parametrize('finish,category', [('SAFETY', 'safety_block'), ('MAX_TOKENS', 'truncated_output')])
def test_finish_reason_overrules_otherwise_accepted_output(finish, category):
    result = {'http_status': 200, 'raw_envelope': json.dumps({'candidates': [{'finishReason': finish}]})}
    failure = classify_failure(result, {'accepted': True})
    assert failure.category == category


def test_unknown_tokens_stay_unknown():
    assert token_counts({'prompt_tokens': 0, 'completion_tokens': 4}) == {'input': 0, 'output': 4}
    assert token_counts({'prompt_tokens': True}) == {'input': None, 'output': None}


def test_replay_is_new_source_reviewed_job(tmp_path, ontology, chunk):
    from app.extraction.recovery import prepare_replay
    instance = runner(tmp_path, ontology, chunk, ['{}'])
    execute(instance)
    dead = instance.telemetry.folder / 'dead-letter.jsonl'
    with pytest.raises(ValueError, match='source-bound review'):
        prepare_replay(instance.pipeline, dead, {}, tmp_path / 'new-job')
    reviews = {chunk['chunk_id']: {'reviewed': True,
                'source_binding': identity([instance.pipeline.plan['plan_hash'], chunk['chunk_id'], chunk['text']])}}
    plan = prepare_replay(instance.pipeline, dead, reviews, tmp_path / 'new-job')
    assert plan['chunks'] == instance.pipeline.plan['chunks']
    assert list(instance.pipeline.job.glob('chunk-*.json'))


def test_concurrency_cannot_bypass_conservative_default(tmp_path, ontology, chunk):
    instance = runner(tmp_path, ontology, chunk, [])
    with pytest.raises(ValueError, match='concurrency 1'):
        asyncio.run(instance.run(2, {}, True, concurrency=8))


def test_replay_rechecks_evidence_not_editable_dead_letter_flags(tmp_path, ontology, chunk):
    from app.extraction.pipeline import CallFailure
    from app.extraction.recovery import prepare_replay
    instance = runner(tmp_path, ontology, chunk, [CallFailure('http_failure', 429)])
    execute(instance)
    path = instance.telemetry.folder / 'dead-letter.jsonl'
    entry = read_events(path)[0]
    entry.update(replayable=True, ambiguous=False, category='schema_error')
    path.write_text(json.dumps(entry) + '\n')
    reviews = {chunk['chunk_id']: {'reviewed': True,
        'source_binding': identity([instance.pipeline.plan['plan_hash'], chunk['chunk_id'], chunk['text']])}}
    with pytest.raises(ValueError, match='not eligible'):
        prepare_replay(instance.pipeline, path, reviews, tmp_path / 'forged-replay')


def test_live_report_refresh_does_not_change_job_evidence(tmp_path, ontology, chunk, payload_data):
    from app.extraction.reporting import write_dashboard
    instance = runner(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    execute(instance)
    before = {path.name: path.read_bytes() for path in instance.pipeline.job.glob('*.json')}
    folder = instance.telemetry.folder
    write_dashboard(instance.pipeline, folder)
    after = {path.name: path.read_bytes() for path in instance.pipeline.job.glob('*.json')}
    assert before == after
    assert json.loads((folder / 'summary.json').read_text())['schema_validity'] == 1


def test_report_html_escapes_external_strings_and_redacts_keys(tmp_path, ontology, chunk, payload_data, monkeypatch):
    from app.extraction.dashboard import dashboard_html
    from app.extraction.observability import scrub
    monkeypatch.setenv('GEMINI_API_KEY', 'configured-secret-value')
    summary = {'attempts': [], 'basis': '<script>attack</script>', 'comparison_warning': 'none'}
    for key in ('attempts_started', 'attempts_finished', 'unfinished_attempts', 'accepted_attempts',
                'retry_attempts', 'fallback_attempts', 'cache_hits', 'checkpoint_skips',
                'mean_request_seconds', 'p95_request_seconds', 'latency_sample_count', 'cost_usd',
                'schema_validity', 'schema_denominator', 'retry_rate', 'retry_denominator'):
        summary[key] = None
    summary['tokens'] = None
    document = dashboard_html(summary, [], {})
    assert '<script>attack</script>' not in document
    assert '&lt;script&gt;' in document
    assert 'configured-secret-value' not in json.dumps(scrub({'configured-secret-value': 'Bearer abcdefgh'}))


def test_report_folder_binding_blocks_wrong_job(tmp_path, ontology, chunk, payload_data):
    from app.extraction.reporting import write_dashboard
    instance = runner(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    execute(instance)
    manifest = instance.telemetry.folder / 'manifest.json'
    saved = json.loads(manifest.read_text())
    saved['plan_hash'] = 'wrong'
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='different immutable job'):
        write_dashboard(instance.pipeline, instance.telemetry.folder)


def test_legacy_sync_path_also_cannot_fallback_on_first_429(tmp_path, ontology, chunk):
    from app.extraction.pipeline import CallFailure
    instance = runner(tmp_path, ontology, chunk, [CallFailure('http_failure', 429)], ['unused'], retries=2)
    instance.pipeline.run(10, 1e12, True, True)
    assert instance.pipeline.primary.calls == 1
    assert instance.pipeline.fallback.calls == 0


def test_pacing_records_wait_and_total_budget_stops_before_network(tmp_path, ontology, chunk, payload_data):
    instance = runner(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    result = execute(instance, maximum=0)
    assert result['status'] == 'STOPPED'
    assert instance.pipeline.primary.calls == 0


def test_cancellation_waits_for_transport_and_preserves_claim_result(tmp_path, ontology, chunk, payload_data):
    import threading
    instance = runner(tmp_path, ontology, chunk, [])
    entered = threading.Event()
    finish = threading.Event()
    original = instance.pipeline.primary.call

    def blocked(prompt, schema):
        entered.set()
        assert finish.wait(3)
        return {'raw_text': json.dumps(payload_data), 'raw_envelope': '{}',
                'outcome': 'returned', 'http_status': 200, 'request_seconds': .1}

    instance.pipeline.primary.call = blocked

    async def scenario():
        task = asyncio.create_task(instance.run(2, {'gemini': {'rpm': 1e12, 'tpm': None}}, True))
        await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert (instance.pipeline.job / 'attempt-1-result.json').exists()
    assert not (instance.pipeline.job / 'run-lock.json').exists()
    assert 'cancellation_wait' in (instance.telemetry.folder / 'events.jsonl').read_text()
    second = AsyncRunner(instance.pipeline, tmp_path / 'reports')
    assert execute(second)['status'] == 'OK'
    assert len(list(instance.pipeline.job.glob('attempt-*-claim.json'))) == 1


def test_server_retry_then_fallback_and_exact_attempt_budget(tmp_path, ontology, chunk, payload_data):
    from app.extraction.pipeline import CallFailure
    instance = runner(tmp_path, ontology, chunk,
                      [CallFailure('http_failure', 503), '{}'], [json.dumps(payload_data)], retries=1)
    result = execute(instance, maximum=3)
    assert result['status'] == 'OK'
    assert instance.pipeline.primary.calls == 2 and instance.pipeline.fallback.calls == 1
    summary = json.loads((instance.telemetry.folder / 'summary.json').read_text())
    assert summary['retry_attempts'] == 1 and summary['fallback_attempts'] == 1
    assert summary['schema_denominator'] == 3


@pytest.mark.parametrize('outcome,expected', [('transport_timeout', 'timeout'), ('transport_reset', 'unknown_error')])
def test_classifications_preserve_ambiguous_transport(outcome, expected):
    failure = classify_failure({'outcome': outcome}, {'accepted': False})
    assert failure.category == expected and failure.ambiguous and not failure.retryable


def test_empty_response_is_not_silently_retried(tmp_path, ontology, chunk):
    instance = runner(tmp_path, ontology, chunk, [''], ['unused'], retries=2)
    result = execute(instance)
    assert result['status'] == 'STOPPED'
    assert instance.pipeline.primary.calls == 1 and instance.pipeline.fallback.calls == 0
    assert read_events(instance.telemetry.folder / 'dead-letter.jsonl')[0]['category'] == 'empty_output'


def test_usage_cannot_smuggle_unrelated_provider_content_into_logs(tmp_path, ontology, chunk, payload_data):
    instance = runner(tmp_path, ontology, chunk, [])
    def call(prompt, schema):
        return {'raw_text': json.dumps(payload_data), 'outcome': 'returned', 'http_status': 200,
                'request_seconds': .5, 'usage': {'prompt_tokens': 4, 'completion_tokens': 2,
                                              'private': 'unrelated sensitive text'}}
    instance.pipeline.primary.call = call
    execute(instance)
    assert 'unrelated sensitive text' not in (instance.telemetry.folder / 'events.jsonl').read_text()
    summary = json.loads((instance.telemetry.folder / 'summary.json').read_text())
    assert summary['tokens']['total'] == {'input': 4, 'output': 2}


def test_paid_scope_not_accepted_by_free_only_guard(tmp_path, ontology, chunk):
    instance = runner(tmp_path, ontology, chunk, [])
    instance.pipeline.primary.recipe['execution_mode'] = 'live_transport'
    instance.pipeline.plan['pricing'] = {'gemini': {'input_per_million_usd': '1',
        'output_per_million_usd': '1', 'free_tier_no_billing_verified': False}}
    with pytest.raises(ValueError, match='free-only'):
        instance.verify_free_scope(False)


def test_oversized_token_window_blocks_before_claim(tmp_path, ontology, chunk, payload_data):
    instance = runner(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    with pytest.raises(ValueError, match='token window'):
        asyncio.run(instance.run(2, {'gemini': {'rpm': 1e12, 'tpm': 1}}, True))
    assert instance.pipeline.primary.calls == 0
    assert not list(instance.pipeline.job.glob('attempt-*-claim.json'))


def test_source_changed_checkpoint_stops_before_new_calls(tmp_path, ontology, chunk, payload_data):
    instance = runner(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    execute(instance)
    checkpoint = next(instance.pipeline.job.glob('chunk-*.json'))
    saved = json.loads(checkpoint.read_text())
    saved['entities'][0]['canonical_name'] = 'Tampered'
    checkpoint.write_text(json.dumps(saved))
    second = AsyncRunner(instance.pipeline, tmp_path / 'reports')
    with pytest.raises(ValueError, match='checkpoint content'):
        execute(second)
    assert instance.pipeline.primary.calls == 1


def test_other_runner_lock_is_never_removed(tmp_path, ontology, chunk):
    instance = runner(tmp_path, ontology, chunk, [])
    lock = instance.pipeline.job / 'run-lock.json'
    lock.write_text('{"owner":"another-runner"}')
    with pytest.raises(FileExistsError):
        execute(instance)
    assert lock.exists()
    assert instance.pipeline.primary.calls == 0


def test_provider_wire_settings_are_explicit_and_truncation_rejected():
    from app.extraction.providers import JSONProvider
    gemini = JSONProvider('gemini', 'explicit-model', 512, timeout=90)
    fallback = JSONProvider('openrouter', 'vendor/model:free', 512, timeout=90)
    schema = {'type': 'object'}
    assert gemini.timeout == fallback.timeout == 90
    assert gemini.body('same source', schema)['generationConfig']['responseFormat']['text']['schema'] == schema
    body = fallback.body('same source', schema)
    assert body['response_format']['json_schema']['schema'] == schema
    assert body['response_format']['json_schema']['strict'] is True
    assert body['provider']['require_parameters'] is True
    # Temperature is intentionally not invented or silently changed.
    assert 'temperature' not in body
    with pytest.raises(ValueError):
        fallback.output_text({'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]})
    for timeout in (float('inf'), float('nan'), 0, True):
        with pytest.raises(ValueError):
            JSONProvider('gemini', 'explicit-model', 512, timeout=timeout)


def test_malformed_candidate_envelope_has_clear_classification():
    failure = classify_failure({'raw_envelope': '{"candidates":42}', 'http_status': 200}, {'accepted': False})
    assert failure.category == 'schema_error'


def test_quality_report_requires_source_bound_labels(tmp_path, ontology, chunk, payload_data):
    from app.extraction.reporting import write_dashboard
    instance = runner(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    execute(instance)
    with pytest.raises(ValueError):
        write_dashboard(instance.pipeline, instance.telemetry.folder,
                        labels={chunk['chunk_id']: {'reviewed': True, 'source_hash': 'wrong'}})


def test_secondary_dashboard_failure_does_not_hide_original_error(tmp_path, ontology, chunk, monkeypatch):
    instance = runner(tmp_path, ontology, chunk, [RuntimeError('original extraction failure')])
    def broken_report(*args, **kwargs):
        raise OSError('report disk failed')
    monkeypatch.setattr('app.extraction.async_runner.write_dashboard', broken_report)
    with pytest.raises(RuntimeError, match='original extraction failure'):
        execute(instance)
    assert not (instance.pipeline.job / 'run-lock.json').exists()
