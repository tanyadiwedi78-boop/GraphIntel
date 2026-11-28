"""Offline tests for golden-run observability: telemetry, summary, categories.

No network, no model calls. Telemetry tests use an in-memory exporter; the
OTLP path is exercised against an unreachable local port to prove export
failure never breaks a run.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from app.extraction import chunk_metrics, golden_telemetry
from scripts import summarize_golden_run
from scripts.run_golden_extraction import run

otel = pytest.importorskip('opentelemetry')
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def base_config(tmp_path, chunks_document):
    chunks_path = tmp_path / 'chunks.json'
    chunks_path.write_text(json.dumps(chunks_document), encoding='utf-8')
    config = {
        'models': {'fake': 'fake/model:free'},
        'active_model': 'fake',
        'reasoning': {'effort': 'none'},
        'max_output_tokens': 4096,
        'chunks_path': str(chunks_path),
        'fewshot_path': str(ROOT / 'configs' / 'golden_fewshot_examples_v5.json'),
        'ontology_path': str(ROOT / 'configs' / 'ontology.json'),
        'reports_dir': str(tmp_path / 'reports'),
        'chunk_limit': None,
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    return config, config_path


def fake_response(payload_json, latency=12.5):
    return {'text': payload_json, 'latency_ms': latency,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20,
                      'prompt_tokens_details': {'cached_tokens': 0},
                      'completion_tokens_details': {'reasoning_tokens': 0},
                      'cost': 0.0},
            'provider': 'FakeProvider', 'openrouter_metadata': None,
            'http_status': 200}


VALID = json.dumps({
    'entities': [{'canonical_name': 'Acme', 'entity_type': 'Company',
                  'aliases': [], 'description': 'Acme develops the Widget platform.',
                  'source_chunk_id': 'c1', 'confidence': 0.9,
                  'evidence': 'Acme develops the Widget platform.'}],
    'relationships': []})


# --- metrics rows -----------------------------------------------------------

def test_record_row_carries_chunk_category():
    row = chunk_metrics.record_row('c1', 'm:free', fake_response('{}'), True, 3, 1, 12.5,
                                   chunk_category='dense')
    assert row['chunk_category'] == 'dense'
    plain = chunk_metrics.record_row('c1', 'm:free', fake_response('{}'), True, 3, 1, 12.5)
    assert plain['chunk_category'] == 'unassigned'


# --- telemetry ---------------------------------------------------------------

def test_telemetry_off_by_default_returns_none():
    assert golden_telemetry.start({}) is None
    assert golden_telemetry.start({'telemetry': {'enabled': False}}) is None


def test_telemetry_spans_carry_metadata_only(tmp_path):
    chunks = [{'chunk_id': 'c1', 'text': 'Acme develops the Widget platform.',
               'golden_category': 'clean'}]
    config, config_path = base_config(tmp_path, chunks)
    config['telemetry'] = {'enabled': True}
    config_path.write_text(json.dumps(config), encoding='utf-8')

    exporter = InMemorySpanExporter()
    telemetry = golden_telemetry.start(config, exporter=exporter)
    assert telemetry is not None

    import scripts.run_golden_extraction as runner
    real_start = golden_telemetry.start
    runner.golden_telemetry.start = lambda cfg: real_start(cfg, exporter=exporter)
    try:
        assert run(config_path, completer=lambda prompt: fake_response(VALID)) == 0
    finally:
        runner.golden_telemetry.start = real_start

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes)
    assert attributes['chunk.id'] == 'c1'
    assert attributes['chunk.category'] == 'clean'
    assert attributes['llm.model'] == 'fake/model:free'
    assert attributes['schema.valid'] is True
    assert attributes['llm.tokens.input'] == 100
    assert attributes['llm.cost_usd'] == 0.0
    assert attributes['extraction.mechanical_problems'] == 0
    assert attributes['extraction.semantic_review'] == 'not_reviewed'
    assert span.status.status_code.name == 'OK'
    leaked = json.dumps({key: str(value) for key, value in attributes.items()})
    assert 'Acme develops the Widget platform' not in leaked
    assert 'Acme develops' not in str(span.name)


def test_failed_chunk_span_is_error_and_redacted(tmp_path):
    chunks = [{'chunk_id': 'c1', 'text': 'Acme develops the Widget platform.'}]
    config, config_path = base_config(tmp_path, chunks)
    config['telemetry'] = {'enabled': True}
    config_path.write_text(json.dumps(config), encoding='utf-8')

    from scripts.run_golden_extraction import GoldenCallError

    def failing(prompt):
        raise GoldenCallError(500, 'http 500')

    exporter = InMemorySpanExporter()
    import scripts.run_golden_extraction as runner
    real_start = golden_telemetry.start
    runner.golden_telemetry.start = lambda cfg: real_start(cfg, exporter=exporter)
    try:
        assert run(config_path, completer=failing) in (0, 1)
    finally:
        runner.golden_telemetry.start = real_start

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code.name == 'ERROR'
    dumped = json.dumps({key: str(value) for key, value in dict(spans[0].attributes).items()})
    assert 'Acme develops' not in dumped
    assert dict(spans[0].attributes)['error.type'] == 'http 500'


def test_error_label_never_carries_message_body():
    exporter = InMemorySpanExporter()
    telemetry = golden_telemetry.start({'telemetry': {'enabled': True}}, exporter=exporter)
    row = {'chunk_id': 'c1', 'model': 'm:free', 'served_provider': 'P',
           'schema_valid': False,
           'error': 'schema validation failed: raw answer text with sk-or-secretkey123 quoted'}
    telemetry.record_chunk(row, 1.0, 2.0)
    span = exporter.get_finished_spans()[0]
    attributes = dict(span.attributes)
    assert attributes['error.type'] == 'schema validation failed'
    dumped = json.dumps({key: str(value) for key, value in attributes.items()})
    assert 'sk-or-secretkey123' not in dumped
    assert 'raw answer text' not in dumped
    assert span.status.status_code.name == 'ERROR'


def test_export_failure_never_breaks_extraction(tmp_path, capsys):
    chunks = [{'chunk_id': 'c1', 'text': 'Acme develops the Widget platform.'}]
    config, config_path = base_config(tmp_path, chunks)
    # Nothing listens on this port: every export attempt fails.
    config['telemetry'] = {'enabled': True,
                           'otlp_endpoint': 'http://127.0.0.1:9/v1/traces'}
    config_path.write_text(json.dumps(config), encoding='utf-8')

    assert run(config_path, completer=lambda prompt: fake_response(VALID)) == 0
    run_folder = next((tmp_path / 'reports').iterdir())
    rows = chunk_metrics.read_rows(run_folder / 'metrics.jsonl')
    assert len(rows) == 1 and rows[0]['schema_valid'] is True
    assert 'Telemetry' in capsys.readouterr().out


# --- run summary ---------------------------------------------------------------

def write_run(folder, rows, summary):
    folder.mkdir(parents=True)
    with open(folder / 'metrics.jsonl', 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    (folder / 'results.json').write_text(
        json.dumps({'summary': summary, 'results': []}), encoding='utf-8')


def metric_row(chunk_id, ok, category='clean', latency=10.0, cost=0.0,
               provider='FakeProvider', prompt_tokens=100):
    row = {'chunk_id': chunk_id, 'chunk_category': category, 'model': 'm:free',
           'served_provider': provider, 'prompt_tokens': prompt_tokens,
           'completion_tokens': 20, 'cached_tokens': 0, 'reasoning_tokens': 0,
           'cost_usd': cost, 'schema_valid': ok, 'entity_count': 1, 'edge_count': 0,
           'latency_ms': latency}
    if not ok:
        row['error'] = 'http 500'
    return row


def test_summary_counts_failed_attempts_in_schema_rate(tmp_path):
    rows = [metric_row('c1', True), metric_row('c2', False, latency=99.0)]
    write_run(tmp_path / 'run-x', rows, {'model': 'm:free', 'prompt_version': 'v'})
    report = summarize_golden_run.summarize(tmp_path / 'run-x')
    assert report['first_pass_schema'] == {'passed': 1, 'attempted': 2, 'rate': 0.5,
                                           'note': 'Denominator counts every attempted chunk, including failed calls.'}
    assert report['latency']['slowest_chunks'][0]['chunk_id'] == 'c2'
    assert report['categories']['clean'] == {'attempted': 2, 'schema_pass': 1}


def test_summary_zero_cost_is_honest_zero_without_savings(tmp_path):
    rows = [metric_row('c1', True), metric_row('c2', True)]
    write_run(tmp_path / 'run-x', rows, {'model': 'm:free'})
    report = summarize_golden_run.summarize(tmp_path / 'run-x')
    assert report['cost']['total_usd'] == 0.0
    assert report['cost']['per_1000_chunks_projection_usd'] == 0.0
    assert 'free models' in report['cost']['note']
    assert '%' not in json.dumps(report['cost'])


def test_summary_missing_cost_stays_unknown(tmp_path):
    row = metric_row('c1', True)
    row['cost_usd'] = 'unavailable'
    write_run(tmp_path / 'run-x', [row], {'model': 'm:free'})
    report = summarize_golden_run.summarize(tmp_path / 'run-x')
    assert report['cost']['total_usd'] == 'unknown'
    assert 'unknown' in report['cost']['note']


def test_summary_token_projection_and_providers(tmp_path):
    rows = [metric_row('c1', True, prompt_tokens=100),
            metric_row('c2', True, prompt_tokens=300, provider='OtherCo')]
    write_run(tmp_path / 'run-x', rows, {'model': 'm:free'})
    report = summarize_golden_run.summarize(tmp_path / 'run-x')
    tokens = report['tokens']['prompt_tokens']
    assert tokens['total'] == 400
    assert tokens['per_1000_chunks_projection'] == 200000.0
    assert report['served_providers'] == {'FakeProvider': 1, 'OtherCo': 1}
    assert report['grounding']['semantic_review'].startswith('not reviewed')
