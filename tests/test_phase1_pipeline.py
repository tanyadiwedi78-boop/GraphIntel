"""Offline vertical-path/fault tests, NEVER provider-quality benchmark results."""
import copy
import json
from pathlib import Path

import pytest

from app.evaluation.extraction_benchmark import identity
from app.extraction.pipeline import Pipeline, prepare, load, save_new, validated_record
from app.extraction.providers import CallFailure, JSONProvider
from app.graph.construction import graph_plan, resolve
from tests.test_anthropic_batch import ontology
from tests.test_extraction_contracts import payload_data


class FakeProvider:
    def __init__(self, values, provider='gemini'):
        self.values = list(values)
        self.recipe = {'provider': provider, 'model': 'offline-fixture', 'execution_mode': 'offline_fixture'}
        self.calls = 0

    def call(self, prompt, schema):
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return {'raw_text': value, 'raw_envelope': '{"offline_fixture": true}',
                'http_status': 200, 'request_seconds': 0.25,
                'usage': {'fixture': True}, 'outcome': 'returned', 'cost_usd': None}


@pytest.fixture
def chunk():
    return {'chunk_id': 'c1', 'text': 'Acme develops Widget.', 'document_hash': 'original',
            'accession': 'fixture-only', 'form': '10-K', 'period': '2023'}


def pipe(tmp_path, ontology, chunk, values, fallback_values=None, cache=None, name='job'):
    primary = FakeProvider(values)
    fallback = FakeProvider(fallback_values, 'openai') if fallback_values is not None else None
    job = tmp_path / name
    prepare(job, [chunk], ontology, primary.recipe, fallback.recipe if fallback else None)
    return Pipeline(job, ontology, primary, fallback, sleep=lambda seconds: None, cache_dir=cache)


def records(p):
    return [load(path) for path in p.job.glob('chunk-*.json')]


def reviews(record):
    result = {}
    for item in record['entities'] + record['relationships']:
        result[identity(item)] = {'reviewed': True, 'source_hash': record['source_hash'], 'supported': True}
    return {record['chunk_id']: result}


def test_vertical_path_extract_report_resolve_graph_and_resume(tmp_path, ontology, chunk, payload_data):
    p = pipe(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    report = p.run(1, 1, live_authorized=True)
    assert report['json_schema_pass_rate']['value'] == 1
    assert report['entity_precision']['status'] == 'Not measured'
    assert report['evidence_accuracy']['status'] == 'Not measured'
    assert report['cost_per_1000_chunks']['status'] == 'Not measured'
    assert p.primary.calls == 1
    record = records(p)[0]
    expected = {'c1': {'reviewed': True, 'source_hash': identity(chunk['text']),
                      'entities': payload_data['entities'], 'relationships': payload_data['relationships']}}
    report = p.report(expected, reviews(record))
    assert report['entity_precision']['value'] == 1
    assert report['entity_recall']['value'] == 1
    assert report['relationship_precision']['value'] == 1
    assert report['relationship_recall']['value'] == 1
    assert report['evidence_accuracy']['value'] == 1
    assert len(resolve([record])['mentions']) == 2
    graph = graph_plan([record], [], reviews(record))
    assert not graph['writes_performed']
    assert 'CLAIM_TARGET' in graph['statements'][-1]['query']
    assert 'Acme' not in graph['statements'][-1]['query']
    p.run(1, 1, live_authorized=True)
    assert p.primary.calls == 1


def test_no_execution_authority_no_claim(tmp_path, ontology, chunk):
    p = pipe(tmp_path, ontology, chunk, [])
    with pytest.raises(ValueError, match='authorized'):
        p.run(1, 1)
    assert not list(p.job.glob('*claim*'))


def test_zero_budget_zero_requests(tmp_path, ontology, chunk):
    p = pipe(tmp_path, ontology, chunk, [])
    p.run(0, 1, True)
    assert p.primary.calls == 0


def test_known_429_explicit_fallback_and_total_budget(tmp_path, ontology, chunk, payload_data):
    p = pipe(tmp_path, ontology, chunk, [CallFailure('http_failure', 429)], [json.dumps(payload_data)])
    report = p.run(2, 100, True, True)
    assert p.primary.calls == p.fallback.calls == 1
    assert records(p)[0]['provider'] == 'openai'
    assert report['rate_limit_failure_rate']['value'] == 0.5
    assert report['calls']['attempts'] == 2


@pytest.mark.parametrize('code', [400, 401, 402, 403, 404, 408, 500, 503, None])
def test_no_fallback_on_auth_quality_or_ambiguous_failure(tmp_path, ontology, chunk, code):
    p = pipe(tmp_path, ontology, chunk, [CallFailure('failure', code)], ['unused'])
    p.run(2, 100, True, True)
    assert p.fallback.calls == 0
    assert not records(p)[0]['accepted']


@pytest.mark.parametrize('budget,authorized', [(1, True), (2, False)])
def test_fallback_needs_budget_and_authority(tmp_path, ontology, chunk, budget, authorized):
    p = pipe(tmp_path, ontology, chunk, [CallFailure('http_failure', 429)], ['unused'])
    p.run(budget, 100, True, authorized)
    assert p.fallback.calls == 0


def test_retry_after_above_bound_stops(tmp_path, ontology, chunk):
    p = pipe(tmp_path, ontology, chunk, [CallFailure('http_failure', 429, 120)], ['unused'])
    p.run(2, 100, True, True, max_fallback_wait=60)
    assert p.fallback.calls == 0


def test_crash_claim_blocks_resume_and_is_not_deleted(tmp_path, ontology, chunk):
    p = pipe(tmp_path, ontology, chunk, [RuntimeError('injected crash')])
    with pytest.raises(RuntimeError):
        p.run(2, 100, True)
    assert len(list(p.job.glob('*claim*'))) == 1
    with pytest.raises(ValueError, match='unreconciled'):
        p.run(2, 100, True)
    assert p.primary.calls == 1
    report = p.report()
    assert report['rate_limit_failure_rate']['unknown_status_attempts'] == 1
    assert report['rate_limit_failure_rate']['value'] is None


def test_response_saved_before_crash_recovered_without_call(tmp_path, ontology, chunk, payload_data):
    p = pipe(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    p.attempt(p.primary, chunk, 1)
    p.run(1, 100, True)
    assert p.primary.calls == 1
    assert records(p)[0]['recovered_from_checkpoint']


@pytest.mark.parametrize('raw', ['{}', '{broken', '{"entities":[],"relationships":[],"extra":1}'])
def test_invalid_output_is_terminal_not_repaired(tmp_path, ontology, chunk, raw):
    p = pipe(tmp_path, ontology, chunk, [raw], ['unused'])
    p.run(5, 100, True, True)
    p.run(5, 100, True, True)
    assert p.primary.calls == 1 and p.fallback.calls == 0
    assert not records(p)[0]['accepted']


def test_quote_presence_does_not_authorize_graph(tmp_path, ontology, chunk, payload_data):
    payload_data['entities'][0]['description'] = 'Unsupported claim despite a real quote.'
    p = pipe(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    p.run(1, 100, True)
    with pytest.raises(ValueError, match='semantic review'):
        graph_plan(records(p), [], {})


def test_cache_exact_recipe_source_and_no_cached_latency_reused(tmp_path, ontology, chunk, payload_data):
    cache = tmp_path / 'cache'
    first = pipe(tmp_path, ontology, chunk, [json.dumps(payload_data)], cache=cache, name='first')
    first.run(1, 100, True)
    second = pipe(tmp_path, ontology, chunk, [], cache=cache, name='second')
    report = second.run(0, 100, True)
    assert second.primary.calls == 0
    assert report['cache']['hits'] == 1
    assert report['extraction_latency']['status'] == 'Not measured'
    changed = dict(chunk, document_hash='changed')
    third = pipe(tmp_path, ontology, changed, [], cache=cache, name='third')
    report = third.run(0, 100, True)
    assert report['cache']['hits'] == 0


def test_cache_corruption_blocks_not_calls(tmp_path, ontology, chunk, payload_data):
    cache = tmp_path / 'cache'
    p = pipe(tmp_path, ontology, chunk, [json.dumps(payload_data)], cache=cache)
    p.run(1, 100, True)
    path = next(cache.glob('*.json'))
    value = load(path)
    value['record']['raw_text'] = '{"entities":[],"relationships":[]}'
    path.write_text(json.dumps(value))
    second = pipe(tmp_path, ontology, chunk, [], cache=cache, name='second')
    with pytest.raises(ValueError, match='cache identity'):
        second.run(1, 100, True)
    assert second.primary.calls == 0


def test_local_names_never_auto_merge(chunk, ontology, payload_data):
    first = validated_record(json.dumps(payload_data), chunk, ontology)
    second_chunk = dict(chunk, chunk_id='c2')
    second_payload = copy.deepcopy(payload_data)
    for item in second_payload['entities'] + second_payload['relationships']:
        item['source_chunk_id'] = 'c2'
    second = validated_record(json.dumps(second_payload), second_chunk, ontology)
    resolved = resolve([first, second])
    assert len(set(resolved['mention_to_entity'].values())) == 4
    assert resolved['resolution_candidates']
    assert all(not c['auto_merge'] for c in resolved['resolution_candidates'])
    ids = resolved['resolution_candidates'][0]['mention_ids']
    reviewed = resolve([first, second], [{'reviewed': True, 'mention_ids': ids}])
    assert len(set(reviewed['mention_to_entity'].values())) == 3


def test_stale_review_and_cross_type_merge_rejected(chunk, ontology, payload_data):
    record = validated_record(json.dumps(payload_data), chunk, ontology)
    evidence = reviews(record)
    for review in evidence['c1'].values():
        review['source_hash'] = 'stale'
    with pytest.raises(ValueError):
        graph_plan([record], [], evidence)
    ids = list(resolve([record])['mention_to_entity'])
    with pytest.raises(ValueError, match='cross-type'):
        resolve([record], [{'reviewed': True, 'mention_ids': ids}])


def test_plan_snapshot_and_provider_recipe_cannot_drift(tmp_path, ontology, chunk):
    p = pipe(tmp_path, ontology, chunk, [])
    value = load(p.job / 'plan.json')
    value['chunks'][0]['text'] = 'changed'
    (p.job / 'plan.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='plan content changed'):
        Pipeline(p.job, ontology, p.primary)


@pytest.mark.parametrize('provider', ['gemini', 'openai'])
def test_response_requires_one_completed_nonrefusal_text(provider):
    p = JSONProvider(provider, 'explicit-model', 100)
    with pytest.raises(ValueError):
        p.output_text({'status': 'incomplete'})
    if provider == 'gemini':
        envelope = {'status': 'completed', 'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': '{}'}]}]}
    else:
        envelope = {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{}'}]}]}
    assert p.output_text(envelope) == '{}'


def test_real_apple_collective_definition_fixture_not_model_benchmark(tmp_path, ontology):
    from app.corpus.filling_parser import to_blocks
    fixture = Path(__file__).parent / 'fixtures' / 'aapl_10k_2023_head_excerpt.html'
    _, text = to_blocks(fixture.read_text(encoding='utf-8'), '.htm')
    start = text.index('Each of the terms')
    source = text[start:text.index('\n', start)]
    assert 'wholly owned subsidiaries' in source
    chunk = {'chunk_id': 'apple-collective-definition', 'text': source,
             'accession': '0000320193-23-000106', 'document_hash': identity(fixture.read_text()),
             'source_url': 'https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm'}
    # Manually authored regression output, NOT generated extraction or gold-label claim.
    output = {'entities': [{'canonical_name': 'Apple Inc.', 'entity_type': 'Company',
                           'aliases': [], 'description': 'Apple Inc. is named in a collective definition.',
                           'source_chunk_id': chunk['chunk_id'], 'confidence': 0.8, 'evidence': source}],
              'relationships': []}
    p = pipe(tmp_path, ontology, chunk, [json.dumps(output)])
    report = p.run(1, 100, True)
    record = records(p)[0]
    assert record['accepted']
    assert record['relationships'] == []
    assert report['run_kind'] == 'offline_fixture'
    assert report['evidence_accuracy']['status'] == 'Not measured'
    assert 'not model benchmarks' in report['warning']
    # Mere quote containment still cannot establish entity/alias truth.
    with pytest.raises(ValueError):
        graph_plan([record], [], {})


def test_corrupted_checkpoint_cannot_be_used_for_report_or_resume(tmp_path, ontology, chunk, payload_data):
    p = pipe(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    p.run(1, 100, True)
    path = next(p.job.glob('chunk-*.json'))
    value = load(path)
    value['entities'][0]['canonical_name'] = 'tampered'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='checkpoint content'):
        p.report()
    with pytest.raises(ValueError, match='checkpoint content'):
        p.run(2, 100, True)
    assert p.primary.calls == 1


def test_real_transport_needs_token_quota_even_with_execution_flag(tmp_path, ontology, chunk):
    primary = JSONProvider('gemini', 'explicit-model', 100)
    prepare(tmp_path / 'job', [chunk], ontology, primary.recipe)
    p = Pipeline(tmp_path / 'job', ontology, primary)
    with pytest.raises(ValueError, match='token-per-minute'):
        p.run(1, 100, True)
    assert not list(p.job.glob('*claim*'))


def test_tpm_preflight_blocks_oversized_request_without_claim(tmp_path, ontology, chunk, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'offline-not-a-key')
    primary = JSONProvider('gemini', 'explicit-model', 100)
    prepare(tmp_path / 'job', [chunk], ontology, primary.recipe)
    p = Pipeline(tmp_path / 'job', ontology, primary, sleep=lambda seconds: None)
    with pytest.raises(ValueError, match='token window'):
        p.run(1, 100, True, tokens_per_minute=1)
    assert not list(p.job.glob('*claim*'))


def test_cli_offline_prepare_report_resolve(tmp_path):
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    chunks = tmp_path / 'chunks.jsonl'
    chunks.write_text(json.dumps({'chunk_id': 'c1', 'text': 'Original source.'}) + '\n')
    job = tmp_path / 'job'
    base = [sys.executable, 'scripts/run_extraction.py', 'phase1']
    for stage, flags in [
        ('prepare', ['--chunks', str(chunks), '--gemini-model', 'explicit-not-selected-model', '--max-output-tokens', '100']),
        ('report', []), ('resolve', [])]:
        completed = subprocess.run(base + [stage, '--job', str(job), '--output', str(tmp_path / (stage + '.json'))] + flags,
                                   cwd=root, text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr
    assert load(tmp_path / 'report.json')['extraction_latency']['status'] == 'Not measured'
    assert load(tmp_path / 'resolve.json')['mentions'] == []


class OfflineEmbedder:
    fingerprint = 'deterministic-offline-vectors-not-a-real-model'

    def __call__(self, texts):
        values = []
        for text in texts:
            if 'Widget' in text or 'widget' in text:
                values.append([0.0, 1.0])
            else:
                values.append([1.0, 0.0])
        return values


def test_global_resolution_reuses_persistent_registry_and_edges(chunk, ontology, payload_data):
    from app.graph.resolution import global_resolve
    record = validated_record(json.dumps(payload_data), chunk, ontology)
    model = OfflineEmbedder()
    calibration = {'model_fingerprint': model.fingerprint, 'threshold': 0.9,
                   'calibration_hash': 'offline-fixture-only'}
    registry = {'version': 0, 'target': 'offline-db', 'entities': []}
    first = global_resolve([record], registry, model, calibration)
    assert all(d['decision'] == 'new' for d in first['decisions'])
    assert len(first['canonical_entities']) == 2
    # Fake vectors deliberately map Acme's description to Widget too; type fence
    # prevents a Company/Technology merge even when vector similarity is 1.
    updated = {'version': 1, 'target': 'offline-db', 'entities': first['canonical_entities']}
    second = global_resolve([record], updated, model, calibration)
    assert all(d['decision'] == 'reuse' for d in second['decisions'])
    assert first['mention_to_entity'] == second['mention_to_entity']
    plan = graph_plan([record], [], reviews(record), second)
    assert plan['registry_version'] == 1
    assert plan['registry_target'] == 'offline-db'
    assert any('[r:DEVELOPS' in s['query'] for s in plan['statements'])
    assert any('SET e += $props' in s['query'] for s in plan['statements'])


def test_ambiguous_global_match_does_not_allocate_duplicate(chunk, ontology, payload_data):
    from app.graph.resolution import global_resolve
    record = validated_record(json.dumps(payload_data), chunk, ontology)
    model = OfflineEmbedder()
    calibration = {'model_fingerprint': model.fingerprint, 'threshold': 0.9}
    base = {'canonical_name': 'Acme', 'entity_type': 'Company', 'aliases': ['ACME'],
            'description': 'Existing company.', 'embedding': [0.0, 1.0], 'model_fingerprint': model.fingerprint}
    registry = {'version': 2, 'target': 'offline-db', 'entities': [dict(base, id='a'), dict(base, id='b')]}
    result = global_resolve([record], registry, model, calibration)
    assert result['status'] == 'Needs identity review'
    assert result['writes_permitted'] is False
    assert 'canonical_entities' not in result
    mention_id = result['unresolved'][0]['mention_id']
    reviewed = global_resolve([record], registry, model, calibration,
                             [{'reviewed': True, 'mention_id': mention_id, 'entity_id': 'a'}])
    assert reviewed['mention_to_entity'][mention_id] == 'a'


def test_threshold_calibration_requires_both_classes_and_separation(payload_data):
    from app.graph.resolution import calibrate
    model = OfflineEmbedder()
    left, right = copy.deepcopy(payload_data['entities'])
    right['entity_type'] = left['entity_type']
    def pair(a, b, same):
        return {'left': a, 'right': b, 'same_entity': same, 'reviewed': True, 'pair_hash': identity([a, b])}
    # A separate model fixture with explicit vectors avoids claiming semantic accuracy.
    class CalibrationModel:
        fingerprint = 'offline'
        def __call__(self, texts):
            return [[1.0, 0.0] if 'acme' in t else [0.0, 1.0] for t in texts]
    model = CalibrationModel()
    pairs = [pair(left, left, True), pair(left, right, False)]
    result = calibrate(pairs, model)
    assert result['threshold'] == 1.0
    assert result['calibration_hash'] == identity(pairs)
    with pytest.raises(ValueError, match='AND nonmatches'):
        calibrate(pairs[:1], model)
    with pytest.raises(ValueError, match='no separating threshold'):
        calibrate([pair(left, left, True), pair(left, left, False)], model)


def test_neo4j_target_authority_and_stale_registry_fail_before_data(chunk, ontology, payload_data):
    from app.graph.resolution import global_resolve
    from app.graph.neo4j_store import write_graph
    record = validated_record(json.dumps(payload_data), chunk, ontology)
    model = OfflineEmbedder()
    resolved = global_resolve([record], {'version': 2, 'target': 'expected-db', 'entities': []}, model,
                              {'model_fingerprint': model.fingerprint, 'threshold': 0.9})
    with pytest.raises(ValueError, match='authority'):
        write_graph(None, 'neo4j', 'expected-db', [record], resolved, reviews(record))
    with pytest.raises(ValueError, match='another target'):
        write_graph(None, 'neo4j', 'wrong-db', [record], resolved, reviews(record), authorized=True)
    class Result:
        def __init__(self, rows=()): self.rows = list(rows)
        def __iter__(self): return iter(self.rows)
        def single(self): return self.rows[0] if self.rows else None
        def consume(self): return None
    class Tx:
        def __init__(self): self.queries = []; self.committed = False
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def run(self, query, params=None, **kwargs):
            self.queries.append(query)
            return Result([{'version': 3}])  # concurrent update since snapshot
        def commit(self): self.committed = True
    class Session:
        def __init__(self): self.tx = Tx()
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def begin_transaction(self): return self.tx
        def run(self, query, *args, **kwargs):
            return Result([{'name': name, 'type': 'UNIQUENESS', 'labelsOrTypes': [label], 'properties': ['id']}
                for name, label in [('entity_id', 'Entity'), ('mention_id', 'Mention'),
                                    ('citation_id', 'Citation'), ('registry_id', 'GraphIntelRegistry')]])
    class Driver:
        def __init__(self): self.s = Session()
        def session(self, **kwargs): return self.s
    driver = Driver()
    with pytest.raises(ValueError, match='registry changed'):
        write_graph(driver, 'neo4j', 'expected-db', [record], resolved, reviews(record), authorized=True)
    assert not driver.s.tx.committed
    assert len(driver.s.tx.queries) == 1  # only version guard, no data MERGEs


def test_reconciled_cost_ledger_binds_exact_attempt_without_mutating_outputs(tmp_path, ontology, chunk, payload_data):
    p = pipe(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    p.run(1, 100, True)
    attempts, unknown = p.attempts()
    before = json.dumps(attempts, sort_keys=True)
    ledger = {identity(attempts[0]): {'reviewed': True, 'cost_usd': '0.02', 'billing_reference': 'offline ledger test'}}
    report = p.report(cost_ledger=ledger)
    assert report['cost_per_1000_chunks']['value_usd'] == '20.00'
    assert json.dumps(p.attempts()[0], sort_keys=True) == before
    stale = {'wrong-hash': next(iter(ledger.values()))}
    assert p.report(cost_ledger=stale)['cost_per_1000_chunks']['status'] == 'Not measured'
    ledger[identity(attempts[0])]['reviewed'] = False
    with pytest.raises(ValueError, match='actual charge'):
        p.report(cost_ledger=ledger)
