"""Offline vertical-path/fault tests, NEVER provider-quality benchmark results."""
import copy
import json
from pathlib import Path

import pytest

from app.evaluation.extraction_benchmark import identity
from app.extraction.pipeline import Pipeline, prepare, load, save_new, validated_record, retry_policy
from app.extraction.providers import CallFailure, JSONProvider
from app.graph.construction import graph_plan, resolve
from app.graph.ontology import load_ontology
from tests.test_extraction_contracts import payload_data

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def ontology():
    return load_ontology(ROOT / "configs" / "ontology.json")


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
    return {'chunk_id': 'c1', 'text': 'Acme develops Widget.', 'document_hash': 'a' * 64,
            'accession': 'fixture-only', 'form': '10-K', 'period': '2023'}


def pipe(tmp_path, ontology, chunk, values, fallback_values=None, cache=None, name='job'):
    primary = FakeProvider(values)
    fallback = FakeProvider(fallback_values, 'openai') if fallback_values is not None else None
    job = tmp_path / name
    prepare(job, [chunk], ontology, primary.recipe, fallback.recipe if fallback else None, policy=retry_policy(0))
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
def test_invalid_output_retries_known_output_then_checkpoints(tmp_path, ontology, chunk, raw, payload_data):
    primary = FakeProvider([raw, raw, json.dumps(payload_data)])
    job = tmp_path / 'retry-job'
    prepare(job, [chunk], ontology, primary.recipe, policy=retry_policy(2))
    p = Pipeline(job, ontology, primary, sleep=lambda seconds: None)
    p.run(3, 100, True)
    p.run(3, 100, True)
    assert primary.calls == 3
    assert records(p)[0]['accepted']
    assert p.report()['schema_per_attempt']['value'] == pytest.approx(1 / 3)
    assert p.report()['first_attempt']['json_schema_pass_rate']['value'] == 0


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
    changed = dict(chunk, document_hash='b' * 64)
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
    from datetime import datetime, timezone
    pricing = {'gemini': {'model': 'explicit-model', 'verified': True, 'source': 'offline-test',
        'account_reference': 'offline-test', 'currency': 'USD', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'max_input_tokens': 100000, 'max_output_tokens': 1000, 'input_per_million_usd': '0', 'output_per_million_usd': '0', 'free_tier_no_billing_verified': True}}
    limits = {'authority_reference': 'offline-test-not-user-authority', 'total_usd': '0',
              'documents': {chunk['document_hash']: '0'}}
    prepare(tmp_path / 'job', [chunk], ontology, primary.recipe, pricing=pricing, cost_limits=limits)
    p = Pipeline(tmp_path / 'job', ontology, primary, sleep=lambda seconds: None)
    with pytest.raises(ValueError, match='token window'):
        p.run(1, 100, True, tokens_per_minute=1)
    assert not list(p.job.glob('*claim*'))


def test_cli_offline_prepare_report_resolve(tmp_path):
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    chunks = tmp_path / 'chunks.jsonl'
    chunks.write_text(json.dumps({'chunk_id': 'c1', 'text': 'Original source.', 'source_sha256': 'a' * 64}) + '\n')
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
    with pytest.raises(ValueError, match='duplicate calibration pair'):
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


def test_retry_then_fallback_exhausts_primary_not_first_error(tmp_path, ontology, chunk, payload_data):
    primary = FakeProvider([CallFailure('http_failure', 429), '{}', '{}'])
    fallback = FakeProvider([json.dumps(payload_data)], 'openai')
    job = tmp_path / 'policy'
    prepare(job, [chunk], ontology, primary.recipe, fallback.recipe, retry_policy(2))
    waits = []
    p = Pipeline(job, ontology, primary, fallback, sleep=waits.append)
    report = p.run(4, 100, True, True)
    assert primary.calls == 3 and fallback.calls == 1
    assert records(p)[0]['provider'] == 'openai'
    assert waits[1] >= 2 and waits[2] >= 4
    assert report['calls']['attempts'] == 4
    assert report['first_attempt']['returned_output_coverage']['value'] == 0


def test_call_limit_pauses_known_retries_not_terminal(tmp_path, ontology, chunk, payload_data):
    primary = FakeProvider(['{}', json.dumps(payload_data)])
    job = tmp_path / 'partial'
    prepare(job, [chunk], ontology, primary.recipe, policy=retry_policy(2))
    p = Pipeline(job, ontology, primary, sleep=lambda seconds: None)
    p.run(1, 100, True)
    assert records(p) == []
    p.run(2, 100, True)
    assert records(p)[0]['accepted'] and primary.calls == 2


def test_transport_unknown_never_retried_or_fallen_back(tmp_path, ontology, chunk):
    primary = FakeProvider([CallFailure('transport_unknown')])
    fallback = FakeProvider(['unused'], 'openai')
    job = tmp_path / 'unknown'
    prepare(job, [chunk], ontology, primary.recipe, fallback.recipe)
    p = Pipeline(job, ontology, primary, fallback, sleep=lambda seconds: None)
    p.run(10, 100, True, True)
    p.run(10, 100, True, True)
    assert primary.calls == 1 and fallback.calls == 0


def pricing_for(model='explicit-model', input_rate='1', output_rate='2'):
    from datetime import datetime, timezone
    return {'model': model, 'verified': True, 'source': 'offline source', 'currency': 'USD',
            'account_reference': 'offline-only', 'checked_at': datetime.now(timezone.utc).isoformat(),
            'max_input_tokens': 100000, 'max_output_tokens': 1000, 'input_per_million_usd': input_rate, 'output_per_million_usd': output_rate}


def test_document_budgets_block_before_claim_and_prices_cannot_be_omitted(tmp_path, ontology, chunk, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'not-a-real-key')
    primary = JSONProvider('gemini', 'explicit-model', 100)
    limits = {'authority_reference': 'offline fixture', 'total_usd': '0', 'documents': {chunk['document_hash']: '0'}}
    job = tmp_path / 'dollars'
    prepare(job, [chunk], ontology, primary.recipe, pricing={'gemini': pricing_for()}, cost_limits=limits)
    p = Pipeline(job, ontology, primary)
    with pytest.raises(ValueError, match='document cost exceeds'):
        p.run(1, 100, True, tokens_per_minute=100000)
    assert not list(job.glob('*claim*'))
    other = tmp_path / 'unpriced'
    prepare(other, [chunk], ontology, primary.recipe)
    with pytest.raises(ValueError, match='cost limits'):
        Pipeline(other, ontology, primary).run(1, 100, True, tokens_per_minute=100000)


def test_zero_cost_needs_verified_free_account_and_stale_price_blocked(tmp_path, ontology, chunk):
    from app.extraction.pipeline import request_reservation, common_schema
    primary = JSONProvider('gemini', 'explicit-model', 100)
    price = pricing_for(input_rate='0', output_rate='0')
    with pytest.raises(ValueError, match='no-billing'):
        request_reservation(chunk, ontology, common_schema(ontology), primary.recipe, {'gemini': price})
    price['free_tier_no_billing_verified'] = True
    assert request_reservation(chunk, ontology, common_schema(ontology), primary.recipe, {'gemini': price})['usd'] == '0'
    price['checked_at'] = '2020-01-01T00:00:00Z'
    with pytest.raises(ValueError, match='stale'):
        request_reservation(chunk, ontology, common_schema(ontology), primary.recipe, {'gemini': price})


def test_source_sha256_is_explicit_document_group_and_missing_hash_rejected(tmp_path, ontology, chunk):
    primary = JSONProvider('gemini', 'explicit-model', 100)
    original = dict(chunk)
    original['source_sha256'] = original.pop('document_hash')
    plan = prepare(tmp_path / 'sha', [original], ontology, primary.recipe)
    assert list(plan['document_budgets']) == [original['source_sha256']]
    original.pop('source_sha256')
    with pytest.raises(ValueError, match='SHA256'):
        prepare(tmp_path / 'missing', [original], ontology, primary.recipe)


def test_fair_independent_comparison_no_fallback_no_cache(tmp_path, ontology, chunk, payload_data):
    from app.extraction.pipeline import prepare_comparison, compare_reports
    left = FakeProvider([json.dumps(payload_data)])
    right = FakeProvider(['{}'], 'openai')
    labels = {'c1': {'reviewed': True, 'source_hash': identity(chunk['text']),
                     'entities': payload_data['entities'], 'relationships': payload_data['relationships']}}
    root = tmp_path / 'paired'
    prepare_comparison(root, [chunk], ontology, left.recipe, right.recipe, labels, retry_policy(0))
    for provider in [left, right]:
        Pipeline(root / provider.recipe['provider'], ontology, provider, sleep=lambda n: None).run(1, 100, True)
    report = compare_reports(root, ontology, [left, right])
    assert report['reports']['gemini']['entity_recall']['value'] == 1
    assert report['reports']['openai']['entity_recall']['value'] == 0
    assert report['reports']['openai']['first_attempt']['json_schema_pass_rate']['value'] == 0
    with pytest.raises(ValueError, match='forbids'):
        Pipeline(root / 'gemini', ontology, left, cache_dir=tmp_path / 'cache').run(1, 100, True)
    altered = load(root / 'openai' / 'plan.json')
    altered['chunks'][0]['text'] = 'different source'
    altered.pop('plan_hash')
    altered['plan_hash'] = identity(altered)
    (root / 'openai' / 'plan.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='unfair'):
        compare_reports(root, ontology, [left, right])


class FakeBatch(FakeProvider):
    def __init__(self, values, result_items=None, fail_submit=False):
        super().__init__(values)
        self.result_items = result_items
        self.fail_submit = fail_submit
    def body(self, prompt, schema):
        return {'prompt': prompt, 'schema': schema}
    def batch_submit(self, items, display_name):
        self.items = items
        self.calls += 1
        if self.fail_submit:
            raise CallFailure('batch_transport_unknown')
        return {'name': 'batches/offline'}
    def batch_collect(self, receipt, ids):
        if self.result_items is not None:
            return {'done': True, 'snapshot': {'offline': True}, 'results': self.result_items}
        return {'done': True, 'snapshot': {'offline': True}, 'results': [
            {'custom_id': key, 'body': {'text': self.values.pop(0)}, 'http_status': None, 'error': None} for key in ids]}
    def output_text(self, body):
        return body['text']


def test_native_batch_roundtrip_no_invented_chunk_latency_or_429(tmp_path, ontology, chunk, payload_data):
    primary = FakeBatch([json.dumps(payload_data)])
    job = tmp_path / 'batch'
    prepare(job, [chunk], ontology, primary.recipe, execution_mode='native_batch')
    p = Pipeline(job, ontology, primary)
    submitted = p.batch_submit(3, True, max_requests=1000, max_input_units=100000)
    assert submitted['items'] == 1
    with pytest.raises(ValueError, match='pending'):
        p.batch_submit(3, True, max_requests=1000, max_input_units=100000)
    result = p.batch_collect(1)
    assert records(p)[0]['accepted']
    assert result['report']['extraction_latency']['status'] == 'Not measured'
    assert result['report']['rate_limit_failure_rate']['value'] is None
    assert p.batch_collect(1)['already_collected']
    assert not p.batch_submit(3, True, max_requests=1000, max_input_units=100000)['submitted']
    assert primary.calls == 1


def test_native_batch_ambiguous_create_retains_reservations_blocks_resubmit(tmp_path, ontology, chunk):
    primary = FakeBatch([], fail_submit=True)
    job = tmp_path / 'batch'
    prepare(job, [chunk], ontology, primary.recipe, execution_mode='native_batch')
    p = Pipeline(job, ontology, primary)
    with pytest.raises(CallFailure):
        p.batch_submit(3, True, max_requests=1000, max_input_units=100000)
    with pytest.raises(ValueError, match='ambiguous'):
        p.batch_submit(3, True, max_requests=1000, max_input_units=100000)
    assert len(list(job.glob('attempt-*-claim.json'))) == 1 and primary.calls == 1


def test_native_batch_missing_result_not_reprocessed(tmp_path, ontology, chunk):
    primary = FakeBatch([], result_items=[])
    job = tmp_path / 'batch'
    prepare(job, [chunk], ontology, primary.recipe, execution_mode='native_batch')
    p = Pipeline(job, ontology, primary)
    p.batch_submit(3, True, max_requests=1000, max_input_units=100000)
    assert p.batch_collect(1)['unreconciled_items'] == 1
    with pytest.raises(ValueError, match='unreconciled'):
        p.batch_submit(3, True, max_requests=1000, max_input_units=100000)
    assert primary.calls == 1


def test_native_batch_known_invalid_can_retry_next_wave(tmp_path, ontology, chunk, payload_data):
    primary = FakeBatch(['{}', json.dumps(payload_data)])
    job = tmp_path / 'batch'
    prepare(job, [chunk], ontology, primary.recipe, execution_mode='native_batch')
    p = Pipeline(job, ontology, primary, sleep=lambda n: None)
    for wave in [1, 2]:
        p.batch_submit(3, True, max_requests=1000, max_input_units=100000)
        p.batch_collect(wave)
    assert records(p)[0]['accepted'] and primary.calls == 2


def test_gemini_generatecontent_body_common_schema_and_output(payload_data):
    provider = JSONProvider('gemini', 'explicit-model', 100)
    body = provider.body('same prompt', {'type': 'object'})
    assert body['contents'][0]['parts'][0]['text'] == 'same prompt'
    assert body['generationConfig']['responseFormat']['text']['schema'] == {'type': 'object'}
    envelope = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(payload_data)}]}}]}
    assert json.loads(provider.output_text(envelope)) == payload_data
    envelope['candidates'][0]['finishReason'] = 'MAX_TOKENS'
    with pytest.raises(ValueError):
        provider.output_text(envelope)


def test_malformed_known_provider_envelope_is_saved_not_hidden_crash(monkeypatch):
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b'[]'
    class Opener:
        def open(self, request, timeout): return Response()
    monkeypatch.setenv('GEMINI_API_KEY', 'offline-not-a-key')
    p = JSONProvider('gemini', 'explicit-model', 100, opener=Opener())
    result = p.call('prompt', {'type': 'object'})
    assert result['outcome'] == 'unusable_response'
    assert result['http_status'] == 200 and result['raw_text'] is None


def test_native_provider_batch_payloads_and_reads_are_documented_shapes(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'offline-not-a-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-not-a-key')
    class Provider(JSONProvider):
        def __init__(self, name):
            super().__init__(name, 'explicit-model', 100)
            self.requests = []
        def api(self, path, body=None, multipart=None):
            self.requests.append((path, body, multipart))
            if path == 'files': return '{"id":"file_offline"}'
            if path == 'batches': return '{"id":"batch_offline"}'
            return '{"name":"batches/offline"}'
    g = Provider('gemini')
    items = [{'custom_id': 'i1', 'body': g.body('identical prompt', {'type': 'object'})}]
    g.batch_submit(items, 'offline')
    path, body, _ = g.requests[0]
    assert path == 'models/explicit-model:batchGenerateContent'
    assert body['batch']['input_config']['requests']['requests'][0]['metadata']['key'] == 'i1'
    o = Provider('openai')
    o.batch_submit([{'custom_id': 'i1', 'body': o.body('identical prompt', {'type': 'object'})}], 'offline')
    assert o.requests[0][0] == 'files'
    assert b'"url": "/v1/responses"' in o.requests[0][2][0]
    assert o.requests[1][1]['completion_window'] == '24h'
    assert o.requests[1][1]['input_file_id'] == 'file_offline'


def test_durable_dollar_reservations_cannot_be_reset_by_restart(tmp_path, ontology, chunk, monkeypatch):
    from app.extraction.pipeline import request_reservation, common_schema
    monkeypatch.setenv('GEMINI_API_KEY', 'offline-not-a-key')
    p = JSONProvider('gemini', 'explicit-model', 100)
    pricing = {'gemini': pricing_for()}
    reservation = request_reservation(chunk, ontology, common_schema(ontology), p.recipe, pricing)
    amount = reservation['usd']
    job = tmp_path / 'reserved'
    limits = {'authority_reference': 'offline-only', 'total_usd': amount, 'documents': {chunk['document_hash']: amount}}
    prepare(job, [chunk], ontology, p.recipe, policy=retry_policy(0), pricing=pricing, cost_limits=limits)
    pipe = Pipeline(job, ontology, p)
    save_new(job / 'attempt-1-claim.json', {'reservation': reservation})
    with pytest.raises(ValueError, match='document dollar'):
        pipe.check_reserved([reservation])
    pipe2 = Pipeline(job, ontology, p)
    with pytest.raises(ValueError, match='document dollar'):
        pipe2.check_reserved([reservation])


def test_native_batch_exhausted_invalid_is_checkpointed_without_an_extra_submit(tmp_path, ontology, chunk):
    primary = FakeBatch(['{}'])
    job = tmp_path / 'batch'
    prepare(job, [chunk], ontology, primary.recipe, policy=retry_policy(0), execution_mode='native_batch')
    p = Pipeline(job, ontology, primary)
    p.batch_submit(1, True, max_requests=1000, max_input_units=100000)
    report = p.batch_collect(1)['report']
    assert len(records(p)) == 1 and not records(p)[0]['accepted']
    assert report['returned_output_coverage']['value'] == 1
    assert report['json_schema_pass_rate']['value'] == 0


def test_frozen_literal_validation_shares_provider_enum(ontology, payload_data):
    from app.extraction.contracts import ontology_contract
    from app.extraction.pipeline import common_schema
    from pydantic import ValidationError
    model = ontology_contract(ontology)
    assert common_schema(ontology) == model.model_json_schema()
    payload_data['entities'][0]['entity_type'] = 'Person'
    with pytest.raises(ValidationError):
        model.model_validate(payload_data)


def test_f1_raw_quote_and_reviewed_name_groups(chunk, payload_data):
    from app.evaluation.extraction_benchmark import benchmark
    prediction = copy.deepcopy(payload_data)
    prediction['entities'][0]['canonical_name'] = 'Acme Corp'
    prediction['relationships'][0]['source_entity'] = 'Acme Corp'
    record = dict(prediction, accepted=True, raw_text=json.dumps(prediction))
    label = dict(payload_data, reviewed=True, source_hash=identity(chunk['text']))
    exact = benchmark({'c1': chunk['text']}, {'c1': record}, {'c1': label}, expanded_contract=True)
    assert exact['entity_f1']['value'] == 0.5
    assert exact['relationship_f1']['value'] == 0
    assert exact['exact_quote_containment_rate']['value'] == 1
    label['name_groups'] = [['Acme', 'Acme Corp']]
    reviewed = benchmark({'c1': chunk['text']}, {'c1': record}, {'c1': label}, expanded_contract=True)
    assert reviewed['entity_f1']['value'] == reviewed['relationship_f1']['value'] == 1
    label['name_groups'].append(['Other', 'Acme'])
    with pytest.raises(ValueError, match='overlapping'):
        benchmark({'c1': chunk['text']}, {'c1': record}, {'c1': label}, expanded_contract=True)


def test_f1_missing_output_zero_not_unmeasured(chunk, payload_data):
    from app.evaluation.extraction_benchmark import benchmark
    label = dict(payload_data, reviewed=True, source_hash=identity(chunk['text']))
    report = benchmark({'c1': chunk['text']}, {}, {'c1': label}, expanded_contract=True)
    assert report['entity_f1']['value'] == report['relationship_f1']['value'] == 0


def test_normalized_and_raw_quote_rates_stay_separate(chunk, payload_data):
    from app.evaluation.extraction_benchmark import benchmark
    payload_data['relationships'][0]['evidence'] = 'Acme  develops Widget.'
    record = dict(payload_data, accepted=True, raw_text=json.dumps(payload_data))
    result = benchmark({'c1': chunk['text']}, {'c1': record}, expanded_contract=True)
    assert result['quote_containment_rate']['value'] == 1
    assert result['exact_quote_containment_rate']['value'] == 0
    assert result['evidence_accuracy']['value'] is None


def test_long_empty_output_review_signal_never_paid_fallback(tmp_path, ontology, chunk):
    chunk['text'] = 'Boilerplate ' * 151
    p = pipe(tmp_path, ontology, chunk, [json.dumps({'entities': [], 'relationships': []})], ['unused'])
    report = p.run(2, 100, True, True)
    assert report['quality_review'][0]['long_empty_output'] is True
    assert report['quality_review'][0]['minimum_self_reported_confidence'] is None
    assert p.primary.calls == 1 and p.fallback.calls == 0


def test_low_confidence_is_visible_not_an_automatic_quality_verdict(tmp_path, ontology, chunk, payload_data):
    payload_data['entities'][0]['confidence'] = 0.1
    p = pipe(tmp_path, ontology, chunk, [json.dumps(payload_data)], ['unused'])
    report = p.run(2, 100, True, True)
    assert report['quality_review'][0]['minimum_self_reported_confidence'] == 0.1
    assert p.fallback.calls == 0


def test_heldout_canonicalization_uses_actual_resolver_and_blocks_leaks(payload_data):
    from app.graph.resolution import calibrate, evaluate_resolution
    class Model:
        fingerprint = 'synthetic-pair-model'
        def __call__(self, texts):
            return [[1., 0.] if 'acme' in text else [0., 1.] for text in texts]
    model = Model()
    left = dict(payload_data['entities'][0])
    other = dict(left, canonical_name='Other', description='An unrelated company.')
    def pair(a, b, expected):
        return dict(left=a, right=b, same_entity=expected, reviewed=True, pair_hash=identity([a,b]))
    calibration = calibrate([pair(left, left, True), pair(left, other, False)], model)
    right = dict(left, canonical_name='Acme Corp')
    different = dict(other, canonical_name='Different')
    pairs = [pair(left, right, True), pair(left, different, False)]
    report = evaluate_resolution(pairs, model, calibration)
    assert report['canonicalization_accuracy']['value'] == report['match_f1']['value'] == 1
    assert not report['writes_performed']
    with pytest.raises(ValueError, match='overlap'):
        evaluate_resolution([pair(left, left, True)], model, calibration)
    with pytest.raises(ValueError, match='overlap'):
        evaluate_resolution([pair(other, left, False)], model, calibration)
    with pytest.raises(ValueError, match='AND nonmatches'):
        evaluate_resolution(pairs[:1], model, calibration)
    with pytest.raises(ValueError, match='pair hashes'):
        evaluate_resolution(pairs, model, dict(calibration, pair_hashes=[]))


def test_old_schema_plan_is_report_only(tmp_path, ontology, chunk):
    p = pipe(tmp_path, ontology, chunk, [])
    plan = load(p.job / 'plan.json')
    plan['version'] = 'extraction-contract-v2-review'
    plan.pop('plan_hash')
    plan['plan_hash'] = identity(plan)
    (p.job / 'plan.json').write_text(json.dumps(plan))
    old = Pipeline(p.job, ontology, p.primary)
    with pytest.raises(ValueError):
        old.run(1, 100, True)
    assert p.primary.calls == 0



def test_fresh_extraction_without_any_legacy_modules(tmp_path, chunk):
    import subprocess
    import sys
    import shutil
    root = Path(__file__).resolve().parents[1]
    needed = ['app/extraction/contracts.py', 'app/extraction/providers.py',
              'app/extraction/pipeline.py', 'app/evaluation/extraction_benchmark.py',
              'app/graph/construction.py', 'app/graph/resolution.py',
              'app/graph/neo4j_store.py', 'app/graph/ontology.py',
              'configs/ontology.json', 'scripts/run_extraction.py']
    for name in needed:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    # There is NO old runner.py/client.py/validate.py/Anthropic module in this tree.
    for stage in ('--help', 'demo'):
        command = [sys.executable, 'scripts/run_extraction.py', 'phase1', stage]
        if stage == 'demo':
            command += ['--output', 'review.json']
        got = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert got.returncode == 0, got.stderr
    output = json.loads((tmp_path / 'review.json').read_text())
    assert output['fixture_validation']['accepted'] is True
    assert output['invented_quote_rejected'] is True
    assert output['provider_calls'] == output['database_calls'] == 0
    (tmp_path / 'chunks.jsonl').write_text(json.dumps(chunk) + '\n')
    got = subprocess.run([sys.executable, 'scripts/run_extraction.py', 'phase1', 'prepare',
        '--job', 'new-job', '--chunks', 'chunks.jsonl', '--gemini-model', 'offline-shape-only',
        '--max-output-tokens', '64', '--output', 'prepared.json'], cwd=tmp_path, capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert (tmp_path / 'new-job' / 'plan.json').is_file()



def test_raw_attempt_schema_metrics_use_frozen_pydantic_literals(tmp_path, ontology, chunk, payload_data):
    payload_data['entities'][0]['entity_type'] = 'Person'
    p = pipe(tmp_path, ontology, chunk, [json.dumps(payload_data)])
    report = p.run(1, 100, True)
    assert report['json_schema_pass_rate']['value'] == 0
    assert report['first_attempt']['json_schema_pass_rate']['value'] == 0
    assert report['schema_per_attempt']['value'] == 0



def test_entrypoint_without_args_is_safe_guidance_even_without_pipeline(tmp_path):
    import subprocess
    import sys
    import shutil
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts/run_extraction.py', scripts / 'run_extraction.py')
    result = subprocess.run([sys.executable, 'scripts/run_extraction.py'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert 'No extraction started.' in result.stdout
    assert 'phase1 --help' in result.stdout
    result = subprocess.run([sys.executable, 'scripts/run_extraction.py', 'prepare'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Historical provider commands' in result.stdout
    assert 'Traceback' not in result.stderr
