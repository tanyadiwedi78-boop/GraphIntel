"""Synthetic mechanics only. These are NOT real extraction benchmarks."""
import json

import pytest

from app.evaluation.extraction_benchmark import benchmark, identity


SOURCE = 'Acme develops Widget. Acme mentions Other.'
EDGE = {'source': 'Acme', 'target': 'Widget', 'relation': 'DEVELOPS',
        'evidence_quote': 'Acme develops Widget.', 'confidence': 0.9}
ENTITIES = [{'name': 'Acme', 'type': 'Company'}, {'name': 'Widget', 'type': 'Technology'}]


def observation(edge=None):
    edge = edge or EDGE
    data = {'entities': ENTITIES, 'relationships': [edge]}
    return {'raw_text': json.dumps(data), 'accepted': True, **data}


def labels(source=SOURCE):
    return {'c': {'reviewed': True, 'source_hash': identity(source),
                  'entities': ENTITIES, 'relationships': [EDGE]}}


def review(edge=EDGE, source=SOURCE, supported=True):
    return {'c': {identity(edge): {'reviewed': True, 'source_hash': identity(source),
                                  'supported': supported}}}


def test_no_runs_remain_unmeasured_not_zero_or_pass():
    report = benchmark({'c': SOURCE}, {})
    for metric in ('json_schema_pass_rate', 'entity_precision', 'entity_recall',
                   'relationship_precision', 'relationship_recall', 'evidence_accuracy',
                   'extraction_latency', 'cost_per_1000_chunks', 'rate_limit_failure_rate'):
        assert report[metric]['status'] == 'Not measured'
    assert report['returned_output_coverage']['value'] == 0
    assert report['coverage']['planned'] == 1


def test_valid_shape_and_contained_quote_do_not_prove_semantic_truth():
    # The quote is real but it does not support DEVELOPS(Other).
    edge = dict(EDGE, target='Other', evidence_quote='Acme mentions Other.')
    report = benchmark({'c': SOURCE}, {'c': observation(edge)}, labels(), review(edge, supported=False))
    assert report['json_schema_pass_rate']['value'] == 1
    assert report['quote_containment_rate']['value'] == 1
    assert report['evidence_accuracy']['value'] == 0
    assert report['relationship_precision']['value'] == 0
    assert report['entity_precision']['value'] == 1


def test_containment_without_semantic_review_is_not_evidence_accuracy():
    report = benchmark({'c': SOURCE}, {'c': observation()})
    assert report['quote_containment_rate']['value'] == 1
    assert report['evidence_accuracy']['value'] is None
    assert report['evidence_review_coverage']['value'] == 0


def test_missing_output_counts_as_missed_on_reviewed_labels():
    report = benchmark({'c': SOURCE}, {}, labels())
    assert report['entity_recall']['value'] == 0
    assert report['relationship_recall']['value'] == 0
    assert report['entity_precision']['value'] is None
    assert report['coverage']['reviewed_labels'] == 1


def test_reviewed_empty_chunk_counts_false_positive():
    truth = labels()
    truth['c'].update(entities=[], relationships=[])
    report = benchmark({'c': SOURCE}, {'c': observation()}, truth)
    assert report['entity_precision']['value'] == 0
    assert report['relationship_precision']['value'] == 0
    assert report['entity_recall']['value'] is None


def test_raw_schema_fail_before_normalized_accepted_output():
    item = observation()
    item['raw_text'] = 'invalid original output'
    report = benchmark({'c': SOURCE}, {'c': item}, labels())
    assert report['json_schema_pass_rate']['value'] == 0
    assert report['entity_precision']['value'] == 1


@pytest.mark.parametrize('target', ['labels', 'evidence'])
def test_stale_source_labels_or_evidence_rejected(target):
    truth = labels('a different original document') if target == 'labels' else labels()
    evidence = review(source='different') if target == 'evidence' else review()
    with pytest.raises(ValueError, match='original chunk'):
        benchmark({'c': SOURCE}, {'c': observation()}, truth, evidence)


def test_changed_edge_cannot_reuse_semantic_review():
    changed = dict(EDGE, confidence=0.5)
    report = benchmark({'c': SOURCE}, {'c': observation(changed)}, labels(), review())
    assert report['evidence_accuracy']['value'] is None


def test_absent_quote_cannot_become_supported_even_with_bad_positive_label():
    edge = dict(EDGE, evidence_quote='not actually in the original text')
    report = benchmark({'c': SOURCE}, {'c': observation(edge)}, labels(), review(edge))
    assert report['evidence_accuracy']['value'] == 0
    assert report['quote_containment_rate']['value'] == 0


def test_unknown_post_status_does_not_become_zero_rate_failures():
    report = benchmark({'c': SOURCE}, {}, submission_attempts=[200, 429, None])
    metric = report['rate_limit_failure_rate']
    assert metric['value'] is None
    assert metric['unknown_status_attempts'] == 1
    assert metric['known_429_count'] == 1
    assert metric['denominator'] == 3


def test_known_rate_failures_and_sample_cost_extrapolation():
    report = benchmark({'c': SOURCE}, {'c': observation()},
                       submission_attempts=[200, 429], sample_cost_usd='0.10')
    assert report['rate_limit_failure_rate']['value'] == 0.5
    assert report['cost_per_1000_chunks']['value_usd'] == '100.00'
    assert 'extrapolation' in report['cost_per_1000_chunks']['status']


def test_batch_wall_time_not_fabricated_request_latency():
    report = benchmark({'c': SOURCE}, {'c': observation()}, batch_timing={'elapsed_seconds': 180})
    assert report['batch_wall_time']['elapsed_seconds'] == 180
    assert report['extraction_latency']['mean_seconds'] is None


def test_actual_request_latency_can_be_reported():
    item = observation()
    item['latency_seconds'] = 0
    report = benchmark({'c': SOURCE}, {'c': item})
    assert report['extraction_latency']['mean_seconds'] == 0


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True])
def test_invalid_measurements_rejected(value):
    with pytest.raises(ValueError):
        benchmark({'c': SOURCE}, {}, sample_cost_usd=value)
    item = observation()
    item['latency_seconds'] = value
    with pytest.raises(ValueError):
        benchmark({'c': SOURCE}, {'c': item})


def test_duplicate_predictions_not_double_correct():
    item = observation()
    item['entities'] = ENTITIES + ENTITIES
    report = benchmark({'c': SOURCE}, {'c': item}, labels())
    assert report['entity_precision']['value'] == 0.5
    assert report['entity_recall']['value'] == 1


def test_unknown_chunk_rejected():
    with pytest.raises(ValueError, match='unknown chunk'):
        benchmark({'c': SOURCE}, {'wrong': observation()})
