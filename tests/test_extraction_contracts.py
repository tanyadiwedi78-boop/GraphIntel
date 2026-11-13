"""Contract fixtures test boundaries, not factual extraction performance."""
import copy

import pytest
from pydantic import ValidationError

from app.extraction.contracts import ExtractionPayload, check_source_contract
from tests.test_anthropic_batch import ontology


@pytest.fixture
def payload_data():
    return {'entities': [
        {'canonical_name': 'Acme', 'entity_type': 'Company', 'aliases': [],
         'description': 'A company developing Widget.', 'source_chunk_id': 'c1',
         'confidence': 0.8, 'evidence': 'Acme develops Widget.'},
        {'canonical_name': 'Widget', 'entity_type': 'Technology', 'aliases': [],
         'description': 'A technology Acme develops.', 'source_chunk_id': 'c1',
         'confidence': 0.8, 'evidence': 'Acme develops Widget.'}],
        'relationships': [{'source_entity': 'Acme', 'target_entity': 'Widget',
         'relationship_type': 'DEVELOPS', 'source_chunk_id': 'c1',
         'confidence': 0.8, 'evidence': 'Acme develops Widget.'}]}


def check(data, ontology):
    return check_source_contract(ExtractionPayload.model_validate(data),
                                 {'chunk_id': 'c1', 'text': 'Acme develops Widget.'}, ontology)


def test_mechanical_pass_not_semantic_truth_or_resolved_graph(payload_data, ontology):
    result = check(payload_data, ontology)
    assert result['mechanical_checks_pass'] is True
    assert result['semantic_evidence_status'] == 'Not reviewed'
    assert result['global_entity_resolution_status'] == 'Not performed'
    assert result['graph_write_permitted'] is False


@pytest.mark.parametrize('field', ['canonical_name', 'entity_type', 'aliases', 'description',
                                 'source_chunk_id', 'confidence', 'evidence'])
def test_each_entity_field_required_no_invented_defaults(payload_data, field):
    del payload_data['entities'][0][field]
    with pytest.raises(ValidationError):
        ExtractionPayload.model_validate(payload_data)


@pytest.mark.parametrize('field', ['source_entity', 'target_entity', 'relationship_type',
                                 'source_chunk_id', 'confidence', 'evidence'])
def test_each_relationship_field_required(payload_data, field):
    del payload_data['relationships'][0][field]
    with pytest.raises(ValidationError):
        ExtractionPayload.model_validate(payload_data)


@pytest.mark.parametrize('confidence', [-1, 1.01, float('nan'), float('inf'), True, '0.8'])
def test_strict_finite_confidence(payload_data, confidence):
    payload_data['entities'][0]['confidence'] = confidence
    with pytest.raises(ValidationError):
        ExtractionPayload.model_validate(payload_data)


def test_unknown_fields_not_dropped(payload_data):
    payload_data['entities'][0]['guess'] = 'invented'
    with pytest.raises(ValidationError):
        ExtractionPayload.model_validate(payload_data)


@pytest.mark.parametrize('part', ['entities', 'relationships'])
def test_wrong_provenance_and_nonexistent_quote_rejected(payload_data, ontology, part):
    payload_data[part][0].update(source_chunk_id='c2', evidence='An invented quote.')
    result = check(payload_data, ontology)
    assert not result['mechanical_checks_pass']
    assert len(result['problems']) == 2


def test_wrong_direction_rejected(payload_data, ontology):
    payload_data['relationships'][0].update(source_entity='Widget', target_entity='Acme')
    result = check(payload_data, ontology)
    assert not result['mechanical_checks_pass']


def test_real_quote_with_false_description_does_not_become_reviewed(payload_data, ontology):
    payload_data['entities'][0]['description'] = 'An unsupported description.'
    result = check(payload_data, ontology)
    assert result['mechanical_checks_pass']
    assert result['semantic_evidence_status'] == 'Not reviewed'
    assert not result['graph_write_permitted']
