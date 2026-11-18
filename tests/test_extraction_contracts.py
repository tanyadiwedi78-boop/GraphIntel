"""Contract fixtures test boundaries, not factual extraction performance."""
import copy

import pytest
from pydantic import ValidationError

from pathlib import Path

from app.extraction.contracts import ExtractionPayload, check_source_contract
from app.graph.ontology import load_ontology

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def ontology():
    return load_ontology(ROOT / "configs" / "ontology.json")


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


CAUTIONARY_TEXT = ('Each of the terms the "Company" and "Apple" as used herein refers '
    'collectively to Apple Inc. For example, statements in this Form 10-Q regarding the '
    'potential future impact of the COVID-19 pandemic on the Company business and results '
    'of operations are forward-looking statements. The COVID-19 pandemic has materially '
    'disrupted the Company supply chain and retail operations. The forward-looking '
    'statements in this report also name the iPhone 13.')

CAUTIONARY_QUOTE = ('statements in this Form 10-Q regarding the potential future impact of '
    'the COVID-19 pandemic on the Company business and results of operations are '
    'forward-looking statements')
SUBSTANTIVE_QUOTE = ('The COVID-19 pandemic has materially disrupted the Company supply '
    'chain and retail operations')


def cautionary_chunk():
    return {'chunk_id': 'c1', 'text': CAUTIONARY_TEXT}


def apple_covid_payload(edge_evidence, relationship_type='FACES'):
    return {'entities': [
        {'canonical_name': 'Apple', 'entity_type': 'Company', 'aliases': [],
         'description': 'The registrant.', 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': 'Each of the terms the "Company" and "Apple" as used herein refers collectively to Apple Inc.'},
        {'canonical_name': 'COVID-19 pandemic', 'entity_type': 'Risk', 'aliases': [],
         'description': 'A named risk.', 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': CAUTIONARY_QUOTE}],
        'relationships': [{'source_entity': 'Apple', 'target_entity': 'COVID-19 pandemic',
         'relationship_type': relationship_type, 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': edge_evidence}]}


def test_cautionary_only_evidence_holds_faces_edge(ontology):
    result = check_source_contract(ExtractionPayload.model_validate(
        apple_covid_payload(CAUTIONARY_QUOTE)), cautionary_chunk(), ontology)
    assert result['mechanical_checks_pass'] is True
    assert result['problems'] == []
    assert len(result['cautionary_holds']) == 1
    assert result['cautionary_holds'][0]['relationship_type'] == 'FACES'


def test_substantive_risk_statement_edge_not_held(ontology):
    result = check_source_contract(ExtractionPayload.model_validate(
        apple_covid_payload(SUBSTANTIVE_QUOTE)), cautionary_chunk(), ontology)
    assert result['cautionary_holds'] == []
    assert result['mechanical_checks_pass'] is True


def test_mentions_edge_exempt_from_cautionary_hold(ontology):
    data = {'entities': [
        {'canonical_name': 'Apple', 'entity_type': 'Company', 'aliases': [],
         'description': 'The registrant.', 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': 'Each of the terms the "Company" and "Apple" as used herein refers collectively to Apple Inc.'},
        {'canonical_name': 'iPhone 13', 'entity_type': 'Product', 'aliases': [],
         'description': 'A named product.', 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': 'The forward-looking statements in this report also name the iPhone 13.'}],
        'relationships': [{'source_entity': 'Apple', 'target_entity': 'iPhone 13',
         'relationship_type': 'MENTIONS', 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': 'The forward-looking statements in this report also name the iPhone 13.'}]}
    result = check_source_contract(ExtractionPayload.model_validate(data),
                                   cautionary_chunk(), ontology)
    assert result['cautionary_holds'] == []
    assert result['mechanical_checks_pass'] is True


def test_entities_never_held_for_cautionary_evidence(ontology):
    data = apple_covid_payload(SUBSTANTIVE_QUOTE)
    data['relationships'] = []
    result = check_source_contract(ExtractionPayload.model_validate(data),
                                   cautionary_chunk(), ontology)
    # The COVID entity cites the cautionary sentence; entities are unaffected.
    assert result['cautionary_holds'] == []
    assert result['mechanical_checks_pass'] is True


def test_validated_record_quarantines_held_edges_but_keeps_everything(ontology):
    from app.extraction.pipeline import validated_record
    data = apple_covid_payload(CAUTIONARY_QUOTE)
    data['entities'].append(
        {'canonical_name': 'iPhone 13', 'entity_type': 'Product', 'aliases': [],
         'description': 'A named product.', 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': 'The forward-looking statements in this report also name the iPhone 13.'})
    data['relationships'].append(
        {'source_entity': 'Apple', 'target_entity': 'iPhone 13',
         'relationship_type': 'MENTIONS', 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': 'The forward-looking statements in this report also name the iPhone 13.'})
    import json
    record = validated_record(json.dumps(data), cautionary_chunk(), ontology)
    assert record['accepted'] is True
    assert len(record['entities']) == 3
    assert [edge['relationship_type'] for edge in record['relationships']] == ['MENTIONS']
    assert [edge['relationship_type'] for edge in record['held_relationships']] == ['FACES']
    assert 'FACES' in record['raw_text']
    assert record['semantic_evidence_status'] == 'Not reviewed'


def ungrounded_payload(relationship_type='DEVELOPS'):
    return {'entities': [
        {'canonical_name': 'Apple', 'entity_type': 'Company', 'aliases': [],
         'description': 'The registrant.', 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': 'Each of the terms the "Company" and "Apple" as used herein refers collectively to Apple Inc.'},
        {'canonical_name': 'Widget Pro', 'entity_type': 'Technology', 'aliases': [],
         'description': 'An invented name.', 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': SUBSTANTIVE_QUOTE}],
        'relationships': [{'source_entity': 'Apple', 'target_entity': 'Widget Pro',
         'relationship_type': relationship_type, 'source_chunk_id': 'c1', 'confidence': 0.9,
         'evidence': SUBSTANTIVE_QUOTE}]}


def test_absent_endpoint_name_holds_edge_and_lists_ungrounded(ontology):
    result = check_source_contract(ExtractionPayload.model_validate(ungrounded_payload()),
                                   cautionary_chunk(), ontology)
    assert result['mechanical_checks_pass'] is True
    assert result['cautionary_holds'] == []
    assert len(result['grounding_holds']) == 1
    assert result['grounding_holds'][0]['target_entity'] == 'Widget Pro'
    assert result['ungrounded_entity_names'] == ['Widget Pro']


def test_alias_grounding_avoids_hold(ontology):
    data = ungrounded_payload()
    data['entities'][1]['aliases'] = ['iPhone 13']
    result = check_source_contract(ExtractionPayload.model_validate(data),
                                   cautionary_chunk(), ontology)
    assert result['grounding_holds'] == []
    assert result['ungrounded_entity_names'] == []


def test_name_grounding_is_case_insensitive(ontology):
    data = ungrounded_payload()
    data['entities'][1]['canonical_name'] = 'iphone 13'
    data['relationships'][0]['target_entity'] = 'iphone 13'
    result = check_source_contract(ExtractionPayload.model_validate(data),
                                   cautionary_chunk(), ontology)
    assert result['grounding_holds'] == []
    assert result['mechanical_checks_pass'] is True


def test_grounding_hold_keeps_entities_and_raw_text(ontology):
    import json
    from app.extraction.pipeline import validated_record
    record = validated_record(json.dumps(ungrounded_payload()), cautionary_chunk(), ontology)
    assert record['accepted'] is True
    assert len(record['entities']) == 2
    assert record['relationships'] == []
    assert [edge['target_entity'] for edge in record['held_relationships']] == ['Widget Pro']
    assert 'Widget Pro' in record['raw_text']
