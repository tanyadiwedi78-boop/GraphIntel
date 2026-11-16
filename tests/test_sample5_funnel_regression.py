"""Real-data funnel regression: actual sample5 chunks, labels and v2 output.

Fixtures are SEC filing excerpts recovered from the user's own job records.
These tests pin funnel behavior on REAL data, never synthetic stand-ins, and
they are offline checks: they prove guard/scorer behavior, not live model
quality.
"""
import json
from pathlib import Path

import pytest

from app.evaluation.extraction_benchmark import benchmark, identity
from app.extraction.pipeline import validated_record
from app.graph.ontology import load_ontology

ROOT = Path(__file__).resolve().parent.parent
CHUNK_059 = '000032019322000059-b4de7f98ebaba76d-04e605fe-document-015'
KNOWN_059_HASH = 'c92d032193e07d284017ff52b7be85409cec9926c24371e5358b81cae8519db8'


@pytest.fixture
def chunks():
    return json.loads((ROOT / 'tests' / 'fixtures' / 'sample5-chunks.json').read_text(encoding='utf-8'))


@pytest.fixture
def labels():
    return json.loads((ROOT / 'tests' / 'fixtures' / 'labels-sample5.json').read_text(encoding='utf-8'))


@pytest.fixture
def ontology():
    return load_ontology(ROOT / 'configs' / 'ontology.json')


def test_real_chunk_hashes_bind_labels(chunks, labels):
    assert set(chunks) == set(labels) and len(chunks) == 5
    for chunk_id, text in chunks.items():
        assert labels[chunk_id]['source_hash'] == identity(text)
    # Independent anchor from the user's own saved extraction record.
    assert identity(chunks[CHUNK_059]) == KNOWN_059_HASH


def test_real_v2_output_guard_holds_faces_edge_keeps_entities(chunks, ontology):
    saved = json.loads((ROOT / 'tests' / 'fixtures' / 'sample5-v2-output-059.json').read_text(encoding='utf-8'))
    chunk = {'chunk_id': saved['chunk_id'], 'text': chunks[saved['chunk_id']]}
    record = validated_record(saved['raw_text'], chunk, ontology)
    assert record['accepted'] is True
    assert [entity['canonical_name'] for entity in record['entities']] == ['Apple', 'COVID-19 pandemic']
    assert record['relationships'] == []
    assert [edge['relationship_type'] for edge in record['held_relationships']] == ['FACES']
    assert record['raw_text'] == saved['raw_text']
    assert record['semantic_evidence_status'] == 'Not reviewed'


def test_key_structure_matches_source_pages(chunks, labels):
    for chunk_id, label in labels.items():
        assert label['relationships'] == []  # all five pages assert no relations
        types = [entity['entity_type'] for entity in label['entities']]
        if chunk_id.endswith('document-000'):  # 8-K cover pages
            assert types == ['Company']
        else:  # 10-Q boilerplate pages
            assert types == ['Company', 'Risk']


def test_scorer_binds_all_five_and_perfect_output_scores_full(chunks, labels, ontology):
    unbound = benchmark(chunks, {}, labels, None, expanded_contract=True, ontology=ontology)
    assert unbound['coverage']['reviewed_labels'] == 5
    perfect = {}
    for chunk_id, label in labels.items():
        perfect[chunk_id] = {'accepted': True,
                             'entities': [dict(entity) for entity in label['entities']],
                             'relationships': []}
    scored = benchmark(chunks, perfect, labels, None, expanded_contract=True, ontology=ontology)
    assert scored['entity_precision']['value'] == 1
    assert scored['entity_recall']['value'] == 1
    assert scored['relationship_precision']['status'] == 'Not measured'  # no true or predicted edges
