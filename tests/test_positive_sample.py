import json
from pathlib import Path
import pytest
from app.evaluation.extraction_benchmark import identity
from app.graph.ontology import load_ontology
from app.extraction.pipeline import check_label_binding
ROOT = Path(__file__).resolve().parent.parent
@pytest.fixture
def ontology():
    return load_ontology(ROOT / "configs/ontology.json")

FIXTURE_HTML = ROOT / 'tests' / 'fixtures' / 'aapl_10k_2023_head_excerpt.html'


def positive_chunks():
    texts = json.loads((ROOT / 'tests' / 'fixtures' / 'positive-sample-chunks.json').read_text(encoding='utf-8'))
    result = []
    for chunk_id in sorted(texts):
        result.append({'chunk_id': chunk_id, 'text': texts[chunk_id], 'document_hash': 'b' * 64})
    return result


def positive_labels():
    return json.loads((ROOT / 'tests' / 'fixtures' / 'labels-positive-sample.json').read_text(encoding='utf-8'))


def test_positive_sample_text_is_verbatim_fixture_excerpt():
    # Provenance guard: every chunk must be verbatim visible text of the real
    # Apple 10-K excerpt fixture after the same plain normalization used to
    # build the sample. A drifted or invented passage fails here.
    import re
    raw = FIXTURE_HTML.read_text(encoding='utf-8')
    text = re.sub(r'<[^>]+>', ' ', raw)
    for old, new in [('&#160;', ' '), ('&#8217;', "'"), ('&#8220;', '"'), ('&#8221;', '"'),
                     ('&#8482;', ''), ('&#174;', ''), ('&#8209;', '-'), ('&amp;', '&')]:
        text = text.replace(old, new)
    text = re.sub(r'\s+', ' ', text).strip()
    for chunk in positive_chunks():
        assert chunk['text'] in text, chunk['chunk_id']


def test_positive_sample_labels_bind_chunks_and_stay_ontological(ontology):
    chunks = positive_chunks()
    labels = positive_labels()
    binding = check_label_binding(chunks, labels)
    assert binding['reviewed'] == 3 and binding['unreviewed'] == 0 and binding['bound'] == 3
    with_relationships = 0
    without_relationships = 0
    for chunk in chunks:
        label = labels[chunk['chunk_id']]
        assert label['user_confirmed'] is False
        for entity in label['entities']:
            assert entity['entity_type'] in ontology.entity_types
        types = {entity['canonical_name']: entity['entity_type'] for entity in label['entities']}
        for relationship in label['relationships']:
            assert ontology.allows(relationship['relationship_type'],
                                   types[relationship['source_entity']],
                                   types[relationship['target_entity']])
        if label['relationships']:
            with_relationships += 1
        else:
            without_relationships += 1
    # The sample must measure recall on positives AND restraint on a negative.
    assert with_relationships == 2 and without_relationships == 1


def test_positive_sample_identity_pairs_are_well_formed():
    from app.evaluation.extraction_benchmark import identity
    pairs = json.loads((ROOT / 'tests' / 'fixtures' / 'identity-pairs-positive-sample.json').read_text(encoding='utf-8'))
    matches = 0
    nonmatches = 0
    for pair in pairs:
        # The same field rules calibrate() enforces, checked without a model.
        assert pair['reviewed'] is True
        assert pair['user_confirmed'] is False
        assert type(pair['same_entity']) is bool
        assert pair['pair_hash'] == identity([pair['left'], pair['right']])
        assert pair['left']['entity_type'] == pair['right']['entity_type']
        assert pair['left']['description'] and pair['right']['description']
        if pair['same_entity']:
            matches += 1
        else:
            nonmatches += 1
    # calibrate() refuses a one-sided set; the fixture must carry both.
    assert matches >= 1 and nonmatches >= 1
