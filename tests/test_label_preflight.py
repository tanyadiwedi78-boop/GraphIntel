"""Offline label preflight: a labels/source mismatch fails BEFORE any live call.

Uses the real sample5 fixtures so the check is exercised on actual SEC filing
chunks and their reviewed key. These tests prove preflight behavior only;
they say nothing about live model quality.
"""
import copy
import json
from pathlib import Path

import pytest

from app.extraction.pipeline import check_label_binding, load, prepare, retry_policy
from app.graph.ontology import load_ontology

ROOT = Path(__file__).resolve().parent.parent
RECIPE = {'provider': 'gemini', 'model': 'offline-fixture', 'execution_mode': 'offline_fixture'}


@pytest.fixture
def ontology():
    return load_ontology(ROOT / 'configs' / 'ontology.json')


@pytest.fixture
def chunks():
    texts = json.loads((ROOT / 'tests' / 'fixtures' / 'sample5-chunks.json').read_text(encoding='utf-8'))
    result = []
    for chunk_id in sorted(texts):
        result.append({'chunk_id': chunk_id, 'text': texts[chunk_id], 'document_hash': 'a' * 64})
    return result


@pytest.fixture
def labels():
    return json.loads((ROOT / 'tests' / 'fixtures' / 'labels-sample5.json').read_text(encoding='utf-8'))


def test_real_labels_bind_real_chunks(chunks, labels):
    summary = check_label_binding(chunks, labels)
    assert summary['reviewed'] == 5
    assert summary['bound'] == 5
    assert summary['unreviewed'] == 0


def test_changed_text_fails_loudly(chunks, labels):
    chunks[0]['text'] = chunks[0]['text'] + ' one added word'
    with pytest.raises(ValueError) as caught:
        check_label_binding(chunks, labels)
    assert chunks[0]['chunk_id'] in str(caught.value)
    assert 'does not match' in str(caught.value)


def test_missing_chunk_fails_loudly(chunks, labels):
    labels['no-such-chunk'] = {'reviewed': True, 'source_hash': 'b' * 64,
                               'entities': [], 'relationships': []}
    with pytest.raises(ValueError) as caught:
        check_label_binding(chunks, labels)
    assert 'no-such-chunk' in str(caught.value)
    assert 'no chunk with this ID' in str(caught.value)


def test_all_problems_reported_at_once(chunks, labels):
    chunks[0]['text'] = chunks[0]['text'] + ' x'
    chunks[1]['text'] = chunks[1]['text'] + ' y'
    with pytest.raises(ValueError) as caught:
        check_label_binding(chunks, labels)
    assert chunks[0]['chunk_id'] in str(caught.value)
    assert chunks[1]['chunk_id'] in str(caught.value)


def test_unreviewed_label_needs_no_hash(chunks, labels):
    labels = copy.deepcopy(labels)
    first = sorted(labels)[0]
    labels[first] = {'reviewed': False}
    summary = check_label_binding(chunks, labels)
    assert summary['reviewed'] == 4
    assert summary['unreviewed'] == 1


def test_prepare_records_binding_in_saved_plan(tmp_path, chunks, labels, ontology):
    plan = prepare(tmp_path / 'job', chunks, ontology, dict(RECIPE),
                   policy=retry_policy(0), labels=labels)
    assert plan['label_binding']['reviewed'] == 5
    assert plan['label_binding']['bound'] == 5
    saved = load(tmp_path / 'job' / 'plan.json')
    assert saved['label_binding'] == plan['label_binding']


def test_prepare_rejects_unbound_labels_and_saves_nothing(tmp_path, chunks, labels, ontology):
    chunks[2]['text'] = chunks[2]['text'] + ' drift'
    with pytest.raises(ValueError) as caught:
        prepare(tmp_path / 'job', chunks, ontology, dict(RECIPE),
                policy=retry_policy(0), labels=labels)
    assert chunks[2]['chunk_id'] in str(caught.value)
    assert not (tmp_path / 'job' / 'plan.json').exists()


def test_prepare_without_labels_has_no_binding_section(tmp_path, chunks, ontology):
    plan = prepare(tmp_path / 'job', chunks, ontology, dict(RECIPE), policy=retry_policy(0))
    assert 'label_binding' not in plan
