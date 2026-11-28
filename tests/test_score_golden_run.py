"""Offline tests for the golden-run scorer. No network, no model calls.

The scorer's honesty rules are what matter here: incomplete labels never
penalize extras (unknown stays unverified, never auto-hallucination),
quote containment and name presence are review signals rather than
correctness verdicts, a relationship needs a reviewed triple rather than
two present names, and type mismatches are excluded from typed metrics
while remaining visible in name-only ones.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from scripts.score_golden_run import (alias_map_of, entities_match, load_labels,
                                      prf, relationship_key, score_chunk)

CHUNK = {'chunk_id': 'c1',
         'text': 'Acme develops the Widget platform. Beta Group acquired Acme in 2021.'}

PRED_ENT_OK = {'canonical_name': 'acme', 'entity_type': 'Company', 'aliases': ['Acme Inc'],
               'description': 'd', 'source_chunk_id': 'c1', 'confidence': 0.9,
               'evidence': 'Acme develops the Widget platform.'}
REF_ENT = {'canonical_name': 'Acme', 'entity_type': 'Company', 'aliases': ['Acme Inc']}

PRED_REL_OK = {'source_entity': 'Beta Group', 'target_entity': 'Acme',
               'relationship_type': 'ACQUIRED', 'source_chunk_id': 'c1',
               'confidence': 0.9, 'evidence': 'Beta Group acquired Acme in 2021.'}
REF_REL = {'source_entity': 'Beta Group', 'target_entity': 'Acme',
           'relationship_type': 'ACQUIRED'}


def label(entities=None, relationships=None, complete=False):
    return {'complete': complete, 'entities': entities or [],
            'relationships': relationships or [], 'reviewer': 'test'}


def payload(entities=None, relationships=None):
    return {'entities': entities or [], 'relationships': relationships or []}


# --- matching and normalization --------------------------------------------

def test_entity_match_ignores_case_whitespace_and_uses_aliases():
    assert entities_match(PRED_ENT_OK, REF_ENT)
    messy = dict(PRED_ENT_OK, canonical_name='  ACME   Inc ')
    assert entities_match(messy, REF_ENT)
    assert not entities_match(dict(PRED_ENT_OK, canonical_name='Other', aliases=[]), REF_ENT)


def test_relationship_key_resolves_reference_aliases():
    alias_map = alias_map_of([REF_ENT])
    predicted = dict(PRED_REL_OK, target_entity='Acme Inc')
    assert relationship_key(predicted, alias_map) == relationship_key(REF_REL, alias_map)


def test_prf_denominators_and_undefined_cases():
    assert prf(1, 1, 2) == {'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
                            'matched': 1, 'wrong': 1, 'reference': 2}
    empty = prf(0, 0, 0)
    assert empty['precision'] is None and empty['recall'] is None and empty['f1'] is None


# --- typed vs name-only entity metrics --------------------------------------

def test_type_mismatch_stays_in_name_only_but_leaves_typed():
    scored = score_chunk(CHUNK, payload([PRED_ENT_OK], []),
                         label([dict(REF_ENT, entity_type='Person')], []))
    assert scored['entities']['counts']['matched'] == 1
    assert scored['entities']['type_mismatches'] == ['acme']
    typed = scored['entities']['typed_counts']
    assert typed['matched'] == 0 and typed['wrong'] == 1 and typed['recall'] == 0.0


def test_perfect_chunk_scores_one_on_both_views():
    scored = score_chunk(CHUNK, payload([PRED_ENT_OK], [PRED_REL_OK]),
                         label([REF_ENT], [REF_REL]))
    assert scored['entities']['counts']['f1'] == 1.0
    assert scored['entities']['typed_counts']['f1'] == 1.0
    assert scored['relationships']['counts']['f1'] == 1.0


# --- extras under incomplete vs complete labels ------------------------------

def test_incomplete_labels_keep_unknown_extras_unverified_even_ungrounded():
    ghost = dict(PRED_ENT_OK, canonical_name='Phantom Corp', aliases=[],
                 evidence='No such sentence in the chunk.')
    scored = score_chunk(CHUNK, payload([PRED_ENT_OK, ghost], []),
                         label([REF_ENT], [], complete=False))
    counts = scored['entities']['counts']
    assert counts['precision'] == 1.0 and counts['wrong'] == 0
    extra = scored['entities']['extras'][0]
    assert extra['class'] == 'unverified'
    assert extra['name_present'] is False and extra['evidence_verbatim'] is False


def test_incomplete_labels_keep_grounded_extras_unverified():
    extra = dict(PRED_ENT_OK, canonical_name='Beta Group', aliases=[])
    scored = score_chunk(CHUNK, payload([PRED_ENT_OK, extra], []),
                         label([REF_ENT], [], complete=False))
    assert scored['entities']['counts']['precision'] == 1.0
    assert scored['entities']['extras'][0]['class'] == 'unverified'
    assert scored['entities']['extras'][0]['name_present'] is True


def test_complete_labels_count_extras_with_review_signals():
    extra = dict(PRED_ENT_OK, canonical_name='Beta Group', aliases=[])
    scored = score_chunk(CHUNK, payload([PRED_ENT_OK, extra], []),
                         label([REF_ENT], [], complete=True))
    counts = scored['entities']['counts']
    assert counts['precision'] == 0.5 and counts['wrong'] == 1
    signals = scored['entities']['extras'][0]
    assert signals['class'] == 'false_positive'
    assert signals['name_present'] is True and signals['evidence_verbatim'] is True


def test_absent_name_with_verbatim_evidence_is_not_auto_ungrounded():
    normalized = dict(PRED_ENT_OK, canonical_name='JPMorgan Chase', aliases=[],
                      evidence='Acme develops the Widget platform.')
    scored = score_chunk(CHUNK, payload([normalized], []),
                         label([], [], complete=True))
    signals = scored['entities']['extras'][0]
    assert signals['class'] == 'false_positive'  # complete labels: still counts
    assert signals['name_present'] is False      # but the reason is reviewable
    assert signals['evidence_verbatim'] is True  # quote exists; may be a valid normalization


def test_noise_chunk_counts_every_prediction():
    scored = score_chunk(CHUNK, payload([PRED_ENT_OK], [PRED_REL_OK]),
                         label([], [], complete=True))
    assert scored['entities']['counts']['wrong'] == 1
    assert scored['relationships']['counts']['wrong'] == 1


def test_relationship_needs_reviewed_triple_not_two_present_names():
    invented = dict(PRED_REL_OK, relationship_type='OWNS')
    scored = score_chunk(CHUNK, payload([], [invented]),
                         label([], [], complete=True))
    counts = scored['relationships']['counts']
    assert counts['wrong'] == 1 and counts['matched'] == 0
    signals = scored['relationships']['extras'][0]
    assert signals['name_present'] is True  # both names appear; still not grounded


def test_missed_reference_lowers_recall_only():
    scored = score_chunk(CHUNK, payload([], []), label([REF_ENT], [], complete=True))
    counts = scored['entities']['counts']
    assert counts['recall'] == 0.0 and counts['precision'] is None


def test_evidence_quote_missing_from_source_is_flagged_on_match():
    bad = dict(PRED_ENT_OK, evidence='This sentence is not in the chunk.')
    scored = score_chunk(CHUNK, payload([bad], []), label([REF_ENT], []))
    assert scored['entities']['evidence_problems'] == ['acme']


# --- labels file shape -------------------------------------------------------

def test_load_labels_rejects_wrong_version(tmp_path):
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps({'version': 'nope', 'chunks': {}}))
    with pytest.raises(ValueError):
        load_labels(path)


def test_load_labels_defaults_optional_fields(tmp_path):
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps({'version': 'golden20-labels-v1',
                                'chunks': {'c1': {'entities': [REF_ENT]}}}))
    chunks = load_labels(path)
    assert chunks['c1']['complete'] is False
    assert chunks['c1']['relationships'] == []


def test_label_coverage_is_visible(tmp_path):
    import subprocess, sys
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'results.json').write_text(json.dumps({
        'summary': {'model': 'm', 'prompt_version': 'v'},
        'results': [
            {'chunk_id': 'c1', 'ok': True, 'problems': [],
             'payload': payload([PRED_ENT_OK]), 'raw_text': None},
            {'chunk_id': 'c2', 'ok': True, 'problems': [],
             'payload': payload([PRED_ENT_OK]), 'raw_text': None},
        ]}))
    (run / 'metrics.jsonl').write_text(
        json.dumps({'chunk_id': 'c1', 'schema_valid': True}) + '\n'
        + json.dumps({'chunk_id': 'c2', 'schema_valid': True}) + '\n')
    chunks = tmp_path / 'chunks.json'
    chunks.write_text(json.dumps([
        {'chunk_id': 'c1', 'text': CHUNK['text']}, {'chunk_id': 'c2', 'text': CHUNK['text']}]))
    labels = tmp_path / 'labels.json'
    labels.write_text(json.dumps({'version': 'golden20-labels-v1', 'chunks': {
        'c1': {'complete': True, 'entities': [REF_ENT], 'relationships': []},
        'c2': {'complete': False, 'entities': [REF_ENT], 'relationships': []}}}))
    result = subprocess.run(
        [sys.executable, 'scripts/score_golden_run.py', str(run), str(labels), str(chunks)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stderr
    report = json.loads(next(run.glob('score-*.json')).read_text())
    coverage = report['label_coverage']
    assert coverage['complete_labels'] == 1 and coverage['incomplete_labels'] == 1
    assert 'Label coverage: 1 complete, 1 incomplete' in result.stdout


# --- quote containment and certified noise (28 Nov) ---------------------------

def test_quote_containment_counts_all_predictions():
    paraphrased = dict(PRED_ENT_OK, evidence='Acme makes widgets (paraphrase)')
    result = score_chunk(CHUNK, payload(entities=[PRED_ENT_OK, paraphrased],
                                        relationships=[PRED_REL_OK]),
                         label(entities=[REF_ENT], relationships=[REF_REL], complete=True))
    containment = result['quote_containment']
    assert containment['checked'] == 3
    assert containment['verbatim'] == 2
    assert containment['rate'] == round(2 / 3, 4)


def test_reviewed_empty_chunk_counts_false_extractions():
    result = score_chunk(CHUNK, payload(entities=[PRED_ENT_OK]),
                         label(complete=True))
    assert result['reviewed_empty'] is True
    assert result['false_extractions_on_reviewed_empty'] == 1


def test_unreviewed_empty_labels_certify_nothing():
    result = score_chunk(CHUNK, payload(entities=[PRED_ENT_OK]),
                         label(complete=False))
    assert result['reviewed_empty'] is False
    assert result['false_extractions_on_reviewed_empty'] == 0
