"""Score one golden-20 run folder against reviewed labels. Offline, no calls.

Usage:

    python scripts/score_golden_run.py reports/golden20/run-<stamp>-<model> <labels.json> [chunks.json]

Definitions, kept honest:

- Entities and relationships get SEPARATE precision/recall/F1. Entities get
  two views: name-only (a match shares any normalized name or alias with a
  reviewed label) and typed (a match must also carry the same normalized
  entity_type). A type mismatch is never silently counted correct.
- Matching normalizes names, types and quotes (whitespace squashed, case
  folded). Reference aliases count as the same entity.
- A prediction matching no reviewed label is a false positive ONLY when the
  reviewer marked that chunk's labels complete. Under incomplete labels it
  is an unverified extra: reported, outside precision, never called a
  hallucination. Unknown extras stay unknown.
- Quote containment is a mechanical check, not semantic correctness. Every
  prediction carries two review signals: evidence_verbatim (its evidence
  quote appears in the chunk text) and name_present (its name literally
  appears). Neither alone proves grounding: a name can be absent yet valid
  (normalization, abbreviation) and a present name proves nothing about a
  claimed relationship. There is no hallucination rate from lexical proxies.
- A relationship passes only by matching a reviewed triple (alias-resolved
  source, target, type). Both names appearing in the source is never enough.
- A chunk with complete, empty labels is the noise case: every prediction
  there is a false positive.
- Chunks without labels are skipped and listed, never scored as zero.

The score file is a NEW file inside the run folder; an existing score with
the same name is never overwritten.

Labels file shape (fresh file per review round, e.g.
data/golden/golden20-labels-20261127.json):

    {
      "version": "golden20-labels-v1",
      "chunks": {
        "chunk-001": {
          "reviewer": "agent-reviewed",
          "user_confirmed": false,
          "complete": false,
          "entities": [{"canonical_name": "...", "entity_type": "...",
                        "aliases": ["..."]}],
          "relationships": [{"source_entity": "...", "target_entity": "...",
                             "relationship_type": "..."}]
        }
      }
    }
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_FOLDER))

from app.extraction.chunk_metrics import read_rows
from app.extraction.contracts import squash_spaces
from scripts.run_golden_extraction import load_chunks

LABELS_VERSION = 'golden20-labels-v1'


def normalize(text: Any) -> str:
    """One canonical form for names, types and quotes."""
    return squash_spaces(str(text or '')).casefold()


def load_labels(path: Path) -> Dict[str, Any]:
    """Read the labels file and check its shape loudly."""
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('version') != LABELS_VERSION:
        raise ValueError('labels version must be ' + LABELS_VERSION + ' in ' + str(path))
    chunks = document.get('chunks')
    if not isinstance(chunks, dict):
        raise ValueError('labels file needs a "chunks" object in ' + str(path))
    for chunk_id, record in chunks.items():
        record.setdefault('entities', [])
        record.setdefault('relationships', [])
        record.setdefault('complete', False)
    return chunks


def entity_keys(entity: Dict[str, Any]) -> set:
    """All normalized names one entity answers to: canonical plus aliases."""
    keys = {normalize(entity.get('canonical_name'))}
    for alias in entity.get('aliases') or []:
        keys.add(normalize(alias))
    keys.discard('')
    return keys


def entities_match(prediction: Dict[str, Any], reference: Dict[str, Any]) -> bool:
    """True when predicted and reference share any name or alias."""
    return bool(entity_keys(prediction) & entity_keys(reference))


def alias_map_of(entities: List[Dict[str, Any]]) -> Dict[str, str]:
    """Every normalized alias and name pointing at the normalized canonical."""
    mapping = {}
    for entity in entities:
        canonical = normalize(entity.get('canonical_name'))
        for key in entity_keys(entity):
            mapping[key] = canonical
    return mapping


def relationship_key(relationship: Dict[str, Any], alias_map) -> tuple:
    source = normalize(relationship.get('source_entity'))
    target = normalize(relationship.get('target_entity'))
    if alias_map:
        source = alias_map.get(source, source)
        target = alias_map.get(target, target)
    return (source, target, normalize(relationship.get('relationship_type')))


def match_items(predicted: List[Dict[str, Any]], reference: List[Dict[str, Any]],
                matches) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy one-to-one matching. Returns (pairs, unmatched_pred, unmatched_ref)."""
    pairs = []
    used_predictions = set()
    unmatched_ref = []
    for ref_index, ref_item in enumerate(reference):
        found = None
        for pred_index, pred_item in enumerate(predicted):
            if pred_index in used_predictions:
                continue
            if matches(pred_item, ref_item):
                found = pred_index
                break
        if found is None:
            unmatched_ref.append(ref_index)
        else:
            used_predictions.add(found)
            pairs.append((found, ref_index))
    unmatched_pred = [i for i in range(len(predicted)) if i not in used_predictions]
    return pairs, unmatched_pred, unmatched_ref


def prf(matched: int, wrong: int, reference: int) -> Dict[str, Any]:
    """Precision/recall/F1 with visible denominators; None when undefined."""
    precision = None
    recall = None
    if matched + wrong > 0:
        precision = round(matched / (matched + wrong), 4)
    if reference > 0:
        recall = round(matched / reference, 4)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'matched': matched, 'wrong': wrong, 'reference': reference}


def review_signals(item: Dict[str, Any], name_keys: set, source: str) -> Dict[str, Any]:
    """Mechanical review signals for one prediction. Not a correctness verdict."""
    quote = normalize(item.get('evidence'))
    return {
        'item': item,
        'evidence_verbatim': bool(quote) and quote in source,
        'name_present': any(key in source for key in name_keys) if name_keys else False,
    }


def score_entities(predicted: List[Dict[str, Any]], reference: List[Dict[str, Any]],
                   source: str, complete: bool) -> Dict[str, Any]:
    """Score the entity side: name-only and typed metrics, separately."""
    pairs, unmatched_pred, unmatched_ref = match_items(predicted, reference, entities_match)
    typed_pairs = []
    type_mismatches = []
    for pred_index, ref_index in pairs:
        pred_type = normalize(predicted[pred_index].get('entity_type'))
        ref_type = normalize(reference[ref_index].get('entity_type'))
        if ref_type and pred_type != ref_type:
            type_mismatches.append(predicted[pred_index].get('canonical_name'))
        else:
            typed_pairs.append((pred_index, ref_index))
    extras, wrong = classify_extras(predicted, unmatched_pred, entity_keys, source, complete)
    evidence_problems = []
    for pred_index, _ in pairs:
        signals = review_signals(predicted[pred_index], set(), source)
        if not signals['evidence_verbatim']:
            evidence_problems.append(predicted[pred_index].get('canonical_name'))
    return {
        'counts': prf(len(pairs), wrong, len(reference)),
        'typed_counts': prf(len(typed_pairs), wrong + len(type_mismatches), len(reference)),
        'type_mismatches': type_mismatches,
        'missed_reference': [reference[i] for i in unmatched_ref],
        'extras': extras,
        'evidence_problems': evidence_problems,
    }


def score_relationships(predicted: List[Dict[str, Any]], reference: List[Dict[str, Any]],
                        alias_map, source: str, complete: bool) -> Dict[str, Any]:
    """Score the relationship side against reviewed triples only.

    Name presence in the source is kept as a review signal on extras; it
    never passes a relationship, because two present names prove nothing
    about the claimed edge.
    """
    pairs, unmatched_pred, unmatched_ref = match_items(
        predicted, reference,
        lambda pred, ref: relationship_key(pred, alias_map) == relationship_key(ref, alias_map))

    def endpoint_keys(rel):
        return {normalize(rel.get('source_entity')), normalize(rel.get('target_entity'))} - {''}

    extras, wrong = classify_extras(predicted, unmatched_pred, endpoint_keys, source, complete)
    evidence_problems = []
    for pred_index, _ in pairs:
        signals = review_signals(predicted[pred_index], set(), source)
        if not signals['evidence_verbatim']:
            evidence_problems.append(predicted[pred_index].get('relationship_type'))
    return {
        'counts': prf(len(pairs), wrong, len(reference)),
        'missed_reference': [reference[i] for i in unmatched_ref],
        'extras': extras,
        'evidence_problems': evidence_problems,
    }


def classify_extras(predicted: List[Dict[str, Any]], unmatched: List[int],
                    keys_of, source: str, complete: bool):
    """Split unmatched predictions into review classes. Returns (extras, wrong).

    Complete labels: every unmatched prediction is a false positive (wrong).
    Incomplete labels: every unmatched prediction is an unverified extra
    (outside precision). Both carry evidence_verbatim and name_present as
    review signals; neither signal is a correctness verdict by itself.
    """
    extras = []
    for index in unmatched:
        signals = review_signals(predicted[index], keys_of(predicted[index]), source)
        signals['class'] = 'false_positive' if complete else 'unverified'
        extras.append(signals)
    wrong = len(unmatched) if complete else 0
    return extras, wrong


def quote_containment(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Share of ALL predictions whose evidence is a verbatim quote.

    A mechanical check over every predicted entity and relationship, matched
    or not. Quote containment is not semantic grounding: a contained quote can
    still fail to support the claim, and a paraphrase can still be right.
    """
    checked = 0
    verbatim = 0
    for item in (payload.get('entities') or []) + (payload.get('relationships') or []):
        checked += 1
        quote = normalize(item.get('evidence'))
        if quote and quote in source:
            verbatim += 1
    return {'checked': checked, 'verbatim': verbatim,
            'rate': round(verbatim / checked, 4) if checked else None}


def score_chunk(chunk: Dict[str, Any], payload: Dict[str, Any],
                label: Dict[str, Any]) -> Dict[str, Any]:
    """Score one chunk's saved payload against its reviewed labels."""
    source = normalize(chunk.get('text'))
    complete = bool(label.get('complete'))
    reference_entities = label.get('entities') or []
    alias_map = alias_map_of(reference_entities)
    predicted_entities = payload.get('entities') or []
    predicted_relationships = payload.get('relationships') or []
    reviewed_empty = complete and not reference_entities and not (label.get('relationships') or [])
    return {
        'reference_complete': complete,
        'entities': score_entities(predicted_entities,
                                   reference_entities, source, complete),
        'relationships': score_relationships(predicted_relationships,
                                             label.get('relationships') or [],
                                             alias_map, source, complete),
        'quote_containment': quote_containment(payload, source),
        # A chunk is certified noise only when review marked its labels
        # complete AND empty. Boilerplate category alone certifies nothing.
        'reviewed_empty': reviewed_empty,
        'false_extractions_on_reviewed_empty': (
            len(predicted_entities) + len(predicted_relationships)) if reviewed_empty else 0,
    }


def merge_counts(total: Dict[str, int], side: Dict[str, Any], typed: bool) -> None:
    counts = side['typed_counts'] if typed else side['counts']
    total['matched'] += counts['matched']
    total['wrong'] += counts['wrong']
    total['reference'] += counts['reference']
    total['evidence_problems'] += len(side['evidence_problems'])


def format_prf(side: Dict[str, Any]) -> str:
    return ('P=' + str(side['precision']) + ' R=' + str(side['recall'])
            + ' F1=' + str(side['f1']) + ' (matched ' + str(side['matched'])
            + ', wrong ' + str(side['wrong']) + ', reference ' + str(side['reference']) + ')')


def main() -> int:
    if len(sys.argv) < 3:
        print('Usage: python scripts/score_golden_run.py RUN_FOLDER LABELS_FILE [CHUNKS_FILE]')
        return 2
    run_folder = Path(sys.argv[1])
    labels = load_labels(Path(sys.argv[2]))
    config = json.loads((PROJECT_FOLDER / 'configs' / 'golden_extraction.json').read_text(encoding='utf-8'))
    chunks_path = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_FOLDER / config['chunks_path']
    chunks = {str(c['chunk_id']): c for c in load_chunks(chunks_path)}
    document = json.loads((run_folder / 'results.json').read_text(encoding='utf-8'))
    summary = document.get('summary') or {}
    metrics_path = run_folder / 'metrics.jsonl'
    metrics = read_rows(metrics_path) if metrics_path.exists() else []

    per_chunk = []
    unscored = []
    totals = {'entities': {'matched': 0, 'wrong': 0, 'reference': 0, 'evidence_problems': 0},
              'entities_typed': {'matched': 0, 'wrong': 0, 'reference': 0, 'evidence_problems': 0},
              'relationships': {'matched': 0, 'wrong': 0, 'reference': 0, 'evidence_problems': 0}}
    unverified = {'entities': 0, 'relationships': 0}
    containment = {'checked': 0, 'verbatim': 0}
    noise_false_extractions = 0
    reviewed_empty_chunks = []
    for result in document.get('results') or []:
        chunk_id = str(result.get('chunk_id'))
        label = labels.get(chunk_id)
        if label is None:
            unscored.append(chunk_id)
            continue
        if not result.get('payload'):
            per_chunk.append({'chunk_id': chunk_id, 'schema_valid': False,
                              'note': 'no valid payload; nothing to score'})
            continue
        scored = score_chunk(chunks.get(chunk_id, {'chunk_id': chunk_id, 'text': ''}),
                             result['payload'], label)
        scored['chunk_id'] = chunk_id
        per_chunk.append(scored)
        merge_counts(totals['entities'], scored['entities'], typed=False)
        merge_counts(totals['entities_typed'], scored['entities'], typed=True)
        merge_counts(totals['relationships'], scored['relationships'], typed=False)
        for side in ('entities', 'relationships'):
            for extra in scored[side]['extras']:
                if extra['class'] == 'unverified':
                    unverified[side] += 1
        containment['checked'] += scored['quote_containment']['checked']
        containment['verbatim'] += scored['quote_containment']['verbatim']
        if scored['reviewed_empty']:
            reviewed_empty_chunks.append(chunk_id)
            noise_false_extractions += scored['false_extractions_on_reviewed_empty']

    schema_rows = len(metrics)
    schema_pass = sum(1 for row in metrics if row.get('schema_valid') is True)
    scored_chunks = [c for c in per_chunk if 'entities' in c]
    complete_chunks = [c for c in scored_chunks if c['reference_complete']]
    label_coverage = {
        'scored_chunks': len(scored_chunks),
        'complete_labels': len(complete_chunks),
        'incomplete_labels': len(scored_chunks) - len(complete_chunks),
        'unscored_chunks': len(unscored),
        'note': 'Precision on incomplete-label chunks excludes unverified extras; '
                'a high precision there is not a complete evaluation.',
    }
    uncertified_boilerplate = sorted(
        str(chunk.get('chunk_id'))
        for chunk in chunks.values()
        if chunk.get('golden_category') == 'boilerplate'
        and str(chunk.get('chunk_id')) not in reviewed_empty_chunks)
    report = {
        'run_folder': str(run_folder),
        'scored_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': summary.get('model'),
        'prompt_version': summary.get('prompt_version'),
        'schema_pass': schema_pass,
        'schema_attempted': schema_rows,
        'label_coverage': label_coverage,
        'entities_name_only': prf(totals['entities']['matched'], totals['entities']['wrong'],
                                  totals['entities']['reference']),
        'entities_typed': prf(totals['entities_typed']['matched'], totals['entities_typed']['wrong'],
                              totals['entities_typed']['reference']),
        'relationships': prf(totals['relationships']['matched'], totals['relationships']['wrong'],
                             totals['relationships']['reference']),
        'unverified_entity_extras': unverified['entities'],
        'unverified_relationship_extras': unverified['relationships'],
        'quote_containment': {
            'checked': containment['checked'],
            'verbatim': containment['verbatim'],
            'rate': round(containment['verbatim'] / containment['checked'], 4)
            if containment['checked'] else None,
            'note': 'Mechanical verbatim-quote share over all predictions. Not semantic '
                    'grounding: a contained quote can still fail to support the claim.',
        },
        'noise_review': {
            'reviewed_empty_chunks': reviewed_empty_chunks,
            'false_extractions_on_reviewed_empty': noise_false_extractions,
            'uncertified_boilerplate_chunks': uncertified_boilerplate,
            'note': 'A chunk is certified noise only after review marks its labels '
                    'complete and empty. Boilerplate category alone certifies nothing; '
                    'unreviewed boilerplate chunks are listed, not judged.',
        },
        'evidence_problems': sum(totals[key]['evidence_problems'] for key in totals),
        'per_chunk': per_chunk,
        'unscored_chunks': unscored,
        'caveats': [
            'Metrics are against reviewed labels. Labels marked incomplete '
            'keep unmatched predictions as unverified extras outside '
            'precision; they are not counted as hallucinations.',
            'evidence_verbatim and name_present are mechanical review '
            'signals, not proof of grounding; a present name does not '
            'support a relationship, and an absent name can still be valid.',
            'Schema pass is shape/ontology/provenance, not accuracy.',
            'Unverified extras need source review before they count for or '
            'against a model.',
        ],
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    out_path = run_folder / ('score-' + stamp + '.json')
    if out_path.exists():
        print('Score file already exists; refusing to overwrite: ' + str(out_path))
        return 1
    out_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    print('Model: ' + str(report['model']) + ' | prompt ' + str(report['prompt_version']))
    print('Schema pass: ' + str(schema_pass) + ' of ' + str(schema_rows))
    print('Label coverage: ' + str(label_coverage['complete_labels']) + ' complete, '
          + str(label_coverage['incomplete_labels']) + ' incomplete of '
          + str(label_coverage['scored_chunks']) + ' scored chunks'
          + ' (incomplete chunks exclude unverified extras from precision)')
    print('entities (name-only): ' + format_prf(report['entities_name_only']))
    print('entities (typed):     ' + format_prf(report['entities_typed']))
    print('relationships:        ' + format_prf(report['relationships']))
    print('Unverified extras (not penalized): entities '
          + str(report['unverified_entity_extras']) + ', relationships '
          + str(report['unverified_relationship_extras']))
    containment_note = report['quote_containment']
    print('Quote containment (mechanical, not grounding): '
          + str(containment_note['verbatim']) + ' of ' + str(containment_note['checked'])
          + ' predictions carry a verbatim quote')
    noise = report['noise_review']
    print('Reviewed-empty (certified noise) chunks: ' + str(len(noise['reviewed_empty_chunks']))
          + ', false extractions on them: '
          + str(noise['false_extractions_on_reviewed_empty']))
    if noise['uncertified_boilerplate_chunks']:
        print('Boilerplate chunks not yet certified by review: '
              + ', '.join(noise['uncertified_boilerplate_chunks']))
    if unscored:
        print('Unscored chunks (no labels): ' + ', '.join(unscored))
    print('Score saved: ' + str(out_path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
