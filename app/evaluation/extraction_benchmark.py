"""Offline extraction metrics. Shape, containment and reviewed truth stay separate.

Inputs must come from the same immutable job. This module makes no model calls,
retries, embedding calls or database writes. Missing observations stay unknown.
"""
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal, InvalidOperation

from app.extraction.validate import ExtractionResponse, squash_spaces


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def measured_ratio(numerator, denominator, basis):
    return {'value': numerator / denominator if denominator else None,
            'numerator': numerator, 'denominator': denominator,
            'status': 'Measured' if denominator else 'Not measured', 'basis': basis}


def item_key(item, fields):
    return tuple(' '.join(str(item[field]).casefold().split()) for field in fields)


def precision_recall(predicted, expected, fields):
    predictions = Counter(item_key(item, fields) for item in predicted)
    truth = Counter(item_key(item, fields) for item in expected)
    correct = sum((predictions & truth).values())
    return correct, sum(predictions.values()), sum(truth.values())


def empty_metrics():
    return {'correct': 0, 'predicted': 0, 'expected': 0}


def benchmark(chunks, observations, labels=None, evidence_reviews=None,
              submission_attempts=None, sample_cost_usd=None, batch_timing=None, expanded_contract=False):
    """One observation per planned chunk, including missing/failed chunks.

    observations: chunk_id -> {raw_text: str, accepted: bool, entities: [],
    relationships: [], latency_seconds: number or None}. accepted means only
    local pipeline acceptance, NEVER factual truth. Review labels must bind the
    source hash. Evidence decisions must bind both source hash and exact predicted
    relationship hash. Missing review cannot silently count as false or true.
    submission_attempts=None means history unavailable, not zero failures.
    """
    from app.extraction.contracts import ExtractionPayload
    schema = ExtractionPayload if expanded_contract else ExtractionResponse
    entity_fields = ('canonical_name', 'entity_type') if expanded_contract else ('name', 'type')
    edge_fields = ('relationship_type', 'source_entity', 'target_entity') if expanded_contract else ('relation', 'source', 'target')
    evidence_field = 'evidence' if expanded_contract else 'evidence_quote'
    labels = labels or {}
    evidence_reviews = evidence_reviews or {}
    if set(observations) - set(chunks):
        raise ValueError('observation has an unknown chunk ID')
    coverage = {'planned': len(chunks), 'returned_text': 0, 'schema_pass': 0,
                'accepted': 0, 'reviewed_labels': 0, 'unreviewed_labels': 0,
                'predicted_relationships': 0, 'reviewed_evidence': 0,
                'contained_quotes': 0, 'latency_observations': 0}
    entity = empty_metrics()
    relationship = empty_metrics()
    supported = 0
    latencies = []
    for chunk_id, source in chunks.items():
        if not isinstance(source, str):
            raise ValueError('chunk text must be a string')
        source_hash = identity(source)
        observation = observations.get(chunk_id, {})
        raw_text = observation.get('raw_text')
        if isinstance(raw_text, str):
            coverage['returned_text'] += 1
            try:
                schema.model_validate_json(raw_text)
                coverage['schema_pass'] += 1
            except (ValueError, TypeError):
                pass
        accepted = observation.get('accepted') is True
        if accepted:
            coverage['accepted'] += 1
        predictions = observation.get('entities', []) if accepted else []
        relations = observation.get('relationships', []) if accepted else []
        label = labels.get(chunk_id, {})
        if label.get('reviewed') is True:
            if label.get('source_hash') != source_hash:
                raise ValueError('reviewed labels do not match the original chunk')
            coverage['reviewed_labels'] += 1
            for target, predicted, expected, fields in (
                (entity, predictions, label['entities'], entity_fields),
                (relationship, relations, label['relationships'], edge_fields)):
                correct, count, total = precision_recall(predicted, expected, fields)
                target['correct'] += correct
                target['predicted'] += count
                target['expected'] += total
        else:
            coverage['unreviewed_labels'] += 1
        for edge in relations:
            coverage['predicted_relationships'] += 1
            quote = edge.get(evidence_field)
            contained = isinstance(quote, str) and bool(quote.strip()) and squash_spaces(quote) in squash_spaces(source)
            coverage['contained_quotes'] += int(contained)
            review = evidence_reviews.get(chunk_id, {}).get(identity(edge))
            if review and review.get('reviewed') is True:
                if review.get('source_hash') != source_hash:
                    raise ValueError('evidence review does not match original chunk')
                if type(review.get('supported')) is not bool:
                    raise ValueError('reviewed evidence needs an explicit supported boolean')
                coverage['reviewed_evidence'] += 1
                # Even a positive human label cannot rescue an absent quote.
                supported += int(review['supported'] and contained)
        latency = observation.get('latency_seconds')
        if latency is not None:
            if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
                raise ValueError('latency must be finite nonnegative measured seconds')
            latencies.append(latency)
    coverage['latency_observations'] = len(latencies)
    report = {'coverage': coverage,
        'json_schema_pass_rate': measured_ratio(coverage['schema_pass'], coverage['returned_text'],
            'Raw returned text before normalization; not ontology or factual correctness. See return coverage.'),
        'returned_output_coverage': measured_ratio(coverage['returned_text'], len(chunks), 'All planned chunks.'),
        'evidence_accuracy': measured_ratio(supported, coverage['reviewed_evidence'],
            'Explicit reviewed semantic support AND quote containment. Unreviewed edges are excluded, not assumed correct.'),
        'evidence_review_coverage': measured_ratio(coverage['reviewed_evidence'], coverage['predicted_relationships'],
            'Reviewed accepted relationship predictions.'),
        'quote_containment_rate': measured_ratio(coverage['contained_quotes'], coverage['predicted_relationships'],
            'Whitespace-normalized substring test only; NOT semantic evidence accuracy.'),
        'extraction_latency': {'status': 'Measured' if latencies else 'Not measured',
            'mean_seconds': sum(latencies) / len(latencies) if latencies else None,
            'count': len(latencies), 'basis': 'Only actual per-request measurements; never inferred from batch wall time.'},
        'batch_wall_time': batch_timing or {'status': 'Not measured'},
        'cost_per_1000_chunks': {'status': 'Not measured', 'value_usd': None,
            'basis': 'Observed cost / all planned sample chunks * 1000. Sample extrapolation, not a corpus quote or invoice.'},
        'rate_limit_failure_rate': {'status': 'Not measured', 'value': None,
            'basis': 'HTTP 429 responses / all observed submission HTTP attempts; GET polls excluded.'},
        'warning': 'Offline tests and schema acceptance do not measure factual accuracy. No provider selection is implied.'}
    for name, totals in (('entity', entity), ('relationship', relationship)):
        report[name + '_precision'] = measured_ratio(totals['correct'], totals['predicted'],
            'Accepted predictions on source-bound reviewed labels, including reviewed-empty chunks.')
        report[name + '_recall'] = measured_ratio(totals['correct'], totals['expected'],
            'Expected items across reviewed chunks; missing or rejected outputs count as missed.')
    if sample_cost_usd is not None:
        try:
            cost = Decimal(str(sample_cost_usd))
        except InvalidOperation:
            raise ValueError('sample cost must be measured nonnegative USD') from None
        if not cost.is_finite() or cost < 0 or isinstance(sample_cost_usd, bool):
            raise ValueError('sample cost must be measured nonnegative USD')
        if chunks:
            report['cost_per_1000_chunks'].update(status='Measured sample extrapolation',
                value_usd=str(cost * 1000 / len(chunks)), sample_cost_usd=str(cost), sample_chunks=len(chunks))
    if submission_attempts is not None:
        # Unknown response must remain in denominator and explicit uncertainty.
        rate_failures = 0
        unknown = 0
        for status in submission_attempts:
            if status is None:
                unknown += 1
            elif type(status) is not int or not 100 <= status <= 599:
                raise ValueError('HTTP attempt status must be a code or None for unknown')
            elif status == 429:
                rate_failures += 1
        metric = measured_ratio(rate_failures, len(submission_attempts),
            'Observed HTTP 429 / submission attempts; unknown-status attempts are reported, never called non-429.')
        metric['unknown_status_attempts'] = unknown
        if unknown:
            metric.update(value=None, status='Not measured completely', known_429_count=rate_failures)
        report['rate_limit_failure_rate'] = metric
    return report
