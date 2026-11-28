"""Summarize one golden-20 run folder: operations view, offline, no calls.

Usage:

    python scripts/summarize_golden_run.py reports/golden20/run-<stamp>-<model>

Reads the run's metrics.jsonl and results.json and prints plus saves an
operations summary: first-pass schema rate, token totals and a per-1,000-chunk
projection, latency with the slowest chunks, served-provider distribution and
a per-category breakdown.

Honesty rules:
- The schema-rate denominator is every attempted chunk, including failed ones.
- Cost totals use only provider-reported numbers. If any row's cost is
  unreported, the cost total and projection are "unknown", never a guess.
  Provider-reported zero is reported as zero (free models report 0). A
  percentage saving is never computed: there is no reference run to compare.
- The token projection is arithmetic on this run's average, not a price quote.
- Mechanical checks (schema, quote containment) stay separate from reviewed
  semantic quality; this script reports the mechanical side only.
- Served-provider counts describe which provider served each call. The runner
  has no failover logic; nothing here claims failover behavior.

Writes summary-<stamp>.json inside the run folder and never overwrites.
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))

from app.extraction import chunk_metrics  # noqa: E402

UNAVAILABLE = chunk_metrics.UNAVAILABLE
PER_CHUNKS = 1000


def numbers(rows: List[Dict[str, Any]], field: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(value)
    return values


def token_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    attempted = len(rows)
    out: Dict[str, Any] = {}
    for field in ('prompt_tokens', 'completion_tokens', 'cached_tokens', 'reasoning_tokens'):
        values = numbers(rows, field)
        entry: Dict[str, Any] = {'total': sum(values) if values else 0,
                                 'reported_rows': len(values)}
        if len(values) < attempted:
            entry['unreported_rows'] = attempted - len(values)
        if attempted and len(values) == attempted:
            entry['per_1000_chunks_projection'] = round(sum(values) / attempted * PER_CHUNKS, 1)
        else:
            entry['per_1000_chunks_projection'] = 'unknown (unreported rows)'
        out[field] = entry
    return out


def cost_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    attempted = len(rows)
    values = numbers(rows, 'cost_usd')
    if len(values) < attempted:
        return {'total_usd': 'unknown',
                'note': str(attempted - len(values))
                        + ' row(s) did not report cost; missing costs stay unknown.'}
    total = sum(values)
    note = ('All rows reported cost. Values are provider-reported; free models '
            'report 0. No percentage saving is computed: there is no reference '
            'run to compare against.')
    return {'total_usd': round(total, 6),
            'per_1000_chunks_projection_usd': round(total / attempted * PER_CHUNKS, 4)
            if attempted else 'unknown',
            'note': note}


def latency_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    timed = [(row.get('chunk_id'), row.get('latency_ms'))
             for row in rows
             if isinstance(row.get('latency_ms'), (int, float))]
    if not timed:
        return {'note': 'no latency reported'}
    values = [item[1] for item in timed]
    slowest = sorted(timed, key=lambda item: item[1], reverse=True)[:3]
    return {'min_ms': min(values),
            'median_ms': statistics.median(values),
            'max_ms': max(values),
            'slowest_chunks': [{'chunk_id': c, 'latency_ms': v} for c, v in slowest]}


def distribution(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or UNAVAILABLE)
        counts[key] = counts.get(key, 0) + 1
    return counts


def category_breakdown(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    breakdown: Dict[str, Dict[str, int]] = {}
    for row in rows:
        category = str(row.get('chunk_category') or 'unassigned')
        entry = breakdown.setdefault(category, {'attempted': 0, 'schema_pass': 0})
        entry['attempted'] += 1
        if row.get('schema_valid') is True:
            entry['schema_pass'] += 1
    return breakdown


def summarize(run_folder: Path) -> Dict[str, Any]:
    rows = chunk_metrics.read_rows(run_folder / 'metrics.jsonl')
    results_path = run_folder / 'results.json'
    run_summary: Dict[str, Any] = {}
    if results_path.exists():
        document = json.loads(results_path.read_text(encoding='utf-8'))
        run_summary = document.get('summary') or {}
    attempted = len(rows)
    passed = sum(1 for row in rows if row.get('schema_valid') is True)
    return {
        'run_folder': str(run_folder),
        'summarized_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': run_summary.get('model'),
        'prompt_version': run_summary.get('prompt_version'),
        'chunks_attempted': attempted,
        'first_pass_schema': {
            'passed': passed,
            'attempted': attempted,
            'rate': round(passed / attempted, 4) if attempted else None,
            'note': 'Denominator counts every attempted chunk, including failed calls.',
        },
        'tokens': token_summary(rows),
        'cost': cost_summary(rows),
        'latency': latency_summary(rows),
        'served_providers': distribution(rows, 'served_provider'),
        'categories': category_breakdown(rows),
        'grounding': {
            'mechanical': 'schema validation and quote containment only (see score file '
                          'for containment once labels exist)',
            'semantic_review': 'not reviewed by this script; reviewed-label quality comes '
                               'from scripts/score_golden_run.py',
        },
        'caveats': [
            'Served-provider counts show which provider answered each call. The runner '
            'has no failover logic; nothing here observes or claims failover.',
            'Token and cost projections are arithmetic on this run, not a price quote.',
            'A percentage cost saving is undefined without a reference run.',
        ],
    }


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python scripts/summarize_golden_run.py RUN_FOLDER')
        return 2
    run_folder = Path(sys.argv[1])
    if not (run_folder / 'metrics.jsonl').exists():
        print('No metrics.jsonl in ' + str(run_folder))
        return 2
    report = summarize(run_folder)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    out_path = run_folder / ('summary-' + stamp + '.json')
    if out_path.exists():
        print('Summary file already exists; refusing to overwrite: ' + str(out_path))
        return 1
    out_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    schema = report['first_pass_schema']
    print('Model: ' + str(report['model']) + ' | prompt ' + str(report['prompt_version']))
    print('First-pass schema: ' + str(schema['passed']) + ' of ' + str(schema['attempted'])
          + ' attempted chunks (failed calls count)')
    tokens = report['tokens']
    print('Tokens: input ' + str(tokens['prompt_tokens']['total'])
          + ', output ' + str(tokens['completion_tokens']['total'])
          + ', cached ' + str(tokens['cached_tokens']['total'])
          + ', reasoning ' + str(tokens['reasoning_tokens']['total']))
    print('Projection per 1,000 chunks (arithmetic on this run): input '
          + str(tokens['prompt_tokens']['per_1000_chunks_projection'])
          + ', output ' + str(tokens['completion_tokens']['per_1000_chunks_projection']))
    print('Cost: ' + str(report['cost']['total_usd']) + ' USD total - '
          + str(report['cost'].get('note', '')))
    latency = report['latency']
    if 'slowest_chunks' in latency:
        slow = ', '.join(str(item['chunk_id']) + ' (' + str(item['latency_ms']) + ' ms)'
                         for item in latency['slowest_chunks'])
        print('Latency: median ' + str(latency['median_ms']) + ' ms; slowest: ' + slow)
    print('Served providers: ' + json.dumps(report['served_providers'], sort_keys=True))
    print('Categories: ' + json.dumps(report['categories'], sort_keys=True))
    print('Grounding: mechanical checks here; reviewed semantic quality comes from the scorer.')
    print('Summary saved: ' + str(out_path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
