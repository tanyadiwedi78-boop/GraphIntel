"""Per-chunk extraction metrics, written as JSONL, one row per chunk attempt.

Honesty rules baked in:
- A field the provider did not report is the string "unavailable", never a
  made-up zero. OpenRouter's usage accounting documents which fields a
  response should carry; whether a given model actually sends them is only
  known from a real response.
- verify_log fails loudly when any usage or cost field is unavailable, so a
  silent gap can never pass as a complete log. A provider-reported 0 (for a
  free model's cost, or zero cached tokens) is a real number and passes.

Columns, in the order requested for the evaluation:
chunk_id, model, served_provider, prompt_tokens, completion_tokens,
cached_tokens, reasoning_tokens, cost_usd, schema_valid, entity_count,
edge_count, latency_ms.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

UNAVAILABLE = 'unavailable'

METRIC_FIELDS = ['chunk_id', 'model', 'served_provider', 'prompt_tokens',
                 'completion_tokens', 'cached_tokens', 'reasoning_tokens',
                 'cost_usd', 'schema_valid', 'entity_count', 'edge_count',
                 'latency_ms']

# Fields that must be provider-reported numbers/strings for the log to pass
# verification. schema_valid is reported separately as a check, not a gap.
REQUIRED_REPORTED = ['served_provider', 'prompt_tokens', 'completion_tokens',
                     'cached_tokens', 'reasoning_tokens', 'cost_usd']


def _reported(value: Any) -> Any:
    """Keep a real reported value (0 counts); turn anything missing into UNAVAILABLE."""
    if value is None or isinstance(value, bool):
        return UNAVAILABLE
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and value.strip():
        return value
    return UNAVAILABLE


def _walk(mapping: Any, path: List[str]) -> Any:
    current = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def served_provider_of(response: Dict[str, Any]) -> Any:
    """The upstream provider that actually served the request, when reported.

    OpenRouter chat completions can carry it top-level as "provider" or inside
    openrouter_metadata.endpoints.available entries flagged selected. Neither
    is guaranteed on every response, so absence is UNAVAILABLE, not a guess.
    """
    top = response.get('provider')
    if isinstance(top, str) and top.strip():
        return top
    available = _walk(response, ['openrouter_metadata', 'endpoints', 'available'])
    if isinstance(available, list):
        for entry in available:
            if isinstance(entry, dict) and entry.get('selected') and entry.get('provider'):
                return entry['provider']
    return UNAVAILABLE


def record_row(chunk_id: str, model: str, response: Optional[Dict[str, Any]],
               schema_valid: bool, entity_count: Any, edge_count: Any,
               latency_ms: Any, error: Optional[str] = None) -> Dict[str, Any]:
    """Build one metrics row from a provider response (or a failed attempt)."""
    usage = response.get('usage') if isinstance(response, dict) else None
    row = {
        'chunk_id': chunk_id,
        'model': model,
        'served_provider': served_provider_of(response) if isinstance(response, dict) else UNAVAILABLE,
        'prompt_tokens': _reported(_walk(usage, ['prompt_tokens'])),
        'completion_tokens': _reported(_walk(usage, ['completion_tokens'])),
        'cached_tokens': _reported(_walk(usage, ['prompt_tokens_details', 'cached_tokens'])),
        'reasoning_tokens': _reported(_walk(usage, ['completion_tokens_details', 'reasoning_tokens'])),
        'cost_usd': _reported(_walk(usage, ['cost'])),
        'schema_valid': bool(schema_valid),
        'entity_count': entity_count if isinstance(entity_count, int) else UNAVAILABLE,
        'edge_count': edge_count if isinstance(edge_count, int) else UNAVAILABLE,
        'latency_ms': latency_ms if isinstance(latency_ms, (int, float)) else UNAVAILABLE,
    }
    if error:
        row['error'] = error
    return row


def append_row(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSONL row. Rows are immutable evidence; never rewrite one."""
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, sort_keys=False) + '\n')


def read_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def verify_log(path: Path) -> List[str]:
    """Problems found in a metrics log; an empty list means the log passes.

    Fails loudly when any required usage/cost field is missing or unavailable
    on any row, and when any chunk's schema check failed. This is the check a
    reviewer runs before trusting the numbers; it never edits the log.
    """
    problems = []
    for row in read_rows(path):
        chunk_id = row.get('chunk_id', '?')
        for field in REQUIRED_REPORTED:
            value = row.get(field)
            if value is None or value == UNAVAILABLE:
                problems.append(chunk_id + ': ' + field + ' was not reported by the provider')
        if row.get('schema_valid') is not True:
            detail = row.get('error', 'schema validation failed')
            problems.append(chunk_id + ': schema_valid=false (' + str(detail) + ')')
    return problems
