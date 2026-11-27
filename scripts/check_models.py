"""Preflight check for the golden-20 model picks, run before any paid call.

Reads the OpenRouter catalog and the per-model endpoint list (both public,
no API key needed, no completion is sent) and checks every model in the
golden config:

  1. the model id is still listed (free-tier ids vanish without notice);
  2. ONE SAME endpoint jointly declares structured_outputs, response_format
     and reasoning control, and prices prompt and completion at zero -
     different endpoints passing different criteria does not count;
  3. reasoning is not mandatory, so the configured effort "none" is not
     rejected outright.

A model can also pass with a WARNING when its metadata does not confirm
that effort "none" is accepted (supported_efforts missing or lacking
"none"). A warning means: the two-chunk smoke run decides, and if the
response rejects the reasoning setting, stop and report - do not quietly
rerun with changed settings.

Passing proves listing and declared support only. Extraction quality and
real reasoning behavior are proven by the smoke run, not here.
Exit code 1 means at least one model failed; run nothing until it is fixed.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CATALOG_URL = 'https://openrouter.ai/api/v1/models'
ENDPOINTS_URL = 'https://openrouter.ai/api/v1/models/{model_id}/endpoints'
REQUIRED_PARAMETERS = ('structured_outputs', 'response_format', 'reasoning')
DEFAULT_CONFIG_PATH = 'configs/golden_extraction.json'


def fetch_json(url: str) -> Dict[str, Any]:
    """GET one public JSON document with a clear failure message."""
    request = Request(url, headers={'Accept': 'application/json'})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as error:
        raise RuntimeError('GET ' + url + ' failed: HTTP ' + str(error.code)) from None
    except URLError as error:
        raise RuntimeError('GET ' + url + ' failed: ' + str(error.reason)) from None


def endpoint_is_free(endpoint: Dict[str, Any]) -> bool:
    pricing = endpoint.get('pricing') or {}
    return float(pricing.get('prompt') or 0) == 0 and float(pricing.get('completion') or 0) == 0


def joint_endpoint(endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The first endpoint that alone satisfies schema, format, reasoning and $0."""
    for endpoint in endpoints:
        supported = endpoint.get('supported_parameters') or []
        if all(name in supported for name in REQUIRED_PARAMETERS) and endpoint_is_free(endpoint):
            return endpoint
    return {}


def check_model(entry: Dict[str, Any], endpoints: List[Dict[str, Any]]):
    """Return (problems, warnings) for one catalog entry and its endpoints."""
    problems = []
    warnings = []
    pricing = entry.get('pricing') or {}
    if float(pricing.get('prompt') or 0) != 0 or float(pricing.get('completion') or 0) != 0:
        problems.append('no longer free at model level')
    if not endpoints:
        problems.append('no serving endpoints today')
        return problems, warnings
    if not joint_endpoint(endpoints):
        problems.append('no single endpoint jointly offers strict JSON schema, '
                        'response_format, reasoning control and zero pricing')
    reasoning = entry.get('reasoning') or {}
    if reasoning.get('mandatory'):
        problems.append('reasoning is mandatory; effort "none" would be rejected')
        return problems, warnings
    efforts = reasoning.get('supported_efforts')
    if efforts is None:
        warnings.append('effort "none" not confirmed in metadata; smoke run decides')
    elif 'none' not in efforts:
        warnings.append('supported_efforts ' + str(efforts) + ' lacks "none"; smoke run decides')
    return problems, warnings


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config = json.loads(open(config_path, encoding='utf-8').read())
    model_ids = sorted(set(config['models'].values()))
    catalog = fetch_json(CATALOG_URL)['data']
    by_id = {}
    for entry in catalog:
        by_id[entry['id']] = entry
    failures = 0
    for model_id in model_ids:
        entry = by_id.get(model_id)
        if entry is None:
            print('FAIL ' + model_id + ': not in the OpenRouter catalog today')
            failures += 1
            continue
        endpoints_data = fetch_json(ENDPOINTS_URL.format(model_id=model_id))
        endpoints = (endpoints_data.get('data') or {}).get('endpoints') or []
        problems, warnings = check_model(entry, endpoints)
        if problems:
            failures += 1
            for problem in problems:
                print('FAIL ' + model_id + ': ' + problem)
        else:
            print('PASS ' + model_id + ' (' + str(len(endpoints)) + ' endpoint(s), one jointly capable, free)')
        for warning in warnings:
            print('WARN ' + model_id + ': ' + warning)
    if failures:
        print(str(failures) + ' of ' + str(len(model_ids)) + ' models failed; fix the config before any run.')
        return 1
    print('All ' + str(len(model_ids)) + ' models pass preflight as of this check. Recheck on run day.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
