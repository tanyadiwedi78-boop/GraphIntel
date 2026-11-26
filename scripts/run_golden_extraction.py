"""Run the golden-20 extraction through OpenRouter and log per-chunk metrics.

One command:

    python scripts/run_golden_extraction.py

Everything else lives in configs/golden_extraction.json. Change the model in
that one file. Set "chunk_limit": 2 there for the first live smoke check,
then back to null for the full 20 chunks.

Needs OPENROUTER_API_KEY in the environment (the .env on your machine). The
script never asks for a key and never prints one.

What a run does, one chunk at a time, in order:
  1. builds the frozen few-shot prompt (golden-fewshot-v4),
  2. calls the configured free model with a strict JSON schema,
  3. validates the answer against the frozen contract and the source rules,
  4. appends one metrics row (tokens, cached/reasoning tokens, reported cost,
     served provider, schema pass, counts, latency),
  5. saves the raw parsed answer as evidence.

Safety: a 401/402/403/429 stops the whole run at once - no retries, no
guessing about quota. Every run writes NEW files under
reports/golden20/run-<timestamp>-<model>/ and refuses to reuse a folder, so
no result is ever overwritten.

At the end the metrics log is verified: any usage or cost field the provider
did not report, or any chunk that failed schema validation, fails the check
loudly. Numbers that were never reported are never shown as zeros.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_FOLDER))

from app.extraction import chunk_metrics
from app.extraction.contracts import check_source_contract, ontology_contract
from app.extraction.golden_openrouter import GoldenCallError, complete
from app.extraction.golden_prompt import GOLDEN_PROMPT_VERSION, build_golden_prompt
from app.graph.ontology import load_ontology

CONFIG_PATH = PROJECT_FOLDER / 'configs' / 'golden_extraction.json'
STOP_STATUSES = {401, 402, 403, 429}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('active_model') not in config.get('models', {}):
        raise ValueError('active_model must be a key under models in ' + str(path))
    return config


def load_chunks(path: Path) -> list:
    document = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(document, dict) and isinstance(document.get('chunks'), list):
        return document['chunks']
    if isinstance(document, dict):
        return [{'chunk_id': key, 'text': value} for key, value in document.items()]
    if isinstance(document, list):
        return document
    raise ValueError('unrecognized chunk file shape in ' + str(path))


def fresh_run_folder(reports_dir: Path, model_id: str) -> Path:
    slug = ''.join(ch if ch.isalnum() else '-' for ch in model_id)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    folder = reports_dir / ('run-' + stamp + '-' + slug)
    if folder.exists():
        raise ValueError('run folder already exists; refusing to overwrite: ' + str(folder))
    folder.mkdir(parents=True)
    return folder


def run(config_path: Path = CONFIG_PATH, completer=None) -> int:
    config = load_config(config_path)
    model_id = config['models'][config['active_model']]
    ontology = load_ontology(PROJECT_FOLDER / config['ontology_path'])
    contract = ontology_contract(ontology)
    schema = contract.model_json_schema()

    fewshot_document = json.loads(
        (PROJECT_FOLDER / config['fewshot_path']).read_text(encoding='utf-8'))
    examples = fewshot_document['examples']

    chunks = load_chunks(PROJECT_FOLDER / config['chunks_path'])
    limit = config.get('chunk_limit')
    if isinstance(limit, int) and limit > 0:
        chunks = chunks[:limit]
    if not chunks:
        print('No chunks to run. Check chunks_path in ' + str(config_path))
        return 2

    run_folder = fresh_run_folder(PROJECT_FOLDER / config['reports_dir'], model_id)
    metrics_path = run_folder / 'metrics.jsonl'
    results_path = run_folder / 'results.json'

    if completer is None:
        api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        if not api_key:
            print('OPENROUTER_API_KEY is not set; no request sent.')
            print('Set it from your .env first, then rerun.')
            return 2

        def completer(prompt):
            return complete(model_id, prompt, schema,
                            max_output_tokens=config['max_output_tokens'],
                            api_key=api_key,
                            reasoning=config.get('reasoning'))

    results = []
    stopped_early = False
    for chunk in chunks:
        chunk_id = str(chunk['chunk_id'])
        prompt = build_golden_prompt(chunk, ontology, examples)
        try:
            response = completer(prompt)
        except GoldenCallError as error:
            row = chunk_metrics.record_row(chunk_id, model_id, None, False,
                                           'unavailable', 'unavailable',
                                           'unavailable',
                                           error='http ' + str(error.status))
            chunk_metrics.append_row(metrics_path, row)
            results.append({'chunk_id': chunk_id, 'ok': False,
                            'error': 'http ' + str(error.status)})
            if error.status in STOP_STATUSES:
                print('Stop: the provider answered HTTP ' + str(error.status)
                      + ' on chunk ' + chunk_id + '.')
                print('No more live calls this run. The saved rows show how far it got.')
                stopped_early = True
                break
            print('Chunk ' + chunk_id + ' failed (HTTP ' + str(error.status)
                  + '); recorded, moving on.')
            continue

        raw_text = response.get('text')
        payload = None
        problems = ['empty or unusable response text']
        if raw_text:
            try:
                payload = contract.model_validate_json(raw_text)
                checks = check_source_contract(payload, chunk, ontology)
                problems = checks['problems']
            except ValueError as error:
                problems = ['schema validation failed: ' + str(error)[:200]]

        ok = payload is not None and not problems
        row = chunk_metrics.record_row(
            chunk_id, model_id, response, ok,
            len(payload.entities) if payload else 'unavailable',
            len(payload.relationships) if payload else 'unavailable',
            response.get('latency_ms'),
            error=None if ok else '; '.join(problems)[:200])
        chunk_metrics.append_row(metrics_path, row)
        results.append({'chunk_id': chunk_id, 'ok': ok, 'problems': problems,
                        'payload': json.loads(raw_text) if payload else None,
                        'raw_text': None if payload else raw_text})

    summary = {
        'model': model_id,
        'prompt_version': GOLDEN_PROMPT_VERSION,
        'chunks_attempted': len(results),
        'chunks_ok': sum(1 for item in results if item['ok']),
        'stopped_early': stopped_early,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'note': 'Schema pass means shape, ontology and provenance checks passed. '
                'It is not an accuracy claim; accuracy comes from scoring these '
                'answers against reviewed labels.',
    }
    results_path.write_text(json.dumps({'summary': summary, 'results': results},
                                       indent=2) + '\n', encoding='utf-8')

    print('Run saved under ' + str(run_folder))
    print('Chunks ok: ' + str(summary['chunks_ok']) + ' of '
          + str(summary['chunks_attempted']))

    problems = chunk_metrics.verify_log(metrics_path)
    if problems:
        print('METRICS VERIFICATION FAILED - ' + str(len(problems)) + ' problem(s):')
        for problem in problems:
            print('  ' + problem)
        print('Fields the provider did not report stay "unavailable"; they are never filled in.')
        return 1
    print('Metrics verification passed: every usage and cost field was reported.')
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
