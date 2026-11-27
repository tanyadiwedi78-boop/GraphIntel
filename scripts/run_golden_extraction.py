"""Run the golden-20 extraction through OpenRouter and log per-chunk metrics.

One command:

    python scripts/run_golden_extraction.py

Everything else lives in configs/golden_extraction.json. Change the model in
that one file. Set "chunk_limit": 2 there for the first live smoke check,
then back to null for the full 20 chunks.

Needs OPENROUTER_API_KEY in the environment (the .env on your machine). The
script never asks for a key and never prints one.

What a run does, one chunk at a time, in order:
  1. builds the frozen few-shot prompt (the version GOLDEN_PROMPT_VERSION names),
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
from app.extraction.golden_openrouter import GoldenCallError, OpenRouterClient
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


class GoldenRun:
    """One evaluation run: shared settings in one place, one method per step."""

    def __init__(self, config: dict, completer=None) -> None:
        self.config = config
        self.model_id = config['models'][config['active_model']]
        self.ontology = load_ontology(PROJECT_FOLDER / config['ontology_path'])
        self.contract = ontology_contract(self.ontology)
        self.schema = self.contract.model_json_schema()
        self.examples = self.load_examples()
        self.completer = completer
        self.results = []
        self.stopped_early = False

    def load_examples(self) -> list:
        path = PROJECT_FOLDER / self.config['fewshot_path']
        return json.loads(path.read_text(encoding='utf-8'))['examples']

    def select_chunks(self) -> list:
        chunks = load_chunks(PROJECT_FOLDER / self.config['chunks_path'])
        limit = self.config.get('chunk_limit')
        if isinstance(limit, int) and limit > 0:
            chunks = chunks[:limit]
        return chunks

    def ensure_completer(self) -> bool:
        """Build the live client unless a test already supplied a completer."""
        if self.completer is not None:
            return True
        api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        if not api_key:
            print('OPENROUTER_API_KEY is not set; no request sent.')
            print('Set it from your .env first, then rerun.')
            return False
        client = OpenRouterClient(self.model_id, api_key,
                                  self.config['max_output_tokens'],
                                  reasoning=self.config.get('reasoning'))
        self.completer = lambda prompt: client.complete(prompt, self.schema)
        return True

    def validate(self, raw_text, chunk):
        """Return (payload or None, list of problems)."""
        if not raw_text:
            return None, ['empty or unusable response text']
        try:
            payload = self.contract.model_validate_json(raw_text)
        except ValueError as error:
            return None, ['schema validation failed: ' + str(error)[:200]]
        checks = check_source_contract(payload, chunk, self.ontology)
        return payload, checks['problems']

    def record_failure(self, metrics_path: Path, chunk_id: str, error: GoldenCallError) -> None:
        label = 'http ' + str(error.status)
        row = chunk_metrics.record_row(chunk_id, self.model_id, None, False,
                                       'unavailable', 'unavailable', 'unavailable',
                                       error=label)
        chunk_metrics.append_row(metrics_path, row)
        self.results.append({'chunk_id': chunk_id, 'ok': False, 'error': label})

    def record_answer(self, metrics_path: Path, chunk: dict, response: dict) -> None:
        chunk_id = str(chunk['chunk_id'])
        raw_text = response.get('text')
        payload, problems = self.validate(raw_text, chunk)
        ok = payload is not None and not problems
        row = chunk_metrics.record_row(
            chunk_id, self.model_id, response, ok,
            len(payload.entities) if payload else 'unavailable',
            len(payload.relationships) if payload else 'unavailable',
            response.get('latency_ms'),
            error=None if ok else '; '.join(problems)[:200])
        chunk_metrics.append_row(metrics_path, row)
        self.results.append({'chunk_id': chunk_id, 'ok': ok, 'problems': problems,
                             'payload': json.loads(raw_text) if payload else None,
                             'raw_text': None if payload else raw_text})

    def run_chunk(self, metrics_path: Path, chunk: dict) -> None:
        chunk_id = str(chunk['chunk_id'])
        prompt = build_golden_prompt(chunk, self.ontology, self.examples)
        try:
            response = self.completer(prompt)
        except GoldenCallError as error:
            self.record_failure(metrics_path, chunk_id, error)
            if error.status in STOP_STATUSES:
                print('Stop: the provider answered HTTP ' + str(error.status)
                      + ' on chunk ' + chunk_id + '.')
                print('No more live calls this run. The saved rows show how far it got.')
                self.stopped_early = True
            else:
                print('Chunk ' + chunk_id + ' failed (HTTP ' + str(error.status)
                      + '); recorded, moving on.')
            return
        self.record_answer(metrics_path, chunk, response)

    def summary(self) -> dict:
        return {
            'model': self.model_id,
            'prompt_version': GOLDEN_PROMPT_VERSION,
            'fewshot_path': self.config['fewshot_path'],
            'fewshot_example_ids': [str(example.get('chunk_id')) for example in self.examples],
            'chunks_attempted': len(self.results),
            'chunks_ok': sum(1 for item in self.results if item['ok']),
            'stopped_early': self.stopped_early,
            'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'note': 'Schema pass means shape, ontology and provenance checks passed. '
                    'It is not an accuracy claim; accuracy comes from scoring these '
                    'answers against reviewed labels.',
        }

    def execute(self) -> int:
        chunks = self.select_chunks()
        if not chunks:
            print('No chunks to run. Check chunks_path in the config.')
            return 2
        if not self.ensure_completer():
            return 2
        run_folder = fresh_run_folder(PROJECT_FOLDER / self.config['reports_dir'], self.model_id)
        metrics_path = run_folder / 'metrics.jsonl'
        for chunk in chunks:
            self.run_chunk(metrics_path, chunk)
            if self.stopped_early:
                break
        summary = self.summary()
        (run_folder / 'results.json').write_text(
            json.dumps({'summary': summary, 'results': self.results}, indent=2) + '\n',
            encoding='utf-8')
        print('Run saved under ' + str(run_folder))
        print('Chunks ok: ' + str(summary['chunks_ok']) + ' of ' + str(summary['chunks_attempted']))
        return self.report_verification(metrics_path)

    def report_verification(self, metrics_path: Path) -> int:
        problems = chunk_metrics.verify_log(metrics_path)
        if problems:
            print('METRICS VERIFICATION FAILED - ' + str(len(problems)) + ' problem(s):')
            for problem in problems:
                print('  ' + problem)
            print('Fields the provider did not report stay "unavailable"; they are never filled in.')
            return 1
        print('Metrics verification passed: every usage and cost field was reported.')
        return 0


def run(config_path: Path = CONFIG_PATH, completer=None) -> int:
    return GoldenRun(load_config(config_path), completer).execute()


if __name__ == '__main__':
    raise SystemExit(run())
