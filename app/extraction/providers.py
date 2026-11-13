"""One HTTP request per call, no hidden SDK retries or provider repair calls."""
import json
import http.client
import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError
from urllib.request import Request, build_opener

from app.extraction.client import NoRedirect, number_or_none


class CallFailure(Exception):
    def __init__(self, kind, status=None, retry_after=None):
        super().__init__(kind)  # Never echo provider bodies, headers or secrets.
        self.kind = kind
        self.status = status
        self.retry_after = retry_after


class JSONProvider:
    def __init__(self, provider, model, max_output_tokens, timeout=120, opener=None):
        if provider not in ('gemini', 'openai') or not model.strip():
            raise ValueError('explicit provider and model required')
        if type(max_output_tokens) is not int or max_output_tokens <= 0 or timeout <= 0:
            raise ValueError('positive output limit and timeout required')
        self.provider = provider
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.opener = opener or build_opener(NoRedirect())
        self.recipe = {'provider': provider, 'model': model,
                       'max_output_tokens': max_output_tokens, 'api_version': 'structured-v2',
                       'execution_mode': 'live_transport'}

    def ready(self):
        variable = 'GEMINI_API_KEY' if self.provider == 'gemini' else 'OPENAI_API_KEY'
        if not os.environ.get(variable, '').strip():
            raise ValueError('provider key missing; no request sent')

    def call(self, prompt, schema):
        if self.provider == 'gemini':
            url = 'https://generativelanguage.googleapis.com/v1beta/interactions'
            key = os.environ.get('GEMINI_API_KEY', '').strip()
            headers = {'x-goog-api-key': key}
            body = {'model': self.model, 'input': prompt, 'store': False,
                    'generation_config': {'max_output_tokens': self.max_output_tokens},
                    'response_format': {'type': 'text', 'mime_type': 'application/json', 'schema': schema}}
        else:
            url = 'https://api.openai.com/v1/responses'
            key = os.environ.get('OPENAI_API_KEY', '').strip()
            headers = {'Authorization': 'Bearer ' + key}
            body = {'model': self.model, 'input': prompt, 'store': False,
                    'max_output_tokens': self.max_output_tokens,
                    'text': {'format': {'type': 'json_schema', 'name': 'extraction',
                                        'strict': True, 'schema': schema}}}
        if not key:
            raise ValueError('provider key missing; no request sent')
        headers['Content-Type'] = 'application/json'
        request = Request(url, data=json.dumps(body).encode(), headers=headers, method='POST')
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read().decode('utf-8')
                status = response.status
        except HTTPError as error:
            retry_after = None
            kind = 'http_failure'
            header = error.headers.get('Retry-After') if error.headers else None
            if header is not None:
                try:
                    retry_after = number_or_none(float(header))
                    if retry_after is None:
                        raise ValueError('invalid delay')
                except ValueError:
                    try:
                        date = parsedate_to_datetime(header)
                        retry_after = max(0, (date - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError, OverflowError):
                        kind = 'http_failure_retry_after_unparseable'
            raise CallFailure(kind, error.code, retry_after) from None
        except (OSError, EOFError, http.client.HTTPException):
            raise CallFailure('transport_unknown') from None
        elapsed = time.monotonic() - started
        try:
            data = json.loads(raw)
            text = self.output_text(data)
        except (ValueError, KeyError, TypeError):
            # A completed HTTP request with unusable content is not a safe retry.
            return {'raw_envelope': raw, 'raw_text': None, 'http_status': status,
                    'request_seconds': elapsed, 'usage': None, 'outcome': 'unusable_response',
                    'cost_usd': None}
        return {'raw_envelope': raw, 'raw_text': text, 'http_status': status,
                'request_seconds': elapsed, 'usage': data.get('usage'),
                'outcome': 'returned', 'cost_usd': None}

    def output_text(self, data):
        if data.get('status') != 'completed':
            raise ValueError('incomplete or refused')
        texts = []
        if self.provider == 'gemini':
            for step in data['steps']:
                if step.get('type') == 'model_output':
                    for part in step.get('content', []):
                        if part.get('type') == 'text':
                            texts.append(part['text'])
        else:
            for output in data['output']:
                if output.get('type') == 'message':
                    for part in output.get('content', []):
                        if part.get('type') == 'refusal':
                            raise ValueError('refused')
                        if part.get('type') == 'output_text':
                            texts.append(part['text'])
        if len(texts) != 1 or not isinstance(texts[0], str):
            raise ValueError('exactly one structured output required')
        return texts[0]
