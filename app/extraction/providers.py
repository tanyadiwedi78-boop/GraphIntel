"""One HTTP request per call, no hidden SDK retries or provider repair calls."""
import json
import math
import http.client
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError
from urllib.request import Request, build_opener, HTTPRedirectHandler

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def number_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


class CallFailure(Exception):
    def __init__(self, kind, status=None, retry_after=None):
        super().__init__(kind)  # Never echo provider bodies, headers or secrets.
        self.kind = kind
        self.status = status
        self.retry_after = retry_after


class JSONProvider:
    def __init__(self, provider, model, max_output_tokens, timeout=120, opener=None):
        if provider not in ('gemini', 'openai') or not re.fullmatch(r'[A-Za-z0-9._-]+', model):
            raise ValueError('explicit provider and model required')
        if type(max_output_tokens) is not int or max_output_tokens <= 0 or timeout <= 0:
            raise ValueError('positive output limit and timeout required')
        self.provider = provider
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.opener = opener or build_opener(NoRedirect())
        self.recipe = {'provider': provider, 'model': model,
                       'max_output_tokens': max_output_tokens, 'api_version': 'structured-v3-generate-content',
                       'execution_mode': 'live_transport'}

    def ready(self):
        variable = 'GEMINI_API_KEY' if self.provider == 'gemini' else 'OPENAI_API_KEY'
        if not os.environ.get(variable, '').strip():
            raise ValueError('provider key missing; no request sent')

    def call(self, prompt, schema):
        url, headers = self.endpoint()
        key = os.environ.get(self.key_variable(), '').strip()
        body = self.body(prompt, schema)
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
        data = None
        try:
            data = json.loads(raw)
            text = self.output_text(data)
        except (ValueError, KeyError, TypeError, AttributeError):
            # A completed HTTP request with unusable content is not a safe retry.
            return {'raw_envelope': raw, 'raw_text': None, 'http_status': status,
                    'request_seconds': elapsed, 'usage': data.get('usage') or data.get('usageMetadata') if isinstance(data, dict) else None, 'outcome': 'unusable_response',
                    'cost_usd': None}
        return {'raw_envelope': raw, 'raw_text': text, 'http_status': status,
                'request_seconds': elapsed, 'usage': data.get('usage') or data.get('usageMetadata'),
                'outcome': 'returned', 'cost_usd': None}

    def key_variable(self):
        return 'GEMINI_API_KEY' if self.provider == 'gemini' else 'OPENAI_API_KEY'

    def endpoint(self):
        key = os.environ.get(self.key_variable(), '').strip()
        if self.provider == 'gemini':
            return ('https://generativelanguage.googleapis.com/v1beta/models/' + self.model + ':generateContent',
                    {'x-goog-api-key': key})
        return 'https://api.openai.com/v1/responses', {'Authorization': 'Bearer ' + key}

    def body(self, prompt, schema):
        # Same common JSON schema/prompt for sync and native batch. Only the
        # provider's documented request envelope differs. No tool/search calls.
        if self.provider == 'gemini':
            return {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                    'generationConfig': {'maxOutputTokens': self.max_output_tokens,
                        'responseFormat': {'text': {'mimeType': 'application/json', 'schema': schema}}}}
        return {'model': self.model, 'input': prompt, 'store': False,
                'max_output_tokens': self.max_output_tokens,
                'text': {'format': {'type': 'json_schema', 'name': 'extraction',
                                    'strict': True, 'schema': schema}}}

    def output_text(self, data):
        if not isinstance(data, dict):
            raise ValueError('provider envelope must be an object')
        texts = []
        if self.provider == 'gemini' and 'candidates' in data:
            candidates = data['candidates']
            if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
                raise ValueError('incomplete or refused')
            for part in candidates[0]['content']['parts']:
                if 'text' in part and not part.get('thought'):
                    texts.append(part['text'])
        elif self.provider == 'gemini':
            # Read old saved Interactions envelopes, never silently send to it.
            if data.get('status') != 'completed':
                raise ValueError('incomplete or refused')
            for step in data['steps']:
                if step.get('type') == 'model_output':
                    for part in step.get('content', []):
                        if part.get('type') == 'text':
                            texts.append(part['text'])
        else:
            if data.get('status') != 'completed':
                raise ValueError('incomplete or refused')
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

    def api(self, path, body=None, multipart=None):
        """One batch control request. No retries, redirects or unsafe URL input."""
        self.ready()
        base = ('https://generativelanguage.googleapis.com/v1beta/' if self.provider == 'gemini'
                else 'https://api.openai.com/v1/')
        if not re.fullmatch(r'[A-Za-z0-9_./:-]+', path) or '..' in path:
            raise ValueError('invalid provider resource ID')
        headers = self.endpoint()[1]
        data = None
        if body is not None:
            headers['Content-Type'] = 'application/json'
            data = json.dumps(body).encode()
        if multipart is not None:
            data, headers['Content-Type'] = multipart
        request = Request(base + path, headers=headers, data=data,
                          method='POST' if data is not None else 'GET')
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return response.read().decode('utf-8')
        except HTTPError as error:
            raise CallFailure('batch_http_failure', error.code) from None
        except (OSError, EOFError, http.client.HTTPException):
            raise CallFailure('batch_transport_unknown') from None

    def batch_submit(self, items, display_name):
        """Native provider batch, one inference per item, never packed together.

        Gemini uses bounded inline generateContent requests. OpenAI uses its
        JSONL Files upload then Batch API. A lost create response is ambiguous;
        the caller's durable claim blocks re-submission, including after upload.
        """
        if self.provider == 'gemini':
            requests = [{'request': item['body'], 'metadata': {'key': item['custom_id']}} for item in items]
            body = {'batch': {'display_name': display_name,
                             'input_config': {'requests': {'requests': requests}}}}
            if len(json.dumps(body).encode()) > 16 * 1024 * 1024:
                raise ValueError('inline batch exceeds local 16 MiB safety bound')
            return json.loads(self.api('models/' + self.model + ':batchGenerateContent', body))
        import uuid
        boundary = 'GraphIntel' + uuid.uuid4().hex
        lines = ''.join(json.dumps({'custom_id': item['custom_id'], 'method': 'POST',
                         'url': '/v1/responses', 'body': item['body']}) + '\n' for item in items)
        raw = ('--' + boundary + '\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
               '--' + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="requests.jsonl"\r\n'
               'Content-Type: application/jsonl\r\n\r\n' + lines + '\r\n--' + boundary + '--\r\n').encode()
        uploaded = json.loads(self.api('files', multipart=(raw, 'multipart/form-data; boundary=' + boundary)))
        return json.loads(self.api('batches', {'input_file_id': uploaded['id'], 'endpoint': '/v1/responses',
                          'completion_window': '24h', 'metadata': {'graphintel_job': display_name}}))

    def batch_collect(self, receipt, expected_ids):
        """One read-only poll. Never tight-loop; missing items remain unknown."""
        name = receipt.get('name') if self.provider == 'gemini' else receipt.get('id')
        prefix = 'batches/' if self.provider == 'gemini' else 'batch_'
        if not isinstance(name, str) or not name.startswith(prefix):
            raise ValueError('batch receipt missing provider ID')
        state = json.loads(self.api(name if self.provider == 'gemini' else 'batches/' + name))
        if self.provider == 'gemini':
            if not state.get('done'):
                return {'done': False, 'snapshot': state}
            results = []
            inline = state.get('response', {}).get('inlinedResponses', [])
            for position, item in enumerate(inline):
                key = item.get('metadata', {}).get('key')
                # Google documents inline results in request order. IDs win;
                # order is used only for the documented inline-response shape.
                if key is None and position < len(expected_ids):
                    key = expected_ids[position]
                results.append({'custom_id': key, 'body': item.get('response'),
                                'error': item.get('error'), 'http_status': None})
        else:
            if state.get('status') not in ('completed', 'failed', 'expired', 'cancelled'):
                return {'done': False, 'snapshot': state}
            results = []
            for field in ('output_file_id', 'error_file_id'):
                if state.get(field):
                    for line in self.api('files/' + state[field] + '/content').splitlines():
                        if line.strip():
                            item = json.loads(line)
                            response = item.get('response') or {}
                            results.append({'custom_id': item.get('custom_id'), 'body': response.get('body'),
                                            'http_status': response.get('status_code'), 'error': item.get('error')})
        return {'done': True, 'snapshot': state, 'results': results}
