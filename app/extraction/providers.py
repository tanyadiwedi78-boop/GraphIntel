"""One HTTP request per call, no hidden SDK retries or provider repair calls."""
import json
import math
import http.client
import os
import re
import socket
import ssl
import time
from typing import Any, Optional
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
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
    def __init__(self, kind, status=None, retry_after=None, detail=None, provider_codes=None):
        super().__init__(kind)  # Never echo provider bodies, headers or secrets.
        self.kind = kind
        self.status = status
        self.retry_after = retry_after
        self.detail = detail
        self.provider_codes = provider_codes


def provider_error_codes(error):
    # Google's error body carries machine-readable codes beside prose. Keep
    # only the codes: never the message text, which can echo request data.
    try:
        envelope = json.loads(error.read().decode('utf-8'))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    detail = envelope.get('error') if isinstance(envelope, dict) else None
    if not isinstance(detail, dict):
        return None
    codes = {}
    if isinstance(detail.get('status'), str):
        codes['status'] = detail['status']
    reasons = []
    for entry in detail.get('details') or []:
        if isinstance(entry, dict) and isinstance(entry.get('reason'), str):
            reasons.append(entry['reason'])
    if reasons:
        codes['reasons'] = reasons
    return codes or None


def transport_cause(error):
    # urllib wraps socket/ssl failures in URLError; unwrap to the real cause.
    if isinstance(error, URLError):
        return error.reason
    return error


def transport_kind(cause):
    # Named categories for the failures we can recognize. Anything else stays
    # honestly unknown instead of guessing a cause.
    if isinstance(cause, socket.gaierror):
        return 'transport_dns'
    if isinstance(cause, ssl.SSLError):
        return 'transport_tls'
    if isinstance(cause, (socket.timeout, TimeoutError)):
        return 'transport_timeout'
    if isinstance(cause, ConnectionRefusedError):
        return 'transport_refused'
    if isinstance(cause, ConnectionResetError):
        return 'transport_reset'
    return 'transport_unknown'


def transport_detail(cause):
    # Only the exception class and errno: enough to act on, never message
    # prose, which can contain URLs, provider text or local paths.
    return {'exception': type(cause).__name__, 'errno': getattr(cause, 'errno', None)}


class JSONProvider:
    def __init__(self, provider: str, model: str, max_output_tokens: int,
                 timeout: float = 120, opener: Optional[Any] = None) -> None:
        # OpenRouter model ids look like "vendor/name:free", so they keep / and :.
        pattern = r'[A-Za-z0-9._/:~-]+' if provider == 'openrouter' else r'[A-Za-z0-9._-]+'
        if provider not in ('gemini', 'openai', 'openrouter') or not re.fullmatch(pattern, model):
            raise ValueError('explicit provider and model required')
        if type(max_output_tokens) is not int or max_output_tokens <= 0 or isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
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
        if not os.environ.get(self.key_variable(), '').strip():
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
            raise CallFailure(kind, error.code, retry_after,
                              provider_codes=provider_error_codes(error)) from None
        except (OSError, EOFError, http.client.HTTPException) as error:
            cause = transport_cause(error)
            raise CallFailure(transport_kind(cause), detail=transport_detail(cause)) from None
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

    def check_dns(self, resolver=None):
        """Read-only name resolution check before any live request is claimed.

        Returns None when the provider host resolves, otherwise a plain
        problem sentence. A failure here means the network or DNS is broken;
        running would only burn an attempt on a guaranteed failure.
        """
        host = self.endpoint()[0].split('/')[2]
        resolve = resolver or socket.getaddrinfo
        try:
            resolve(host, 443)
        except OSError as error:
            return ('cannot resolve ' + host + ' (' + type(error).__name__ + ' errno '
                    + str(getattr(error, 'errno', None)) + '): the network or DNS lookup '
                    'is failing. Fix connectivity first; no request was sent.')
        return None

    def key_variable(self):
        if self.provider == 'gemini':
            return 'GEMINI_API_KEY'
        if self.provider == 'openrouter':
            return 'OPENROUTER_API_KEY'
        return 'OPENAI_API_KEY'

    def endpoint(self):
        key = os.environ.get(self.key_variable(), '').strip()
        if self.provider == 'gemini':
            return ('https://generativelanguage.googleapis.com/v1beta/models/' + self.model + ':generateContent',
                    {'x-goog-api-key': key})
        if self.provider == 'openrouter':
            return ('https://openrouter.ai/api/v1/chat/completions',
                    {'Authorization': 'Bearer ' + key})
        return 'https://api.openai.com/v1/responses', {'Authorization': 'Bearer ' + key}

    def body(self, prompt, schema):
        # Same common JSON schema/prompt for sync and native batch. Only the
        # provider's documented request envelope differs. No tool/search calls.
        if self.provider == 'gemini':
            return {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                    'generationConfig': {'maxOutputTokens': self.max_output_tokens,
                        # Documented wire value is the enum name APPLICATION_JSON,
                        # not the MIME string (aistudio.google.com generate-content
                        # reference: responseFormat.text.mimeType string enum).
                        'responseFormat': {'text': {'mimeType': 'APPLICATION_JSON', 'schema': schema}}}}
        if self.provider == 'openrouter':
            return {'model': self.model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': self.max_output_tokens,
                    'response_format': {'type': 'json_schema',
                                        'json_schema': {'name': 'extraction', 'strict': True,
                                                        'schema': schema}},
                    # Route only to endpoints that can enforce the schema.
                    'provider': {'require_parameters': True}}
        return {'model': self.model, 'input': prompt, 'store': False,
                'max_output_tokens': self.max_output_tokens,
                'text': {'format': {'type': 'json_schema', 'name': 'extraction',
                                    'strict': True, 'schema': schema}}}

    def output_text(self, data: dict) -> str:
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
        elif self.provider == 'openrouter':
            choices = data.get('choices')
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError('incomplete or refused')
            if choices[0].get('finish_reason') not in (None, 'stop'):
                raise ValueError('incomplete or refused')
            message = choices[0].get('message') or {}
            if message.get('refusal'):
                raise ValueError('refused')
            if not isinstance(message.get('content'), str):
                raise ValueError('exactly one structured output required')
            texts.append(message['content'])
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
        if self.provider == 'openrouter':
            raise ValueError('native batch is not wired for openrouter; prepare a sync job')
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
        except (OSError, EOFError, http.client.HTTPException) as error:
            cause = transport_cause(error)
            raise CallFailure('batch_transport_unknown', detail=transport_detail(cause)) from None

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
