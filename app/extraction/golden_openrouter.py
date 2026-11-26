"""Minimal OpenRouter chat-completions transport for the golden-20 run.

Kept separate from app/extraction/providers.py so the existing pipeline's
request shape is untouched. One OpenRouterClient holds the settings that stay
the same for every chunk (model, key, output limit, reasoning, timeout), so a
call only needs the prompt and the schema.

It sends exactly the settings the evaluation needs: temperature 0, a strict
JSON-schema response format, provider.require_parameters so a model that
cannot honor the schema fails instead of silently downgrading, and the
reasoning setting from the config (effort "none" by default, so hidden
reasoning tokens are off where the model allows it).

No retries here: the runner decides. A 429 stops the whole run immediately.
"""
from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError
from urllib.request import Request, build_opener

DEFAULT_URL = 'https://openrouter.ai/api/v1/chat/completions'


class GoldenCallError(Exception):
    """One failed HTTP attempt. status and retry_after are evidence, not advice."""

    def __init__(self, status: Optional[int], detail: str,
                 retry_after: Optional[float] = None) -> None:
        super().__init__(detail)
        self.status = status
        self.retry_after = retry_after


class OpenRouterClient:
    def __init__(self, model: str, api_key: str, max_output_tokens: int,
                 reasoning: Optional[Dict[str, Any]] = None, timeout: float = 120,
                 opener: Any = None, url: str = DEFAULT_URL) -> None:
        self.model = model
        self.api_key = api_key.strip()
        self.max_output_tokens = max_output_tokens
        self.reasoning = reasoning
        self.timeout = timeout
        self.opener = opener or build_opener()
        self.url = url

    def request_body(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        body = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0,
            'max_tokens': self.max_output_tokens,
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'phase1_extraction',
                    'strict': True,
                    'schema': schema,
                },
            },
            'provider': {'require_parameters': True},
        }
        if self.reasoning:
            body['reasoning'] = self.reasoning
        return body

    def build_request(self, prompt: str, schema: Dict[str, Any]) -> Request:
        body = self.request_body(prompt, schema)
        return Request(self.url, data=json.dumps(body).encode(), method='POST', headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.api_key,
            # Documented opt-in (default disabled) that puts routing metadata on
            # the response as openrouter_metadata, so the metrics log can record
            # the upstream provider that actually served the request.
            # https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
            'X-OpenRouter-Metadata': 'enabled',
        })

    def complete(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """One chat completion. Returns the parsed envelope plus measured latency."""
        if not self.api_key:
            raise GoldenCallError(None, 'provider key missing; no request sent')
        request = self.build_request(prompt, schema)
        started = time.monotonic()
        raw, status = self.send(request)
        latency_ms = (time.monotonic() - started) * 1000
        return self.parse(raw, status, latency_ms)

    def send(self, request: Request) -> tuple:
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return response.read().decode('utf-8'), response.status
        except HTTPError as error:
            raise self.http_failure(error) from None
        except (OSError, socket.error) as error:
            raise GoldenCallError(None, 'transport failure: ' + str(error)) from None

    def http_failure(self, error: HTTPError) -> GoldenCallError:
        retry_after = None
        header = error.headers.get('Retry-After') if error.headers else None
        if header is not None:
            try:
                retry_after = float(header)
            except ValueError:
                retry_after = None
        try:
            detail = error.read().decode('utf-8')[:500]
        except Exception:
            detail = ''
        return GoldenCallError(error.code, detail or 'HTTP ' + str(error.code), retry_after)

    def parse(self, raw: str, status: int, latency_ms: float) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
        except ValueError:
            raise GoldenCallError(status, 'response was not JSON; not retried') from None
        text = None
        try:
            text = data['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            pass
        return {
            'text': text,
            'usage': data.get('usage'),
            'provider': data.get('provider'),
            'openrouter_metadata': data.get('openrouter_metadata'),
            'http_status': status,
            'latency_ms': latency_ms,
        }
