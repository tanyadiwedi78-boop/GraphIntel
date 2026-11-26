"""Minimal OpenRouter chat-completions transport for the golden-20 run.

Kept separate from app/extraction/providers.py so the existing pipeline's
request shape is untouched. This transport sends exactly the settings the
evaluation needs: temperature 0, a strict JSON-schema response format,
provider.require_parameters so a model that cannot honor the schema fails
instead of silently downgrading, and the reasoning setting from the config
(effort "none" by default, so hidden reasoning tokens are off where the model
allows it).

No retries here: the runner decides. A 429 stops the whole run immediately.
"""
from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, Optional
from urllib.request import Request, build_opener
from urllib.error import HTTPError

DEFAULT_URL = 'https://openrouter.ai/api/v1/chat/completions'


class GoldenCallError(Exception):
    """One failed HTTP attempt. status and retry_after are evidence, not advice."""

    def __init__(self, status: Optional[int], detail: str,
                 retry_after: Optional[float] = None) -> None:
        super().__init__(detail)
        self.status = status
        self.retry_after = retry_after


def request_body(model: str, prompt: str, schema: Dict[str, Any],
                 max_output_tokens: int, reasoning: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0,
        'max_tokens': max_output_tokens,
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
    if reasoning:
        body['reasoning'] = reasoning
    return body


def complete(model: str, prompt: str, schema: Dict[str, Any], *,
             max_output_tokens: int, api_key: str,
             reasoning: Optional[Dict[str, Any]] = None,
             timeout: float = 120, opener: Any = None,
             url: str = DEFAULT_URL) -> Dict[str, Any]:
    """One chat completion. Returns the parsed envelope plus measured latency."""
    if not api_key.strip():
        raise GoldenCallError(None, 'provider key missing; no request sent')
    body = request_body(model, prompt, schema, max_output_tokens, reasoning)
    request = Request(url, data=json.dumps(body).encode(), method='POST', headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key.strip(),
        # Documented opt-in (default disabled) that puts routing metadata on the
        # response as openrouter_metadata, so the metrics log can record the
        # upstream provider that actually served the request.
        # https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
        'X-OpenRouter-Metadata': 'enabled',
    })
    opener = opener or build_opener()
    started = time.monotonic()
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read().decode('utf-8')
            status = response.status
    except HTTPError as error:
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
        raise GoldenCallError(error.code, detail or 'HTTP ' + str(error.code),
                              retry_after) from None
    except (OSError, socket.error) as error:
        raise GoldenCallError(None, 'transport failure: ' + str(error)) from None
    latency_ms = (time.monotonic() - started) * 1000
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
