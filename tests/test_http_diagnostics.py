"""Safe HTTP failure diagnostics on the current JSONProvider.

The retired pilot-CLI diagnostic rendering and download handling are
quarantined in legacy_anthropic_tests/. What survives here: a failure carries
its status and a parsed Retry-After, never server prose or response bodies.
"""
import io
import json
from urllib.error import HTTPError

import pytest

from app.extraction.providers import JSONProvider, CallFailure, NoRedirect
from tests.test_extraction_client import RecordingOpener, http_error


def client(monkeypatch, *answers, provider='gemini'):
    monkeypatch.setenv('GEMINI_API_KEY' if provider == 'gemini' else 'OPENAI_API_KEY', 'k-secret')
    return JSONProvider(provider, 'm1', 128, opener=RecordingOpener(*answers))


@pytest.mark.parametrize('code', [400, 401, 403, 404, 429, 500, 503])
def test_http_failure_preserves_status_and_kind(monkeypatch, code):
    c = client(monkeypatch, http_error(code))
    with pytest.raises(CallFailure) as caught:
        c.call('p', {'type': 'object'})
    assert caught.value.kind == 'http_failure'
    assert caught.value.status == code


def test_server_prose_never_leaks_into_the_error(monkeypatch):
    body = b'{"error": {"message": "internal trace: secret-path/xyx", "status": "INTERNAL"}}'
    error = HTTPError('https://example.invalid', 500, 'oops', {}, io.BytesIO(body))
    c = client(monkeypatch, error)
    with pytest.raises(CallFailure) as caught:
        c.call('p', {'type': 'object'})
    text = str(caught.value)
    assert 'secret-path' not in text and 'INTERNAL' not in text and 'oops' not in text


def test_malformed_error_body_does_not_hide_status(monkeypatch):
    error = HTTPError('https://example.invalid', 429, 'Rate limited', {}, io.BytesIO(b'\xff\xfe not json'))
    c = client(monkeypatch, error)
    with pytest.raises(CallFailure) as caught:
        c.call('p', {'type': 'object'})
    assert caught.value.status == 429


def test_unparseable_retry_after_is_flagged_not_followed(monkeypatch):
    error = HTTPError('https://example.invalid', 429, 'x', {'Retry-After': 'whenever'}, io.BytesIO(b''))
    c = client(monkeypatch, error)
    with pytest.raises(CallFailure) as caught:
        c.call('p', {'type': 'object'})
    assert caught.value.kind == 'http_failure_retry_after_unparseable'
    assert caught.value.retry_after is None


def test_transport_wraps_oserror_once(monkeypatch):
    c = client(monkeypatch, OSError('connection reset by peer'))
    with pytest.raises(CallFailure) as caught:
        c.call('p', {'type': 'object'})
    assert caught.value.kind == 'transport_unknown'
    assert caught.value.__cause__ is None  # one clean wrap, no chained prose


def test_default_opener_refuses_redirects():
    c = JSONProvider('gemini', 'm1', 128)
    assert any(isinstance(handler, NoRedirect) for handler in c.opener.handlers)


def test_batch_control_failure_is_marked_not_confused_with_sync(monkeypatch):
    c = client(monkeypatch, http_error(503))
    with pytest.raises(CallFailure) as caught:
        c.api('batches/abc')
    assert caught.value.kind == 'batch_http_failure'
    assert caught.value.status == 503


def test_batch_control_rejects_untrusted_resource_ids(monkeypatch):
    c = client(monkeypatch)
    for bad in ('../secrets', 'batches/x y', 'batches/x?key=1', ''):
        with pytest.raises(ValueError, match='invalid provider resource ID'):
            c.api(bad)


def test_suite_network_guard_blocks_external_sockets():
    """Active proof, not structure: an external connection attempt fails here."""
    import socket as real_socket
    with pytest.raises(RuntimeError, match='Offline tests forbid external'):
        real_socket.socket().connect(('203.0.113.1', 443))
    with pytest.raises(RuntimeError, match='Offline tests forbid external'):
        real_socket.getaddrinfo('example.com', 443)


def test_suite_network_guard_reaches_python_subprocesses():
    """The guard is inherited by child Python processes via sitecustomize."""
    import subprocess
    import sys
    code = ("import socket\n"
            "try:\n"
            "    socket.socket().connect(('203.0.113.1', 443))\n"
            "    print('UNGUARDED')\n"
            "except RuntimeError:\n"
            "    print('GUARDED')\n")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.stdout.strip() == 'GUARDED'
