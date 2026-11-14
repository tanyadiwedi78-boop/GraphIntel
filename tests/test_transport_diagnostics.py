"""Named transport failure categories with sanitized, actionable detail.

A failed attempt used to collapse every socket/ssl problem into
'transport_unknown', which left a real run undiagnosable. Now a recognized
cause is named (dns, tls, timeout, refused, reset) and the attempt record
carries only the exception class and errno: never message prose, which can
contain URLs, provider text or local paths. Everything unrecognized stays
honestly unknown. Nothing here touches the network; openers and resolvers
are fakes.
"""
import json
import socket
import ssl
from urllib.error import URLError

import pytest

from app.extraction.pipeline import Pipeline, prepare
from tests.conftest import FakeProvider, demo_chunks
from tests.test_extraction_client import RecordingOpener


# Import provider symbols inside each test: test_model.py reloads the
# providers module mid-suite, so module-level references can go stale.


def client(monkeypatch, *answers):
    from app.extraction.providers import JSONProvider
    monkeypatch.setenv('GEMINI_API_KEY', 'k-secret')
    return JSONProvider('gemini', 'm1', 128, opener=RecordingOpener(*answers))


@pytest.mark.parametrize('cause,kind', [
    (socket.gaierror(11001, 'getaddrinfo failed'), 'transport_dns'),
    (ssl.SSLError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed'), 'transport_tls'),
    (socket.timeout('timed out'), 'transport_timeout'),
    (TimeoutError(110, 'Connection timed out'), 'transport_timeout'),
    (ConnectionRefusedError(111, 'Connection refused'), 'transport_refused'),
    (ConnectionResetError(104, 'Connection reset by peer'), 'transport_reset'),
    (OSError('something else entirely'), 'transport_unknown'),
])
def test_recognized_causes_are_named_and_unknown_stays_unknown(monkeypatch, cause, kind):
    from app.extraction.providers import CallFailure
    c = client(monkeypatch, cause)
    with pytest.raises(CallFailure) as caught:
        c.call('p', {'type': 'object'})
    assert caught.value.kind == kind
    assert caught.value.status is None
    assert caught.value.detail == {'exception': type(cause).__name__, 'errno': cause.errno}


def test_urlerror_unwraps_to_the_real_socket_cause(monkeypatch):
    from app.extraction.providers import CallFailure
    wrapped = URLError(socket.gaierror(-2, 'Name or service not known'))
    c = client(monkeypatch, wrapped)
    with pytest.raises(CallFailure) as caught:
        c.call('p', {'type': 'object'})
    assert caught.value.kind == 'transport_dns'
    assert caught.value.detail['exception'] == 'gaierror'


def test_urlerror_wrapping_tls_is_tls_not_unknown(monkeypatch):
    from app.extraction.providers import CallFailure
    wrapped = URLError(ssl.SSLError(1, 'certificate verify failed'))
    c = client(monkeypatch, wrapped)
    with pytest.raises(CallFailure) as caught:
        c.call('p', {'type': 'object'})
    assert caught.value.kind == 'transport_tls'


def test_detail_never_carries_message_prose(monkeypatch):
    from app.extraction.providers import CallFailure
    cause = socket.gaierror(11001, 'lookup secret-host.internal/path failed')
    c = client(monkeypatch, cause)
    with pytest.raises(CallFailure) as caught:
        c.call('p', {'type': 'object'})
    failure = caught.value
    assert str(failure) == 'transport_dns'  # the exception text is only the kind
    assert 'secret-host' not in json.dumps(failure.detail)
    assert failure.__cause__ is None  # one clean wrap, no chained prose


def test_check_dns_reports_the_problem_and_sends_nothing(monkeypatch):
    from app.extraction.providers import JSONProvider
    monkeypatch.setenv('GEMINI_API_KEY', 'k-secret')
    c = JSONProvider('gemini', 'm1', 128, opener=RecordingOpener())

    def failing(host, port):
        raise socket.gaierror(11001, 'getaddrinfo failed')

    problem = c.check_dns(resolver=failing)
    assert 'generativelanguage.googleapis.com' in problem
    assert 'gaierror' in problem and '11001' in problem
    assert 'no request was sent' in problem
    assert c.opener.requests == []  # the check itself never opens a connection

    def resolving(host, port):
        return [(2, 1, 6, '', ('142.250.0.1', 443))]

    assert c.check_dns(resolver=resolving) is None


def test_openai_host_is_checked_by_name(monkeypatch):
    from app.extraction.providers import JSONProvider
    monkeypatch.setenv('OPENAI_API_KEY', 'k-secret')
    c = JSONProvider('openai', 'm1', 128, opener=RecordingOpener())
    seen = []
    c.check_dns(resolver=lambda host, port: seen.append(host) or [(2, 1, 6, '', ('1.2.3.4', 443))])
    assert seen == ['api.openai.com']


def test_named_transport_failure_persists_detail_and_stops_the_run(tmp_path, ontology):
    # The exact class object Pipeline catches, whatever reload order ran.
    from app.extraction.pipeline import CallFailure
    chunks = demo_chunks(2)
    detail = {'exception': 'gaierror', 'errno': 11001}
    primary = FakeProvider([CallFailure('transport_dns', detail=detail),
                            CallFailure('transport_dns', detail=detail)])
    fallback = FakeProvider(['unused'], 'openai')
    job = tmp_path / 'dns'
    prepare(job, chunks, ontology, primary.recipe, fallback.recipe)
    p = Pipeline(job, ontology, primary, fallback, sleep=lambda seconds: None)
    p.run(10, 100, True, True)
    assert primary.calls == 1 and fallback.calls == 0  # never retried, never fallen back
    saved = json.loads((job / 'attempt-1-result.json').read_text())
    assert saved['outcome'] == 'transport_dns'
    assert saved['transport_detail'] == detail
    checkpoints = list(job.glob('chunk-*.json'))
    assert len(checkpoints) == 1  # the whole run stopped at the broken network
    record = json.loads(checkpoints[0].read_text())
    assert record['terminal_outcome'] == 'transport_dns'
    assert record['transport_detail'] == detail
    assert record['accepted'] is False
    # A resume does not resubmit the reconciled chunk-1 attempt; it moves to
    # the remaining chunk, which fails the same way while DNS is still broken.
    p.run(10, 100, True, True)
    assert primary.calls == 2 and fallback.calls == 0
    resumed = json.loads((job / 'attempt-2-result.json').read_text())
    assert resumed['chunk_id'] == 'c2' and resumed['outcome'] == 'transport_dns'


def test_helpers_classify_without_a_client():
    from app.extraction.providers import transport_detail, transport_kind
    assert transport_kind(socket.gaierror()) == 'transport_dns'
    assert transport_detail(OSError(5, 'io'))['errno'] == 5
    assert transport_kind(Exception('not transport')) == 'transport_unknown'
