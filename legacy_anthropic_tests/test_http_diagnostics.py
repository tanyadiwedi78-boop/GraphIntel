"""No live provider calls. Lost past responses cannot be reconstructed."""
import io
import json
from urllib.error import HTTPError

import pytest

from app.extraction import anthropic_batch as batch
from tests.test_anthropic_batch import FakeAPI, ontology
from tests.test_single_request_pilot import pilot, send


def http_error(code=400, body=None, headers=None):
    if body is None:
        body = {'error': {'type': 'invalid_request_error',
                          'message': 'secret-key and private prompt must not appear'},
                'request_id': 'req_body123'}
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return HTTPError('https://api.anthropic.com/v1/messages/batches', code,
                     'secret reason', headers or {}, io.BytesIO(raw))


@pytest.mark.parametrize('code', [400, 402, 403, 429, 500, 504, 529])
def test_submit_http_failure_preserves_claim_and_blocks_retry(pilot, code):
    folder, plan = pilot
    before = (folder / 'plan.json').read_bytes()
    api = FakeAPI([http_error(code)])
    with pytest.raises(batch.ProviderHTTPFailure):
        send(pilot, api)
    diagnostic = json.loads((folder / 'submission-error.json').read_text())
    assert diagnostic['http_status'] == code
    assert diagnostic['request_id'] == 'req_body123'
    assert diagnostic['method'] == 'POST'
    assert diagnostic['path'] == batch.BATCH_PATH
    assert diagnostic['submission_outcome'] == 'unreconciled_do_not_retry'
    assert 'secret' not in json.dumps(diagnostic)
    assert 'private prompt' not in json.dumps(diagnostic)
    assert (folder / 'submission-claim.json').exists()
    assert not (folder / 'receipt.json').exists()
    assert not (folder / 'submission-response.txt').exists()
    assert (folder / 'plan.json').read_bytes() == before
    with pytest.raises(ValueError, match='already has'):
        send(pilot, api)
    assert len(api.calls) == 1


@pytest.mark.parametrize('raw', [b'not JSON secret', b'x' * 70000,
                               b'[]', b'{"error":null}',
                               b'{"error":{"type":["secret"]}}'])
def test_malformed_error_does_not_hide_status_or_leak(raw):
    diagnostic = batch.safe_http_failure(http_error(500, raw), 'POST', batch.BATCH_PATH).diagnostic
    assert diagnostic['http_status'] == 500
    assert diagnostic['error_type'] is None
    assert 'secret' not in json.dumps(diagnostic)


def test_whitelist_not_server_controlled_prose():
    error = http_error(429, {'error': {'type': 'rate_limit_error',
        'message': 'secret', 'details': {'error_code': 'enforced_spend_limit_reached'}},
        'request_id': 'secret'}, {'request-id': 'req_header123', 'x-api-key': 'secret'})
    diagnostic = batch.safe_http_failure(error, 'GET', batch.BATCH_PATH).diagnostic
    assert diagnostic['request_id'] == 'req_header123'
    assert diagnostic['error_type'] == 'rate_limit_error'
    assert diagnostic['error_code'] == 'enforced_spend_limit_reached'
    assert 'secret' not in json.dumps(diagnostic)


def test_unknown_type_and_untrusted_ids_omitted():
    diagnostic = batch.safe_http_failure(http_error(400,
        {'error': {'type': 'private-prompt', 'details': {'error_code': 'secret'}},
         'request_id': 'req_bad\nsecret'}, {'request-id': 'req_bad\nsecret'}),
        'POST', batch.BATCH_PATH).diagnostic
    assert diagnostic['request_id'] is None
    assert diagnostic['error_type'] is None
    assert diagnostic['error_code'] is None


def test_transport_wraps_http_error_once(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'secret-key')
    api = batch.BatchAPI()
    calls = []
    def fail(request, timeout):
        calls.append(request)
        raise http_error()
    monkeypatch.setattr(api.opener, 'open', fail)
    with pytest.raises(batch.ProviderHTTPFailure) as caught:
        api.request('GET', batch.BATCH_PATH + '?limit=10')
    assert len(calls) == 1
    assert caught.value.diagnostic['method'] == 'GET'
    assert caught.value.diagnostic['path'].endswith('?limit=10')


def test_cli_reports_safe_diagnostics(pilot, monkeypatch, capsys):
    folder, plan = pilot
    failure = batch.safe_http_failure(http_error(), 'POST', batch.BATCH_PATH)
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(batch, 'submit_job', fail)
    assert batch.main(['submit', '--job-dir', str(folder), '--approve-plan', plan['plan_id'],
                       '--single-request-actual-charge', '--allow-paid',
                       '--approve-chunk-id', 'approved-source']) == 1
    output = capsys.readouterr().out
    assert '400' in output and 'req_body123' in output
    assert 'secret' not in output and 'private prompt' not in output


def test_failed_diagnostic_write_does_not_hide_http_error(pilot, monkeypatch):
    original_write = batch.write_new
    def fail_diagnostic(path, value):
        if path.name == 'submission-error.json':
            raise OSError('private disk error text')
        return original_write(path, value)
    monkeypatch.setattr(batch, 'write_new', fail_diagnostic)
    with pytest.raises(batch.ProviderHTTPFailure) as caught:
        send(pilot, FakeAPI([http_error(500)]))
    assert caught.value.diagnostic['http_status'] == 500
    assert caught.value.diagnostic['diagnostic_save_failed'] is True
    assert (pilot[0] / 'submission-claim.json').exists()
    assert 'private disk' not in str(caught.value)


def test_download_http_error_is_safe_and_no_partial_created(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'secret')
    api = batch.BatchAPI()
    def fail(request, timeout):
        raise http_error(403)
    monkeypatch.setattr(api.opener, 'open', fail)
    with pytest.raises(batch.ProviderHTTPFailure) as caught:
        api.download_results(batch.BATCH_PATH + '/msgbatch_test/results', tmp_path / 'results.jsonl')
    assert caught.value.diagnostic['http_status'] == 403
    assert caught.value.diagnostic['submission_outcome'] == 'not_a_submission_attempt'
    assert not list(tmp_path.iterdir())
