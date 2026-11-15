"""The connection check reports plain verdicts and never sends anything."""
import importlib.util
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parent.parent / 'scripts' / 'check_connection.py'
    spec = importlib.util.spec_from_file_location('check_connection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module(monkeypatch):
    loaded = load_module()
    calls = []

    def fake_resolve(name, port):
        calls.append(('resolve', name))
        return [(2, 1, 6, '', ('142.250.0.1', 443))]

    def fake_connect(address, timeout=None):
        calls.append(('connect', address[0]))

        class Bare:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Bare()

    monkeypatch.setattr('socket.getaddrinfo', fake_resolve)
    monkeypatch.setattr('socket.create_connection', fake_connect)
    return loaded, calls


def test_all_ready_reports_ready_and_checks_every_host(module, capsys):
    loaded, calls = module
    assert loaded.main() == 0
    out = capsys.readouterr().out
    assert 'Connection looks ready' in out
    assert out.count('resolves') == 2 and out.count('opens') == 2
    assert 'no request sent, no key used' in out
    assert calls == [('resolve', 'generativelanguage.googleapis.com'),
                     ('connect', 'generativelanguage.googleapis.com'),
                     ('resolve', 'api.openai.com'),
                     ('connect', 'api.openai.com')]


def test_dns_failure_skips_the_connection_and_says_nothing_was_sent(module, monkeypatch, capsys):
    loaded, calls = module

    def failing(name, port):
        raise OSError(11001, 'getaddrinfo failed')

    monkeypatch.setattr('socket.getaddrinfo', failing)
    assert loaded.main() == 1
    out = capsys.readouterr().out
    assert 'PROBLEM: cannot look up generativelanguage.googleapis.com (OSError errno 11001)' in out
    assert 'Nothing was sent' in out
    assert 'Connection is NOT ready' in out
    assert ('connect', 'generativelanguage.googleapis.com') not in calls  # no doomed attempt


def test_blocked_connection_is_named_separately(module, monkeypatch, capsys):
    loaded, calls = module

    def refused(address, timeout=None):
        raise ConnectionRefusedError(111, 'Connection refused')

    monkeypatch.setattr('socket.create_connection', refused)
    assert loaded.main() == 1
    out = capsys.readouterr().out
    assert 'resolves but the connection will not open (ConnectionRefusedError errno 111)' in out
    assert 'firewall' in out
