"""Fake HTTP server only. No real API key, network, or Groq call.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.extraction import groq_client
from app.extraction.client import ChunkFailedError, ProviderError
from app.extraction.groq_client import GroqClient
from app.graph.ontology import load_ontology

FAKE_KEY = "fake-key-for-tests"


def groq_response(text='{"entities": [], "relationships": []}', finish_reason="stop"):
    return {
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 20,
                  "total_tokens": 50},
    }


@pytest.fixture
def server():
    state = {"requests": [], "status": 200, "body": groq_response()}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, dict(self.headers), body))
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            if state["status"] == 429:
                self.send_header("Retry-After", "7")
            self.end_headers()
            self.wfile.write(json.dumps(state["body"]).encode("utf-8"))

        def log_message(self, format, *args):
            pass

    http = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(http.server_port), state
    finally:
        http.shutdown()
        thread.join()
        http.server_close()


@pytest.fixture
def client(server, monkeypatch):
    url, state = server
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setattr(groq_client, "ENDPOINT", url)
    return GroqClient(load_ontology("configs/ontology.json"), assume_free=True,
                      max_requests_per_minute=1000, max_tokens_per_minute=1000000)


def test_constructor_makes_no_call_and_request_carries_schema(client, server):
    url, state = server
    assert state["requests"] == []
    assert client("test prompt") == '{"entities": [], "relationships": []}'
    path, headers, request = state["requests"][0]
    sent = {k.lower(): v for k, v in headers.items()}
    assert sent["authorization"] == "Bearer " + FAKE_KEY
    assert request["model"] == "openai/gpt-oss-20b"
    assert request["messages"] == [{"role": "user", "content": "test prompt"}]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["schema"] == client.schema
    assert request["response_format"]["json_schema"]["strict"] is False
    assert request["max_completion_tokens"] == 1024
    assert request["stream"] is False
    assert client.last_metrics["provider"] == "groq"
    assert client.last_metrics["prompt_tokens"] == 30
    assert client.last_metrics["completion_tokens"] == 20
    assert client.last_metrics["thought_tokens"] is None
    assert client.last_metrics["cached_tokens"] is None
    assert client.last_metrics["output_tokens_per_sec_end_to_end"] is not None
    assert client.last_metrics["latency_ms"] >= 0


def test_assume_free_reports_unknown_cost_not_zero(client, server):
    client("p")
    assert client.last_metrics["cost_basis"] == "assumed_free_unverified"
    assert client.last_metrics["estimated_cost_usd"] is None


def test_unconfigured_client_blocks_before_any_request(server, monkeypatch):
    url, state = server
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setattr(groq_client, "ENDPOINT", url)
    blocked = GroqClient(load_ontology("configs/ontology.json"))
    with pytest.raises(ProviderError) as info:
        blocked("p")
    assert info.value.status_code == 402
    assert blocked.last_metrics["error_type"] == "cost_blocked"
    assert blocked.cost_mode() == "unpriced_unknown"
    assert state["requests"] == []


def test_missing_key_is_a_setup_error(server, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError) as info:
        GroqClient(load_ontology("configs/ontology.json"), assume_free=True,
                   max_requests_per_minute=1000, max_tokens_per_minute=1000000)
    assert "GROQ_API_KEY" in str(info.value)


def test_http_error_status_and_retry_after(client, server):
    url, state = server
    state["status"] = 429
    state["body"] = {"error": {"message": "rate limited"}}
    with pytest.raises(ProviderError) as info:
        client("p")
    assert info.value.status_code == 429
    assert info.value.retry_after_seconds == 7
    assert "rate limited" not in str(info.value)
    assert client.last_metrics["error_type"] == "http_429"


def test_setup_error_status_passthrough(client, server):
    url, state = server
    state["status"] = 401
    state["body"] = {"error": {"message": "bad key"}}
    with pytest.raises(ProviderError) as info:
        client("p")
    assert info.value.status_code == 401


def test_cut_off_answer_is_a_content_failure(client, server):
    url, state = server
    state["body"] = groq_response(text='{"entities": [', finish_reason="length")
    with pytest.raises(ChunkFailedError) as info:
        client("p")
    assert "output token limit" in str(info.value)
    assert client.last_metrics["error_type"] == "output_cut_off"


def test_invalid_envelope_is_a_provider_error(client, server):
    url, state = server
    state["body"] = {"unexpected": True}
    with pytest.raises(ProviderError) as info:
        client("p")
    assert info.value.status_code == 502
    assert client.last_metrics["error_type"] == "response_envelope"


def test_model_name_marks_groq_and_differs_from_gemini(client):
    assert client.model_name.startswith("groq:openai/gpt-oss-20b:")
    assert "gemini" not in client.model_name
