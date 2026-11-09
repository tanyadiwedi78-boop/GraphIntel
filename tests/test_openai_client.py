"""Fake HTTP server only. No real API key, network, or OpenAI call.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.extraction import openai_client
from app.extraction.client import ChunkFailedError, ProviderError
from app.extraction.openai_client import OpenAIClient
from app.graph.ontology import load_ontology

FAKE_KEY = "fake-key-for-tests"


def openai_response(text='{"entities": [], "relationships": []}', finish_reason="stop"):
    return {
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 20,
                  "total_tokens": 50,
                  "completion_tokens_details": {"reasoning_tokens": 12},
                  "prompt_tokens_details": {"cached_tokens": 5}},
    }


@pytest.fixture
def server():
    state = {"requests": [], "status": 200, "body": openai_response()}

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
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(openai_client, "ENDPOINT", url)
    return OpenAIClient(load_ontology("configs/ontology.json"), model_id="test-model-1",
                        allow_paid=True,
                        max_requests_per_minute=1000, max_tokens_per_minute=1000000)


def test_constructor_makes_no_call_and_request_carries_schema(client, server):
    url, state = server
    assert state["requests"] == []
    assert client("test prompt") == '{"entities": [], "relationships": []}'
    path, headers, request = state["requests"][0]
    sent = {k.lower(): v for k, v in headers.items()}
    assert sent["authorization"] == "Bearer " + FAKE_KEY
    assert request["model"] == "test-model-1"
    assert request["messages"] == [{"role": "user", "content": "test prompt"}]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["schema"] == client.schema
    assert request["response_format"]["json_schema"]["strict"] is False
    assert request["max_completion_tokens"] == 1024
    # Model-neutral by default: no reasoning effort is sent unless configured,
    # because models differ and this build invents no per-model defaults.
    assert "reasoning_effort" not in request
    assert request["stream"] is False
    assert client.last_metrics["provider"] == "openai"
    assert client.last_metrics["prompt_tokens"] == 30
    assert client.last_metrics["completion_tokens"] == 20
    assert client.last_metrics["thought_tokens"] == 12
    assert client.last_metrics["cached_tokens"] == 5
    assert client.last_metrics["output_tokens_per_sec_end_to_end"] is not None
    assert client.last_metrics["latency_ms"] >= 0


def test_reasoning_effort_is_sent_only_when_configured(server, monkeypatch):
    url, state = server
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(openai_client, "ENDPOINT", url)
    reasoning_client = OpenAIClient(load_ontology("configs/ontology.json"),
                                    model_id="test-model-1", allow_paid=True,
                                    reasoning_effort="low",
                                    max_requests_per_minute=1000,
                                    max_tokens_per_minute=1000000)
    reasoning_client("p")
    path, headers, request = state["requests"][0]
    assert request["reasoning_effort"] == "low"


def test_paid_approval_still_reports_unknown_cost_not_zero(client, server):
    client("p")
    assert client.last_metrics["cost_basis"] == "paid_approved_by_flag"
    assert client.last_metrics["estimated_cost_usd"] is None


def test_unapproved_client_blocks_before_any_request(server, monkeypatch):
    url, state = server
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(openai_client, "ENDPOINT", url)
    blocked = OpenAIClient(load_ontology("configs/ontology.json"), model_id="test-model-1",
                           max_requests_per_minute=1000, max_tokens_per_minute=1000000)
    with pytest.raises(ProviderError) as info:
        blocked("p")
    assert info.value.status_code == 402
    assert blocked.last_metrics["error_type"] == "cost_blocked"
    assert blocked.cost_mode() == "paid_not_approved"
    assert state["requests"] == []


def test_paid_approval_needs_explicit_rate_limits(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    with pytest.raises(ValueError) as info:
        OpenAIClient(load_ontology("configs/ontology.json"), model_id="test-model-1",
                     allow_paid=True)
    assert "max_requests_per_minute" in str(info.value)


def test_missing_key_is_a_setup_error(server, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError) as info:
        OpenAIClient(load_ontology("configs/ontology.json"), model_id="test-model-1",
                     allow_paid=True,
                     max_requests_per_minute=1000, max_tokens_per_minute=1000000)
    assert "OPENAI_API_KEY" in str(info.value)


def test_missing_model_id_is_a_setup_error(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    with pytest.raises(ValueError) as info:
        OpenAIClient(load_ontology("configs/ontology.json"), model_id="",
                     allow_paid=True,
                     max_requests_per_minute=1000, max_tokens_per_minute=1000000)
    assert "model_id" in str(info.value)


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
    state["body"] = openai_response(text='{"entities": [', finish_reason="length")
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


def test_model_name_marks_openai_and_differs_per_model(client):
    assert client.model_name.startswith("openai:test-model-1:")
    assert "gemini" not in client.model_name
    assert "groq" not in client.model_name
    other = OpenAIClient(load_ontology("configs/ontology.json"), model_id="test-model-2",
                         allow_paid=True,
                         max_requests_per_minute=1000, max_tokens_per_minute=1000000)
    assert other.model_name != client.model_name
