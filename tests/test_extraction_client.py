"""Fake HTTP server only. No real API key, network, or Gemini call.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.extraction import client as gemini_client
from app.extraction.client import ChunkFailedError, GeminiClient, ProviderError, extraction_json_schema
from app.graph.ontology import load_ontology

FAKE_KEY = "fake-key-for-tests"


def gemini_response(text='{"entities": [], "relationships": []}', status="completed"):
    return {
        "status": status,
        "steps": [{"type": "model_output",
                   "content": [{"type": "text", "text": text}]}],
        "usage": {"total_input_tokens": 30, "total_output_tokens": 20,
                  "total_thought_tokens": 22, "total_cached_tokens": 4,
                  "total_tokens": 76},
    }


@pytest.fixture
def server():
    state = {"requests": [], "status": 200, "body": gemini_response()}

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
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(gemini_client, "ENDPOINT", url)
    return GeminiClient(load_ontology("configs/ontology.json"), assume_free=True,
                        max_requests_per_minute=1000, max_tokens_per_minute=1000000)


def test_schema_required_enums_and_documented_fields_only():
    ontology = load_ontology("configs/ontology.json")
    schema = extraction_json_schema(ontology)
    assert schema["required"] == ["entities", "relationships"]
    entity = schema["properties"]["entities"]["items"]
    relationship = schema["properties"]["relationships"]["items"]
    assert entity["properties"]["type"]["enum"] == list(ontology.entity_types)
    assert set(entity["required"]) == set(entity["properties"].keys())
    assert relationship["properties"]["relation"]["enum"] == sorted(ontology.relations)
    assert relationship["properties"]["confidence"]["maximum"] == 1
    assert set(relationship["required"]) == set(relationship["properties"].keys())


def test_constructor_makes_no_call_and_request_carries_schema(client, server):
    url, state = server
    assert state["requests"] == []
    assert client("test prompt") == '{"entities": [], "relationships": []}'
    path, headers, request = state["requests"][0]
    sent = {k.lower(): v for k, v in headers.items()}
    assert sent["x-goog-api-key"] == FAKE_KEY
    assert request["model"] == "gemini-3.5-flash-lite"
    assert request["input"] == "test prompt"
    assert request["response_format"]["mime_type"] == "application/json"
    assert request["response_format"]["schema"] == client.schema
    assert request["generation_config"]["max_output_tokens"] == 1024
    assert request["store"] is False
    assert client.last_metrics["prompt_tokens"] == 30
    assert client.last_metrics["completion_tokens"] == 20
    assert client.last_metrics["thought_tokens"] == 22
    assert client.last_metrics["cached_tokens"] == 4
    assert client.last_metrics["output_tokens_per_sec_end_to_end"] is not None
    assert client.last_metrics["latency_ms"] >= 0


def test_assume_free_reports_unknown_cost_not_zero(client, server):
    client("p")
    assert client.last_metrics["cost_basis"] == "assumed_free_unverified"
    assert client.last_metrics["estimated_cost_usd"] is None


def test_unconfigured_client_blocks_before_any_request(server, monkeypatch):
    url, state = server
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(gemini_client, "ENDPOINT", url)
    bare = GeminiClient(load_ontology("configs/ontology.json"))
    with pytest.raises(ProviderError) as caught:
        bare("p")
    assert caught.value.status_code == 402
    assert "not implemented" in str(caught.value)
    assert bare.last_metrics["error_type"] == "cost_blocked"
    assert bare.last_metrics["cost_basis"] == "unpriced_unknown"
    assert state["requests"] == []


def test_assume_free_without_limits_is_refused(server, monkeypatch):
    url, state = server
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(gemini_client, "ENDPOINT", url)
    ontology = load_ontology("configs/ontology.json")
    with pytest.raises(ValueError) as caught:
        GeminiClient(ontology, assume_free=True)
    assert "per-minute" in str(caught.value)
    with pytest.raises(ValueError):
        GeminiClient(ontology, assume_free=True, max_requests_per_minute=10)
    assert state["requests"] == []


def test_pacing_waits_before_sending(server, monkeypatch):
    url, state = server
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(gemini_client, "ENDPOINT", url)
    clock = {"now": 1000.0}
    sleeps = []
    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] = clock["now"] + seconds
    paced = GeminiClient(load_ontology("configs/ontology.json"), assume_free=True,
                         max_requests_per_minute=1, max_tokens_per_minute=1000000,
                         sleep_func=fake_sleep, now_func=lambda: clock["now"])
    paced("first")
    assert sleeps == []
    paced("second")
    assert sleeps == [1] * 60  # waited out the full 60-second window first
    assert paced.last_metrics["paced_wait_seconds"] == 60
    assert len(state["requests"]) == 2


def test_single_prompt_over_token_limit_fails_fast(server, monkeypatch):
    url, state = server
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(gemini_client, "ENDPOINT", url)
    paced = GeminiClient(load_ontology("configs/ontology.json"), assume_free=True,
                         max_requests_per_minute=1000, max_tokens_per_minute=4)
    with pytest.raises(ValueError) as caught:
        paced("0123456789")  # estimate 5 tokens > the 4-token limit
    assert "never pass" in str(caught.value)
    assert state["requests"] == []


def test_token_pacing_uses_conservative_estimate(server, monkeypatch):
    url, state = server
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(gemini_client, "ENDPOINT", url)
    clock = {"now": 1000.0}
    sleeps = []
    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] = clock["now"] + seconds
    paced = GeminiClient(load_ontology("configs/ontology.json"), assume_free=True,
                         max_requests_per_minute=1000, max_tokens_per_minute=9,
                         sleep_func=fake_sleep, now_func=lambda: clock["now"])
    paced("0123456789")  # estimate: 10 chars / 2 = 5 tokens
    assert paced.last_metrics["estimated_input_tokens"] == 5
    paced("0123456789")  # 5 + 5 would exceed the 9-token window
    assert sleeps == [1] * 60
    assert len(state["requests"]) == 2

def test_missing_usage_is_null_not_zero(client, server):
    url, state = server
    state["body"] = gemini_response()
    state["body"].pop("usage")
    client("p")
    assert client.last_metrics["prompt_tokens"] is None
    assert client.last_metrics["estimated_cost_usd"] is None


@pytest.mark.parametrize("status", [400, 429, 503])
def test_http_errors_have_one_attempt_and_safe_message(client, server, status):
    url, state = server
    state["status"] = status
    state["body"] = {"error": "arbitrary server body"}
    with pytest.raises(ProviderError) as caught:
        client("p")
    assert caught.value.status_code == status
    assert len(state["requests"]) == 1
    assert "arbitrary server body" not in str(caught.value)
    if status == 429:
        assert caught.value.retry_after_seconds == 7


def test_bad_envelope_is_not_a_valid_empty_answer(client, server):
    url, state = server
    state["body"] = {"not_steps": []}
    with pytest.raises(ProviderError):
        client("p")
    assert client.last_metrics["error_type"] == "response_envelope"


def test_cut_off_answer_fails_the_chunk(client, server):
    # Official docs: hitting max_tokens returns status "incomplete".
    url, state = server
    state["body"] = gemini_response(status="incomplete")
    with pytest.raises(ChunkFailedError) as caught:
        client("p")
    assert "output token limit" in str(caught.value)
    assert client.last_metrics["error_type"] == "output_cut_off"


def test_continuation_request_fails_the_chunk(client, server):
    # Official docs: "requires_continuation" needs another request with a
    # continuation_token; it is not an output-limit signal.
    url, state = server
    state["body"] = gemini_response(status="requires_continuation")
    with pytest.raises(ChunkFailedError) as caught:
        client("p")
    assert "continuation" in str(caught.value)
    assert client.last_metrics["error_type"] == "needs_continuation"


def test_dropped_connection_is_a_retryable_transport_error(client):
    import http.client as http_client
    def drop(request, timeout=None):
        raise http_client.RemoteDisconnected("dropped")
    client.opener.open = drop
    with pytest.raises(ProviderError) as caught:
        client("p")
    assert caught.value.status_code == 503
    assert client.last_metrics["error_type"] == "transport_or_timeout"


def test_cut_off_chunk_fails_but_the_run_continues(client, server, tmp_path):
    from app.extraction.runner import RunLimits, run_extraction
    from tests.test_extraction import make_chunk
    url, state = server
    state["body"] = gemini_response(status="incomplete")
    ontology = load_ontology("configs/ontology.json")
    summary = run_extraction([make_chunk(1), make_chunk(2)], client, ontology,
                             client.model_name, str(tmp_path / "out"),
                             RunLimits(max_calls=10))
    # Each chunk got exactly one attempt: no retry with an identical request,
    # and the first chunk's failure did not stop the run.
    assert len(state["requests"]) == 2
    assert summary["failed"] == 2
    assert summary["calls_used"] == 2
    assert summary["stopped_early"] is False


def test_missing_key_is_refused_without_leaking(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ontology = load_ontology("configs/ontology.json")
    with pytest.raises(ValueError) as caught:
        GeminiClient(ontology)
    assert FAKE_KEY not in str(caught.value)


def test_recipe_changes_with_model_and_generation_limit(client):
    ontology = load_ontology("configs/ontology.json")
    other_model = GeminiClient(ontology, model_id="other-gemini-model", assume_free=True,
                               max_requests_per_minute=10, max_tokens_per_minute=1000)
    other_limit = GeminiClient(ontology, max_output_tokens=256, assume_free=True,
                               max_requests_per_minute=10, max_tokens_per_minute=1000)
    assert client.model_name != other_model.model_name
    assert client.model_name != other_limit.model_name
    assert FAKE_KEY not in client.model_name


def test_adapter_runner_and_jsonl_connected(client, tmp_path):
    from app.extraction.runner import RunLimits, run_extraction
    from app.extraction.runner import JsonlEvents
    from tests.test_extraction import make_chunk
    path = tmp_path / "events.jsonl"
    summary = run_extraction([make_chunk(1)], client, load_ontology("configs/ontology.json"),
                             client.model_name, str(tmp_path / "out"), RunLimits(max_calls=5),
                             log_event=JsonlEvents(path, "run-test"))
    assert summary["valid"] == 1  # empty relations list is a valid empty answer
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert rows[0]["model"].startswith("gemini:")
    assert rows[0]["metrics"]["prompt_tokens"] == 30
    assert rows[0]["metrics"]["cost_basis"] == "assumed_free_unverified"
    assert rows[-1]["empty"] is True
