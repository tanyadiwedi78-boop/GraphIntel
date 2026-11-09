"""Provider router tests: fake clients only. No network, no real API key.
"""
import json

import pytest

from app.extraction.client import ChunkFailedError, ProviderError
from app.extraction.router import ProviderRouter
from app.extraction.runner import RunLimits, run_extraction
from app.graph.ontology import load_ontology
from tests.fake_model import good_answer

QUOTE = "Acme develops the Widget Engine"


class FakeClient:
    """A pretend provider client with the contract the router routes between."""

    def __init__(self, model_name, provider, steps):
        self.model_name = model_name
        self.provider = provider
        self.steps = steps
        self.calls = []
        self.last_metrics = {}

    def cost_mode(self):
        return "test_cost_mode"

    def __call__(self, prompt):
        self.calls.append(prompt)
        self.last_metrics = {"provider": self.provider, "error_type": None}
        position = len(self.calls) - 1
        if position >= len(self.steps):
            raise AssertionError("FakeClient " + self.provider + " called more times than planned")
        step = self.steps[position]
        if isinstance(step, ProviderError):
            self.last_metrics["error_type"] = "http_" + str(step.status_code)
            raise step
        return step


def make_chunk(number):
    return {
        "chunk_id": f"chunk-{number}",
        "text": f"Item {number}. {QUOTE} for\n customers.",
        "accession": "0000000000-26-000001",
        "form": "10-K",
        "period": "2026-01-31",
    }


def quick_limits(**changes):
    values = dict(max_calls=100, max_attempts_per_chunk=4, base_wait_seconds=1.0,
                  max_wait_seconds=60.0, jitter_fraction=0.0, stop_after_failed_chunks=5)
    values.update(changes)
    return RunLimits(**values)


def test_router_needs_at_least_one_client():
    with pytest.raises(ValueError):
        ProviderRouter([])


def test_single_client_answers_and_delegates_identity():
    gemini = FakeClient("gemini:model-x:tag", "gemini", ["answer"])
    router = ProviderRouter([gemini])
    assert router("prompt") == "answer"
    assert router.model_name == "gemini:model-x:tag"
    assert router.cost_mode() == "test_cost_mode"
    assert router.route_events == []


def test_first_short_rate_limit_stays_on_the_same_provider():
    """A per-minute 429 clears after a small wait: never route on the first blip."""
    primary = FakeClient("gemini:model-x:tag", "gemini",
                         [ProviderError(429, "rate limited", 5.0), "recovered"])
    fallback = FakeClient("openai:model-y:tag", "openai", ["should not be served yet"])
    router = ProviderRouter([primary, fallback])
    with pytest.raises(ProviderError):
        router("prompt")
    assert router.route_events == []
    assert router.model_name == "gemini:model-x:tag"
    assert fallback.calls == []
    # The runner waits with backoff and retries: the same provider recovered.
    assert router("prompt") == "recovered"
    assert router.route_events == []
    assert fallback.calls == []


def test_repeated_provider_wide_failure_fails_over():
    """A persistent outage routes after failover_after eligible errors in a row."""
    primary = FakeClient("gemini:model-x:tag", "gemini",
                         [ProviderError(429, "rate limited", 5.0),
                          ProviderError(429, "still limited", 5.0)])
    fallback = FakeClient("groq:model-y:tag", "groq", ["answer from fallback"])
    router = ProviderRouter([primary, fallback])
    with pytest.raises(ProviderError):
        router("prompt")
    assert router.route_events == []
    with pytest.raises(ProviderError):
        router("prompt")
    assert router.route_events == [{
        "from_model": "gemini:model-x:tag",
        "to_model": "groq:model-y:tag",
        "reason": "provider-wide failure 2 times in a row",
    }]
    # The failed attempt's metrics say where the run moved to, and last_metrics
    # still belongs to the client that FAILED, not the one taking over.
    assert primary.last_metrics["failover_to"] == "groq:model-y:tag"
    assert router.last_metrics["provider"] == "gemini"
    # The next attempt lands on the fallback.
    assert router("prompt") == "answer from fallback"
    assert router.model_name == "groq:model-y:tag"
    assert fallback.last_metrics["served_after_failover"] is True
    # Failover is one-way for the rest of the run: no probing back.
    assert router.active_index == 1


def test_long_requested_wait_fails_over_at_once():
    """A wait longer than the run accepts (daily quota) is a confirmed outage."""
    primary = FakeClient("gemini:model-x:tag", "gemini",
                         [ProviderError(429, "daily quota spent", 3600.0)])
    fallback = FakeClient("groq:model-y:tag", "groq", ["answer from fallback"])
    router = ProviderRouter([primary, fallback], max_wait_seconds=60.0)
    with pytest.raises(ProviderError):
        router("prompt")
    assert len(router.route_events) == 1
    assert "wait longer than 60" in router.route_events[0]["reason"]
    assert router.model_name == "groq:model-y:tag"


def test_success_resets_the_failure_count():
    primary = FakeClient("gemini:model-x:tag", "gemini",
                         [ProviderError(429, "blip", 5.0), "fine",
                          ProviderError(429, "another blip", 5.0)])
    fallback = FakeClient("groq:model-y:tag", "groq", ["should not be served"])
    router = ProviderRouter([primary, fallback])
    with pytest.raises(ProviderError):
        router("prompt")
    assert router("prompt") == "fine"
    with pytest.raises(ProviderError):
        router("prompt")
    assert router.route_events == []
    assert fallback.calls == []


def test_server_error_and_timeout_are_route_eligible():
    for status in [500, 503, 408]:
        primary = FakeClient("p", "gemini", [ProviderError(status, "down"),
                                             ProviderError(status, "still down")])
        fallback = FakeClient("f", "groq", ["ok"])
        router = ProviderRouter([primary, fallback])
        with pytest.raises(ProviderError):
            router("prompt")
        assert router.route_events == []  # first failure never routes
        with pytest.raises(ProviderError):
            router("prompt")
        assert len(router.route_events) == 1
        assert router.model_name == "f"


def test_setup_problems_never_fail_over():
    for status in [401, 402, 403, 404]:
        primary = FakeClient("p", "gemini", [ProviderError(status, "setup problem"),
                                             ProviderError(status, "setup problem")])
        fallback = FakeClient("f", "openai", ["should never be served"])
        router = ProviderRouter([primary, fallback])
        with pytest.raises(ProviderError) as info:
            router("prompt")
        with pytest.raises(ProviderError):
            router("prompt")
        assert info.value.status_code == status
        assert fallback.calls == []
        assert router.route_events == []
        assert router.model_name == "p"


def test_content_failure_never_fails_over():
    primary = FakeClient("p", "gemini", [ChunkFailedError(400, "answer cut off")])
    fallback = FakeClient("f", "openai", ["should never be served"])
    router = ProviderRouter([primary, fallback])
    with pytest.raises(ChunkFailedError):
        router("prompt")
    assert fallback.calls == []
    assert router.route_events == []


def test_no_fallback_left_reraises_without_routing():
    primary = FakeClient("p", "gemini", [ProviderError(429, "rate limited"),
                                         ProviderError(429, "still limited")])
    router = ProviderRouter([primary])
    with pytest.raises(ProviderError):
        router("prompt")
    with pytest.raises(ProviderError):
        router("prompt")
    assert router.route_events == []
    assert router.model_name == "p"


def test_failover_run_records_each_attempts_provider_and_skips_on_rerun(tmp_path):
    """Mid-chunk failover: every attempt names its provider, the saved record
    names the fallback that produced it, and a rerun makes ZERO calls."""
    ontology = load_ontology("configs/ontology.json")
    primary = FakeClient("gemini:model-x:tag", "gemini",
                         [ProviderError(429, "rate limited", 5.0),
                          ProviderError(429, "still limited", 5.0)])
    fallback = FakeClient("groq:model-y:tag", "groq", [good_answer(QUOTE)])
    router = ProviderRouter([primary, fallback])
    events = []
    summary = run_extraction([make_chunk(1)], router, ontology, router.model_name,
                             str(tmp_path), quick_limits(), lambda seconds: None,
                             log_event=events.append)
    assert summary["valid"] == 1
    # Attempts 1 and 2 failed on the primary; attempt 3 was served by the fallback.
    attempts = []
    for event in events:
        if event["event"] == "attempt":
            attempts.append(event)
    assert len(attempts) == 3
    assert attempts[0]["model"] == "gemini:model-x:tag"
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["metrics"]["provider"] == "gemini"
    assert attempts[1]["model"] == "gemini:model-x:tag"
    assert attempts[1]["status"] == "failed"
    assert attempts[1]["metrics"]["failover_to"] == "groq:model-y:tag"
    assert attempts[2]["model"] == "groq:model-y:tag"
    assert attempts[2]["status"] == "valid"
    finals = []
    for event in events:
        if event["event"] == "chunk_final":
            finals.append(event)
    assert finals[0]["model"] == "groq:model-y:tag"
    assert finals[0]["status"] == "valid"
    # The saved record names the provider that actually produced it.
    saved_files = list(tmp_path.glob("chunk-1__*.json"))
    assert len(saved_files) == 1
    record = json.loads(saved_files[0].read_text(encoding="utf-8"))
    assert record["model"] == "groq:model-y:tag"
    assert record["status"] == "valid"

    # Rerun with the ORIGINAL primary only: the chunk is done, whichever
    # provider did it. Zero calls, skipped as accepted from the fallback.
    primary_again = FakeClient("gemini:model-x:tag", "gemini", ["should not be called"])
    rerun_events = []
    rerun = run_extraction([make_chunk(1)], primary_again, ontology, primary_again.model_name,
                           str(tmp_path), quick_limits(), lambda seconds: None,
                           log_event=rerun_events.append)
    assert rerun["skipped_done"] == 1
    assert rerun["calls_used"] == 0
    assert primary_again.calls == []
    skip_event = rerun_events[0]
    assert skip_event["status"] == "skipped"
    assert skip_event["accepted_from_model"] == "groq:model-y:tag"
