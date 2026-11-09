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
    # Within this chunk the confirmed outage stays on fallback. A later
    # chunk will prefer primary unless its circuit has opened.
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
                           log_event=rerun_events.append, approved_models=router.approved_models)
    assert rerun["skipped_done"] == 1
    assert rerun["calls_used"] == 0
    assert primary_again.calls == []
    skip_event = rerun_events[0]
    assert skip_event["status"] == "skipped"
    assert skip_event["accepted_from_model"] == "groq:model-y:tag"


def test_request_failover_without_a_fallback_returns_false():
    only = FakeClient("gemini:model-x:tag", "gemini", ["unused"])
    router = ProviderRouter([only])
    assert router.request_failover("schema_invalid_after_repair") is False
    assert router.active_index == 0
    assert router.route_events == []


def test_request_failover_switches_provider_and_records_the_reason():
    primary = FakeClient("gemini:model-x:tag", "gemini", ["unused"])
    fallback = FakeClient("openai:model-y:tag", "openai", ["unused"])
    router = ProviderRouter([primary, fallback])
    assert router.request_failover("schema_invalid_after_repair") is True
    assert router.active_index == 1
    assert router.model_name == "openai:model-y:tag"
    assert router.route_events == [{
        "from_model": "gemini:model-x:tag",
        "to_model": "openai:model-y:tag",
        "reason": "schema_invalid_after_repair",
    }]


def test_schema_invalid_after_one_repair_routes_to_the_fallback(tmp_path):
    """Her fallback_on policy: an answer that fails validation, then fails the
    ONE repair attempt too, moves the chunk to the fallback - which faces the
    same prompt, schema and validation. Attempts stay exactly counted:
    1 base + 1 repair on the primary, then the fallback's first try."""
    ontology = load_ontology("configs/ontology.json")
    primary = FakeClient("gemini:model-x:tag", "gemini", ["bad answer", "still bad"])
    fallback = FakeClient("openai:model-y:tag", "openai", [good_answer(QUOTE)])
    router = ProviderRouter([primary, fallback])
    events = []
    summary = run_extraction([make_chunk(1)], router, ontology, router.model_name,
                             str(tmp_path), quick_limits(max_attempts_per_chunk=3),
                             lambda seconds: None, log_event=events.append)
    assert summary["valid"] == 1
    assert len(primary.calls) == 2   # base + 1 repair, never more
    assert len(fallback.calls) == 1
    assert fallback.calls[0] == primary.calls[0]  # fallback gets the SAME base prompt
    attempts = []
    for event in events:
        if event["event"] == "attempt":
            attempts.append(event)
    assert len(attempts) == 3
    assert attempts[0]["model"] == "gemini:model-x:tag"
    assert attempts[0]["status"] == "invalid"
    assert attempts[0]["retry_with_feedback"] is False
    assert attempts[1]["model"] == "gemini:model-x:tag"
    assert attempts[1]["status"] == "invalid"
    assert attempts[1]["retry_with_feedback"] is True   # the repair attempt
    assert attempts[2]["model"] == "openai:model-y:tag"
    assert attempts[2]["status"] == "valid"
    assert attempts[2]["retry_with_feedback"] is False  # fresh base prompt on the fallback
    assert router.route_events[-1]["reason"] == "schema_invalid_after_repair"
    saved_files = list(tmp_path.glob("chunk-1__*.json"))
    assert len(saved_files) == 1
    record = json.loads(saved_files[0].read_text(encoding="utf-8"))
    assert record["model"] == "openai:model-y:tag"
    assert record["status"] == "valid"
    assert record["fallback_reason"] == "schema_invalid_after_repair"
    assert record["primary_error_type"] == "schema_invalid"


def test_schema_invalid_after_repair_without_fallback_keeps_repairing(tmp_path):
    """With no fallback there is nowhere to route: the chunk keeps repairing
    on the same provider until the attempt cap, exactly as before."""
    ontology = load_ontology("configs/ontology.json")
    only = FakeClient("gemini:model-x:tag", "gemini", ["bad", "bad", good_answer(QUOTE)])
    router = ProviderRouter([only])
    summary = run_extraction([make_chunk(1)], router, ontology, router.model_name,
                             str(tmp_path), quick_limits(max_attempts_per_chunk=3),
                             lambda seconds: None)
    assert summary["valid"] == 1
    assert len(only.calls) == 3
    assert router.route_events == []


def test_both_providers_blocked_parks_chunks_and_replay_recovers(tmp_path):
    """Her failure test, mocked: every provider call fails provider-wide, so
    every chunk is parked pending (dead letter) and the run continues. A rerun
    against a healthy provider recovers all of them - nothing is lost."""
    ontology = load_ontology("configs/ontology.json")
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    # Chunk 1 fails over to the fallback on its second attempt; chunks 2 and
    # 3 start on the fallback directly. Every call fails provider-wide.
    primary = FakeClient("gemini:model-x:tag", "gemini", [ProviderError(503)] * 6)
    fallback = FakeClient("openai:model-y:tag", "openai", [ProviderError(503)] * 4)
    router = ProviderRouter([primary, fallback])
    summary = run_extraction(chunks, router, ontology, router.model_name,
                             str(tmp_path), quick_limits(max_attempts_per_chunk=2),
                             lambda seconds: None)
    assert len(primary.calls) == 4
    assert len(fallback.calls) == 2
    assert summary["valid"] == 0
    assert summary["failed"] == 3
    assert summary["parked_transient"] == 3
    assert summary["stopped_early"] is False
    healthy = FakeClient("gemini:model-x:tag", "gemini", [good_answer(QUOTE)] * 3)
    replay = run_extraction(chunks, healthy, ontology, healthy.model_name,
                            str(tmp_path), quick_limits(), lambda seconds: None)
    assert replay["valid"] == 3
    assert replay["parked_transient"] == 0


class ManualClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def opened_primary(recovery):
    clock = ManualClock()
    primary = FakeClient("gemini:p:recipe", "gemini", [ProviderError(503)] * 5 + recovery)
    fallback = FakeClient("openai:f:recipe", "openai", ["fallback"] * 20)
    router = ProviderRouter([primary, fallback], clock=clock)
    for number in range(5):
        router.begin_chunk()
        with pytest.raises(ProviderError):
            router("prompt")
    return router, primary, fallback, clock


def test_provider_circuit_counts_failed_calls_not_parked_chunks():
    router, primary, fallback, clock = opened_primary(["recovered"])
    assert len(primary.calls) == 5
    assert router.failures == [5, 0]
    assert len(router.circuit_events) == 1
    assert router.circuit_events[0]["consecutive_failures"] == 5
    assert router.open_until[0] == 220.0
    assert router.open_until[1] == 0.0


def test_open_primary_is_skipped_while_fallback_serves_without_sleep():
    router, primary, fallback, clock = opened_primary(["recovered"])
    for number in range(3):
        router.begin_chunk()
        assert router.prepare_call() == 0
        assert router("prompt") == "fallback"
    assert len(primary.calls) == 5
    assert len(fallback.calls) == 3
    assert router.failures[0] == 5  # fallback success cannot reset primary
    clock.advance(119.999)
    router.begin_chunk()
    assert router("prompt") == "fallback"
    assert len(primary.calls) == 5


def test_half_open_primary_returns_at_exact_cooldown_and_success_resets():
    router, primary, fallback, clock = opened_primary(["recovered"])
    clock.advance(120)
    router.begin_chunk()
    assert router.model_name == primary.model_name
    assert router("prompt") == "recovered"
    assert router.failures[0] == 0
    assert router.open_until[0] == 0
    assert router.half_open[0] is False


def test_failed_half_open_probe_reopens_only_primary():
    router, primary, fallback, clock = opened_primary([ProviderError(503)])
    clock.advance(120)
    router.prepare_call()
    with pytest.raises(ProviderError):
        router("prompt")
    assert router.open_until[0] == 340.0
    assert router.open_until[1] == 0
    assert router("prompt") == "fallback"
    assert len(router.circuit_events) == 2


def test_fallback_circuit_does_not_clear_or_open_primary():
    clock = ManualClock()
    primary = FakeClient("gemini:p:r", "gemini", ["primary"])
    fallback = FakeClient("openai:f:r", "openai", [ProviderError(503)] * 5)
    router = ProviderRouter([primary, fallback], clock=clock)
    router.active_index = 1
    for number in range(5):
        with pytest.raises(ProviderError):
            router("prompt")
    assert router.open_until == [0.0, 220.0]
    router.begin_chunk()
    assert router("prompt") == "primary"


def test_both_circuits_open_awaits_earliest_only():
    clock = ManualClock()
    primary = FakeClient("gemini:p:r", "gemini", ["recovered"])
    fallback = FakeClient("openai:f:r", "openai", [])
    router = ProviderRouter([primary, fallback], clock=clock)
    router.open_until = [220.0, 240.0]
    assert router.prepare_call() == 120.0
    assert primary.calls == fallback.calls == []
    clock.advance(120)
    assert router.prepare_call() == 0
    assert router("prompt") == "recovered"


def test_runner_serves_fallback_through_primary_cooldown_no_global_sleep(tmp_path):
    clock = ManualClock()
    primary = FakeClient("gemini:p:r", "gemini", [ProviderError(503)] * 5)
    fallback = FakeClient("openai:f:r", "openai", [good_answer(QUOTE)] * 8)
    router = ProviderRouter([primary, fallback], clock=clock)
    waits = []
    events = []
    summary = run_extraction([make_chunk(i) for i in range(8)], router,
                            load_ontology("configs/ontology.json"), router.model_name,
                            str(tmp_path), quick_limits(), waits.append,
                            clock=clock, log_event=events.append)
    assert summary["valid"] == 8
    assert len(primary.calls) == 5
    assert len(fallback.calls) == 8
    assert summary["circuit_trips"] == 1
    assert 120.0 not in waits
    assert summary["parked_transient"] == 0
    for path in tmp_path.glob("*.json"):
        record = json.loads(path.read_text())
        assert record["model"] == fallback.model_name


def test_each_provider_gets_its_own_retry_budget_and_saved_attempt_aliases(tmp_path):
    primary = FakeClient("gemini:p:r", "gemini", [ProviderError(503)] * 3)
    fallback = FakeClient("openai:f:r", "openai",
                          [ProviderError(503), ProviderError(503), good_answer(QUOTE)])
    router = ProviderRouter([primary, fallback], failover_after=3,
                            max_attempts={primary.model_name: 3, fallback.model_name: 3})
    summary = run_extraction([make_chunk(1)], router, load_ontology("configs/ontology.json"),
                            router.model_name, str(tmp_path), quick_limits(max_attempts_per_chunk=3),
                            lambda seconds: None)
    assert summary["valid"] == 1
    assert len(primary.calls) == len(fallback.calls) == 3
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["attempt_number"] == 6
    assert record["provider_used"] == "openai"
    assert record["valid_first_try"] is False
    assert len(record["attempt_records"]) == 6
    assert record["cost_usd"] is None
    assert record["tokens_in"] is None  # unknown is never fabricated as zero
    assert record["tokens_out"] is None
    for number, event in enumerate(record["attempt_records"], 1):
        assert event["attempt_number"] == number
        assert event["cost_usd"] is None
        assert "latency_ms" in event
        assert "tokens_in" in event
        assert "tokens_out" in event
    for edge in record["relationships"]:
        assert edge["provider"] == "openai"
        assert edge["model"] == fallback.model_name
        assert edge["prompt_version"] == record["prompt_version"]


def test_configured_fallback_cause_can_disable_schema_route():
    primary = FakeClient("gemini:p:r", "gemini", [])
    fallback = FakeClient("openai:f:r", "openai", [])
    router = ProviderRouter([primary, fallback], fallback_on=["rate_limit"])
    assert router.request_failover("schema_invalid_after_repair") is False
    assert router.active_index == 0


def test_configured_fallback_cause_can_disable_server_error_route():
    primary = FakeClient("gemini:p:r", "gemini", [ProviderError(503)] * 2)
    fallback = FakeClient("openai:f:r", "openai", [])
    router = ProviderRouter([primary, fallback], fallback_on=["rate_limit"])
    for number in range(2):
        with pytest.raises(ProviderError):
            router("prompt")
    assert router.route_events == []
    assert fallback.calls == []


def test_twenty_chunks_both_unavailable_park_and_replay_without_loss(tmp_path):
    clock = ManualClock()
    primary = FakeClient("gemini:p:r", "gemini", [ProviderError(503)] * 100)
    fallback = FakeClient("openai:f:r", "openai", [ProviderError(503)] * 100)
    router = ProviderRouter([primary, fallback], failover_after=3, clock=clock,
                            max_attempts={primary.model_name: 3, fallback.model_name: 3})
    chunks = [make_chunk(i) for i in range(20)]
    summary = run_extraction(chunks, router, load_ontology("configs/ontology.json"),
                            router.model_name, str(tmp_path), quick_limits(max_calls=200),
                            clock.advance, clock=clock)
    assert summary["parked_transient"] == 20
    assert summary["stopped_early"] is False
    assert summary["calls_used"] <= 120
    assert len(list(tmp_path.glob("*.json"))) >= 20
    healthy = FakeClient("gemini:p:r", "gemini", [good_answer(QUOTE)] * 20)
    replay = run_extraction(chunks, healthy, load_ontology("configs/ontology.json"),
                           healthy.model_name, str(tmp_path), quick_limits(), clock.advance)
    assert replay["valid"] == 20
    assert replay["parked_transient"] == 0


def test_rejected_repairs_are_eligible_schema_failures_and_valid_resets():
    clock = ManualClock()
    primary = FakeClient("gemini:p:r", "gemini", ["bad"] * 10 + ["good"])
    fallback = FakeClient("openai:f:r", "openai", [])
    router = ProviderRouter([primary, fallback], clock=clock)
    for number in range(5):
        router.begin_chunk()
        router("base")
        router.validated_response(False)
        router("repair")
        router.validated_response(False, repair_exhausted=True)
        router.request_failover("schema_invalid_after_repair")
    assert router.failures[0] == 5
    assert router.open_until[0] == 220
    clock.advance(120)
    router.begin_chunk()
    router("base")
    router.validated_response(True)
    assert router.failures[0] == 0
