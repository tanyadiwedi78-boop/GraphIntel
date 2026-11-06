"""Run and chunk timing: fake clock, no network, no real waiting."""
import pytest

from app.extraction.client import ProviderError
from app.extraction.runner import RunLimits, run_extraction
from app.graph.ontology import load_ontology
from tests.fake_model import FakeModel, good_answer
from tests.test_extraction import QUOTE, make_chunk, no_sleep_recorder


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now = self.now + seconds


class SlowModel:
    """Each call advances the fake clock by a fixed amount."""

    def __init__(self, clock, seconds_per_call, steps):
        self.clock = clock
        self.seconds_per_call = seconds_per_call
        self.inner = FakeModel(steps)

    def __call__(self, prompt):
        self.clock.advance(self.seconds_per_call)
        return self.inner(prompt)


def quick_limits(**changes):
    values = dict(max_calls=100, max_attempts_per_chunk=3, base_wait_seconds=1.0,
                  max_wait_seconds=60.0, jitter_fraction=0.0, stop_after_failed_chunks=5)
    values.update(changes)
    return RunLimits(**values)


def run_with_clock(chunks, model, tmp_path, clock, **limits):
    events = []
    sleep, waits = no_sleep_recorder()
    summary = run_extraction(chunks, model, load_ontology("configs/ontology.json"),
                             "timing-test", str(tmp_path), quick_limits(**limits),
                             sleep=sleep, clock=clock, log_event=events.append)
    return summary, events, waits


def test_run_and_chunk_times_come_from_the_clock(tmp_path):
    clock = FakeClock()
    model = SlowModel(clock, 0.4, [good_answer(QUOTE), good_answer(QUOTE)])
    summary, events, _ = run_with_clock([make_chunk(1), make_chunk(2)], model, tmp_path, clock)

    assert summary["valid"] == 2
    assert summary["processed_chunks"] == 2
    assert summary["elapsed_seconds"] == pytest.approx(0.8)
    assert summary["chunks_per_minute"] == pytest.approx(150.0)
    assert summary["chunk_median_seconds"] == pytest.approx(0.4)
    assert summary["chunk_p95_seconds"] == pytest.approx(0.4)
    finals = [e for e in events if e["event"] == "chunk_final"]
    assert [f["chunk_seconds"] for f in finals] == [pytest.approx(0.4), pytest.approx(0.4)]
    assert [f["retry_wait_seconds"] for f in finals] == [0.0, 0.0]


def test_retry_backoff_is_tracked_separately(tmp_path):
    clock = FakeClock()
    model = SlowModel(clock, 0.1, [ProviderError(429), good_answer(QUOTE)])
    summary, events, waits = run_with_clock([make_chunk(1)], model, tmp_path, clock)

    assert waits == [1.0]                      # backoff after the 429: base 1.0 s, no jitter
    assert summary["retry_wait_seconds"] == pytest.approx(1.0)
    final = [e for e in events if e["event"] == "chunk_final"][0]
    assert final["retry_wait_seconds"] == pytest.approx(1.0)
    attempt = [e for e in events if e["event"] == "attempt"][0]
    assert attempt["metrics"]["error_type"] == "http_429"
    assert "pacing" not in final               # provider pacing stays in per-attempt metrics only


def test_skipped_chunks_do_not_count_as_fresh_speed(tmp_path):
    clock = FakeClock()
    model = SlowModel(clock, 0.5, [good_answer(QUOTE)])
    run_with_clock([make_chunk(1)], model, tmp_path, clock)

    clock2 = FakeClock()
    model2 = SlowModel(clock2, 0.5, [good_answer(QUOTE)])
    summary, events, _ = run_with_clock([make_chunk(1), make_chunk(2)], model2, tmp_path, clock2)

    assert summary["skipped_done"] == 1
    assert summary["processed_chunks"] == 1
    assert summary["chunks_per_minute"] == pytest.approx(120.0)   # 1 fresh chunk in 0.5 s
    skipped = [e for e in events if e.get("status") == "skipped"][0]
    assert "chunk_seconds" not in skipped


def test_early_stop_still_reports_elapsed(tmp_path):
    clock = FakeClock()
    summary, events, _ = run_with_clock([make_chunk(1)], FakeModel([]), tmp_path, clock, max_calls=0)
    assert summary["stopped_early"] is True
    assert summary["elapsed_seconds"] == 0.0
    assert summary["processed_chunks"] == 0
    assert summary["chunk_median_seconds"] is None
    assert summary["chunk_p95_seconds"] is None
    assert summary["chunks_per_minute"] == 0.0


def test_median_and_p95_over_several_fresh_chunks(tmp_path):
    clock = FakeClock()
    speeds = [0.1, 0.2, 0.3, 0.4]
    model = SlowModel(clock, 0.0, [])
    model.inner = FakeModel([good_answer(QUOTE)] * len(speeds))
    planned = iter(speeds)

    def variable_call(prompt):
        clock.advance(next(planned))
        return model.inner(prompt)

    events = []
    sleep, _ = no_sleep_recorder()
    summary = run_extraction([make_chunk(i + 1) for i in range(len(speeds))], variable_call,
                             load_ontology("configs/ontology.json"), "timing-test", str(tmp_path),
                             quick_limits(), sleep=sleep, clock=clock, log_event=events.append)
    assert summary["chunk_median_seconds"] == pytest.approx(0.25)  # mean of 0.2 and 0.3
    assert summary["chunk_p95_seconds"] == pytest.approx(0.4)      # nearest-rank over 4 values
