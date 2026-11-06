"""Attempt and final-chunk events, including retries, reuse and zero cap."""
import json

from app.extraction.client import ProviderError
from app.extraction.runner import JsonlEvents, RunLimits, run_extraction
from app.graph.ontology import load_ontology
from tests.fake_model import FakeModel, good_answer
from tests.test_extraction import make_chunk, QUOTE


def run_with(model, tmp_path, **limits):
    events = []
    summary = run_extraction([make_chunk(1)], model, load_ontology("configs/ontology.json"),
                             "event-test", str(tmp_path), RunLimits(**limits),
                             sleep=no_wait, log_event=events.append)
    return summary, events


def no_wait(seconds):
    pass


def test_invalid_then_valid_attempts_and_final(tmp_path):
    summary, events = run_with(FakeModel(["bad", good_answer(QUOTE)]), tmp_path)
    assert summary["valid"] == 1
    assert len(events) == 3
    assert events[0]["event"] == "attempt"
    assert events[0]["status"] == "invalid"
    assert events[1]["status"] == "valid"
    assert events[-1]["retries"] == 1
    assert events[-1]["valid_first_try"] is False
    assert events[-1]["relationships"] == 1
    assert events[-1]["entities"] == 2
    assert events[-1]["document_id"] == "0000000000-26-000001"
    assert events[-1]["section"] is None


def test_reuse_emits_no_model_attempt_and_no_new_edge_counts(tmp_path):
    run_with(FakeModel([good_answer(QUOTE)]), tmp_path)
    summary, events = run_with(FakeModel([]), tmp_path)
    assert summary["calls_used"] == 0
    assert len(events) == 1
    assert events[0]["status"] == "skipped"
    assert events[0]["result_reused"] is True
    assert events[0]["valid_first_try"] is None
    assert "relationships" not in events[0]


def test_zero_cap_logs_stop_without_attempt(tmp_path):
    summary, events = run_with(FakeModel([]), tmp_path, max_calls=0)
    assert summary["calls_used"] == 0
    assert len(events) == 1
    assert events[0]["event"] == "chunk_final"
    assert events[0]["status"] == "stopped"


def test_setup_error_emits_attempt_and_stopped_final(tmp_path):
    summary, events = run_with(FakeModel([ProviderError(400)]), tmp_path)
    assert summary["stopped_early"] is True
    assert len(events) == 2
    assert events[0]["metrics"]["error_type"] == "http_400"
    assert events[-1]["status"] == "stopped"


def test_jsonl_writer_does_not_log_prompts(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = JsonlEvents(path, "test-run")
    writer({"event": "attempt", "prompt_tokens": None})
    rows = path.read_text().splitlines()
    assert len(rows) == 1
    saved = json.loads(rows[0])
    assert saved["run_id"] == "test-run"
    assert saved["prompt_tokens"] is None
    assert "prompt" not in saved


def test_strict_validator_rejects_nonfinite_confidence_and_unhashable_types():
    from app.extraction.validate import validate_extraction
    ontology = load_ontology("configs/ontology.json")
    bad = json.loads(good_answer(QUOTE))
    bad["relationships"][0]["confidence"] = float("nan")
    assert not validate_extraction(json.dumps(bad), make_chunk(1), ontology)[0]
    bad = json.loads(good_answer(QUOTE))
    bad["relationships"][0]["relation"] = []
    assert not validate_extraction(json.dumps(bad), make_chunk(1), ontology)[0]
